//! The HVM runtime.
//!
//! This file can be divided into three major logical components:
//! - the **allocator**, which manages the creation and destruction of nodes in
//!   the net
//! - the **linker**, which links ports and wires in the interaction net, in a
//!   thread-safe way
//! - the **interactions**, which define the interaction system used in HVM
//!   (i.e. the agents and their interaction rules)
//!
//! The memory layout is documented within the code, but at a high level:
//! - references into the net are represented by [`Port`]s and [`Wire`]s, which
//!   are often tagged pointers into nodes managed by the allocator
//! - nilary agents are *unboxed* -- they have no backing allocation -- and
//!   their data is stored inline in their principal `Port`
//! - other agents are backed by allocated [`Node`]s, which store the targets of
//!   the *auxiliary ports* of the net (as managed by the linker); the target of
//!   the principal port is left implicit
//! - active pairs are thus stored in a dedicated vector, `net.redexes`

use crate::{
  ops::Op,
  trace,
  trace::Tracer,
  util::{bi_enum, deref},
};
use nohash_hasher::{IntMap, IsEnabled};
use std::{
  alloc::{self, Layout},
  any::{Any, TypeId},
  borrow::Cow,
  fmt,
  hint::unreachable_unchecked,
  marker::PhantomData,
  mem::size_of,
  ops::{Deref, DerefMut},
  sync::{Arc, Barrier},
  thread,
};

#[cfg(feature = "_fuzz")]
use crate::fuzz as atomic;
#[cfg(not(feature = "_fuzz"))]
use std::sync::atomic;

#[cfg(feature = "_fuzz")]
use crate::fuzz::spin_loop;
#[cfg(not(feature = "_fuzz"))]
fn spin_loop() {} // this could use `std::hint::spin_loop`, but in practice it hurts performance

use atomic::{AtomicU64, AtomicUsize, Ordering::Relaxed};

use Tag::*;

mod addr;
mod allocator;
mod def;
mod instruction;
mod interact;
mod linker;
mod net;
mod node;
mod parallel;
mod port;
mod wire;

pub use addr::*;
pub use allocator::*;
pub use def::*;
pub use instruction::*;
pub use linker::*;
pub use net::*;
pub use node::*;
pub use port::*;
pub use wire::*;

pub type Lab = u16;

/// The runtime mode is represented with a generic such that, instead of
/// repeatedly branching on the mode at runtime, the branch can happen at the
/// top-most level, and delegate to monomorphized functions specialized for each
/// particular mode.
///
/// This trait is `unsafe` as it may only be implemented by [`Strict`] and
/// [`Lazy`].
pub unsafe trait Mode: Send + Sync + 'static {
  const LAZY: bool;
}

/// In strict mode, all active pairs are expanded.
pub struct Strict;
unsafe impl Mode for Strict {
  const LAZY: bool = false;
}

/// In lazy mode, only active pairs that are reached from a walk from the root
/// port are expanded.
pub struct Lazy;
unsafe impl Mode for Lazy {
  const LAZY: bool = true;
}

/// Tracks the number of rewrites, categorized by type.
#[derive(Clone, Copy, Debug, Default)]
pub struct Rewrites<T = u64> {
  pub anni: T,
  pub comm: T,
  pub eras: T,
  pub dref: T,
  pub oper: T,
}

type AtomicRewrites = Rewrites<AtomicU64>;

impl Rewrites {
  pub fn add_to(&self, target: &AtomicRewrites) {
    target.anni.fetch_add(self.anni, Relaxed);
    target.comm.fetch_add(self.comm, Relaxed);
    target.eras.fetch_add(self.eras, Relaxed);
    target.dref.fetch_add(self.dref, Relaxed);
    target.oper.fetch_add(self.oper, Relaxed);
  }

  pub fn total(&self) -> u64 {
    self.anni + self.comm + self.eras + self.dref + self.oper
  }
}

impl AtomicRewrites {
  pub fn add_to(&self, target: &mut Rewrites) {
    target.anni += self.anni.load(Relaxed);
    target.comm += self.comm.load(Relaxed);
    target.eras += self.eras.load(Relaxed);
    target.dref += self.dref.load(Relaxed);
    target.oper += self.oper.load(Relaxed);
  }
}

/// A [`Net`] whose mode is determined dynamically, at runtime.
///
/// Use [`dispatch_dyn_net!`] to wrap operations on the inner net.
pub enum DynNet<'a> {
  Lazy(Net<'a, Lazy>),
  Strict(Net<'a, Strict>),
}

impl<'h> DynNet<'h> {
  pub fn new(heap: &'h Heap, lazy: bool) -> Self {
    if lazy { DynNet::Lazy(Net::new(heap)) } else { DynNet::Strict(Net::new(heap)) }
  }
}

#[macro_export]
macro_rules! dispatch_dyn_net {
  ($pat:pat = $expr:expr => $body:expr) => {
    match $expr {
      $crate::run::DynNet::Lazy($pat) => $body,
      $crate::run::DynNet::Strict($pat) => $body,
    }
  };
  ($net:ident => $body:expr) => {
    dispatch_dyn_net! { $net = $net => $body }
  };
  (mut $net:ident => $body:expr) => {
    dispatch_dyn_net! { mut $net = $net => $body }
  };
  (&$net:ident => $body:expr) => {
    dispatch_dyn_net! { $net = &$net => $body }
  };
  (&mut $net:ident => $body:expr) => {
    dispatch_dyn_net! { $net = &mut $net => $body }
  };
}
