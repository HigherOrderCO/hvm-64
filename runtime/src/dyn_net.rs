use super::{Heap, Lazy, Mode, Net, Strict};

use crate::prelude::*;

/// A [`Net`] whose mode is determined dynamically, at runtime.
///
/// Use [`dispatch_dyn_net!`] to wrap operations on the inner net.
pub type DynNet<'a> = DynNetInner<Net<'a, Lazy>, Net<'a, Strict>>;

/// A mutable reference to a [`Net`] whose mode is determined dynamically, at
/// runtime.
///
/// Use [`dispatch_dyn_net!`] to wrap operations on the inner net.
pub type DynNetMut<'r, 'h> = DynNetInner<&'r mut Net<'h, Lazy>, &'r mut Net<'h, Strict>>;

pub enum DynNetInner<L, S> {
  Lazy(L),
  Strict(S),
}

impl<'h> DynNet<'h> {
  pub fn new(heap: &'h Heap, lazy: bool) -> Self {
    if lazy { DynNet::Lazy(Net::new(heap)) } else { DynNet::Strict(Net::new(heap)) }
  }
}

impl<'r, 'h, M: Mode> From<&'r mut Net<'h, M>> for DynNetMut<'r, 'h> {
  fn from(value: &'r mut Net<'h, M>) -> Self {
    value.as_dyn_mut()
  }
}

impl<'h, M: Mode> Net<'h, M> {
  pub fn as_dyn_mut(&mut self) -> DynNetMut<'_, 'h> {
    if M::LAZY {
      DynNetMut::Lazy(unsafe { mem::transmute(self) })
    } else {
      DynNetMut::Strict(unsafe { mem::transmute(self) })
    }
  }

  pub fn into_dyn(self) -> DynNet<'h> {
    if M::LAZY {
      DynNet::Lazy(unsafe { mem::transmute(self) })
    } else {
      DynNet::Strict(unsafe { mem::transmute(self) })
    }
  }
}

#[macro_export]
macro_rules! dispatch_dyn_net {
  ($pat:pat = $expr:expr => $body:expr) => {
    match $expr {
      $crate::DynNetInner::Lazy($pat) => $body,
      $crate::DynNetInner::Strict($pat) => $body,
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
