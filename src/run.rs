// An efficient Interaction Combinator runtime
// ===========================================
// This file implements an efficient interaction combinator runtime. Nodes are represented by 2 aux
// ports (P1, P2), with the main port (P1) omitted. A separate vector, 'rdex', holds main ports,
// and, thus, tracks active pairs that can be reduced in parallel. Pointers are unboxed, meaning
// that Ptr::ERAs, NUMs and REFs don't use any additional space. REFs lazily expand to closed nets when
// they interact with nodes, and are cleared when they interact with Ptr::ERAs, allowing for constant
// space evaluation of recursive functions on Scott encoded datatypes.

use crate::{ops::Op, trace, trace::Tracer, util::bi_enum};
use std::{
  alloc::{self, Layout},
  any::TypeId,
  borrow::Cow,
  fmt,
  hint::unreachable_unchecked,
  ops::Deref,
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

// -------------------
//   Primitive Types
// -------------------

/// A port in the interaction net.
///
/// The bottom three bits of this value are the *tag*, which determines both
/// what kind of port it is (principal vs auxiliary, etc.), as well as the
/// semantics of the remainder of the bits of the value.
///
/// All tags other than `Num` divide the bits of the port as follows:
/// - the top 16 bits are the *label*, accessible with `.lab()`
/// - the middle 45 bits are the non-alignment bits of the *address*, an
///   8-byte-aligned pointer accessible with `.addr()`
/// - the bottom 3 bits are the tag, as always
///
/// See the documentation for each `Tag` as to the semantics of each
/// corresponding type of port.
#[derive(Clone, Eq, PartialEq, PartialOrd, Hash, Default)]
#[repr(transparent)]
#[must_use]
pub struct Port(pub u64);

bi_enum! {
  #[repr(u8)]
  #[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
  pub enum Tag {
    /// `Red` ports are an implementation detail of the atomic linking
    /// algorithm, and don't have a precise analogue in interaction nets.
    ///
    /// See the documentation for the linking algorithm for more.
    Red = 0,
    /// A `Var` port represents an auxiliary port in the net.
    ///
    /// The address of this port represents the wire leaving this port,
    /// accessible with `.wire()`.
    ///
    /// The label of this port is currently unused and always 0.
    Var = 1,
    /// A `Ref` port represents the principal port of a nilary reference node.
    ///
    /// The address of this port is a pointer to the corresponding `Def`.
    ///
    /// The label of this port is always equivalent to `def.labs.min_safe`, and
    /// is used as an optimization for the ref commutation interaction.
    ///
    /// Eraser nodes are represented by a null-pointer `Ref`, available as the
    /// constant `Port::ERA`.
    Ref = 2,
    /// A `Num` port represents the principal port of a U60 node.
    ///
    /// The top 60 bits of the port are the value of this node, and are
    /// accessible with `.num()`.
    ///
    /// The `0b1000` bit is currently unused in this port.
    Num = 3,
    /// An `Op2` port represents the principal port of an Op2 node.
    ///
    /// The label of this port is the corresponding operation, which can be
    /// accessed with `.op()`.
    ///
    /// The address of this port is the address of a two-word allocation,
    /// storing the targets of the wires connected to the two auxiliary ports of
    /// this node.
    Op2 = 4,
    /// An `Op1` port represents the principal port of an Op1 node.
    ///
    /// The label of this port is the corresponding operation, which can be
    /// accessed with `.op()`.
    ///
    /// The address of this port is the address of a two-word allocation. The
    /// first word in this allocation stores the first operand as a `Num` port,
    /// and the second word stores the target of the wire connected to the
    /// auxiliary port of this node.
    Op1 = 5,
    /// A `Mat` port represents the principal port of a Mat node.
    ///
    /// The address of this port is the address of a two-word allocation,
    /// storing the targets of the wires connected to the two auxiliary ports of
    /// the node.
    ///
    /// The label of this port is currently unused and always 0.
    Mat = 6,
    /// A `Ctr` port represents the principal port of an binary interaction
    /// combinator node.
    ///
    /// The label of this port is the label of the combinator; two combinators
    /// annihilate if they have the same label, or commute otherwise.
    ///
    /// The address of this port is the address of a two-word allocation,
    /// storing the targets of the wires connected to the two auxiliary ports of
    /// the node.
    Ctr = 7,
  }
}

use Tag::*;

pub type Lab = u16;

impl fmt::Debug for Port {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    write!(f, "{:016x?} ", self.0)?;
    match *self {
      Port::ERA => write!(f, "[ERA]"),
      Port::FREE => write!(f, "[FREE]"),
      Port::GONE => write!(f, "[GONE]"),
      Port::LOCK => write!(f, "[LOCK]"),
      _ => match self.tag() {
        Num => write!(f, "[Num {}]", self.num()),
        Var | Red | Mat => write!(f, "[{:?} {:?}]", self.tag(), self.addr()),
        Op2 | Op1 | Ctr | Ref => write!(f, "[{:?} {:?} {:?}]", self.tag(), self.lab(), self.addr()),
      },
    }
  }
}

impl Port {
  /// The principal port of an eraser node.
  pub const ERA: Port = Port(Ref as _);
  /// A sentinel value used to indicate free memory; see the allocator for more
  /// details.
  pub const FREE: Port = Port(0x8000_0000_0000_0000);
  /// A sentinel value used to lock a wire; see the linking algorithm for more
  /// details.
  pub const LOCK: Port = Port(0xFFFF_FFFF_FFFF_FFF0);
  /// A sentinel value used in the atomic linking algorithm; see it for more
  /// details.
  pub const GONE: Port = Port(0xFFFF_FFFF_FFFF_FFFF);

  /// Creates a new port with a given tag, label, and addr.
  #[inline(always)]
  pub fn new(tag: Tag, lab: Lab, addr: Addr) -> Self {
    Port(((lab as u64) << 48) | (addr.0 as u64) | (tag as u64))
  }

  /// Creates a new `Var` port with a given addr.
  #[inline(always)]
  pub fn new_var(addr: Addr) -> Self {
    Port::new(Var, 0, addr)
  }

  /// Creates a new `Num` port with a given 60-bit numeric value.
  #[inline(always)]
  pub const fn new_num(val: u64) -> Self {
    Port((val << 4) | (Num as u64))
  }

  /// Creates a new `Ref` port corresponding to a given definition.
  #[inline(always)]
  pub fn new_ref(def: &Def) -> Port {
    Port::new(Ref, def.labs.min_safe(), Addr(def as *const _ as _))
  }

  /// Accesses the tag of this port; this is valid for all ports.
  #[inline(always)]
  pub fn tag(&self) -> Tag {
    unsafe { ((self.0 & 0x7) as u8).try_into().unwrap_unchecked() }
  }

  /// Accesses the label of this port; this is valid for all non-`Num` ports.
  #[inline(always)]
  pub const fn lab(&self) -> Lab {
    (self.0 >> 48) as Lab
  }

  /// Accesses the addr of this port; this is valid for all non-`Num` ports.
  #[inline(always)]
  pub const fn addr(&self) -> Addr {
    Addr((self.0 & 0x0000_FFFF_FFFF_FFF8) as usize as _)
  }

  /// Accesses the operation of this port; this is valid for `Op1` and `Op2`
  /// ports.
  #[inline(always)]
  pub fn op(&self) -> Op {
    unsafe { self.lab().try_into().unwrap_unchecked() }
  }

  /// Accesses the numeric value of this port; this is valid for `Num` ports.
  #[inline(always)]
  pub const fn num(&self) -> u64 {
    self.0 >> 4
  }

  /// Accesses the wire leaving this port; this is valid for `Var` ports and
  /// non-sentinel `Red` ports.
  #[inline(always)]
  pub fn wire(&self) -> Wire {
    Wire::new(self.addr())
  }

  #[inline(always)]
  pub fn is_principal(&self) -> bool {
    self.tag() >= Ref
  }

  /// Given a principal port, returns whether this principal port may be part of
  /// a skippable active pair -- an active pair like `ERA-ERA` that does not
  /// need to be added to the redex list.
  #[inline(always)]
  pub fn is_skippable(&self) -> bool {
    matches!(self.tag(), Num | Ref)
  }

  /// Converts a `Var` port into a `Red` port with the same addr.
  #[inline(always)]
  fn redirect(&self) -> Port {
    Port::new(Red, 0, self.addr())
  }

  /// Converts a `Red` port into a `Var` port with the same addr.
  #[inline(always)]
  fn unredirect(&self) -> Port {
    Port::new(Var, 0, self.addr())
  }
}

pub struct TraverseNode {
  pub lab: Lab,
  pub p1: Wire,
  pub p2: Wire,
}

pub struct TraverseOp1 {
  pub op: Op,
  pub num: Port,
  pub p2: Wire,
}

impl Port {
  #[inline(always)]
  pub fn consume_node(self) -> TraverseNode {
    self.traverse_node()
  }

  #[inline(always)]
  pub fn traverse_node(self) -> TraverseNode {
    TraverseNode { lab: self.lab(), p1: Wire::new(self.addr()), p2: Wire::new(self.addr().other_half()) }
  }

  #[inline(always)]
  pub fn consume_op1(self) -> TraverseOp1 {
    let op = self.op();
    let s = self.consume_node();
    let num = s.p1.swap_target(Port::FREE);
    TraverseOp1 { op, num, p2: s.p2 }
  }

  #[inline(always)]
  pub fn traverse_op1(self) -> TraverseOp1 {
    let op = self.op();
    let s = self.traverse_node();
    let num = s.p1.load_target();
    TraverseOp1 { op, num, p2: s.p2 }
  }
}

/// A memory address to be used in a `Port` or a `Wire`.
///
/// The bottom three bits must be zero; i.e. this address must be at least
/// 8-byte-aligned.
///
/// Additionally, all bits other than the lowest 48 must be zero. On a 32-bit
/// system, this has no effect, but on a 64-bit system, this means that the top
/// 16 bits much be zero.
#[derive(Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[must_use]
pub struct Addr(pub usize);

impl fmt::Debug for Addr {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    write!(f, "{:012x?}", self.0)
  }
}

impl Addr {
  pub const NULL: Addr = Addr(0);

  /// Casts this address into an `&AtomicU64`, which may or may not be valid.
  #[inline(always)]
  pub fn val<'a>(&self) -> &'a AtomicU64 {
    unsafe { &*(self.0 as *const _) }
  }

  /// Casts this address into an `&Def`, which may or may not be valid.
  #[inline(always)]
  pub fn def<'a>(&self) -> &'a Def {
    unsafe { &*(self.0 as *const _) }
  }

  const HALF_MASK: usize = 0b1000;

  /// Given an address to one word of a two-word allocation, returns the address
  /// of the first word of that allocation.
  #[inline(always)]
  fn left_half(&self) -> Self {
    Addr(self.0 & !Addr::HALF_MASK)
  }

  /// Given an address to one word of a two-word allocation, returns the address
  /// of the other word of that allocation.
  #[inline(always)]
  pub fn other_half(&self) -> Self {
    Addr(self.0 ^ Addr::HALF_MASK)
  }
}

#[derive(Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[must_use]
pub struct Wire(pub *const AtomicU64);

impl fmt::Debug for Wire {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    write!(f, "{:012x?}", self.0 as usize)
  }
}

unsafe impl Send for Wire {}

impl Wire {
  #[inline(always)]
  pub fn addr(&self) -> Addr {
    Addr(self.0 as _)
  }

  #[inline(always)]
  pub fn new(addr: Addr) -> Wire {
    Wire(addr.0 as _)
  }

  #[inline(always)]
  fn target<'a>(&self) -> &'a AtomicU64 {
    if cfg!(feature = "_fuzz") {
      assert_ne!(self.0 as usize, 0xfffffffffff0);
      assert_ne!(self.0 as usize, 0);
    }
    unsafe { &*self.0 }
  }

  #[inline(always)]
  pub fn load_target(&self) -> Port {
    let port = Port(self.target().load(Relaxed));
    if cfg!(feature = "_fuzz") {
      assert_ne!(port, Port::FREE);
    }
    port
  }

  #[inline(always)]
  pub fn set_target(&self, port: Port) {
    self.target().store(port.0, Relaxed);
  }

  #[inline(always)]
  pub fn cas_target(&self, expected: Port, value: Port) -> Result<Port, Port> {
    self.target().compare_exchange(expected.0, value.0, Relaxed, Relaxed).map(Port).map_err(Port)
  }

  #[inline(always)]
  pub fn swap_target(&self, value: Port) -> Port {
    let port = Port(self.target().swap(value.0, Relaxed));
    if cfg!(feature = "_fuzz") {
      assert_ne!(port, Port::FREE);
    }
    port
  }

  // Takes a pointer's target.
  #[inline(always)]
  pub fn lock_target(&self) -> Port {
    loop {
      let got = self.swap_target(Port::LOCK);
      if cfg!(feature = "_fuzz") {
        assert_ne!(got, Port::FREE);
      }
      if got != Port::LOCK {
        return got;
      }
      spin_loop();
    }
  }
}

/// A bitset representing the set of labels used in a def.
#[derive(Default, Debug, Clone, PartialEq, Eq)]
pub struct LabSet {
  /// The least label greater than every label in the set.
  ///
  /// `lab >= set.min_safe` implies `!set.has(lab)`.
  pub(crate) min_safe: Lab,
  pub(crate) bits: Cow<'static, [u64]>,
}

impl LabSet {
  pub const NONE: LabSet = LabSet { min_safe: 0, bits: Cow::Borrowed(&[]) };
  pub const ALL: LabSet = LabSet { min_safe: Lab::MAX, bits: Cow::Borrowed(&[u64::MAX; 1024]) };

  pub fn add(&mut self, lab: Lab) {
    self.min_safe = self.min_safe.max(lab + 1);
    let index = (lab >> 6) as usize;
    let bit = lab & 63;
    let bits = self.bits.to_mut();
    if index >= bits.len() {
      bits.resize(index + 1, 0);
    }
    bits[index] |= 1 << bit;
  }

  pub fn min_safe(&self) -> Lab {
    self.min_safe
  }

  pub fn has(&self, lab: Lab) -> bool {
    if lab >= self.min_safe {
      return false;
    }
    let index = (lab >> 6) as usize;
    let bit = lab & 63;
    unsafe { self.bits.get_unchecked(index) & 1 << bit != 0 }
  }

  /// Adds of of the labels in `other` to this set.
  pub fn union(&mut self, other: &LabSet) {
    self.min_safe = self.min_safe.max(other.min_safe);
    let bits = self.bits.to_mut();
    for (a, b) in bits.iter_mut().zip(other.bits.iter()) {
      *a |= b;
    }
    if other.bits.len() > bits.len() {
      bits.extend_from_slice(&other.bits[bits.len() ..])
    }
  }

  pub const fn from_bits(bits: &'static [u64]) -> Self {
    if bits.is_empty() {
      return LabSet::NONE;
    }
    let min_safe = (bits.len() << 6) as u16 - bits[bits.len() - 1].leading_zeros() as u16;
    LabSet { min_safe, bits: Cow::Borrowed(bits) }
  }
}

impl FromIterator<Lab> for LabSet {
  fn from_iter<T: IntoIterator<Item = Lab>>(iter: T) -> Self {
    let mut set = LabSet::default();
    for lab in iter {
      set.add(lab);
    }
    set
  }
}

#[repr(C)]
#[repr(align(16))]
pub struct Def<T: ?Sized = Unknown> {
  pub labs: LabSet,
  ty: TypeId,
  call: unsafe fn(*const Def<T>, &mut Net, port: Port),
  pub data: T,
}

extern "C" {
  #[doc(hidden)]
  pub type Unknown;
}

pub trait AsDef: 'static {
  unsafe fn call(slf: *const Def<Self>, net: &mut Net, port: Port);
}

impl<T> Def<T> {
  pub const fn new(labs: LabSet, data: T) -> Self
  where
    T: AsDef,
  {
    Def { labs, ty: TypeId::of::<T>(), call: T::call, data }
  }

  pub const fn upcast(&self) -> &Def {
    unsafe { &*(self as *const _ as *const _) }
  }
}

impl Def {
  pub unsafe fn downcast<T: 'static>(slf: *const Def) -> Option<*const Def<T>> {
    if (*slf).ty == TypeId::of::<T>() { Some(slf.cast()) } else { None }
  }
  pub unsafe fn call(slf: *const Def, net: &mut Net, port: Port) {
    ((*slf).call)(slf as *const _, net, port)
  }
}

impl<T> Deref for Def<T> {
  type Target = Def;
  fn deref(&self) -> &Self::Target {
    self.upcast()
  }
}

impl<F: Fn(&mut Net, Port) + 'static> AsDef for F {
  unsafe fn call(slf: *const Def<Self>, net: &mut Net, port: Port) {
    unsafe { ((*slf).data)(net, port) }
  }
}

pub struct InterpretedDef {
  pub name: String,
  pub instr: Vec<Instruction>,
}

/// `Def`s, when not pre-compiled, are represented as lists of instructions.
///
/// Each instruction corresponds to a fragment of a net that has a native
/// implementation.
///
/// These net fragments may have several free ports, which are each represented
/// with `TrgId`s.
///
/// Each `TrgId` of an instruction has an associated polarity -- it can either
/// be an input or an output. Because the underlying interaction net model we're
/// using does not have polarity, we also need instructions for linking out-out
/// or in-in.
///
/// Linking two outputs can be done with `Link`, which creates a "cup" wire.
///
/// Linking two inputs is more complicated, due to the way locking works. It can
/// be done with `Wires`, which creates two "cap" wires. One half of each cap
/// can be used for each input. Once those inputs have been fully unlocked, the
/// other halves of each cap can be linked with `Link`. For example:
/// ```ignore
/// let (av, aw, bv, bw) = net.do_wires();
/// some_subnet(net, av, bv);
/// net.link(aw, bw);
/// ```
///
/// Each instruction documents both the native implementation and the polarity
/// of each `TrgId`.
///
/// Some instructions take a `Port`; these must always be statically-valid ports
/// -- that is, `Ref` or `Num` ports.
#[derive(Debug, Clone)]
pub enum Instruction {
  /// `let trg = Trg::port(port);`
  Const { trg: TrgId, port: Port },
  /// `net.link_trg(a, b);`
  Link { a: TrgId, b: TrgId },
  /// `net.link(trg, Trg::port(port));`
  LinkConst { trg: TrgId, port: Port },
  /// `let (lft, rgt) = net.do_ctr(lab, trg);`
  Ctr { lab: Lab, trg: TrgId, lft: TrgId, rgt: TrgId },
  /// `let (lft, rgt) = net.do_op2(lab, trg);`
  Op2 { op: Op, trg: TrgId, lft: TrgId, rgt: TrgId },
  /// `let rgt = net.do_op1(lab, num, trg);`
  Op1 { op: Op, num: u64, trg: TrgId, rgt: TrgId },
  /// `let (lft, rgt) = net.do_mat(trg);`
  Mat { trg: TrgId, lft: TrgId, rgt: TrgId },
  /// `let (av, aw, bv, bw) = net.do_wires();`
  Wires { av: TrgId, aw: TrgId, bv: TrgId, bw: TrgId },
}

/// Part of the net to link to; either a wire or a port.
///
/// To store this compactly, we reuse the `Red` tag to indicate if this is a
/// wire.
#[derive(Clone)]
pub struct Trg(Port);

impl Trg {
  /// Creates a `Trg` from a port.
  #[inline(always)]
  pub fn port(port: Port) -> Self {
    Trg(port)
  }

  /// Creates a `Trg` from a wire.
  #[inline(always)]
  pub fn wire(wire: Wire) -> Self {
    Trg(Port(wire.0 as u64))
  }

  #[inline(always)]
  fn is_wire(&self) -> bool {
    self.0.tag() == Red
  }

  #[inline(always)]
  fn as_wire(self) -> Wire {
    Wire(self.0.0 as _)
  }

  #[inline(always)]
  fn as_port(self) -> Port {
    self.0
  }

  /// Access the target port; if this trg is already a port, this does nothing,
  /// but if it is a wire, it loads the target.
  ///
  /// The returned port is only normative if it is principal; if it is a var or
  /// a redirect, it must not be used for linking, and the original `Trg` should
  /// be used instead.
  #[inline(always)]
  pub fn target(&self) -> Port {
    if self.is_wire() { self.clone().as_wire().load_target() } else { self.0.clone() }
  }
}

/// An index to a `Trg` in a `DefNet`. These essentially serve the function of
/// registers.
///
/// In the compiled mode, each `TrgId` will be compiled to a variable.
///
/// In the interpreted mode, the `TrgId` serves as an index into a vector.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct TrgId {
  /// instead of storing the index directly, we store the byte offset, to save a
  /// shift instruction when indexing into the `Trg` vector in interpreted mode.
  ///
  /// This is always `index * size_of<Trg>`.
  byte_offset: usize,
}

impl TrgId {
  pub fn new(index: usize) -> Self {
    TrgId { byte_offset: index * std::mem::size_of::<Trg>() }
  }
  pub fn index(&self) -> usize {
    self.byte_offset / std::mem::size_of::<Trg>()
  }
}

impl fmt::Display for TrgId {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    write!(f, "t{}", self.index())
  }
}

impl fmt::Debug for TrgId {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    write!(f, "TrgId({})", self.index())
  }
}

/// The memory behind a two-word allocation.
///
/// This must be aligned to 16 bytes so that the left word's address always ends
/// with `0b0000` and the right word's address always ends with `0b1000`.
#[repr(C)]
#[repr(align(16))]
#[derive(Default)]
pub struct Node(pub AtomicU64, pub AtomicU64);

// -----------
//   The Net
// -----------

// A interaction combinator net.
pub struct Net<'a> {
  pub tid: usize,              // thread id
  pub tids: usize,             // thread count
  pub rdex: Vec<(Port, Port)>, // redexes
  pub trgs: Vec<Trg>,
  pub rwts: Rewrites, // rewrite count
  pub root: Wire,
  // allocator
  pub area: &'a [Node],
  pub head: Addr,
  pub next: usize,
  //
  tracer: Tracer,
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

impl<'a> Net<'a> {
  /// Creates an empty net with a given heap.
  pub fn new(area: &'a [Node]) -> Self {
    let mut net = Net::new_with_root(area, Wire(std::ptr::null()));
    net.root = Wire::new(net.alloc());
    net
  }

  fn new_with_root(area: &'a [Node], root: Wire) -> Self {
    Net {
      tid: 0,
      tids: 1,
      rdex: vec![],
      trgs: vec![Trg::port(Port::FREE); 1 << 16],
      rwts: Rewrites::default(),
      root,
      area,
      head: Addr::NULL,
      next: 0,
      tracer: Tracer::default(),
    }
  }

  // Boots a net from a Ref.
  pub fn boot(&mut self, def: &Def) {
    let def = Port::new_ref(def);
    trace!(self.tracer, def);
    self.root.set_target(def);
  }
}

// -------------
//   Allocator
// -------------

impl<'a> Net<'a> {
  pub fn init_heap(size: usize) -> Box<[Node]> {
    unsafe {
      Box::from_raw(core::ptr::slice_from_raw_parts_mut(
        alloc::alloc(Layout::array::<Node>(size).unwrap()) as *mut _,
        size,
      ))
    }
  }

  /// Frees one word of a two-word allocation.
  #[inline(always)]
  pub fn half_free(&mut self, addr: Addr) {
    trace!(self.tracer, addr);
    const FREE: u64 = Port::FREE.0;
    if cfg!(feature = "_fuzz") {
      if cfg!(not(feature = "_fuzz_no_free")) {
        assert_ne!(addr.val().swap(FREE, Relaxed), FREE, "double free");
      }
    } else {
      addr.val().store(FREE, Relaxed);
      if addr.other_half().val().load(Relaxed) == FREE {
        trace!(self.tracer, "other free");
        let addr = addr.left_half();
        if addr.val().compare_exchange(FREE, self.head.0 as u64, Relaxed, Relaxed).is_ok() {
          let old_head = &self.head;
          let new_head = addr;
          trace!(self.tracer, "appended", old_head, new_head);
          self.head = new_head;
        } else {
          trace!(self.tracer, "too slow");
        };
      }
    }
  }

  /// Allocates a two-word node.
  #[inline(never)]
  pub fn alloc(&mut self) -> Addr {
    trace!(self.tracer, self.head);
    let addr = if self.head != Addr::NULL {
      let addr = self.head.clone();
      let next = Addr(self.head.val().load(Relaxed) as usize);
      trace!(self.tracer, next);
      self.head = next;
      addr
    } else {
      let index = self.next;
      self.next += 1;
      Addr(&self.area.get(index).expect("OOM").0 as *const _ as _)
    };
    trace!(self.tracer, addr, self.head);
    addr.val().store(Port::LOCK.0, Relaxed);
    addr.other_half().val().store(Port::LOCK.0, Relaxed);
    addr
  }

  /// If `trg` is a wire, frees the backing memory.
  #[inline(always)]
  fn free_trg(&mut self, trg: Trg) {
    if trg.is_wire() {
      self.half_free(trg.as_wire().addr());
    }
  }
}

pub struct CreatedNode {
  pub p0: Port,
  pub p1: Port,
  pub p2: Port,
}

impl<'a> Net<'a> {
  #[inline(always)]
  pub fn create_node(&mut self, tag: Tag, lab: Lab) -> CreatedNode {
    let addr = self.alloc();
    CreatedNode {
      p0: Port::new(tag, lab, addr.clone()),
      p1: Port::new_var(addr.clone()),
      p2: Port::new_var(addr.other_half()),
    }
  }

  /// Creates a wire pointing to a given port; sometimes necessary to avoid
  /// deadlock.
  #[inline(always)]
  fn create_wire(&mut self, port: Port) -> Wire {
    let addr = self.alloc();
    self.half_free(addr.other_half());
    let wire = Wire::new(addr);
    self.link_port_port(port, Port::new_var(wire.addr()));
    wire
  }
}

// ----------
//   Linker
// ----------

/// When threads interfere, this uses the atomic linking algorithm described in
/// `paper/`.
///
/// Linking wires must be done atomically, but linking ports can be done
/// non-atomically (because they must be locked).
impl<'a> Net<'a> {
  /// Links two ports.
  #[inline(always)]
  pub fn link_port_port(&mut self, a_port: Port, b_port: Port) {
    trace!(self.tracer, a_port, b_port);
    if a_port.is_principal() && b_port.is_principal() {
      self.redux(a_port, b_port);
    } else {
      self.half_link_port_port(a_port.clone(), b_port.clone());
      self.half_link_port_port(b_port, a_port);
    }
  }

  /// Links two wires.
  #[inline(always)]
  pub fn link_wire_wire(&mut self, a_wire: Wire, b_wire: Wire) {
    trace!(self.tracer, a_wire, b_wire);
    let a_port = a_wire.lock_target();
    let b_port = b_wire.lock_target();
    trace!(self.tracer, a_port, b_port);
    if a_port.is_principal() && b_port.is_principal() {
      self.half_free(a_wire.addr());
      self.half_free(b_wire.addr());
      self.redux(a_port, b_port);
    } else {
      self.half_link_wire_port(a_port.clone(), a_wire, b_port.clone());
      self.half_link_wire_port(b_port, b_wire, a_port);
    }
  }

  /// Links a wire to a port.
  #[inline(always)]
  pub fn link_wire_port(&mut self, a_wire: Wire, b_port: Port) {
    trace!(self.tracer, a_wire, b_port);
    let a_port = a_wire.lock_target();
    trace!(self.tracer, a_port);
    if a_port.is_principal() && b_port.is_principal() {
      self.half_free(a_wire.addr());
      self.redux(a_port, b_port);
    } else {
      self.half_link_wire_port(a_port.clone(), a_wire, b_port.clone());
      self.half_link_port_port(b_port, a_port);
    }
  }

  /// Pushes an active pair to the redex queue; `a` and `b` must both be
  /// principal ports.
  #[inline(always)]
  fn redux(&mut self, a: Port, b: Port) {
    trace!(self.tracer, a, b);
    if a.is_skippable() && b.is_skippable() {
      self.rwts.eras += 1;
    } else {
      self.rdex.push((a, b));
    }
  }

  /// Half-links `a_port` to `b_port`, without linking `b_port` back to `a_port`.
  #[inline(always)]
  fn half_link_port_port(&mut self, a_port: Port, b_port: Port) {
    trace!(self.tracer, a_port, b_port);
    if a_port.tag() == Var {
      a_port.wire().set_target(b_port);
    }
  }

  /// Half-links a foreign `a_port` (taken from `a_wire`) to `b_port`, without
  /// linking `b_port` back to `a_port`.
  #[inline(always)]
  fn half_link_wire_port(&mut self, a_port: Port, a_wire: Wire, b_port: Port) {
    trace!(self.tracer, a_port, a_wire, b_port);
    // If 'a_port' is a var...
    if a_port.tag() == Var {
      let got = a_port.wire().cas_target(Port::new_var(a_wire.addr()), b_port.clone());
      // Attempts to link using a compare-and-swap.
      if got.is_ok() {
        trace!(self.tracer, "cas ok");
        self.half_free(a_wire.addr());
      // If the CAS failed, resolve by using redirections.
      } else {
        let got = got.unwrap_err();
        trace!(self.tracer, "cas fail", got);
        if b_port.tag() == Var {
          let port = b_port.redirect();
          a_wire.set_target(port);
          //self.resolve_redirect_var(a_port, a_wire, b_port);
        } else if b_port.is_principal() {
          a_wire.set_target(b_port.clone());
          self.resolve_redirect_pri(a_port, a_wire, b_port);
        } else {
          unreachable!();
        }
      }
    } else {
      self.half_free(a_wire.addr());
    }
  }

  /// Resolves redirects when 'b_port' is a principal port.
  fn resolve_redirect_pri(&mut self, mut a_port: Port, a_wire: Wire, b_port: Port) {
    trace!(self.tracer);
    loop {
      trace!(self.tracer, a_port, a_wire, b_port);
      // Peek the target, which may not be owned by us.
      let mut t_wire = a_port.wire();
      let mut t_port = t_wire.load_target();
      trace!(self.tracer, t_port);
      // If it is taken, we wait.
      if t_port == Port::LOCK {
        spin_loop();
        continue;
      }
      // If target is a redirection, we own it. Clear and move forward.
      if t_port.tag() == Red {
        self.half_free(t_wire.addr());
        a_port = t_port;
        continue;
      }
      // If target is a variable, we don't own it. Try replacing it.
      if t_port.tag() == Var {
        if t_wire.cas_target(t_port.clone(), b_port.clone()).is_ok() {
          trace!(self.tracer, "var cas ok");
          // Clear source location.
          // self.half_free(a_wire.addr());
          // Collect the orphaned backward path.
          t_wire = t_port.wire();
          t_port = t_wire.load_target();
          while t_port != Port::LOCK && t_port.tag() == Red {
            trace!(self.tracer, t_wire, t_port);
            self.half_free(t_wire.addr());
            t_wire = t_port.wire();
            // if t_wire == a_wire {
            //   break;
            // }
            t_port = t_wire.load_target();
          }
          return;
        }
        trace!(self.tracer, "var cas fail");
        // If the CAS failed, the var changed, so we try again.
        continue;
      }

      // If it is a node, two threads will reach this branch.
      if t_port.is_principal() || t_port == Port::GONE {
        // Sort references, to avoid deadlocks.
        let x_wire = if a_wire < t_wire { a_wire.clone() } else { t_wire.clone() };
        let y_wire = if a_wire < t_wire { t_wire.clone() } else { a_wire.clone() };
        trace!(self.tracer, x_wire, y_wire);
        // Swap first reference by Ptr::GONE placeholder.
        let x_port = x_wire.swap_target(Port::GONE);
        // First to arrive creates a redex.
        if x_port != Port::GONE {
          let y_port = y_wire.swap_target(Port::GONE);
          trace!(self.tracer, "fst", x_wire, y_wire, x_port, y_port);
          self.redux(x_port, y_port);
          return;
        // Second to arrive clears up the memory.
        } else {
          trace!(self.tracer, "snd !!!", x_wire, y_wire);
          self.half_free(x_wire.addr());
          while y_wire.cas_target(Port::GONE, Port::LOCK).is_err() {
            spin_loop();
          }
          self.half_free(y_wire.addr());
          return;
        }
      }
      // Shouldn't be reached.
      trace!(self.tracer, t_port, a_wire, a_port, b_port);
      unreachable!()
    }
  }

  /// Resolves redirects when 'b_port' is an aux port.
  // TODO: this is currently broken
  #[allow(unused)]
  fn resolve_redirect_var(&mut self, _: Port, _: Wire, b_port: Port) {
    loop {
      let ste_wire = b_port.clone().wire();
      let ste_port = ste_wire.load_target();
      if ste_port.tag() == Var {
        let trg_wire = ste_port.wire();
        let trg_port = trg_wire.load_target();
        if trg_port.tag() == Red {
          let neo_port = trg_port.unredirect();
          if ste_wire.cas_target(ste_port, neo_port).is_ok() {
            self.half_free(trg_wire.addr());
            continue;
          }
        }
      }
      break;
    }
  }

  /// Links a `Trg` to a port, delegating to the appropriate method based on the
  /// type of `a`.
  #[inline(always)]
  pub fn link_trg_port(&mut self, a: Trg, b: Port) {
    match a.is_wire() {
      true => self.link_wire_port(a.as_wire(), b),
      false => self.link_port_port(a.as_port(), b),
    }
  }

  /// Links two `Trg`s, delegating to the appropriate method based on the type
  /// of `a` and `b`.
  #[inline(always)]
  pub fn link_trg(&mut self, a: Trg, b: Trg) {
    match (a.is_wire(), b.is_wire()) {
      (true, true) => self.link_wire_wire(a.as_wire(), b.as_wire()),
      (true, false) => self.link_wire_port(a.as_wire(), b.as_port()),
      (false, true) => self.link_wire_port(b.as_wire(), a.as_port()),
      (false, false) => self.link_port_port(a.as_port(), b.as_port()),
    }
  }
}

// ---------------------
//   Interaction Rules
// ---------------------

impl<'a> Net<'a> {
  /// Performs an interaction between two connected principal ports.
  #[inline(always)]
  pub fn interact(&mut self, a: Port, b: Port) {
    self.tracer.sync();
    trace!(self.tracer, a, b);
    match (a.tag(), b.tag()) {
      // not actually an active pair
      (Var | Red, _) | (_, Var | Red) => unreachable!(),
      // nil-nil
      (Num | Ref, Num | Ref) => self.rwts.eras += 1,
      // comm 2/2
      (Ctr, Mat) if a.lab() != 0 => self.comm22(a, b),
      (Mat, Ctr) if b.lab() != 0 => self.comm22(a, b),
      (Ctr, Op2) | (Op2, Ctr) => self.comm22(a, b),
      (Ctr, Ctr) if a.lab() != b.lab() => self.comm22(a, b),
      // comm 1/2
      (Op1, Ctr) => self.comm12(a, b),
      (Ctr, Op1) => self.comm12(b, a),
      // anni
      (Mat, Mat) | (Op2, Op2) | (Ctr, Ctr) => self.anni2(a, b),
      (Op1, Op1) => self.anni1(a, b),
      // comm 2/0
      (Ref, Ctr) if b.lab() >= a.lab() => self.comm02(a, b),
      (Ctr, Ref) if a.lab() >= b.lab() => self.comm02(b, a),
      (Num, Ctr) => self.comm02(a, b),
      (Ctr, Num) => self.comm02(b, a),
      (Ref, _) if a == Port::ERA => self.comm02(a, b),
      (_, Ref) if b == Port::ERA => self.comm02(b, a),
      // deref
      (Ref, _) => self.call(a, b),
      (_, Ref) => self.call(b, a),
      // native ops
      (Op2, Num) => self.op2_num(a, b),
      (Num, Op2) => self.op2_num(b, a),
      (Op1, Num) => self.op1_num(a, b),
      (Num, Op1) => self.op1_num(b, a),
      (Mat, Num) => self.mat_num(a, b),
      (Num, Mat) => self.mat_num(b, a),
      // todo: what should the semantics of these be?
      (Mat, Ctr) // b.lab() == 0
      | (Ctr, Mat) // a.lab() == 0
      | (Op2, Op1)
      | (Op1, Op2)
      | (Op2, Mat)
      | (Mat, Op2)
      | (Op1, Mat)
      | (Mat, Op1) => unimplemented!("{:?}-{:?}", a.tag(), b.tag()),
    }
  }

  /// ```text
  ///
  ///         a2 |   | a1
  ///           _|___|_
  ///           \     /
  ///         a  \   /
  ///             \ /
  ///              |
  ///             / \
  ///         b  /   \
  ///           /_____\
  ///            |   |
  ///         b1 |   | b2
  ///
  /// --------------------------- anni2
  ///
  ///         a2 |   | a1
  ///            |   |
  ///             \ /
  ///              X
  ///             / \
  ///            |   |
  ///         b1 |   | b2
  ///
  /// ```
  #[inline(never)]
  pub fn anni2(&mut self, a: Port, b: Port) {
    trace!(self.tracer, a, b);
    self.rwts.anni += 1;
    let a = a.consume_node();
    let b = b.consume_node();
    self.link_wire_wire(a.p1, b.p1);
    self.link_wire_wire(a.p2, b.p2);
  }

  /// ```text
  ///
  ///         a2 |   | a1
  ///           _|___|_
  ///           \     /
  ///         a  \   /
  ///             \ /
  ///              |
  ///             /#\
  ///         b  /###\
  ///           /#####\
  ///            |   |
  ///         b1 |   | b2
  ///
  /// --------------------------- comm22
  ///
  ///     a2 |         | a1
  ///        |         |
  ///       /#\       /#\
  ///  B2  /###\     /###\  B1
  ///     /#####\   /#####\
  ///      |   \     /   |
  ///   p1 | p2 \   / p1 | p2
  ///      |     \ /     |
  ///      |      X      |
  ///      |     / \     |
  ///   p2 | p1 /   \ p2 | p1
  ///     _|___/_   _\___|_
  ///     \     /   \     /
  ///  A1  \   /     \   /  A2
  ///       \ /       \ /
  ///        |         |
  ///     b1 |         | b2
  ///
  /// ```
  #[inline(never)]
  pub fn comm22(&mut self, a: Port, b: Port) {
    trace!(self.tracer, a, b);
    self.rwts.comm += 1;

    let a = a.consume_node();
    let b = b.consume_node();

    let A1 = self.create_node(Ctr, a.lab);
    let A2 = self.create_node(Ctr, a.lab);
    let B1 = self.create_node(Ctr, b.lab);
    let B2 = self.create_node(Ctr, b.lab);

    trace!(self.tracer, A1.p0, A2.p0, B1.p0, B2.p0);
    self.link_port_port(A1.p1, B1.p1);
    self.link_port_port(A1.p2, B2.p1);
    self.link_port_port(A2.p1, B1.p2);
    self.link_port_port(A2.p2, B2.p2);

    trace!(self.tracer);
    self.link_wire_port(a.p1, B1.p0);
    self.link_wire_port(a.p2, B2.p0);
    self.link_wire_port(b.p1, A1.p0);
    self.link_wire_port(b.p2, A2.p0);
  }

  /// ```text
  ///
  ///         a  (---)
  ///              |
  ///              |
  ///             /#\
  ///         b  /###\
  ///           /#####\
  ///            |   |
  ///         b1 |   | b2
  ///
  /// --------------------------- comm02
  ///
  ///     a (---)   (---) a
  ///         |       |
  ///      b1 |       | b2
  ///
  /// ```
  #[inline(never)]
  pub fn comm02(&mut self, a: Port, b: Port) {
    trace!(self.tracer, a, b);
    self.rwts.comm += 1;
    let b = b.consume_node();
    self.link_wire_port(b.p1, a.clone());
    self.link_wire_port(b.p2, a);
  }

  /// ```text
  ///
  ///         a2 |
  ///            |   n
  ///           _|___|_
  ///           \     /
  ///         a  \op1/
  ///             \ /
  ///              |
  ///             / \
  ///         b  /op1\
  ///           /_____\
  ///            |   |
  ///            m   |
  ///                | b2
  ///
  /// --------------------------- anni1
  ///
  ///         a2 |
  ///            |
  ///            |
  ///             \
  ///              \
  ///               \
  ///                |
  ///                |
  ///                | b2
  ///
  /// ```
  #[inline(never)]
  pub fn anni1(&mut self, a: Port, b: Port) {
    trace!(self.tracer, a, b);
    self.rwts.anni += 1;
    let a = a.consume_op1();
    let b = b.consume_op1();
    self.link_wire_wire(a.p2, b.p2);
  }

  /// ```text
  ///
  ///         a2 |   n
  ///           _|___|_
  ///           \     /
  ///         a  \op1/
  ///             \ /
  ///              |
  ///             /#\
  ///         b  /###\
  ///           /#####\
  ///            |   |
  ///         b1 |   | b2
  ///
  /// --------------------------- comm12
  ///
  ///     a2 |
  ///        |
  ///       /#\
  ///  B2  /###\
  ///     /#####\
  ///      |   \
  ///   p1 | p2 \
  ///      |     \
  ///      |      \
  ///      |       \
  ///   p2 |   n    \ p2 n
  ///     _|___|_   _\___|_
  ///     \     /   \     /
  ///  A1  \op1/     \op1/  A2
  ///       \ /       \ /
  ///        |         |
  ///     b1 |         | b2
  ///
  /// ```
  #[inline(never)]
  pub fn comm12(&mut self, a: Port, b: Port) {
    trace!(self.tracer, a, b);
    self.rwts.comm += 1;

    let a = a.consume_op1();
    let b = b.consume_node();

    let A1 = self.create_node(Ctr, a.op as Lab);
    let A2 = self.create_node(Ctr, a.op as Lab);
    let B2 = self.create_node(Ctr, b.lab);

    trace!(self.tracer, B2.p0, A1.p0, A2.p0);
    self.link_port_port(A1.p1, a.num.clone());
    self.link_port_port(A1.p2, B2.p1);
    self.link_port_port(A2.p1, a.num.clone());
    self.link_port_port(A2.p2, B2.p2);

    trace!(self.tracer);
    self.link_wire_port(a.p2, B2.p0);
    self.link_wire_port(b.p1, A1.p0);
    self.link_wire_port(b.p2, A2.p0);
  }

  #[inline(never)]
  /// ```text
  ///
  ///         a  (---)
  ///              |
  ///              |
  ///             / \
  ///         b  /op1\
  ///           /_____\
  ///            |   |
  ///            n   |
  ///                | b2
  ///
  /// --------------------------- comm02
  ///
  ///              (---) a
  ///                |
  ///                | b2
  ///
  /// ```
  pub fn comm01(&mut self, a: Port, b: Port) {
    trace!(self.tracer, a, b);
    self.rwts.comm += 1;
    let b = b.consume_op1();
    self.link_wire_port(b.p2, a);
  }

  /// ```text
  ///                             |
  ///         b   (0)             |         b  (n+1)
  ///              |              |              |
  ///              |              |              |
  ///             / \             |             / \
  ///         a  /mat\            |         a  /mat\
  ///           /_____\           |           /_____\
  ///            |   |            |            |   |
  ///         a1 |   | a2         |         a1 |   | a2
  ///                             |
  /// --------------------------- | --------------------------- mat_num
  ///                             |          _ _ _ _ _
  ///                             |        /           \
  ///                             |    y2 |  (n) y1     |
  ///                             |      _|___|_        |
  ///                             |      \     /        |
  ///               _             |    y  \   /         |
  ///             /   \           |        \ /          |
  ///    x2 (*)  | x1  |          |      x2 |  (*) x1   |
  ///       _|___|_    |          |        _|___|_      |
  ///       \     /    |          |        \     /      |
  ///     x  \   /     |          |      x  \   /       |
  ///         \ /      |          |          \ /        |
  ///          |       |          |           |         |
  ///       a1 |       | a2       |        a1 |         | a2
  ///                             |
  /// ```
  #[inline(never)]
  pub fn mat_num(&mut self, a: Port, b: Port) {
    trace!(self.tracer, a, b);
    self.rwts.oper += 1;
    let a = a.consume_node();
    let b = b.num();
    if b == 0 {
      let x = self.create_node(Ctr, 0);
      trace!(self.tracer, x.p0);
      self.link_port_port(x.p2, Port::ERA);
      self.link_wire_port(a.p2, x.p1);
      self.link_wire_port(a.p1, x.p0);
    } else {
      let x = self.create_node(Ctr, 0);
      let y = self.create_node(Ctr, 0);
      trace!(self.tracer, x.p0, y.p0);
      self.link_port_port(x.p1, Port::ERA);
      self.link_port_port(x.p2, y.p0);
      self.link_port_port(y.p1, Port::new_num(b - 1));
      self.link_wire_port(a.p2, y.p2);
      self.link_wire_port(a.p1, x.p0);
    }
  }

  /// ```text
  ///                   
  ///         b   (n)    
  ///              |      
  ///              |       
  ///             / \       
  ///         a  /op2\       
  ///           /_____\       
  ///            |   |         
  ///         a1 |   | a2       
  ///                            
  /// --------------------------- op2_num
  ///           _ _ _
  ///         /       \
  ///        |   n     |   
  ///       _|___|_    |   
  ///       \     /    |   
  ///     x  \op1/     |   
  ///         \ /      |   
  ///          |       |   
  ///       a1 |       | a2  
  ///                       
  /// ```
  #[inline(never)]
  pub fn op2_num(&mut self, a: Port, b: Port) {
    trace!(self.tracer, a, b);
    self.rwts.oper += 1;
    let a = a.consume_node();
    let x = self.create_node(Op1, a.lab);
    trace!(self.tracer, x.p0);
    self.link_port_port(x.p1, b);
    self.link_wire_port(a.p2, x.p2);
    self.link_wire_port(a.p1, x.p0);
  }

  /// ```text
  ///                   
  ///         b   (m)    
  ///              |      
  ///              |       
  ///             / \       
  ///         a  /op1\       
  ///           /_____\       
  ///            |   |         
  ///            n   |         
  ///                | a2       
  ///                            
  /// --------------------------- op2_num
  ///                       
  ///          (n opr m)
  ///              |         
  ///              | a2
  ///                       
  /// ```
  #[inline(never)]
  pub fn op1_num(&mut self, a: Port, b: Port) {
    trace!(self.tracer, a, b);
    self.rwts.oper += 1;
    let a = a.consume_op1();
    let n = a.num.num();
    let m = b.num();
    let out = a.op.op(n, m);
    self.link_wire_port(a.p2, Port::new_num(out));
  }
}

// ----------------
//   Instructions
// ----------------

impl<'a> Net<'a> {
  /// `trg ~ {#lab x y}`
  #[inline(always)]
  pub(crate) fn do_ctr(&mut self, lab: Lab, trg: Trg) -> (Trg, Trg) {
    let port = trg.target();
    if port.tag() == Ctr && port.lab() == lab {
      self.free_trg(trg);
      let node = port.consume_node();
      self.rwts.anni += 1;
      (Trg::wire(node.p1), Trg::wire(node.p2))
    // TODO: fast copy?
    // } else if port.tag() == Num || port.tag() == Ref && lab >= port.lab() {
    //   self.rwts.comm += 1;
    //   (Trg::port(port.clone()), Trg::port(port))
    } else {
      let n = self.create_node(Ctr, lab);
      self.link_trg_port(trg, n.p0);
      (Trg::port(n.p1), Trg::port(n.p2))
    }
  }

  /// `trg ~ <op x y>`
  #[inline(always)]
  pub(crate) fn do_op2(&mut self, op: Op, trg: Trg) -> (Trg, Trg) {
    let port = trg.target();
    if port.tag() == Num {
      self.rwts.oper += 1;
      self.free_trg(trg);
      let n = self.create_node(Op1, op as Lab);
      n.p1.wire().set_target(Port::new_num(port.num()));
      (Trg::port(n.p0), Trg::port(n.p2))
    } else if port == Port::ERA {
      (Trg::port(Port::ERA), Trg::port(Port::ERA))
    } else {
      let n = self.create_node(Op2, op as Lab);
      self.link_trg_port(trg, n.p0);
      (Trg::port(n.p1), Trg::port(n.p2))
    }
  }

  /// `trg ~ <a op x>`
  #[inline(always)]
  pub(crate) fn do_op1(&mut self, op: Op, a: u64, trg: Trg) -> Trg {
    let port = trg.target();
    if trg.target().tag() == Num {
      self.rwts.oper += 1;
      self.free_trg(trg);
      Trg::port(Port::new_num(op.op(a, port.num())))
    } else if port == Port::ERA {
      Trg::port(Port::ERA)
    } else {
      let n = self.create_node(Op1, op as Lab);
      self.link_trg_port(trg, n.p0);
      n.p1.wire().set_target(Port::new_num(a));
      Trg::port(n.p2)
    }
  }

  /// `trg ~ ?<x y>`
  #[inline(always)]
  pub(crate) fn do_mat(&mut self, trg: Trg) -> (Trg, Trg) {
    let port = trg.target();
    if port.tag() == Num {
      self.rwts.oper += 1;
      self.free_trg(trg);
      let num = port.num();
      let c1 = self.create_node(Ctr, 0);
      if num == 0 {
        self.link_port_port(c1.p2, Port::ERA);
        (Trg::port(c1.p0), Trg::wire(self.create_wire(c1.p1)))
      } else {
        let c2 = self.create_node(Ctr, 0);
        self.link_port_port(c1.p1, Port::ERA);
        self.link_port_port(c1.p2, c2.p0);
        self.link_port_port(c2.p1, Port::new_num(num - 1));
        (Trg::port(c1.p0), Trg::wire(self.create_wire(c2.p2)))
      }
    } else if port == Port::ERA {
      self.rwts.eras += 1;
      self.free_trg(trg);
      (Trg::port(Port::ERA), Trg::port(Port::ERA))
    } else {
      let m = self.create_node(Mat, 0);
      self.link_trg_port(trg, m.p0);
      (Trg::port(m.p1), Trg::port(m.p2))
    }
  }

  #[inline(always)]
  pub(crate) fn do_wires(&mut self) -> (Trg, Trg, Trg, Trg) {
    let a = self.alloc();
    let b = a.other_half();
    (
      Trg::port(Port::new_var(a.clone())),
      Trg::port(Port::new_var(b.clone())),
      Trg::wire(Wire::new(a)),
      Trg::wire(Wire::new(b)),
    )
  }

  /// `trg ~ <op #b x>`
  #[inline(always)]
  #[allow(unused)] // TODO: emit this instruction
  pub(crate) fn do_op2_num(&mut self, op: Op, b: u64, trg: Trg) -> Trg {
    let port = trg.target();
    if port.tag() == Num {
      self.rwts.oper += 2;
      self.free_trg(trg);
      Trg::port(Port::new_num(op.op(port.num(), b)))
    } else if port == Port::ERA {
      Trg::port(Port::ERA)
    } else {
      let n = self.create_node(Op2, op as Lab);
      self.link_trg_port(trg, n.p0);
      n.p1.wire().set_target(Port::new_num(b));
      Trg::port(n.p2)
    }
  }

  /// `trg ~ ?<(x (y z)) out>`
  #[inline(always)]
  #[allow(unused)] // TODO: emit this instruction
  pub(crate) fn do_mat_con_con(&mut self, trg: Trg, out: Trg) -> (Trg, Trg, Trg) {
    let port = trg.target();
    if trg.target().tag() == Num {
      self.rwts.oper += 1;
      self.free_trg(trg);
      let num = port.num();
      if num == 0 {
        (out, Trg::port(Port::ERA), Trg::port(Port::ERA))
      } else {
        (Trg::port(Port::ERA), Trg::port(Port::new_num(num - 1)), out)
      }
    } else if port == Port::ERA {
      self.link_trg_port(out, Port::ERA);
      (Trg::port(Port::ERA), Trg::port(Port::ERA), Trg::port(Port::ERA))
    } else {
      let m = self.create_node(Mat, 0);
      let c1 = self.create_node(Ctr, 0);
      let c2 = self.create_node(Ctr, 0);
      self.link_port_port(m.p1, c1.p0);
      self.link_port_port(c1.p2, c2.p0);
      self.link_trg_port(out, m.p2);
      (Trg::port(c1.p1), Trg::port(c2.p1), Trg::port(c2.p2))
    }
  }

  /// `trg ~ ?<(x y) out>`
  #[inline(always)]
  #[allow(unused)] // TODO: emit this instruction
  pub(crate) fn do_mat_con(&mut self, trg: Trg, out: Trg) -> (Trg, Trg) {
    let port = trg.target();
    if trg.target().tag() == Num {
      self.rwts.oper += 1;
      self.free_trg(trg);
      let num = port.num();
      if num == 0 {
        (out, Trg::port(Port::ERA))
      } else {
        let c2 = self.create_node(Ctr, 0);
        c2.p1.wire().set_target(Port::new_num(num - 1));
        self.link_trg_port(out, c2.p2);
        (Trg::port(Port::ERA), Trg::port(c2.p0))
      }
    } else if port == Port::ERA {
      self.link_trg_port(out, Port::ERA);
      (Trg::port(Port::ERA), Trg::port(Port::ERA))
    } else {
      let m = self.create_node(Mat, 0);
      let c1 = self.create_node(Ctr, 0);
      self.link_port_port(m.p1, c1.p0);
      self.link_trg_port(out, m.p2);
      self.link_trg_port(trg, m.p0);
      (Trg::port(c1.p1), Trg::port(c1.p2))
    }
  }
}

impl<'a> Net<'a> {
  /// Expands a `Ref` node connected to `trg`.
  #[inline(never)]
  pub fn call(&mut self, port: Port, trg: Port) {
    trace!(self.tracer, port, trg);

    let def = port.addr().def();

    if trg.tag() == Ctr && !def.labs.has(trg.lab()) {
      return self.comm02(port, trg);
    }

    self.rwts.dref += 1;

    unsafe { Def::call(port.addr().0 as *const _, self, trg) }
  }
}

impl AsDef for InterpretedDef {
  unsafe fn call(slf: *const Def<Self>, net: &mut Net, trg: Port) {
    let instructions = unsafe { &(*slf).data.instr };

    let mut trgs = unsafe { std::mem::transmute::<_, Trgs>(Trgs(&mut net.trgs[..])) };

    struct Trgs<'a>(&'a mut [Trg]);

    impl<'a> Trgs<'a> {
      #[inline(always)]
      fn get_trg(&self, i: TrgId) -> Trg {
        unsafe { (*self.0.as_ptr().byte_offset(i.byte_offset as _)).clone() }
      }

      #[inline(always)]
      fn set_trg(&mut self, i: TrgId, trg: Trg) {
        unsafe { *self.0.as_mut_ptr().byte_offset(i.byte_offset as _) = trg }
      }
    }

    trgs.set_trg(TrgId::new(0), Trg::port(trg));
    for i in instructions {
      unsafe {
        match *i {
          Instruction::Const { trg, ref port } => trgs.set_trg(trg, Trg::port(port.clone())),
          Instruction::Link { a, b } => net.link_trg(trgs.get_trg(a), trgs.get_trg(b)),
          Instruction::LinkConst { trg, ref port } => {
            if !port.is_principal() {
              unreachable_unchecked()
            }
            net.link_trg_port(trgs.get_trg(trg), port.clone())
          }
          Instruction::Ctr { lab, trg, lft, rgt } => {
            let (l, r) = net.do_ctr(lab, trgs.get_trg(trg));
            trgs.set_trg(lft, l);
            trgs.set_trg(rgt, r);
          }
          Instruction::Op2 { op, trg, lft, rgt } => {
            let (l, r) = net.do_op2(op, trgs.get_trg(trg));
            trgs.set_trg(lft, l);
            trgs.set_trg(rgt, r);
          }
          Instruction::Op1 { op, num, trg, rgt } => {
            let r = net.do_op1(op, num, trgs.get_trg(trg));
            trgs.set_trg(rgt, r);
          }
          Instruction::Mat { trg, lft, rgt } => {
            let (l, r) = net.do_mat(trgs.get_trg(trg));
            trgs.set_trg(lft, l);
            trgs.set_trg(rgt, r);
          }
          Instruction::Wires { av, aw, bv, bw } => {
            let (avt, awt, bvt, bwt) = net.do_wires();
            trgs.set_trg(av, avt);
            trgs.set_trg(bv, awt);
            trgs.set_trg(aw, bvt);
            trgs.set_trg(bw, bwt);
          }
        }
      }
    }
  }
}

// -----------------
//   Normalization
// -----------------

impl<'a> Net<'a> {
  /// Reduces at most `limit` redexes.
  #[inline(always)]
  pub fn reduce(&mut self, limit: usize) -> usize {
    let mut count = 0;
    while let Some((a, b)) = self.rdex.pop() {
      self.interact(a, b);
      count += 1;
      if count >= limit {
        break;
      }
    }
    count
  }

  /// Expands `Ref` nodes in the tree connected to `root`.
  #[inline(always)]
  pub fn expand(&mut self) {
    fn go(net: &mut Net, wire: Wire, len: usize, key: usize) {
      trace!(net.tracer, wire);
      let port = wire.load_target();
      trace!(net.tracer, port);
      if port == Port::LOCK {
        return;
      }
      if port.tag() == Ctr {
        let node = port.traverse_node();
        if len >= net.tids || key % 2 == 0 {
          go(net, node.p1, len.saturating_mul(2), key / 2);
        }
        if len >= net.tids || key % 2 == 1 {
          go(net, node.p2, len.saturating_mul(2), key / 2);
        }
      } else if port.tag() == Ref && port != Port::ERA {
        let got = wire.swap_target(Port::LOCK);
        if got != Port::LOCK {
          trace!(net.tracer, port, wire);
          net.call(port, Port::new_var(wire.addr()));
        }
      }
    }
    go(self, self.root.clone(), 1, self.tid);
  }

  /// Reduces a net to normal form.
  pub fn normal(&mut self) {
    self.expand();
    while !self.rdex.is_empty() {
      self.reduce(usize::MAX);
      self.expand();
    }
  }

  /// Forks the net into `tids` child nets, for parallel operation.
  pub fn fork(&mut self, tids: usize) -> impl Iterator<Item = Self> + '_ {
    let mut redexes = std::mem::take(&mut self.rdex).into_iter();
    (0 .. tids).map(move |tid| {
      let heap_size = self.area.len() / tids;
      let heap_start = heap_size * tid;
      let area = &self.area[heap_start .. heap_start + heap_size];
      let mut net = Net::new_with_root(area, self.root.clone());
      net.next = self.next.saturating_sub(heap_start);
      net.head = if tid == 0 { self.head.clone() } else { Addr::NULL };
      net.tid = tid;
      net.tids = tids;
      net.tracer.set_tid(tid);
      let count = redexes.len() / (tids - tid);
      net.rdex.extend((&mut redexes).take(count));
      net
    })
  }

  // Evaluates a term to normal form in parallel
  pub fn parallel_normal(&mut self) {
    const SHARE_LIMIT: usize = 1 << 12; // max share redexes per split
    const LOCAL_LIMIT: usize = 1 << 18; // max local rewrites per epoch

    // Local thread context
    struct ThreadContext<'a> {
      tid: usize,                             // thread id
      tlog2: usize,                           // log2 of thread count
      tick: usize,                            // current tick
      net: Net<'a>,                           // thread's own net object
      delta: &'a AtomicRewrites,              // global delta rewrites
      share: &'a Vec<(AtomicU64, AtomicU64)>, // global share buffer
      rlens: &'a Vec<AtomicUsize>,            // global redex lengths
      total: &'a AtomicUsize,                 // total redex length
      barry: Arc<Barrier>,                    // synchronization barrier
    }

    // Initialize global objects
    let cores = std::thread::available_parallelism().unwrap().get() as usize;
    let tlog2 = cores.ilog2() as usize;
    let tids = 1 << tlog2;
    let delta = AtomicRewrites::default(); // delta rewrite counter
    let rlens = (0 .. tids).map(|_| AtomicUsize::new(0)).collect::<Vec<_>>();
    let share = (0 .. SHARE_LIMIT * tids).map(|_| Default::default()).collect::<Vec<_>>();
    let total = AtomicUsize::new(0); // sum of redex bag length
    let barry = Arc::new(Barrier::new(tids)); // global barrier

    // Perform parallel reductions
    std::thread::scope(|s| {
      for net in self.fork(tids) {
        let mut ctx = ThreadContext {
          tid: net.tid,
          tick: 0,
          net,
          tlog2,
          delta: &delta,
          share: &share,
          rlens: &rlens,
          total: &total,
          barry: Arc::clone(&barry),
        };
        thread::Builder::new().name(format!("t{:02x?}", ctx.net.tid)).spawn_scoped(s, move || main(&mut ctx)).unwrap();
      }
    });

    // Clear redexes and sum stats
    self.rdex.clear();
    delta.add_to(&mut self.rwts);

    // Main reduction loop
    #[inline(always)]
    fn main(ctx: &mut ThreadContext) {
      loop {
        reduce(ctx);
        ctx.net.expand();
        if count(ctx) == 0 {
          break;
        }
      }
      ctx.net.rwts.add_to(ctx.delta);
    }

    // Reduce redexes locally, then share with target
    #[inline(always)]
    fn reduce(ctx: &mut ThreadContext) {
      loop {
        ctx.net.reduce(LOCAL_LIMIT);
        if count(ctx) == 0 {
          break;
        }
        let tlog2 = ctx.tlog2;
        split(ctx, tlog2);
        ctx.tick += 1;
      }
    }

    // Count total redexes (and populate 'rlens')
    #[inline(always)]
    fn count(ctx: &mut ThreadContext) -> usize {
      ctx.barry.wait();
      ctx.total.store(0, Relaxed);
      ctx.barry.wait();
      ctx.rlens[ctx.tid].store(ctx.net.rdex.len(), Relaxed);
      ctx.total.fetch_add(ctx.net.rdex.len(), Relaxed);
      ctx.barry.wait();
      ctx.total.load(Relaxed)
    }

    // Share redexes with target thread
    #[inline(always)]
    fn split(ctx: &mut ThreadContext, plog2: usize) {
      unsafe {
        let side = (ctx.tid >> (plog2 - 1 - (ctx.tick % plog2))) & 1;
        let shift = (1 << (plog2 - 1)) >> (ctx.tick % plog2);
        let a_tid = ctx.tid;
        let b_tid = if side == 1 { a_tid - shift } else { a_tid + shift };
        let a_len = ctx.net.rdex.len();
        let b_len = ctx.rlens[b_tid].load(Relaxed);
        let send = if a_len > b_len { (a_len - b_len) / 2 } else { 0 };
        let recv = if b_len > a_len { (b_len - a_len) / 2 } else { 0 };
        let send = std::cmp::min(send, SHARE_LIMIT);
        let recv = std::cmp::min(recv, SHARE_LIMIT);
        for i in 0 .. send {
          let init = a_len - send * 2;
          let rdx0 = ctx.net.rdex.get_unchecked(init + i * 2 + 0).clone();
          let rdx1 = ctx.net.rdex.get_unchecked(init + i * 2 + 1).clone();
          //let init = 0;
          //let ref0 = ctx.net.rdex.get_unchecked_mut(init + i * 2 + 0);
          //let rdx0 = *ref0;
          //*ref0    = (Ptr(0), Ptr(0));
          //let ref1 = ctx.net.rdex.get_unchecked_mut(init + i * 2 + 1);
          //let rdx1 = *ref1;
          //*ref1    = (Ptr(0), Ptr(0));
          let targ = ctx.share.get_unchecked(b_tid * SHARE_LIMIT + i);
          *ctx.net.rdex.get_unchecked_mut(init + i) = rdx0;
          targ.0.store(rdx1.0.0, Relaxed);
          targ.1.store(rdx1.1.0, Relaxed);
        }
        ctx.net.rdex.truncate(a_len - send);
        ctx.barry.wait();
        for i in 0 .. recv {
          let got = ctx.share.get_unchecked(a_tid * SHARE_LIMIT + i);
          ctx.net.rdex.push((Port(got.0.load(Relaxed)), Port(got.1.load(Relaxed))));
        }
      }
    }
  }
}
