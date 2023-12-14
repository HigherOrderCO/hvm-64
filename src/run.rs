// An efficient Interaction Combinator runtime
// ===========================================
// This file implements an efficient interaction combinator runtime. Nodes are represented by 2 aux
// ports (P1, P2), with the main port (P1) omitted. A separate vector, 'rdex', holds main ports,
// and, thus, tracks active pairs that can be reduced in parallel. Pointers are unboxed, meaning
// that Ptr::ERAs, NUMs and REFs don't use any additional space. REFs lazily expand to closed nets when
// they interact with nodes, and are cleared when they interact with Ptr::ERAs, allowing for constant
// space evaluation of recursive functions on Scott encoded datatypes.

use crate::{ops::Op, trace, trace::Tracer};
use std::{
  alloc::{self, Layout},
  fmt,
  sync::{
    atomic::{AtomicU64, AtomicUsize, Ordering::Relaxed},
    Arc, Barrier,
  },
  thread,
};

#[repr(u8)]
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Tag {
  Red = 0,
  Var = 1,
  Ref = 2,
  Num = 3,
  Op2 = 4,
  Op1 = 5,
  Mat = 6,
  Ctr = 7,
}

use Tag::*;

impl TryFrom<u8> for Tag {
  type Error = ();

  #[inline(always)]
  fn try_from(value: u8) -> Result<Self, Self::Error> {
    Ok(match value {
      0 => Tag::Red,
      1 => Tag::Var,
      2 => Tag::Ref,
      3 => Tag::Num,
      4 => Tag::Op2,
      5 => Tag::Op1,
      6 => Tag::Mat,
      7 => Tag::Ctr,
      _ => Err(())?,
    })
  }
}

pub type Lab = u16;

#[derive(Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[must_use]
pub struct Wire(pub *const APtr);

impl fmt::Debug for Wire {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    write!(f, "{:012x?}", self.0 as usize)
  }
}

unsafe impl Send for Wire {}

const PORT_MASK: Val = 0b1000;

impl Wire {
  pub const NULL: Wire = Wire(std::ptr::null());

  #[inline(always)]
  pub fn index(&self) -> usize {
    self.0 as usize >> 4
  }

  #[inline(always)]
  pub fn port(&self) -> u8 {
    (((self.0 as usize as Val) & PORT_MASK) >> 3) as u8
  }

  #[inline(always)]
  pub fn local(index: usize, port: u8) -> Wire {
    Wire(((index << 4) | ((port as usize) << 3)) as *const _)
  }

  #[inline(always)]
  pub fn with_port(&self, port: u8) -> Wire {
    Wire(((self.0 as Val) & !PORT_MASK | ((port as Val) << 3)) as _)
  }

  #[inline(always)]
  pub fn p0(&self) -> Wire {
    Wire(((self.0 as Val) & !PORT_MASK) as _)
  }

  #[inline(always)]
  pub fn p1(&self) -> Wire {
    Wire(((self.0 as Val) & !PORT_MASK) as _)
  }

  #[inline(always)]
  pub fn p2(&self) -> Wire {
    Wire(((self.0 as Val) | PORT_MASK) as _)
  }

  #[inline(always)]
  pub fn other(&self) -> Wire {
    Wire(((self.0 as Val) ^ PORT_MASK) as _)
  }

  #[inline(always)]
  pub fn target<'a>(&self) -> &'a APtr {
    unsafe { &*self.0 }
  }

  #[inline(always)]
  pub fn def<'a>(&self) -> &'a Def {
    unsafe { &*(self.0 as *const _) }
  }

  #[inline(always)]
  pub fn var(&self) -> Port {
    Port::new(Var, 0, self.clone())
  }
}

pub type Val = u64;
pub type AVal = AtomicU64;

/// A tagged pointer.
#[derive(Clone, Eq, PartialEq, PartialOrd, Hash, Default)]
#[repr(transparent)]
#[must_use]
pub struct Port(pub Val);

impl fmt::Debug for Port {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    write!(f, "{:016x?} ", self.0)?;
    match self {
      &Port::ERA => write!(f, "[ERA]"),
      &Port::NULL => write!(f, "[Ptr::NULL]"),
      &Port::GONE => write!(f, "[Ptr::GONE]"),
      &Port::LOCK => write!(f, "[Ptr::LOCK]"),
      _ => match self.tag() {
        Num => write!(f, "[Num {}]", self.num()),
        Var | Red | Ref | Mat => write!(f, "[{:?} {:?}]", self.tag(), self.loc()),
        Op2 | Op1 | Ctr => write!(f, "[{:?} {:?} {:?}]", self.tag(), self.lab(), self.loc()),
      },
    }
  }
}

impl Port {
  pub const ERA: Port = Port(Ref as _);
  pub const NULL: Port = Port(0x0000_0000_0000_0000);
  pub const LOCK: Port = Port(0xFFFF_FFFF_FFFF_FFF0);
  pub const GONE: Port = Port(0xFFFF_FFFF_FFFF_FFFF);

  #[inline(always)]
  pub fn new(tag: Tag, lab: Lab, loc: Wire) -> Self {
    Port(((lab as Val) << 48) | (loc.0 as usize as Val) | (tag as Val))
  }

  #[inline(always)]
  pub const fn new_num(val: Val) -> Self {
    Port((val << 4) | (Num as Val))
  }

  #[inline(always)]
  pub fn new_ref(def: &Def) -> Port {
    Port::new(Ref, def.lab, Wire(def as *const _ as _))
  }

  #[inline(always)]
  pub fn tag(&self) -> Tag {
    unsafe { ((self.0 & 0x7) as u8).try_into().unwrap_unchecked() }
  }

  #[inline(always)]
  pub const fn lab(&self) -> Lab {
    (self.0 >> 48) as Lab
  }

  #[inline(always)]
  pub fn op(&self) -> Op {
    unsafe { self.lab().try_into().unwrap_unchecked() }
  }

  #[inline(always)]
  pub const fn loc(&self) -> Wire {
    Wire((self.0 & 0x0000_FFFF_FFFF_FFF8) as usize as _)
  }

  #[inline(always)]
  pub const fn num(&self) -> Val {
    self.0 >> 4
  }

  #[inline(always)]
  pub fn is_null(&self) -> bool {
    return self.0 == 0;
  }

  #[inline(always)]
  pub fn is_pri(&self) -> bool {
    return self.tag() >= Ref;
  }

  #[inline(always)]
  pub fn is_nilary(&self) -> bool {
    return matches!(self.tag(), Num | Ref);
  }

  #[inline(always)]
  pub fn is_ctr(&self, lab: Lab) -> bool {
    return self.tag() == Ctr && self.lab() == lab;
  }

  #[inline(always)]
  pub fn p1(&self) -> Wire {
    self.loc().p1()
  }

  #[inline(always)]
  pub fn p2(&self) -> Wire {
    self.loc().p2()
  }

  #[inline(always)]
  pub fn target(&self) -> &APtr {
    self.loc().target()
  }

  #[inline(always)]
  pub fn redirect(&self) -> Port {
    return Port::new(Red, 0, self.loc());
  }

  #[inline(always)]
  pub fn unredirect(&self) -> Port {
    return Port::new(Var, 0, self.loc());
  }

  #[inline(always)]
  pub fn consume_node(self) -> ConsumedNode {
    ConsumedNode { lab: self.lab(), p1: self.p1(), p2: self.p2() }
  }

  #[inline(always)]
  pub fn consume_op1(self) -> ConsumedOp1 {
    let n = self.p1().target().swap(Port::NULL);
    ConsumedOp1 { op: self.op(), num: n, p2: self.p2() }
  }
}

pub struct ConsumedNode {
  pub lab: Lab,
  pub p1: Wire,
  pub p2: Wire,
}

pub struct ConsumedOp1 {
  pub op: Op,
  pub num: Port,
  pub p2: Wire,
}

pub struct CreatedNode {
  pub p0: Port,
  pub p1: Port,
  pub p2: Port,
}

/// An atomic tagged pointer.
#[repr(transparent)]
#[derive(Default)]
pub struct APtr(pub AVal);

impl APtr {
  #[inline(always)]
  pub fn new(port: Port) -> Self {
    APtr(AtomicU64::new(port.0))
  }

  #[inline(always)]
  pub fn load(&self) -> Port {
    Port(self.0.load(Relaxed))
  }

  #[inline(always)]
  pub fn store(&self, port: Port) {
    self.0.store(port.0, Relaxed);
  }

  #[inline(always)]
  pub fn cas(&self, expected: Port, value: Port) -> Result<Port, Port> {
    self.0.compare_exchange_weak(expected.0, value.0, Relaxed, Relaxed).map(Port).map_err(Port)
  }

  #[inline(always)]
  pub fn cas_strong(&self, expected: Port, value: Port) -> Result<Port, Port> {
    self.0.compare_exchange(expected.0, value.0, Relaxed, Relaxed).map(Port).map_err(Port)
  }

  #[inline(always)]
  pub fn swap(&self, value: Port) -> Port {
    Port(self.0.swap(value.0, Relaxed))
  }

  // Takes a pointer's target.
  #[inline(always)]
  pub fn take(&self) -> Port {
    loop {
      let got = self.swap(Port::LOCK);
      if got != Port::LOCK {
        return got;
      }
    }
  }
}

impl<'a> Net<'a> {
  pub fn init_heap(size: usize) -> Box<[ANode]> {
    unsafe {
      Box::from_raw(std::slice::from_raw_parts_mut::<ANode>(
        alloc::alloc(Layout::array::<ANode>(size).unwrap()) as *mut _,
        size,
      ) as *mut _)
    }
  }

  #[inline(never)]
  pub fn weak_half_free(&mut self, loc: Wire) {
    trace!(self.tracer, loc);
    loc.target().store(Port::NULL);
  }

  #[inline(never)]
  pub fn half_free(&mut self, loc: Wire) {
    trace!(self.tracer, loc);
    loc.target().store(Port::NULL);
    if loc.other().target().load() == Port::NULL {
      trace!(self.tracer, "other free");
      let loc = loc.p0();
      // use a label of 1 to distinguish from Ptr::NULL
      if let Ok(_) = loc.target().cas_strong(Port::NULL, Port::new(Red, 1, self.head.clone())) {
        let old_head = &self.head;
        let new_head = loc;
        trace!(self.tracer, "appended", old_head, new_head);
        self.head = new_head;
      } else {
        trace!(self.tracer, "too slow");
      };
    }
  }

  #[inline(never)]
  pub fn alloc(&mut self) -> Wire {
    trace!(self.tracer, self.head);
    let loc = if self.head != Wire::NULL {
      let loc = self.head.clone();
      let next = self.head.target().load();
      trace!(self.tracer, next);
      self.head = next.loc();
      loc
    } else {
      let index = self.next;
      self.next += 1;
      Wire(&self.area.get(index).expect("OOM").0 as _)
    };
    trace!(self.tracer, loc, self.head);
    loc
  }

  #[inline(never)]
  pub fn safe_alloc(&mut self) -> Wire {
    let loc = self.alloc();
    loc.target().store(Port::LOCK);
    loc.p2().target().store(Port::LOCK);
    loc
  }
}

#[repr(C)]
#[repr(align(16))]
#[derive(Default, Debug, Clone)]
pub struct Node(pub Port, pub Port);

#[repr(C)]
#[repr(align(16))]
#[derive(Default)]
pub struct ANode(pub APtr, pub APtr);

// The global node buffer.
pub type Data = [ANode];

/// Rewrite counter.
#[derive(Clone, Copy, Debug, Default)]
pub struct Rewrites {
  pub anni: usize, // anni rewrites
  pub comm: usize, // comm rewrites
  pub eras: usize, // eras rewrites
  pub dref: usize, // dref rewrites
  pub oper: usize, // oper rewrites
}

impl Rewrites {
  pub fn add_to(&self, target: &AtomicRewrites) {
    target.anni.fetch_add(self.anni, Relaxed);
    target.comm.fetch_add(self.comm, Relaxed);
    target.eras.fetch_add(self.eras, Relaxed);
    target.dref.fetch_add(self.dref, Relaxed);
    target.oper.fetch_add(self.oper, Relaxed);
  }

  // Total rewrite count.
  pub fn total(&self) -> usize {
    return self.anni + self.comm + self.eras + self.dref + self.oper;
  }
}

/// Rewrite counter, atomic.
#[derive(Default)]
pub struct AtomicRewrites {
  pub anni: AtomicUsize, // anni rewrites
  pub comm: AtomicUsize, // comm rewrites
  pub eras: AtomicUsize, // eras rewrites
  pub dref: AtomicUsize, // dref rewrites
  pub oper: AtomicUsize, // oper rewrites
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

#[derive(Clone, Debug)]
#[repr(align(16))]
pub struct Def {
  pub lab: Lab,
  pub inner: DefType,
}

#[derive(Clone, Debug)]
pub enum DefType {
  Native(fn(&mut Net, Port)),
  Net(DefNet),
}

/// A compact closed net, used for dereferences.
#[derive(Clone, Debug, Default)]
pub struct DefNet {
  pub root: Port,
  pub rdex: Vec<(Port, Port)>,
  pub node: Vec<Node>,
}

// A interaction combinator net.
pub struct Net<'a> {
  pub tid: usize,              // thread id
  pub tids: usize,             // thread count
  pub rdex: Vec<(Port, Port)>, // redexes
  pub locs: Vec<Wire>,
  pub rwts: Rewrites, // rewrite count
  pub quik: Rewrites, // quick rewrite count
  pub root: Wire,
  // allocator
  pub area: &'a Data,
  pub head: Wire,
  pub next: usize,
  //
  pub tracer: Tracer,
}

impl<'a> Net<'a> {
  // Creates an empty net with a given heap.
  pub fn new(area: &'a Data) -> Self {
    let mut net = Net::new_with_root(area, Wire::NULL);
    let root = net.safe_alloc();
    net.root = root;
    net
  }

  // Creates an empty net with a given heap.
  pub fn new_with_root(area: &'a Data, root: Wire) -> Self {
    Net {
      tid: 0,
      tids: 1,
      rdex: vec![],
      locs: vec![Wire::NULL; 1 << 16],
      rwts: Rewrites::default(),
      quik: Rewrites::default(),
      root,
      area,
      head: Wire::NULL,
      next: 0,
      tracer: Tracer::new(),
    }
  }

  // Creates a net and boots from a REF.
  pub fn boot(&mut self, def: &Def) {
    let def = Port::new_ref(def);
    trace!(self.tracer, def);
    self.root.target().store(def);
  }

  #[inline(always)]
  pub fn redux(&mut self, a: Port, b: Port) {
    trace!(self.tracer, a, b);
    if a.is_nilary() && b.is_nilary() {
      self.rwts.eras += 1;
    } else {
      self.rdex.push((a, b));
    }
  }

  #[inline(always)]
  pub fn create_node(&mut self, tag: Tag, lab: Lab) -> CreatedNode {
    let loc = self.alloc();
    CreatedNode {
      p0: Port::new(tag, lab, loc.clone()),
      p1: Port::new(Var, 0, loc.clone()),
      p2: Port::new(Var, 0, loc.p2()),
    }
  }

  // Links two pointers, forming a new wire. Assumes ownership.
  #[inline(always)]
  pub fn link_port_port(&mut self, a_port: Port, b_port: Port) {
    trace!(self.tracer, a_port, b_port);
    if a_port.is_pri() && b_port.is_pri() {
      return self.redux(a_port, b_port);
    } else {
      self.half_link_port_port(a_port.clone(), b_port.clone());
      self.half_link_port_port(b_port, a_port);
    }
  }

  // Given two locations, links both stored pointers, atomically.
  #[inline(always)]
  pub fn link_wire_wire(&mut self, a_wire: Wire, b_wire: Wire) {
    trace!(self.tracer, a_wire, b_wire);
    let a_port = a_wire.target().take();
    let b_port = b_wire.target().take();
    trace!(self.tracer, a_port, b_port);
    if a_port.is_pri() && b_port.is_pri() {
      self.half_free(a_wire);
      self.half_free(b_wire);
      return self.redux(a_port, b_port);
    } else {
      self.half_link_wire_port(a_port.clone(), a_wire, b_port.clone());
      self.half_link_wire_port(b_port, b_wire, a_port);
    }
  }

  // Given a location, link the pointer stored to another pointer, atomically.
  #[inline(always)]
  pub fn link_wire_port(&mut self, a_wire: Wire, b_port: Port) {
    trace!(self.tracer, a_wire, b_port);
    let a_port = a_wire.target().take();
    trace!(self.tracer, a_port);
    if a_port.is_pri() && b_port.is_pri() {
      self.half_free(a_wire);
      return self.redux(a_port, b_port);
    } else {
      self.half_link_wire_port(a_port.clone(), a_wire, b_port.clone());
      self.half_link_port_port(b_port, a_port);
    }
  }

  // When two threads interfere, uses the lock-free link algorithm described on the 'paper/'.
  #[inline(always)]
  pub fn half_link_port_port(&mut self, a_port: Port, b_port: Port) {
    trace!(self.tracer, a_port, b_port);
    if a_port.tag() == Var {
      a_port.target().store(b_port);
    }
  }

  // When two threads interfere, uses the lock-free link algorithm described on the 'paper/'.
  #[inline(always)]
  pub fn half_link_wire_port(&mut self, a_port: Port, a_wire: Wire, b_port: Port) {
    trace!(self.tracer, a_port, a_wire, b_port);
    // If 'a_port' is a var...
    if a_port.tag() == Var {
      let got = a_port.target().cas(a_wire.var(), b_port.clone());
      // Attempts to link using a compare-and-swap.
      if got.is_ok() {
        trace!(self.tracer, "cas ok");
        self.half_free(a_wire);
      // If the CAS failed, resolve by using redirections.
      } else {
        trace!(self.tracer, "cas fail", got.unwrap_err());
        if b_port.tag() == Var {
          let port = b_port.redirect();
          a_wire.target().store(port);
          //self.atomic_linker_var(a_port, a_wire, b_port);
        } else if b_port.is_pri() {
          a_wire.target().store(b_port.clone());
          self.atomic_linker_pri(a_port, a_wire, b_port);
        } else {
          unreachable!();
        }
      }
    } else {
      self.half_free(a_wire);
    }
  }

  // Atomic linker for when 'b_port' is a principal port.
  pub fn atomic_linker_pri(&mut self, mut a_port: Port, a_wire: Wire, b_port: Port) {
    trace!(self.tracer);
    loop {
      trace!(self.tracer, a_port, a_wire, b_port);
      // Peek the target, which may not be owned by us.
      let mut t_wire = a_port.loc();
      let mut t_port = t_wire.target().load();
      trace!(self.tracer, t_port);
      // If it is taken, we wait.
      if t_port == Port::LOCK {
        continue;
      }
      // If target is a rewireection, we own it. Clear and move forward.
      if t_port.tag() == Red {
        self.half_free(t_wire);
        a_port = t_port;
        continue;
      }
      // If target is a variable, we don't own it. Try replacing it.
      if t_port.tag() == Var {
        if t_wire.target().cas(t_port.clone(), b_port.clone()).is_ok() {
          trace!(self.tracer, "var cas ok");
          // Clear source location.
          self.half_free(a_wire);
          // Collect the orphaned backward path.
          t_wire = t_port.loc();
          t_port = t_port.target().load();
          while t_port.tag() == Red {
            trace!(self.tracer, t_wire, t_port);
            self.half_free(t_wire);
            t_wire = t_port.loc();
            t_port = t_wire.target().load();
          }
          return;
        }
        trace!(self.tracer, "var cas fail");
        // If the CAS failed, the var changed, so we try again.
        continue;
      }

      // If it is a node, two threads will reach this branch.
      if t_port.is_pri() || t_port == Port::GONE {
        // Sort references, to avoid deadlocks.
        let x_wire = if a_wire < t_wire { a_wire.clone() } else { t_wire.clone() };
        let y_wire = if a_wire < t_wire { t_wire.clone() } else { a_wire.clone() };
        trace!(self.tracer, x_wire, y_wire);
        // Swap first reference by Ptr::GONE placeholder.
        let x_port = x_wire.target().swap(Port::GONE);
        // First to arrive creates a redex.
        if x_port != Port::GONE {
          let y_port = y_wire.target().swap(Port::GONE);
          trace!(self.tracer, "fst", x_wire, y_wire, x_port, y_port);
          self.redux(x_port, y_port);
          return;
        // Second to arrive clears up the memory.
        } else {
          trace!(self.tracer, "snd", x_wire, y_wire);
          self.half_free(x_wire);
          while y_wire.target().cas(Port::GONE, Port::LOCK).is_err() {}
          self.half_free(y_wire);
          return;
        }
      }
      // Shouldn't be reached.
      trace!(self.tracer, t_port, a_wire, a_port, b_port);
      unreachable!()
    }
  }

  // Atomic linker for when 'b_port' is an aux port.
  pub fn atomic_linker_var(&mut self, _: Port, _: Wire, b_port: Port) {
    loop {
      let ste_wire = b_port.clone();
      let ste_port = ste_wire.target().load();
      if ste_port.tag() == Var {
        let trg_wire = ste_port.loc();
        let trg_port = trg_wire.target().load();
        if trg_port.tag() == Red {
          let neo_port = trg_port.unredirect();
          if ste_wire.target().cas(ste_port, neo_port).is_ok() {
            self.half_free(trg_wire);
            continue;
          }
        }
      }
      break;
    }
  }

  // Performs an interaction over a redex.
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
      (Mat, Ctr) // b.tag() == 0
      | (Ctr, Mat) // a.tag() == 0
      | (Op2, Op1)
      | (Op1, Op2)
      | (Op2, Mat)
      | (Mat, Op2)
      | (Op1, Mat)
      | (Mat, Op1) => todo!(),
    }
  }

  #[inline(never)]
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
  pub fn anni2(&mut self, a: Port, b: Port) {
    trace!(self.tracer, a, b);
    self.rwts.anni += 1;
    let a = a.consume_node();
    let b = b.consume_node();
    self.link_wire_wire(a.p1, b.p1);
    self.link_wire_wire(a.p2, b.p2);
  }

  #[inline(never)]
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

  #[inline(never)]
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
  pub fn comm02(&mut self, a: Port, b: Port) {
    trace!(self.tracer, a, b);
    self.rwts.comm += 1;
    let b = b.consume_node();
    self.link_wire_port(b.p1, a.clone());
    self.link_wire_port(b.p2, a);
  }

  #[inline(never)]
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

  #[inline(never)]
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
  /// --------------------------- | -----------X--------------- mat_num
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

  #[inline(never)]
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

  #[inline(never)]
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
  pub fn op1_num(&mut self, a: Port, b: Port) {
    trace!(self.tracer, a, b);
    self.rwts.oper += 1;
    let a = a.consume_op1();
    let n = a.num.num();
    let m = b.num();
    let out = a.op.op(n, m);
    self.link_wire_port(a.p2, Port::new_num(out));
  }

  // Expands a closed net.
  #[inline(never)]
  pub fn call(&mut self, port: Port, trg: Port) {
    trace!(self.tracer, port, trg);
    self.rwts.dref += 1;
    // Intercepts with a native function, if available.
    let def = port.loc().def();
    let net = match &def.inner {
      DefType::Native(native) => return native(self, trg),
      DefType::Net(net) => net,
    };
    let len = net.node.len();
    // Allocate space.
    for i in 0 .. len {
      *unsafe { self.locs.get_unchecked_mut(i) } = self.safe_alloc();
    }
    // Load nodes, adjusted.
    for i in 0 .. len {
      let loc = unsafe { self.locs.get_unchecked(i) }.clone();
      let p1 = self.adjust(&unsafe { net.node.get_unchecked(i) }.0);
      let p2 = self.adjust(&unsafe { net.node.get_unchecked(i) }.1);
      trace!(self.tracer, loc, p1, p2);
      loc.p1().target().store(p1);
      loc.p2().target().store(p2);
    }
    // Load redexes, adjusted.
    for r in &net.rdex {
      let p1 = self.adjust(&r.0);
      let p2 = self.adjust(&r.1);
      trace!(self.tracer, p1, p2);
      self.rdex.push((p1, p2));
    }
    trace!(self.tracer);
    // Load root, adjusted.
    self.link_port_port(self.adjust(&net.root), trg);
  }

  // Adjusts dereferenced pointer locations.
  #[inline(always)]
  fn adjust(&self, port: &Port) -> Port {
    if !port.is_nilary() && !port.is_null() {
      Port::new(
        port.tag(),
        port.lab(),
        (*unsafe { self.locs.get_unchecked(port.loc().index()) }).with_port(port.loc().port()),
      )
    } else {
      port.clone()
    }
  }

  // Reduces all redexes.
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
    return count;
  }

  // Expands heads.
  #[inline(always)]
  pub fn expand(&mut self) {
    fn go(net: &mut Net, wire: Wire, len: usize, key: usize) {
      trace!(net.tracer, wire);
      let port = wire.target().load();
      trace!(net.tracer, port);
      if port == Port::LOCK {
        return;
      }
      if port.tag() == Ctr {
        if len >= net.tids || key % 2 == 0 {
          go(net, port.p1(), len * 2, key / 2);
        }
        if len >= net.tids || key % 2 == 1 {
          go(net, port.p2(), len * 2, key / 2);
        }
      } else if port.tag() == Ref && port != Port::ERA {
        let got = wire.target().swap(Port::LOCK);
        if got != Port::LOCK {
          trace!(net.tracer, port, wire);
          net.call(port, wire.var());
        }
      }
    }
    go(self, self.root.clone(), 1, self.tid);
  }

  // Reduce a net to normal form.
  pub fn normal(&mut self) {
    self.expand();
    while self.rdex.len() > 0 {
      self.reduce(usize::MAX);
      self.expand();
    }
  }

  // Forks into child threads, returning a Net for the (tid/tids)'th thread.
  pub fn fork(&self, tid: usize, tids: usize) -> Self {
    let heap_size = self.area.len() / tids;
    let heap_start = heap_size * tid;
    let area = &self.area[heap_start .. heap_start + heap_size];
    let mut net = Net::new_with_root(area, self.root.clone());
    net.next = self.next.saturating_sub(heap_start);
    net.head = if tid == 0 { self.head.clone() } else { Wire::NULL };
    net.tid = tid;
    net.tids = tids;
    net.tracer.set_tid(tid);
    let from = self.rdex.len() * (tid + 0) / tids;
    let upto = self.rdex.len() * (tid + 1) / tids;
    for i in from .. upto {
      net.rdex.push(self.rdex[i].clone());
    }
    return net;
  }

  // Evaluates a term to normal form in parallel
  pub fn parallel_normal(&mut self) {
    const SHARE_LIMIT: usize = 1 << 12; // max share redexes per split
    const LOCAL_LIMIT: usize = 1 << 18; // max local rewrites per epoch

    // Local thread context
    struct ThreadContext<'a> {
      tid: usize,                   // thread id
      tids: usize,                  // thread count
      tlog2: usize,                 // log2 of thread count
      tick: usize,                  // current tick
      net: Net<'a>,                 // thread's own net object
      delta: &'a AtomicRewrites,    // global delta rewrites
      quick: &'a AtomicRewrites,    // global delta rewrites
      share: &'a Vec<(APtr, APtr)>, // global share buffer
      rlens: &'a Vec<AtomicUsize>,  // global redex lengths
      total: &'a AtomicUsize,       // total redex length
      barry: Arc<Barrier>,          // synchronization barrier
    }

    // Initialize global objects
    let cores = std::thread::available_parallelism().unwrap().get() as usize;
    let tlog2 = cores.ilog2() as usize;
    let tids = 1 << tlog2;
    let delta = AtomicRewrites::default(); // delta rewrite counter
    let quick = AtomicRewrites::default(); // quick rewrite counter
    let rlens = (0 .. tids).map(|_| AtomicUsize::new(0)).collect::<Vec<_>>();
    let share =
      (0 .. SHARE_LIMIT * tids).map(|_| (APtr(AtomicU64::new(0)), APtr(AtomicU64::new(0)))).collect::<Vec<_>>();
    let total = AtomicUsize::new(0); // sum of redex bag length
    let barry = Arc::new(Barrier::new(tids)); // global barrier

    // Perform parallel reductions
    std::thread::scope(|s| {
      for tid in 0 .. tids {
        let mut ctx = ThreadContext {
          tid,
          tids,
          tick: 0,
          net: self.fork(tid, tids),
          tlog2,
          delta: &delta,
          quick: &quick,
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
    quick.add_to(&mut self.quik);

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
      ctx.net.quik.add_to(ctx.quick);
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
      return ctx.total.load(Relaxed);
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
          targ.0.store(rdx1.0);
          targ.1.store(rdx1.1);
        }
        ctx.net.rdex.truncate(a_len - send);
        ctx.barry.wait();
        for i in 0 .. recv {
          let got = ctx.share.get_unchecked(a_tid * SHARE_LIMIT + i);
          ctx.net.rdex.push((got.0.load(), got.1.load()));
        }
      }
    }
  }
}
