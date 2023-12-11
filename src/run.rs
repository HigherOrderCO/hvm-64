// An efficient Interaction Combinator runtime
// ===========================================
// This file implements an efficient interaction combinator runtime. Nodes are represented by 2 aux
// ports (P1, P2), with the main port (P1) omitted. A separate vector, 'rdex', holds main ports,
// and, thus, tracks active pairs that can be reduced in parallel. Pointers are unboxed, meaning
// that Ptr::ERAs, NUMs and REFs don't use any additional space. REFs lazily expand to closed nets when
// they interact with nodes, and are cleared when they interact with Ptr::ERAs, allowing for constant
// space evaluation of recursive functions on Scott encoded datatypes.

use crate::ops::Op;
use std::{
  collections::HashMap,
  default, fmt,
  sync::{
    atomic::{AtomicU64, AtomicUsize, Ordering::Relaxed},
    Arc, Barrier,
  },
};

macro_rules! trace {
  ($($x:tt)*) => {
    if cfg!(feature = "trace")  {
      eprintln!($($x)*);
    }
  };
}

#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
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

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct Loc(pub *const APtr);

impl fmt::Debug for Loc {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    write!(f, "{:012x?}", self.0 as usize)
  }
}

unsafe impl Send for Loc {}

const PORT_MASK: Val = 0b1000;

impl Loc {
  pub const NULL: Loc = Loc(std::ptr::null());

  pub fn index(&self) -> usize {
    self.0 as usize >> 4
  }

  pub fn port(&self) -> u8 {
    (((self.0 as usize as Val) & PORT_MASK) >> 3) as u8
  }

  pub fn local(index: usize, port: u8) -> Loc {
    Loc(((index << 4) | ((port as usize) << 3)) as *const _)
  }

  pub fn with_port(&self, port: u8) -> Loc {
    Loc(((self.0 as Val) & !PORT_MASK | ((port as Val) << 3)) as _)
  }

  pub fn p0(&self) -> Loc {
    Loc(((self.0 as Val) & !PORT_MASK) as _)
  }

  pub fn p1(&self) -> Loc {
    Loc(((self.0 as Val) & !PORT_MASK) as _)
  }

  pub fn p2(&self) -> Loc {
    Loc(((self.0 as Val) | PORT_MASK) as _)
  }

  pub fn other(&self) -> Loc {
    Loc(((self.0 as Val) ^ PORT_MASK) as _)
  }

  pub fn target<'a>(self) -> &'a APtr {
    unsafe { &*self.0 }
  }

  pub fn def<'a>(self) -> &'a Def {
    unsafe { &*(self.0 as *const _) }
  }
}

pub type Val = u64;
pub type AVal = AtomicU64;

/// A tagged pointer.
#[derive(Copy, Clone, Eq, PartialEq, PartialOrd, Hash, Default)]
#[repr(transparent)]
pub struct Ptr(pub Val);

impl fmt::Debug for Ptr {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    write!(f, "{:016x?} ", self.0)?;
    match self {
      &Ptr::ERA => write!(f, "[ERA]"),
      &Ptr::NULL => write!(f, "[Ptr::NULL]"),
      &Ptr::GONE => write!(f, "[Ptr::GONE]"),
      &Ptr::LOCK => write!(f, "[Ptr::LOCK]"),
      _ => match self.tag() {
        Num => write!(f, "[Num {}]", self.num()),
        Var | Red => write!(f, "[{:?} {} {:?}]", self.tag(), self.loc().port(), self.loc()),
        Ref | Mat => write!(f, "[{:?} {:?}]", self.tag(), self.loc()),
        Op2 | Op1 | Ctr => write!(f, "[{:?} {:?} {:?}]", self.tag(), self.lab(), self.loc()),
      },
    }
  }
}

impl Ptr {
  pub const ERA: Ptr = Ptr(Ref as _);
  pub const NULL: Ptr = Ptr(0x0000_0000_0000_0000);
  pub const LOCK: Ptr = Ptr(0xFFFF_FFFF_FFFF_FFF0);
  pub const GONE: Ptr = Ptr(0xFFFF_FFFF_FFFF_FFFF);

  #[inline(always)]
  pub fn new(tag: Tag, lab: Lab, loc: Loc) -> Self {
    Ptr(((lab as Val) << 48) | (loc.0 as usize as Val) | (tag as Val))
  }

  #[inline(always)]
  pub const fn new_num(val: Val) -> Self {
    Ptr((val << 4) | (Num as Val))
  }

  #[inline(always)]
  pub fn new_ref(def: &Def) -> Ptr {
    Ptr::new(Ref, def.lab, Loc(def as *const _ as _))
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
  pub const fn loc(&self) -> Loc {
    Loc((self.0 & 0x0000_FFFF_FFFF_FFF8) as usize as _)
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
  pub fn p1(&self) -> Ptr {
    Ptr::new(Var, 0, self.loc().p1())
  }

  #[inline(always)]
  pub fn p2(&self) -> Ptr {
    Ptr::new(Var, 0, self.loc().p2())
  }

  #[inline(always)]
  pub fn target(&self) -> &APtr {
    self.loc().target()
  }

  #[inline(always)]
  pub fn redirect(&self) -> Ptr {
    return Ptr::new(Red, 0, self.loc());
  }

  #[inline(always)]
  pub fn unredirect(&self) -> Ptr {
    return Ptr::new(Var, 0, self.loc());
  }
}

/// An atomic tagged pointer.
#[repr(transparent)]
#[derive(Default)]
pub struct APtr(pub AVal);

impl APtr {
  #[inline(always)]
  pub fn new(ptr: Ptr) -> Self {
    APtr(AtomicU64::new(ptr.0))
  }

  #[inline(always)]
  pub fn load(&self) -> Ptr {
    Ptr(self.0.load(Relaxed))
  }

  #[inline(always)]
  pub fn store(&self, ptr: Ptr) {
    self.0.store(ptr.0, Relaxed);
  }

  #[inline(always)]
  pub fn cas(&self, expected: Ptr, value: Ptr) -> Result<Ptr, Ptr> {
    self.0.compare_exchange_weak(expected.0, value.0, Relaxed, Relaxed).map(Ptr).map_err(Ptr)
  }

  #[inline(always)]
  pub fn cas_strong(&self, expected: Ptr, value: Ptr) -> Result<Ptr, Ptr> {
    self.0.compare_exchange(expected.0, value.0, Relaxed, Relaxed).map(Ptr).map_err(Ptr)
  }

  #[inline(always)]
  pub fn swap(&self, value: Ptr) -> Ptr {
    Ptr(self.0.swap(value.0, Relaxed))
  }

  // Takes a pointer's target.
  #[inline(always)]
  pub fn take(&self) -> Ptr {
    loop {
      let got = self.swap(Ptr::LOCK);
      if got != Ptr::LOCK {
        std::hint::spin_loop();
        return got;
      }
    }
  }
}

// A handy wrapper around Data.
pub struct Heap<'a> {
  pub head: Loc,
  pub area: &'a Data,
  pub next: usize,
}

impl<'a> Heap<'a> {
  pub fn init(size: usize) -> Box<[ANode]> {
    let mut data = Vec::with_capacity(size);
    data.resize_with(size, Default::default);
    return data.into_boxed_slice();
  }

  pub fn new(data: &'a Data) -> Self {
    Heap { area: data, next: 0, head: Loc::NULL }
  }

  #[inline(always)]
  pub fn half_free(&mut self, loc: Loc) {
    loc.target().store(Ptr::NULL);
    if loc.other().target().load() == Ptr::NULL {
      let loc = loc.p0();
      if let Ok(x) = loc.target().cas_strong(Ptr::NULL, Ptr::new(Red, 1, self.head)) {
        self.head = loc;
      }
    }
  }

  #[inline(always)]
  pub fn alloc(&mut self) -> Loc {
    let loc = if self.head != Loc::NULL {
      let loc = self.head;
      self.head = self.head.target().load().loc();
      loc
    } else {
      let index = self.next;
      self.next += 1;
      Loc(&unsafe { self.area.get_unchecked(index) }.0 as _)
    };
    loc.target().store(Ptr::LOCK);
    loc.p2().target().store(Ptr::LOCK);
    loc
  }
}

#[repr(C)]
#[repr(align(16))]
#[derive(Default, Debug, Clone, Copy)]
pub struct Node(pub Ptr, pub Ptr);

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
  Native(fn(&mut Net, Ptr)),
  Net(DefNet),
}

/// A compact closed net, used for dereferences.
#[derive(Clone, Debug, Default)]
pub struct DefNet {
  pub root: Ptr,
  pub rdex: Vec<(Ptr, Ptr)>,
  pub node: Vec<Node>,
}

// A interaction combinator net.
pub struct Net<'a> {
  pub tid: usize,            // thread id
  pub tids: usize,           // thread count
  pub heap: Heap<'a>,        // allocator
  pub rdex: Vec<(Ptr, Ptr)>, // redexes
  pub locs: Vec<Loc>,
  pub rwts: Rewrites, // rewrite count
  pub quik: Rewrites, // quick rewrite count
  pub root: Loc,
}

impl<'a> Net<'a> {
  // Creates an empty net with a given heap.
  pub fn new(mut heap: Heap<'a>) -> Self {
    let root = heap.alloc();
    Net::new_with_root(heap, root)
  }

  // Creates an empty net with a given heap.
  pub fn new_with_root(heap: Heap<'a>, root: Loc) -> Self {
    Net {
      tid: 0,
      tids: 1,
      heap,
      rdex: vec![],
      locs: vec![Loc::NULL; 1 << 16],
      rwts: Rewrites::default(),
      quik: Rewrites::default(),
      root,
    }
  }

  // Creates a net and boots from a REF.
  pub fn boot(&mut self, def: &Def) {
    self.root.target().store(Ptr::new_ref(def));
  }

  #[inline(always)]
  pub fn redux(&mut self, a: Ptr, b: Ptr) {
    if a.is_nilary() && b.is_nilary() {
      self.rwts.eras += 1;
    } else {
      self.rdex.push((a, b));
    }
  }

  // Links two pointers, forming a new wire. Assumes ownership.
  #[inline(always)]
  pub fn link(&mut self, a_ptr: Ptr, b_ptr: Ptr) {
    if a_ptr.is_pri() && b_ptr.is_pri() {
      return self.redux(a_ptr, b_ptr);
    } else {
      self.linker(a_ptr, b_ptr);
      self.linker(b_ptr, a_ptr);
    }
  }

  // Given two locations, links both stored pointers, atomically.
  #[inline(always)]
  pub fn atomic_link(&mut self, a_dir: Ptr, b_dir: Ptr) {
    trace!("[{:04x}] atomic_link {:?} {:?}", self.tid, a_dir, b_dir);
    let a_ptr = a_dir.target().take();
    let b_ptr = b_dir.target().take();
    trace!("[{:08x}] took {:?} {:?}", self.tid, a_ptr, b_ptr);
    if a_ptr.is_pri() && b_ptr.is_pri() {
      self.heap.half_free(a_dir.loc());
      self.heap.half_free(b_dir.loc());
      return self.redux(a_ptr, b_ptr);
    } else {
      self.atomic_linker(a_ptr, a_dir, b_ptr);
      self.atomic_linker(b_ptr, b_dir, a_ptr);
    }
  }

  // Given a location, link the pointer stored to another pointer, atomically.
  #[inline(always)]
  pub fn half_atomic_link(&mut self, a_dir: Ptr, b_ptr: Ptr) {
    let a_ptr = a_dir.target().take();
    if a_ptr.is_pri() && b_ptr.is_pri() {
      self.heap.half_free(a_dir.loc());
      return self.redux(a_ptr, b_ptr);
    } else {
      self.atomic_linker(a_ptr, a_dir, b_ptr);
      self.linker(b_ptr, a_ptr);
    }
  }

  // When two threads interfere, uses the lock-free link algorithm described on the 'paper/'.
  #[inline(always)]
  pub fn linker(&mut self, a_ptr: Ptr, b_ptr: Ptr) {
    if a_ptr.tag() == Var {
      a_ptr.target().store(b_ptr);
    }
  }

  // When two threads interfere, uses the lock-free link algorithm described on the 'paper/'.
  #[inline(always)]
  pub fn atomic_linker(&mut self, a_ptr: Ptr, a_dir: Ptr, b_ptr: Ptr) {
    // If 'a_ptr' is a var...
    if a_ptr.tag() == Var {
      let got = a_ptr.target().cas(a_dir, b_ptr);
      // Attempts to link using a compare-and-swap.
      if got.is_ok() {
        self.heap.half_free(a_dir.loc());
      // If the CAS failed, resolve by using redirections.
      } else {
        trace!("[{:04x}] cas fail {:016x}", self.tid, got.unwrap_err().0);
        if b_ptr.tag() == Var {
          let ptr = b_ptr.redirect();
          a_dir.target().store(ptr);
          //self.atomic_linker_var(a_ptr, a_dir, b_ptr);
        } else if b_ptr.is_pri() {
          a_dir.target().store(b_ptr);
          self.atomic_linker_pri(a_ptr, a_dir, b_ptr);
        } else {
          unreachable!();
        }
      }
    } else {
      self.heap.half_free(a_dir.loc());
    }
  }

  // Atomic linker for when 'b_ptr' is a principal port.
  pub fn atomic_linker_pri(&mut self, mut a_ptr: Ptr, a_dir: Ptr, b_ptr: Ptr) {
    loop {
      // Peek the target, which may not be owned by us.
      let mut t_dir = a_ptr;
      let mut t_ptr = t_dir.target().load();
      // If it is taken, we wait.
      if t_ptr == Ptr::LOCK {
        std::hint::spin_loop();
        continue;
      }
      // If target is a redirection, we own it. Clear and move forward.
      if t_ptr.tag() == Red {
        self.heap.half_free(t_dir.loc());
        a_ptr = t_ptr;
        continue;
      }
      // If target is a variable, we don't own it. Try replacing it.
      if t_ptr.tag() == Var {
        if t_dir.target().cas(t_ptr, b_ptr).is_ok() {
          trace!("[{:04x}] var", self.tid);
          // Clear source location.
          self.heap.half_free(a_dir.loc());
          // Collect the orphaned backward path.
          t_dir = t_ptr;
          t_ptr = t_ptr.target().load();
          while t_ptr.tag() == Red {
            self.heap.half_free(t_dir.loc());
            t_dir = t_ptr;
            t_ptr = t_dir.target().load();
          }
          return;
        }
        // If the CAS failed, the var changed, so we try again.
        continue;
      }
      // If it is a node, two threads will reach this branch.
      if t_ptr.is_pri() || t_ptr == Ptr::GONE {
        // Sort references, to avoid deadlocks.
        let x_dir = if a_dir < t_dir { a_dir } else { t_dir };
        let y_dir = if a_dir < t_dir { t_dir } else { a_dir };
        // Swap first reference by Ptr::GONE placeholder.
        let x_ptr = x_dir.target().swap(Ptr::GONE);
        // First to arrive creates a redex.
        if x_ptr != Ptr::GONE {
          trace!("[{:04x}] fst {:016x}", self.tid, x_ptr.0);
          let y_ptr = y_dir.target().swap(Ptr::GONE);
          self.redux(x_ptr, y_ptr);
          return;
        // Second to arrive clears up the memory.
        } else {
          trace!("[{:04x}] snd", self.tid);
          self.heap.half_free(x_dir.loc());
          while y_dir.target().cas(Ptr::GONE, Ptr::NULL).is_err() {}
          self.heap.half_free(y_dir.loc());
          return;
        }
      }
      // Shouldn't be reached.
      trace!("[{:04x}] {:016x} | {:016x} {:016x} {:016x}", self.tid, t_ptr.0, a_dir.0, a_ptr.0, b_ptr.0);
      unreachable!()
    }
  }

  // Atomic linker for when 'b_ptr' is an aux port.
  pub fn atomic_linker_var(&mut self, a_ptr: Ptr, a_dir: Ptr, b_ptr: Ptr) {
    loop {
      let ste_dir = b_ptr;
      let ste_ptr = ste_dir.target().load();
      if ste_ptr.tag() == Var {
        let trg_dir = ste_ptr;
        let trg_ptr = trg_dir.target().load();
        if trg_ptr.tag() == Red {
          let neo_ptr = trg_ptr.unredirect();
          if ste_dir.target().cas(ste_ptr, neo_ptr).is_ok() {
            self.heap.half_free(trg_dir.loc());
            continue;
          }
        }
      }
      break;
    }
  }

  // Performs an interaction over a redex.
  #[inline(always)]
  pub fn interact(&mut self, a: Ptr, b: Ptr) {
    trace!("[{:04x}] interact {:?}\n                {:?}", self.tid, a, b);
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
      (Ref, _) if a == Ptr::ERA => self.comm02(a, b),
      (_, Ref) if b == Ptr::ERA => self.comm02(b, a),
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

  pub fn anni2(&mut self, a: Ptr, b: Ptr) {
    self.rwts.anni += 1;
    self.atomic_link(a.p1(), b.p1());
    self.atomic_link(a.p2(), b.p2());
  }

  pub fn anni1(&mut self, a: Ptr, b: Ptr) {
    // todo: is this right?
    self.rwts.anni += 1;
    self.atomic_link(a.p2(), b.p2());
  }

  pub fn comm22(&mut self, a: Ptr, b: Ptr) {
    self.rwts.comm += 1;
    let B1 = Ptr::new(Ctr, b.lab(), self.heap.alloc());
    let B2 = Ptr::new(Ctr, b.lab(), self.heap.alloc());
    let A1 = Ptr::new(Ctr, a.lab(), self.heap.alloc());
    let A2 = Ptr::new(Ctr, a.lab(), self.heap.alloc());
    B1.p1().target().store(A1.p1());
    B1.p2().target().store(A2.p1());
    B2.p1().target().store(A1.p2());
    B2.p2().target().store(A2.p2());
    A1.p1().target().store(B1.p1());
    A1.p2().target().store(B2.p1());
    A2.p1().target().store(B1.p2());
    A2.p2().target().store(B2.p2());
    self.half_atomic_link(a.p1(), B1);
    self.half_atomic_link(a.p2(), B2);
    self.half_atomic_link(b.p1(), A1);
    self.half_atomic_link(b.p2(), A2);
  }

  pub fn comm12(&mut self, a: Ptr, b: Ptr) {
    self.rwts.comm += 1;
    let n = a.p1().target().load();
    self.heap.half_free(a.p1().loc());
    let B2 = Ptr::new(Ctr, b.lab(), self.heap.alloc());
    let A1 = Ptr::new(Ctr, a.lab(), self.heap.alloc());
    let A2 = Ptr::new(Ctr, a.lab(), self.heap.alloc());
    B2.p1().target().store(A1.p2());
    B2.p2().target().store(A2.p2());
    A1.p1().target().store(n);
    A1.p2().target().store(B2.p1());
    A2.p1().target().store(n);
    A2.p2().target().store(B2.p2());
    self.half_atomic_link(a.p2(), B2);
    self.half_atomic_link(b.p1(), A1);
    self.half_atomic_link(b.p2(), A2);
  }

  pub fn comm02(&mut self, a: Ptr, b: Ptr) {
    self.rwts.comm += 1;
    self.half_atomic_link(b.p1(), a);
    self.half_atomic_link(b.p2(), a);
  }

  pub fn comm01(&mut self, a: Ptr, b: Ptr) {
    self.rwts.comm += 1;
    self.heap.half_free(b.p1().loc());
    self.half_atomic_link(b.p2(), a);
  }

  pub fn mat_num(&mut self, a: Ptr, b: Ptr) {
    self.rwts.oper += 1;
    if b.num() == 0 {
      let x = Ptr::new(Ctr, 0, self.heap.alloc());
      x.p2().target().store(Ptr::ERA);
      self.half_atomic_link(a.p2(), x.p1());
      self.half_atomic_link(a.p1(), x);
    } else {
      let x = Ptr::new(Ctr, 0, self.heap.alloc());
      let y = Ptr::new(Ctr, 0, self.heap.alloc());
      x.p1().target().store(Ptr::ERA);
      x.p2().target().store(y);
      y.p1().target().store(Ptr::new_num(b.num() - 1));
      self.half_atomic_link(a.p1(), x);
      self.half_atomic_link(a.p2(), y.p2());
    }
  }

  pub fn op2_num(&mut self, a: Ptr, b: Ptr) {
    self.rwts.oper += 1;
    let x = Ptr::new(Op1, a.lab(), self.heap.alloc());
    x.p1().target().store(b);
    self.half_atomic_link(a.p2(), x.p2());
    self.half_atomic_link(a.p1(), x);
  }

  pub fn op1_num(&mut self, a: Ptr, b: Ptr) {
    self.rwts.oper += 1;
    let op = a.op();
    let v0 = a.p1().target().load().num();
    self.heap.half_free(b.p1().loc());
    let v1 = b.num();
    let v2 = op.op(v0, v1);
    self.half_atomic_link(a.p2(), Ptr::new_num(v2));
  }

  // Expands a closed net.
  #[inline(always)]
  pub fn call(&mut self, ptr: Ptr, trg: Ptr) {
    self.rwts.dref += 1;
    // Intercepts with a native function, if available.
    let def = ptr.loc().def();
    let net = match &def.inner {
      DefType::Native(native) => return native(self, trg),
      DefType::Net(net) => net,
    };
    let len = net.node.len();
    // Allocate space.
    for i in 0 .. len {
      *unsafe { self.locs.get_unchecked_mut(i) } = self.heap.alloc();
    }
    // Load nodes, adjusted.
    for i in 0 .. len {
      let p1 = self.adjust(unsafe { net.node.get_unchecked(i) }.0);
      let p2 = self.adjust(unsafe { net.node.get_unchecked(i) }.1);
      let lc = *unsafe { self.locs.get_unchecked(i) };
      lc.p1().target().store(p1);
      lc.p2().target().store(p2);
    }
    // Load redexes, adjusted.
    for r in &net.rdex {
      let p1 = self.adjust(r.0);
      let p2 = self.adjust(r.1);
      self.rdex.push((p1, p2));
    }
    // Load root, adjusted.
    self.link(self.adjust(net.root), trg);
  }

  // Adjusts dereferenced pointer locations.
  #[inline(always)]
  fn adjust(&self, ptr: Ptr) -> Ptr {
    if !ptr.is_nilary() && !ptr.is_null() {
      let loc = ptr.loc().0 as usize;
      return Ptr::new(
        ptr.tag(),
        ptr.lab(),
        (*unsafe { self.locs.get_unchecked(ptr.loc().index()) }).with_port(ptr.loc().port()),
      );
    } else {
      return ptr;
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
    fn go(net: &mut Net, dir: Ptr, len: usize, key: usize) {
      trace!("[{:04x}] expand dir: {:?}", net.tid, dir);
      let ptr = dir.target().load();
      if ptr == Ptr::LOCK {
        return;
      }
      if ptr.tag() == Ctr {
        if len >= net.tids || key % 2 == 0 {
          go(net, ptr.p1(), len * 2, key / 2);
        }
        if len >= net.tids || key % 2 == 1 {
          go(net, ptr.p2(), len * 2, key / 2);
        }
      } else if ptr.tag() == Ref && ptr != Ptr::ERA {
        let got = dir.target().swap(Ptr::LOCK);
        if got != Ptr::LOCK {
          trace!("[{:08x}] expand {:?} {:?}", net.tid, ptr, dir);
          net.call(ptr, dir);
        }
      }
    }
    go(self, Ptr::new(Var, 0, self.root), 1, self.tid);
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
    let heap_size = self.heap.area.len() / tids;
    let heap_start = heap_size * tid;
    let heap = Heap {
      area: &self.heap.area[heap_start .. heap_start + heap_size],
      next: self.heap.next.saturating_sub(heap_start),
      head: if tid == 0 { self.heap.head } else { Loc::NULL },
    };
    let mut net = Net::new_with_root(heap, self.root);
    net.tid = tid;
    net.tids = tids;
    let from = self.rdex.len() * (tid + 0) / tids;
    let upto = self.rdex.len() * (tid + 1) / tids;
    for i in from .. upto {
      net.rdex.push((self.rdex[i].0, self.rdex[i].1));
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
        s.spawn(move || main(&mut ctx));
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
        let reduced = ctx.net.reduce(LOCAL_LIMIT);
        trace!("[{:04x}] reduced {}", ctx.tid, reduced);
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
          let rdx0 = *ctx.net.rdex.get_unchecked(init + i * 2 + 0);
          let rdx1 = *ctx.net.rdex.get_unchecked(init + i * 2 + 1);
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
