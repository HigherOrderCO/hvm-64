// An efficient Interaction Combinator runtime
// ===========================================
// This file implements an efficient interaction combinator runtime. Nodes are represented by 2 aux
// ports (P1, P2), with the main port (P1) omitted. A separate vector, 'rdex', holds main ports,
// and, thus, tracks active pairs that can be reduced in parallel. Pointers are unboxed, meaning
// that ERAs, NUMs and REFs don't use any additional space. REFs lazily expand to closed nets when
// they interact with nodes, and are cleared when they interact with ERAs, allowing for constant
// space evaluation of recursive functions on Scott encoded datatypes.

pub type Val = u32;

// Core terms.
#[repr(u8)]
#[derive(Eq, PartialEq, PartialOrd, Ord, Clone, Copy)]
pub enum Tag {
  /// Variable to aux port 1
  VR1,
  /// Variable to aux port 2
  VR2,
  /// Redirect to aux port 1
  RD1,
  /// Redirect to aux port 2
  RD2,
  /// Lazy closed net
  REF,
  /// Unboxed eraser
  ERA,
  /// Unboxed number
  NUM,
  /// Binary numeric operation
  OP2,
  /// Unary numeric operation
  OP1,
  /// Numeric if-then-else(MATCH)
  MAT,
  /// Main port of con node(label 0)
  CT0,
  /// Main port of con node(label 1)
  CT1,
  /// Main port of con node(label 2)
  CT2,
  /// Main port of con node(label 3)
  CT3,
  /// Main port of con node(label 4)
  CT4,
  /// Main port of con node(label 5)
  CT5,
}
pub use Tag::*;

// Numeric operations.
pub const USE: u8 = 0x0; // set-next-op
pub const ADD: u8 = 0x1; // addition
pub const SUB: u8 = 0x2; // subtraction
pub const MUL: u8 = 0x3; // multiplication
pub const DIV: u8 = 0x4; // division
pub const MOD: u8 = 0x5; // modulus
pub const EQ : u8 = 0x6; // equal-to
pub const NE : u8 = 0x7; // not-equal-to
pub const LT : u8 = 0x8; // less-than
pub const GT : u8 = 0x9; // greater-than
pub const AND: u8 = 0xA; // logical-and
pub const OR : u8 = 0xB; // logical-or
pub const XOR: u8 = 0xC; // logical-xor
pub const NOT: u8 = 0xD; // logical-not
pub const LSH: u8 = 0xE; // left-shift
pub const RSH: u8 = 0xF; // right-shift

// Root pointer.
pub const ERAS: Address = Address(0x0000_0000 | ERA as Val);
pub const ROOT: Address = Address(0x0000_0000 | VR2 as Val);
pub const NULL: Address = Address(0x0000_0000);

// An auxiliary port.
#[repr(u8)]
#[derive(Debug, Clone, PartialEq, Eq, Hash, Copy)]
pub enum Port {
  P1,
  P2,
}
pub use Port::*;

// A tagged pointer.
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
pub struct Address(pub Val);

type AuxPortAddresses = (Address, Address);
type Redex = (Address, Address);

#[derive(Clone, Debug, Eq, PartialEq, Hash)]
pub struct NodeStorage {
  /// The addresses of the targets of each node's auxiliary ports
  /// Suppose we have a node at address #A, with auxiliary ports P1 and P2
  /// ```
  /// aux_ports[#A] = (#F, #J)
  /// aux_ports[#F] = (.., ..)
  /// ```
  aux_ports: Vec<AuxPortAddresses>,
  next: usize,
  used: usize,
  full: bool,
}

/// Holds the statistics of a given execution 
#[derive(Clone, Debug, Eq, PartialEq, Hash, Default, Copy)]
pub struct Stats {
  pub annihilations: usize,
  pub commutations: usize,
  pub erasures: usize,
  pub derefs: usize,
  pub numeric_ops: usize,
}


// A interaction combinator net.
#[derive(Clone, Debug, Eq, PartialEq, Hash)]
pub struct Net {
  pub redexes: Vec<Redex>,
  pub node_storage: NodeStorage,
  pub stats: Stats,
}

// A compact closed net, used for dereferences.
#[derive(Clone, Debug, Eq, PartialEq, Hash, Default)]
pub struct Def {
  pub redexes: Vec<Redex>,
  pub nodes: Vec<AuxPortAddresses>,
}

// A map of id to definitions (closed nets).
pub type Book = Vec<Def>;

pub fn new_book() -> Book {
  vec![Def::default(); 1 << 24]
}

// Patterns for easier matching on tags
macro_rules! CTR{() => {CT0 | CT1 | CT2 | CT3 | CT4 | CT5}}
macro_rules! VAR{() => {VR1 | VR2}}
macro_rules! RED{() => {RD1 | RD2}}
macro_rules! OPS{() => {OP2 | OP1 | MAT}}
macro_rules! PRI{() => {REF | ERA | NUM | OPS!() | CTR!()}}

impl From<Tag> for u8 {
  #[inline(always)]
  fn from(tag: Tag) -> Self {
    tag as u8
  }
}

impl From<u8> for Tag {
  #[inline(always)]
  fn from(value: u8) -> Self {
    unsafe {
      std::mem::transmute(value)
    }
  }
}

impl From<u32> for Port {
  #[inline(always)]
  fn from(value: u32) -> Self {
    unsafe {
      std::mem::transmute((value as u8) & 1)
    }
  }
}

impl Address {
  #[inline(always)]
  pub fn new(tag: Tag, val: Val) -> Self {
    Address((val << 4) | (tag as Val))
  }

  #[inline(always)]
  pub fn data(&self) -> Val {
    self.0
  }

  #[inline(always)]
  pub fn tag(&self) -> Tag {
    let tag_bits = (self.data() & 0xF) as u8;
    tag_bits.into()
  }

  #[inline(always)]
  pub fn val(&self) -> Val {
    (self.data() >> 4) as Val
  }

  #[inline(always)]
  pub fn is_nil(&self) -> bool {
    self.data() == 0
  }

  #[inline(always)]
  pub fn is_var(&self) -> bool {
    matches!(self.tag(), VAR!())
  }

  #[inline(always)]
  pub fn is_era(&self) -> bool {
    matches!(self.tag(), ERA)
  }

  #[inline(always)]
  pub fn is_ctr(&self) -> bool {
    matches!(self.tag(), CTR!())
  }

  #[inline(always)]
  pub fn is_ref(&self) -> bool {
    matches!(self.tag(), REF)
  }

  #[inline(always)]
  pub fn is_pri(&self) -> bool {
    matches!(self.tag(), PRI!())
  }

  #[inline(always)]
  pub fn is_num(&self) -> bool {
    matches!(self.tag(), NUM)
  }

  #[inline(always)]
  pub fn is_op1(&self) -> bool {
    matches!(self.tag(), OP1)
  }

  #[inline(always)]
  pub fn is_op2(&self) -> bool {
    matches!(self.tag(), OP2)
  }

  #[inline(always)]
  pub fn is_skp(&self) -> bool {
    matches!(self.tag(), ERA | NUM | REF)
  }

  #[inline(always)]
  pub fn is_mat(&self) -> bool {
    matches!(self.tag(), MAT)
  }

  #[inline(always)]
  pub fn has_loc(&self) -> bool {
    matches!(self.tag(), VAR!() | OP2 | OP1 | MAT | CTR!())
  }

  #[inline(always)]
  pub fn adjust(&self, loc: Val) -> Address {
    Address::new(self.tag(), self.val() + if self.has_loc() { loc - 1 } else { 0 })
  }

  // Can this redex be skipped (as an optimization)?
  #[inline(always)]
  pub fn can_skip(a: Address, b: Address) -> bool {
    matches!((a.tag(), b.tag()), (ERA | REF, ERA | REF))
  }
}

impl NodeStorage {
  pub fn new(size: usize) -> NodeStorage {
    NodeStorage {
      aux_ports: vec![(NULL, NULL); size],
      next: 1,
      used: 0,
      full: false,
    }
  }

  #[inline(always)]
  pub fn alloc(&mut self, size: usize) -> Val {
    if size == 0 {
      0
    } else if !self.full && self.next + size <= self.aux_ports.len() {
      self.used += size;
      self.next += size;
      (self.next - size) as Val
    } else {
      self.full = true;
      let mut space = 0;
      loop {
        if self.next >= self.aux_ports.len() {
          space = 0;
          self.next = 1;
        }
        if self.get(self.next as Val, P1).is_nil() {
          space += 1;
        } else {
          space = 0;
        }
        self.next += 1;
        if space == size {
          self.used += size;
          return (self.next - space) as Val;
        }
      }
    }
  }

  #[inline(always)]
  pub fn free(&mut self, index: Val) {
    self.used -= 1;
    self.set(index, P1, NULL);
    self.set(index, P2, NULL);
  }

  #[inline(always)]
  pub fn lock(&self, index: Val) {}

  #[inline(always)]
  pub fn unlock(&self, index: Val) {}

  #[inline(always)]
  pub fn get(&self, index: Val, port: Port) -> Address {
    unsafe {
      let node = self.aux_ports.get_unchecked(index as usize);
      match port {
        P1 => node.0,
        P2 => node.1,
      }
    }
  }

  #[inline(always)]
  pub fn set(&mut self, index: Val, port: Port, value: Address) {
    unsafe {
      let node = self.aux_ports.get_unchecked_mut(index as usize);
      match port {
        P1 => node.0 = value,
        P2 => node.1 = value,
      }
    }
  }

  #[inline(always)]
  pub fn get_root(&self) -> Address {
    self.get(0, P2)
  }

  #[inline(always)]
  pub fn set_root(&mut self, value: Address) {
    self.set(0, P2, value);
  }

  #[inline(always)]
  pub fn compact(&self) -> Vec<AuxPortAddresses> {
    let mut node = vec![];
    loop {
      let p1 = self.aux_ports[node.len()].0;
      let p2 = self.aux_ports[node.len()].1;
      if p1 != NULL || p2 != NULL {
        node.push((p1, p2));
      } else {
        break;
      }
    }
    node
  }
}

impl From<Net> for Def {
  fn from(Net { redexes: redex, node_storage: heap, .. }: Net) -> Self {
    Def { redexes: redex, nodes: heap.compact() } 
  }
}

impl Net {
  // Creates an empty net with given size.
  pub fn new(size: usize) -> Self {
    Net {
      redexes: vec![],
      node_storage: NodeStorage::new(size),
      stats: Stats::default(),
    }
  }

  // Creates a net and boots from a REF.
  pub fn boot(&mut self, root_id: Val) {
    self.node_storage.set_root(Address::new(REF, root_id));
  }

  // Reads back from a def.
  pub fn from_def(def: Def) -> Self {
    let mut net = Net::new(def.nodes.len());
    for (i, &(p1, p2)) in def.nodes.iter().enumerate() {
      net.node_storage.set(i as Val, P1, p1);
      net.node_storage.set(i as Val, P2, p2);
    }
    net.redexes = def.redexes;
    net
  }

  // Gets a pointer's target.
  #[inline(always)]
  pub fn get_target(&self, ptr: Address) -> Address {
    self.node_storage.get(ptr.val(), Port::from(ptr.0))
  }

  // Sets a pointer's target.
  #[inline(always)]
  pub fn set_target(&mut self, ptr: Address, val: Address) {
    self.node_storage.set(ptr.val(), Port::from(ptr.0), val)
  }

  // Links two pointers, forming a new wire.
  pub fn link(&mut self, a: Address, b: Address) {
    // Creates redex A-B
    if a.is_pri() && b.is_pri() {
      if Address::can_skip(a, b) {
        self.stats.erasures += 1;
      } else {
        self.redexes.push((a, b));
      }
      return;
    }
    // Substitutes A
    if a.is_var() {
      self.set_target(a, b);
    }
    // Substitutes B
    if b.is_var() {
      self.set_target(b, a);
    }
  }

  #[inline(always)]
  fn expand_if_ref(&mut self, book: &Book, (mut a, mut b): Redex) -> Redex {
    match (a.tag(), b.tag()) {
      (REF, OPS!() | CTR!()) => a = self.expand_ref(book, a, b),
      (OPS!() | CTR!(), REF) => b = self.expand_ref(book, b, a),
      _ => {}
    }
    (a, b)
  }

  // Performs an interaction over a redex.
  pub fn interact(&mut self, (a, b): Redex) {
    match (a.tag(), b.tag()) {
      // CTR-CTR when same labels
      (CT0, CT0) | (CT1, CT1) | (CT2, CT2) | (CT3, CT3) | (CT4, CT4) | (CT5, CT5)
        => self.annihilate((a, b)),
      // CTR-CTR when different labels
      (CTR!(), CTR!()) => self.commutate((a, b)),
      (CTR!(), ERA)    => self.era2(a),
      (ERA, CTR!())    => self.era2(b),
      (REF, ERA)       => self.stats.erasures += 1,
      (ERA, REF)       => self.stats.erasures += 1,
      (ERA, ERA)       => self.stats.erasures += 1,
      (VAR!(), _)      => self.link(a, b),
      (_, VAR!())      => self.link(b, a),
      (CTR!(), NUM)    => self.copy((a, b)),
      (NUM, CTR!())    => self.copy((b, a)),
      (NUM, ERA)       => self.stats.erasures += 1,
      (ERA, NUM)       => self.stats.erasures += 1,
      (NUM, NUM)       => self.stats.erasures += 1,
      (OP2, NUM)       => self.op2n((a, b)),
      (NUM, OP2)       => self.op2n((b, a)),
      (OP1, NUM)       => self.op1n((a, b)),
      (NUM, OP1)       => self.op1n((b, a)),
      (OP2, CTR!())    => self.commutate((a, b)),
      (CTR!(), OP2)    => self.commutate((b, a)),
      (OP1, CTR!())    => self.pass((a, b)),
      (CTR!(), OP1)    => self.pass((b, a)),
      (OP2, ERA)       => self.era2(a),
      (ERA, OP2)       => self.era2(b),
      (OP1, ERA)       => self.era1(a),
      (ERA, OP1)       => self.era1(b),
      (MAT, NUM)       => self.match_num((a, b)),
      (NUM, MAT)       => self.match_num((b, a)),
      (MAT, CTR!())    => self.commutate((a, b)),
      (CTR!(), MAT)    => self.commutate((b, a)),
      (MAT, ERA)       => self.era2(a),
      (ERA, MAT)       => self.era2(b),

      // because of the deref above this match
      // we know that A and B are not REFs
      (REF, _) => unreachable!(),
      (_, REF) => unreachable!(),

      // undefined numerical interactions resulting from a sort of "type error"
      (OPS!(), OPS!()) => unreachable!(),

      // TODO: this will change when we implement the multi-threaded version
      (RED!(), _) => unreachable!(),
      (_, RED!()) => unreachable!(),
    };
  }

  pub fn conn(&mut self, (a, b): Redex) {
    self.stats.annihilations += 1;
    self.link(self.node_storage.get(a.val(), P2), self.node_storage.get(b.val(), P2));
    self.node_storage.free(a.val());
    self.node_storage.free(b.val());
  }

  pub fn annihilate(&mut self, (a, b): Redex) {
    self.stats.annihilations += 1;
    self.link(self.node_storage.get(a.val(), P1), self.node_storage.get(b.val(), P1));
    self.link(self.node_storage.get(a.val(), P2), self.node_storage.get(b.val(), P2));
    self.node_storage.free(a.val());
    self.node_storage.free(b.val());
  }

  pub fn commutate(&mut self, (a, b): Redex) {
    self.stats.commutations += 1;
    let loc = self.node_storage.alloc(4);
    self.link(self.node_storage.get(a.val(), P1), Address::new(b.tag(), loc));
    self.link(self.node_storage.get(b.val(), P1), Address::new(a.tag(), loc + 2));
    self.link(self.node_storage.get(a.val(), P2), Address::new(b.tag(), loc + 1));
    self.link(self.node_storage.get(b.val(), P2), Address::new(a.tag(), loc + 3));
    self.node_storage.set(loc, P1, Address::new(VR1, loc + 2));
    self.node_storage.set(loc, P2, Address::new(VR1, loc + 3));
    self.node_storage.set(loc + 1, P1, Address::new(VR2, loc + 2));
    self.node_storage.set(loc + 1, P2, Address::new(VR2, loc + 3));
    self.node_storage.set(loc + 2, P1, Address::new(VR1, loc));
    self.node_storage.set(loc + 2, P2, Address::new(VR1, loc + 1));
    self.node_storage.set(loc + 3, P1, Address::new(VR2, loc));
    self.node_storage.set(loc + 3, P2, Address::new(VR2, loc + 1));
    self.node_storage.free(a.val());
    self.node_storage.free(b.val());
  }

  pub fn pass(&mut self, (a, b): Redex) {
    self.stats.commutations += 1;
    let loc = self.node_storage.alloc(3);
    self.link(self.node_storage.get(a.val(), P2), Address::new(b.tag(), loc));
    self.link(self.node_storage.get(b.val(), P1), Address::new(a.tag(), loc+1));
    self.link(self.node_storage.get(b.val(), P2), Address::new(a.tag(), loc+2));
    self.node_storage.set(loc, P1, Address::new(VR2, loc+1));
    self.node_storage.set(loc, P2, Address::new(VR2, loc+2));
    self.node_storage.set(loc + 1, P1, self.node_storage.get(a.val(), P1));
    self.node_storage.set(loc + 1, P2, Address::new(VR1, loc));
    self.node_storage.set(loc + 2, P1, self.node_storage.get(a.val(), P1));
    self.node_storage.set(loc + 2, P2, Address::new(VR2, loc));
    self.node_storage.free(a.val());
    self.node_storage.free(b.val());
  }

  pub fn copy(&mut self, (a, b): Redex) {
    self.stats.commutations += 1;
    self.link(self.node_storage.get(a.val(), P1), b);
    self.link(self.node_storage.get(a.val(), P2), b);
    self.node_storage.free(a.val());
  }

  pub fn era2(&mut self, a: Address) {
    self.stats.erasures += 1;
    self.link(self.node_storage.get(a.val(), P1), ERAS);
    self.link(self.node_storage.get(a.val(), P2), ERAS);
    self.node_storage.free(a.val());
  }

  pub fn era1(&mut self, a: Address) {
    self.stats.erasures += 1;
    self.link(self.node_storage.get(a.val(), P2), ERAS);
    self.node_storage.free(a.val());
  }


  pub fn op2n(&mut self, (op, num): Redex) {
    self.stats.numeric_ops += 1;
    let mut p1 = self.node_storage.get(op.val(), P1);
    // Optimization: perform chained ops at once
    if p1.is_num() {
      let mut rt = num.val();
      let mut p2 = self.node_storage.get(op.val(), P2);
      loop {
        self.stats.numeric_ops += 1;
        rt = self.do_primitive(rt, p1.val());
        // If P2 is OP2, keep looping
        if p2.is_op2() {
          p1 = self.node_storage.get(p2.val(), P1);
          if p1.is_num() {
            p2 = self.node_storage.get(p2.val(), P2);
            self.stats.numeric_ops += 1; // since OP1 is skipped
            continue;
          }
        }
        // If P2 is OP1, flip args and keep looping
        if p2.is_op1() {
          let tmp = rt;
          rt = self.node_storage.get(p2.val(), P1).val();
          p1 = Address::new(NUM, tmp);
          p2 = self.node_storage.get(p2.val(), P2);
          continue;
        }
        break;
      }
      self.link(Address::new(NUM, rt), p2);
      return;
    }
    self.node_storage.set(op.val(), P1, num);
    self.link(Address::new(OP1, op.val()), p1);
  }

  pub fn op1n(&mut self, (a, b): Redex) {
    self.stats.numeric_ops += 1;
    let p1 = self.node_storage.get(a.val(), P1);
    let p2 = self.node_storage.get(a.val(), P2);
    let v0 = p1.val();
    let v1 = b.val();
    let v2 = self.do_primitive(v0, v1);
    self.link(Address::new(NUM, v2), p2);
    self.node_storage.free(a.val());
  }

  pub fn do_primitive(&mut self, a: u32, b: u32) -> u32 {
    let a_opr = (a >> 24) & 0xF;
    let b_opr = (b >> 24) & 0xF; // not used yet
    let a_val = a & 0xFFFFFF;
    let b_val = b & 0xFFFFFF;
    match a_opr as u8 {
      USE => { ((a_val & 0xF) << 24) | b_val }
      ADD => { (a_val.wrapping_add(b_val)) & 0xFFFFFF }
      SUB => { (a_val.wrapping_sub(b_val)) & 0xFFFFFF }
      MUL => { (a_val.wrapping_mul(b_val)) & 0xFFFFFF }
      DIV => { (a_val.wrapping_div(b_val)) & 0xFFFFFF }
      MOD => { (a_val.wrapping_rem(b_val)) & 0xFFFFFF }
      EQ  => { ((a_val == b_val) as Val) & 0xFFFFFF }
      NE  => { ((a_val != b_val) as Val) & 0xFFFFFF }
      LT  => { ((a_val < b_val) as Val) & 0xFFFFFF }
      GT  => { ((a_val > b_val) as Val) & 0xFFFFFF }
      AND => { (a_val & b_val) & 0xFFFFFF }
      OR  => { (a_val | b_val) & 0xFFFFFF }
      XOR => { (a_val ^ b_val) & 0xFFFFFF }
      NOT => { (!b_val) & 0xFFFFFF }
      LSH => { (a_val << b_val) & 0xFFFFFF }
      RSH => { (a_val >> b_val) & 0xFFFFFF }
      _   => { unreachable!() }
    }
  }

  pub fn match_num(&mut self, (mat, num): Redex) {
    self.stats.numeric_ops += 1;
    let branch = self.node_storage.get(mat.val(), P1); // branch
    let ret = self.node_storage.get(mat.val(), P2); // return
    if num.val() == 0 {
      let loc = self.node_storage.alloc(1);
      self.node_storage.set(loc, P2, ERAS);
      self.link(branch, Address::new(CT0, loc));
      self.link(ret, Address::new(VR1, loc));
      self.node_storage.free(mat.val());
    } else {
      let loc = self.node_storage.alloc(2);
      self.node_storage.set(loc, P1, ERAS);
      self.node_storage.set(loc, P2, Address::new(CT0, loc + 1));
      self.node_storage.set(loc+1, P1, Address::new(NUM, num.val() - 1));
      self.link(branch, Address::new(CT0, loc));
      self.link(ret, Address::new(VR2, loc+1));
      self.node_storage.free(mat.val());
    }
  }

  // Expands a closed net on `ptr` and links it with `parent`.
  #[inline(always)]
  pub fn expand_ref(&mut self, book: &Book, ptr: Address, parent: Address) -> Address {
    self.stats.derefs += 1;
    let mut ptr = ptr;
    // FIXME: change "while" to "if" once lang prevents refs from returning refs
    while ptr.is_ref() {
      // Load the closed net.
      let got = unsafe { book.get_unchecked((ptr.val() & 0xFFFFFF) as usize) };
      if got.nodes.len() > 0 {
        let len = got.nodes.len() - 1;
        let loc = self.node_storage.alloc(len);
        // Load nodes, adjusted.
        for i in 0..len as Val {
          unsafe {
            let p1 = got.nodes.get_unchecked(1 + i as usize).0.adjust(loc);
            let p2 = got.nodes.get_unchecked(1 + i as usize).1.adjust(loc);
            self.node_storage.set(loc + i, P1, p1);
            self.node_storage.set(loc + i, P2, p2);
          }
        }
        // Load redexes, adjusted.
        for r in &got.redexes {
          let p1 = r.0.adjust(loc);
          let p2 = r.1.adjust(loc);
          self.redexes.push((p1, p2));
        }
        // Load root, adjusted.
        ptr = got.nodes[0].1.adjust(loc);
        // Link root.
        if ptr.is_var() {
          self.set_target(ptr, parent);
        }
      }
    }
    ptr
  }

  // Reduces all redexes.
  pub fn reduce(&mut self, book: &Book) {
    let mut redex_snapshot = std::mem::take(&mut self.redexes);
    while !redex_snapshot.is_empty() {
      for redex in redex_snapshot.drain(..) {
        let redex = self.expand_if_ref(book, redex);
        self.interact(redex);
      }
      std::mem::swap(&mut self.redexes, &mut redex_snapshot);
    }
  }

  // Reduce a net to normal form.
  pub fn normal(&mut self, book: &Book) {
    self.expand(book, ROOT);
    while self.redexes.len() > 0 {
      self.reduce(book);
      self.expand(book, ROOT);
    }
  }

  // Expands heads.
  pub fn expand(&mut self, book: &Book, dir: Address) {
    let ptr = self.get_target(dir);
    if ptr.is_ctr() {
      self.expand(book, Address::new(VR1, ptr.val()));
      self.expand(book, Address::new(VR2, ptr.val()));
    } else if ptr.is_ref() {
      let exp = self.expand_ref(book, ptr, dir);
      self.set_target(dir, exp);
    }
  }

  // Total rewrite count.
  pub fn rewrites(&self) -> usize {
    let Net { stats, .. } = self;
    stats.annihilations + stats.commutations + stats.erasures + stats.derefs + stats.numeric_ops
  }
}
