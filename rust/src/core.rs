// An efficient Interaction Combinator runtime
// ===========================================
//
// This file implements interaction combinators with an efficient memory format. Nodes store only
// aux ports, with the main port omitted. This segments the graph in trees, including parent-child
// wires (P1|P2->P0). Main wires (P0<->P0) are then stored in a separate vector, called 'acts'
// (active wires), and aux wires (P1|P2->P1|P2) are represented by VAR pointers. The 'acts' vector
// is automatically updated during reduction, which allows us to always keep track of all active
// wires. Pointers contain the tag of the pointed object. This allows for 1. unboxed ERAs, NUMs,
// REFs; 2. omitting labels on nodes (as these are stored on their parent's pointers). This file
// also includes REF pointers, which expand to pre-defined modules (closed nets with 1 free wire).
// This expansion is performed on demand, and ERA-REF pointers are collected, allowing the runtime
// to compute tail-recursive functions with constant memory usage.

use std::{collections::HashMap, hash::BuildHasherDefault};

pub type Tag = u16;
pub type Val = u64;

// Core terms
pub const NIL: Tag = 0x0000; // empty node
pub const REF: Tag = 0x0001; // reference to a definition (closed net)
pub const ERA: Tag = 0x0002; // unboxed eraser
pub const VRR: Tag = 0x0003; // variable pointing to root
pub const VR1: Tag = 0x0004; // variable pointing to aux1 port of node
pub const VR2: Tag = 0x0005; // variable pointing to aux2 port of node
pub const RDR: Tag = 0x0006; // redirection to root
pub const RD1: Tag = 0x0007; // redirection to aux1 port of node
pub const RD2: Tag = 0x0008; // redirection to aux2 port of node
pub const CON: Tag = 0x1000; // points to main port of con node
pub const DUP: Tag = 0x1001; // points to main port of dup node; higher labels also dups

// Numeric terms
pub const NUM: Tag = 0x0100; // unboxed u48

pub const ADX: Tag = 0x0200; // ...
pub const SBX: Tag = 0x0201; // ...
pub const MLX: Tag = 0x0202; // ...
pub const DVX: Tag = 0x0203; // ...

pub const ADY: Tag = 0x0300; // ...
pub const SBY: Tag = 0x0301; // ...
pub const MLY: Tag = 0x0302; // ...
pub const DVY: Tag = 0x0303; // ...
pub const OPX: Tag = ADX;
pub const OPY: Tag = ADY;
pub const OPZ: Tag = DVY+1;

// A node port: 1 or 2. Main ports are omitted.
pub type Port = usize;
pub const P1 : Port = 0;
pub const P2 : Port = 1;

// A tagged pointer. When tag >= VR1, it stores an absolute target location (node index).
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
pub struct Ptr {
  pub data: Val,
}

// A node is just a pair of two delta pointers. It uses 64 bits.
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
pub struct Node {
  pub ports: [Ptr; 2],
}

// A net has:
// - root: a single free wire, used as the entrancy point.
// - acts: a vector of redexes, updated automatically.
// - node: a vector of nodes, with main ports omitted.
// - used: total nodes currently allocated on the graph.
// - rwts: total graph rewrites performed inside this net.
// - next: next pointer to allocate memory (internal).
#[derive(Debug)]
pub struct Net {
  pub root: Ptr,
  pub acts: Vec<(Ptr, Ptr)>,
  pub node: Vec<Node>,
  pub used: usize,
  pub rwts: usize,
      next: usize,
      locs: Vec<u64>,
}

// A book is just a map of definitions, mapping ids to closed nets.
pub struct Book {
  pub defs: HashMap<u64, Net, std::hash::BuildHasherDefault<nohash::NoHashHasher<u64>>>,
}

impl Ptr {
  #[inline(always)]
  pub fn new(tag: Tag, val: Val) -> Self {
    Ptr { data: (((tag as u64) << 48) | (val & 0xFFFF_FFFF_FFFF)) }
  }

  #[inline(always)]
  pub fn tag(&self) -> Tag {
    (self.data >> 48) as Tag
  }

  #[inline(always)]
  pub fn val(&self) -> Val {
    (self.data & 0xFFFF_FFFF_FFFF) as Val
  }

  #[inline(always)]
  pub fn is_var(&self) -> bool {
    return self.tag() >= VRR && self.tag() <= VR2;
  }

  #[inline(always)]
  pub fn is_red(&self) -> bool {
    return self.tag() >= RDR && self.tag() <= RD2;
  }

  #[inline(always)]
  pub fn is_era(&self) -> bool {
    return self.tag() == ERA;
  }

  #[inline(always)]
  pub fn is_num(&self) -> bool {
    return self.tag() == NUM;
  }

  #[inline(always)]
  pub fn is_opx(&self) -> bool {
    return self.tag() >= OPX && self.tag() < OPY;
  }

  #[inline(always)]
  pub fn is_opy(&self) -> bool {
    return self.tag() >= OPY && self.tag() < OPZ;
  }

  #[inline(always)]
  pub fn is_ctr(&self) -> bool {
    return self.tag() >= CON;
  }

  #[inline(always)]
  pub fn is_ref(&self) -> bool {
    return self.tag() == REF;
  }

  #[inline(always)]
  pub fn is_pri(&self) -> bool {
    return self.is_era()
        || self.is_ctr()
        || self.is_ref()
        || self.is_num()
        || self.is_opx()
        || self.is_opy();
  }

  #[inline(always)]
  pub fn has_loc(&self) -> bool {
    return self.is_ctr()
        || self.is_opx()
        || self.is_opy()
        || self.is_var() && self.tag() != VRR
        || self.is_red() && self.tag() != RDR;
  }

  #[inline(always)]
  pub fn target<'a>(&'a self, net: &'a mut Net) -> Option<&mut Ptr> {
    match self.tag() {
      VRR => { Some(&mut net.root) }
      VR1 => { Some(net.at_mut(self.val()).port_mut(P1)) }
      VR2 => { Some(net.at_mut(self.val()).port_mut(P2)) }
      _   => { None }
    }
  }

  #[inline(always)]
  pub fn adjust(&self, locs: &[u64]) -> Ptr {
    unsafe {
      return Ptr::new(self.tag(), if self.has_loc() { *locs.get_unchecked(self.val() as usize) } else { self.val() });
    }
  }
}

impl Node {
  #[inline(always)]
  pub fn new(p1: Ptr, p2: Ptr) -> Self {
    Node { ports: [p1, p2] }
  }

  #[inline(always)]
  pub fn nil() -> Self {
    Self::new(Ptr::new(NIL,0), Ptr::new(NIL,0))
  }

  #[inline(always)]
  pub fn port(&self, port: Port) -> &Ptr {
    unsafe {
      return self.ports.get_unchecked(port as usize);
    }
  }

  #[inline(always)]
  pub fn port_mut(&mut self, port: Port) -> &mut Ptr {
    unsafe {
      return self.ports.get_unchecked_mut(port as usize);
    }
  }
}

impl Book {
  pub fn new() -> Self {
    Book { defs: HashMap::with_hasher(std::hash::BuildHasherDefault::default()) }
  }

  pub fn def(&mut self, id: u64, net: Net) {
    self.defs.insert(id, net);
  }
}

impl Net {
  // Creates an empty net with given size.
  pub fn new(size: usize) -> Self {
    Net {
      root: Ptr::new(NIL, 0),
      acts: vec![],
      node: vec![Node::nil(); size],
      next: 0,
      used: 0,
      rwts: 0,
      locs: vec![0; 1 << 16], // FIXME: should be field of Worker, not Net
    }
  }

  // Creates a net and boots from a REF.
  pub fn init(&mut self, root_id: u64) {
    self.link(Ptr::new(VRR, 0), Ptr::new(REF, root_id));
  }

  // Allocates a consecutive chunk of 'size' nodes. Returns the index.
  #[inline(always)]
  pub fn alloc(&mut self, size: usize) -> Val {
    loop {
      if self.next >= self.node.len() {
        self.next = 0;
      }
      if self.get(self.next as Val, P1).tag() == NIL {
        self.next += 1;
        self.used += 1;
        return (self.next - 1) as Val;
      }
      self.next += 1;
    }
  }

  // Frees the memory used by a single node.
  #[inline(always)]
  pub fn free(&mut self, val: Val) {
    self.used -= 1;
    self.node[val as usize] = Node::nil();
  }

  // Gets node at given index.
  #[inline(always)]
  pub fn at(&self, index: u64) -> &Node {
    unsafe {
      return self.node.get_unchecked(index as usize);
    }
  }

  // Gets node at given index, mutable.
  #[inline(always)]
  pub fn at_mut(&mut self, index: u64) -> &mut Node {
    unsafe {
      return self.node.get_unchecked_mut(index as usize);
    }
  }

  // Gets the pointer stored on the port 1 or 2 of a node.
  #[inline(always)]
  pub fn get(&self, index: Val, port: Port) -> Ptr {
    return *self.at(index).port(port);
  }

  // Sets the pointer stored on the port 1 or 2 of a node.
  #[inline(always)]
  pub fn set(&mut self, index: Val, port: Port, value: Ptr) {
    *self.at_mut(index).port_mut(port) = value;
  }

  // Links two pointers, forming a new wire.
  // - If one of the pointers is a variable, it will move the other value.
  // - Otherwise, this is an redexes, so we add it to 'acts'.
  #[inline(always)]
  pub fn link(&mut self, a: Ptr, b: Ptr) {
    if let Some(a_trg) = a.target(self) {
      *a_trg = b;
    }
    if let Some(b_trg) = b.target(self) {
      *b_trg = a;
    }
    if a.is_pri() && b.is_pri() {
      self.acts.push((a, b));
    }
  }

  // Performs an interaction over a redex.
  #[inline(always)]
  pub fn interact(&mut self, book: &Book, a: &mut Ptr, b: &mut Ptr) {
    // Dereference
    if a.tag() == REF && b.tag() != ERA { *a = self.deref(book, *a); }
    if a.tag() != ERA && b.tag() == REF { *b = self.deref(book, *b); }
    self.rwts += 1;
    // VAR
    if a.is_var() || b.is_var() {
      self.link(*a, *b);
    // OPX-NUM
    } else if a.is_opx() && b.is_num() { // TODO: test
      let v1 = self.get(a.val(), P1);
      self.set(a.val(), P1, *b);
      self.link(Ptr::new(OPY, a.val()), v1);
    // NUM-OPX
    } else if a.is_num() && b.is_opx() { // TODO: test
      self.interact(book, b, a);
    // OPY-NUM
    } else if a.is_opy() && b.is_num() { // TODO: test
      let v0 = self.get(a.val(), P1);
      let rt = self.get(a.val(), P2);
      self.link(Ptr::new(NUM, v0.val() + b.val()), rt);
      self.free(a.val());
    // NUM-OPY
    } else if a.is_num() && b.is_opy() { // TODO: test
      self.interact(book, a, b);
    // OPX-CTR
    } else if a.is_opx() && b.is_ctr() { // TODO: test
      let x1 = self.alloc(1);
      let x2 = self.alloc(1);
      let y1 = self.alloc(1);
      let y2 = self.alloc(1);
      self.set(x1, P1, Ptr::new(VR1, y1));
      self.set(x1, P2, Ptr::new(VR1, y2));
      self.set(x2, P1, Ptr::new(VR2, y1));
      self.set(x2, P2, Ptr::new(VR2, y2));
      self.set(y1, P1, Ptr::new(VR1, x1));
      self.set(y1, P2, Ptr::new(VR1, x2));
      self.set(y2, P1, Ptr::new(VR2, x1));
      self.set(y2, P2, Ptr::new(VR2, x2));
      self.link(self.get(a.val(), P1), Ptr::new(b.tag(), x1));
      self.link(self.get(a.val(), P2), Ptr::new(b.tag(), x2));
      self.link(self.get(b.val(), P1), Ptr::new(a.tag(), y1));
      self.link(self.get(b.val(), P2), Ptr::new(a.tag(), y2));
      self.free(a.val());
      self.free(b.val());
    // CTR-OPX
    } else if a.is_ctr() && b.is_opx() { // TODO: test
      self.interact(book, b, a);
    // OPY-CTR
    } else if a.is_opy() && b.is_ctr() { // TODO: test
      let x1 = self.alloc(1);
      let y1 = self.alloc(1);
      let y2 = self.alloc(1);
      let av = self.get(a.val(), P1);
      self.set(x1, P1, Ptr::new(VR2, y1));
      self.set(x1, P2, Ptr::new(VR2, y2));
      self.set(y1, P2, Ptr::new(VR1, x1));
      self.set(y2, P2, Ptr::new(VR2, x1));
      self.set(y1, P1, av);
      self.set(y2, P1, av);
      self.link(self.get(b.val(), P1), Ptr::new(a.tag(), y1));
      self.link(self.get(b.val(), P2), Ptr::new(a.tag(), y2));
      self.link(self.get(a.val(), P2), Ptr::new(b.tag(), x1));
      self.free(a.val());
      self.free(b.val());
    // CTR-OPX
    } else if a.is_ctr() && b.is_opy() { // TODO: test
      self.interact(book, b, a);
    // CON-CON
    } else if a.is_ctr() && b.is_ctr() && a.tag() == b.tag() {
      let a1 = self.get(a.val(), P1);
      let b1 = self.get(b.val(), P1);
      self.link(a1, b1);
      let a2 = self.get(a.val(), P2);
      let b2 = self.get(b.val(), P2);
      self.link(a2, b2);
      self.free(a.val());
      self.free(b.val());
    // CON-DUP
    } else if a.is_ctr() && b.is_ctr() && a.tag() != b.tag() {
      let x1 = self.alloc(1);
      let x2 = self.alloc(1);
      let y1 = self.alloc(1);
      let y2 = self.alloc(1);
      self.set(x1, P1, Ptr::new(VR1, y1));
      self.set(x1, P2, Ptr::new(VR1, y2));
      self.set(x2, P1, Ptr::new(VR2, y1));
      self.set(x2, P2, Ptr::new(VR2, y2));
      self.set(y1, P1, Ptr::new(VR1, x1));
      self.set(y1, P2, Ptr::new(VR1, x2));
      self.set(y2, P1, Ptr::new(VR2, x1));
      self.set(y2, P2, Ptr::new(VR2, x2));
      self.link(self.get(a.val(), P1), Ptr::new(b.tag(), x1));
      self.link(self.get(a.val(), P2), Ptr::new(b.tag(), x2));
      self.link(self.get(b.val(), P1), Ptr::new(a.tag(), y1));
      self.link(self.get(b.val(), P2), Ptr::new(a.tag(), y2));
      self.free(a.val());
      self.free(b.val());
    // CTR-NUM
    } else if a.is_ctr() && b.is_num() { // TODO: test
      self.link(self.get(a.val(), P1), Ptr::new(NUM, b.val()));
      self.link(self.get(a.val(), P2), Ptr::new(NUM, b.val()));
      self.free(a.val());
    // NUM-CTR
    } else if a.is_num() && b.is_ctr() { // TODO: test
      self.interact(book, b, a);
    // CON-ERA
    } else if a.is_ctr() && b.is_era() {
      self.link(self.get(a.val(), P1), Ptr::new(ERA, 0));
      self.link(self.get(a.val(), P2), Ptr::new(ERA, 0));
      self.free(a.val());
    // ERA-CON
    } else if a.is_era() && b.is_ctr() {
      self.link(self.get(b.val(), P1), Ptr::new(ERA, 0));
      self.link(self.get(b.val(), P2), Ptr::new(ERA, 0));
      self.free(b.val());
    }
  }

  // Expands a REF into its definition (a closed net).
  #[inline(always)]
  pub fn deref(&mut self, book: &Book, ptr: Ptr) -> Ptr {
    let mut ptr = ptr;
    // White ptr is still a REF...
    while ptr.is_ref() {
      // Loads the referenced definition...
      if let Some(got) = book.defs.get(&ptr.val()) {
        // Allocates enough space...
        for i in 0 .. got.node.len() {
          unsafe {
            *self.locs.get_unchecked_mut(i) = self.alloc(1);
          }
        }
        // Loads nodes, adjusting locations...
        for i in 0 .. got.node.len() {
          unsafe {
            let got = got.node.get_unchecked(i).clone();
            let neo = Node::new(got.port(P1).adjust(&self.locs), got.port(P2).adjust(&self.locs));
            *self.at_mut(*self.locs.get_unchecked(i)) = neo;
          }
        }
        // Loads redexes, adjusting locations...
        for got in &got.acts {
          self.acts.push((got.0.adjust(&self.locs), got.1.adjust(&self.locs)));
        }
        // Overwrites 'ptr' with the loaded root pointer, adjusting locations...
        ptr = got.root;
        ptr = ptr.adjust(&self.locs);
      }
    }
    return ptr;
  }

  // Reduces all redexes at the same time.
  pub fn reduce(&mut self, book: &Book) -> usize {
    let rwts = self.acts.len();
    let acts = std::mem::replace(&mut self.acts, vec![]);
    // This loop can be parallelized!
    for (mut a, mut b) in acts {
      self.interact(book, &mut a, &mut b);
    }
    return rwts;
  }

  // Reduces all redexes, until there is none.
  pub fn reduce_all(&mut self, book: &Book) {
    while self.acts.len() > 0 {
      self.reduce(book);
    }
  }

  // Expands all references in a term.
  pub fn normalize(&mut self, book: &Book) {
    self.reduce_all(book);
    let mut stack = vec![Ptr::new(VRR,0)];
    while let Some(loc) = stack.pop() {
      let trg = *loc.target(self).unwrap();
      if trg.is_ctr() {
        stack.push(Ptr::new(VR1, trg.val()));
        stack.push(Ptr::new(VR2, trg.val()));
      } else if trg.tag() == REF {
        let res = self.deref(book, trg);
        self.link(res, loc);
        self.reduce_all(book);
        stack.push(loc);
      }
    }
  }

}
