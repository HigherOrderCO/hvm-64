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

use std::collections::HashMap;

pub type Tag = u16;
pub type Val = u32;

// Core terms
pub const NIL: Tag = 0x0; // empty node
pub const REF: Tag = 0x1; // reference to a definition (closed net)
pub const ERA: Tag = 0x2; // unboxed eraser
pub const VRR: Tag = 0x3; // variable pointing to root
pub const VR1: Tag = 0x4; // variable pointing to aux1 port of node
pub const VR2: Tag = 0x5; // variable pointing to aux2 port of node
pub const RDR: Tag = 0x6; // redirection to root
pub const RD1: Tag = 0x7; // redirection to aux1 port of node
pub const RD2: Tag = 0x8; // redirection to aux2 port of node
pub const NUM: Tag = 0x9; // redirection to aux2 port of node
pub const CON: Tag = 0xA; // points to main port of con node
pub const DUP: Tag = 0xB; // points to main port of dup node; higher labels also dups

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
// - head: a vector of unomralized heads
// - used: total nodes currently allocated on the graph.
// - rwts: total graph rewrites performed inside this net.
// - next: next pointer to allocate memory (internal).
#[derive(Debug)]
pub struct Net {
  pub root: Ptr,
  pub acts: Vec<(Ptr, Ptr)>,
  pub node: Vec<Node>,
  pub isnf: Vec<bool>,
  pub used: usize,
  pub rwts: usize,
      next: usize,
      locs: Vec<u32>,
}

// A book is just a map of definitions, mapping ids to closed nets.
pub struct Book {
  pub defs: HashMap<u32, Net, std::hash::BuildHasherDefault<nohash::NoHashHasher<u32>>>,
}

impl Ptr {
  #[inline(always)]
  pub fn new(tag: Tag, val: Val) -> Self {
    Ptr { data: (((tag as u32) << 28) | (val & 0xFFF_FFFF)) }
  }

  #[inline(always)]
  pub fn tag(&self) -> Tag {
    (self.data >> 28) as Tag
  }

  #[inline(always)]
  pub fn val(&self) -> Val {
    (self.data & 0xFFF_FFFF) as Val
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
        || self.is_num();
  }

  #[inline(always)]
  pub fn has_loc(&self) -> bool {
    return self.is_ctr()
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
  pub fn is_target_normal<'a>(&'a self, net: &'a Net) -> bool {
    return self.is_var() && self.tag() == VRR || net.isnf[self.val() as usize];
  }

  #[inline(always)]
  pub fn adjust(&self, locs: &[u32]) -> Ptr {
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

  pub fn def(&mut self, id: u32, net: Net) {
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
      isnf: vec![false; size],
      next: 0,
      used: 0,
      rwts: 0,
      locs: vec![0; 1 << 16], // FIXME: should be field of Worker, not Net
    }
  }

  // Creates a net and boots from a REF.
  pub fn boot(&mut self, book: &Book, root_id: u32) {
    self.link(book, Ptr::new(VRR, 0), Ptr::new(REF, root_id));
  }

  // Allocates a consecutive chunk of 'size' nodes. Returns the index.
  #[inline(always)]
  pub fn alloc(&mut self) -> Val {
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
  pub fn at(&self, index: u32) -> &Node {
    unsafe {
      return self.node.get_unchecked(index as usize);
    }
  }

  // Gets node at given index, mutable.
  #[inline(always)]
  pub fn at_mut(&mut self, index: u32) -> &mut Node {
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
  pub fn link(&mut self, book: &Book, a: Ptr, b: Ptr) {
    let mut a = a;
    let mut b = b;
    // Dereferences A
    if a.is_ref() && (b.is_ctr() || b.is_target_normal(self)) {
      a = self.deref(book, a, &mut 0);
    }
    // Dereferences B
    if b.is_ref() && (a.is_ctr() || a.is_target_normal(self)) {
      b = self.deref(book, b, &mut 0);
    }
    // Substitutes A
    if a.is_var() {
      *a.target(self).unwrap() = b;
      // Flags normalized B
      if b.is_pri() && a.is_target_normal(self) {
        self.isnf[b.val() as usize] = true;
      }
    }
    // Substitutes B
    if b.is_var() {
      *b.target(self).unwrap() = a;
      // Flags normalized A
      if a.is_pri() && b.is_target_normal(self) {
        self.isnf[a.val() as usize] = true;
      }
    }
    // New redex A-B
    if a.is_pri() && b.is_pri() {
      self.acts.push((a, b));
    }
  }

  // Performs an interaction over a redex.
  #[inline(always)]
  pub fn interact(&mut self, book: &Book, a: &mut Ptr, b: &mut Ptr) {
    // Dereference
    //if a.tag() == REF && b.tag() != ERA { *a = self.deref(book, *a, Ptr::new(NIL,0)); }
    //if a.tag() != ERA && b.tag() == REF { *b = self.deref(book, *b, Ptr::new(NIL,0)); }
    self.rwts += 1;
    // CON-CON
    if a.is_ctr() && b.is_ctr() && a.tag() == b.tag() {
      let a1 = self.get(a.val(), P1);
      let b1 = self.get(b.val(), P1);
      self.link(book, a1, b1);
      let a2 = self.get(a.val(), P2);
      let b2 = self.get(b.val(), P2);
      self.link(book, a2, b2);
      self.free(a.val());
      self.free(b.val());
    // CON-DUP
    } else if a.is_ctr() && b.is_ctr() && a.tag() != b.tag() {
      let x1 = self.alloc();
      let x2 = self.alloc();
      let y1 = self.alloc();
      let y2 = self.alloc();
      self.set(x1, P1, Ptr::new(VR1, y1));
      self.set(x1, P2, Ptr::new(VR1, y2));
      self.set(x2, P1, Ptr::new(VR2, y1));
      self.set(x2, P2, Ptr::new(VR2, y2));
      self.set(y1, P1, Ptr::new(VR1, x1));
      self.set(y1, P2, Ptr::new(VR1, x2));
      self.set(y2, P1, Ptr::new(VR2, x1));
      self.set(y2, P2, Ptr::new(VR2, x2));
      self.link(book, self.get(a.val(), P1), Ptr::new(b.tag(), x1));
      self.link(book, self.get(a.val(), P2), Ptr::new(b.tag(), x2));
      self.link(book, self.get(b.val(), P1), Ptr::new(a.tag(), y1));
      self.link(book, self.get(b.val(), P2), Ptr::new(a.tag(), y2));
      self.free(a.val());
      self.free(b.val());
    // CTR-NUM
    } else if a.is_ctr() && b.is_num() { // TODO: test
      self.link(book, self.get(a.val(), P1), Ptr::new(NUM, b.val()));
      self.link(book, self.get(a.val(), P2), Ptr::new(NUM, b.val()));
      self.free(a.val());
    // NUM-CTR
    } else if a.is_num() && b.is_ctr() { // TODO: test
      self.interact(book, b, a);
    // CON-ERA
    } else if a.is_ctr() && b.is_era() {
      self.link(book, self.get(a.val(), P1), Ptr::new(ERA, 0));
      self.link(book, self.get(a.val(), P2), Ptr::new(ERA, 0));
      self.free(a.val());
    // ERA-CON
    } else if a.is_era() && b.is_ctr() {
      self.link(book, self.get(b.val(), P1), Ptr::new(ERA, 0));
      self.link(book, self.get(b.val(), P2), Ptr::new(ERA, 0));
      self.free(b.val());
    }
  }

  // Expands a REF into its definition (a closed net).
  #[inline(always)]
  pub fn deref(&mut self, book: &Book, ptr: Ptr, loc: &mut usize) -> Ptr {
    let mut ptr = ptr;
    // White ptr is still a REF...
    while ptr.is_ref() {
      // Loads the referenced definition...
      if let Some(got) = book.defs.get(&ptr.val()) {
        let ini = *loc;
        *loc += got.node.len();
        // Allocates enough space...
        for i in 0 .. got.node.len() {
          unsafe {
            *self.locs.get_unchecked_mut(ini + i) = self.alloc();
          }
        }
        // Loads nodes, adjusting locations...
        for i in 0 .. got.node.len() {
          unsafe {
            let got = got.node.get_unchecked(i).clone();
            let neo = Node::new(got.port(P1).adjust(&self.locs[ini..]), got.port(P2).adjust(&self.locs[ini..]));
            *self.at_mut(*self.locs.get_unchecked(ini + i)) = neo;
          }
        }
        // Loads redexes, adjusting locations...
        for got in &got.acts {
          let a = got.0.adjust(&self.locs[ini..]);
          let b = got.1.adjust(&self.locs[ini..]);
          let a = self.deref(book, a, loc);
          let b = self.deref(book, b, loc);
          self.acts.push((a, b));
        }
        // Overwrites 'ptr' with the loaded root pointer, adjusting locations...
        ptr = got.root.adjust(&self.locs[ini..]);
      }
    }
    return ptr;
  }

  // Performs a global parallel rewrite.
  pub fn reduce(&mut self, book: &Book) -> usize {
    let rwts = self.acts.len();
    let acts = std::mem::replace(&mut self.acts, vec![]);
    // This loop can be parallelized!
    for (mut a, mut b) in acts {
      self.interact(book, &mut a, &mut b);
    }
    return rwts;
  }

  // Reduces all redexes until there is none.
  pub fn normal(&mut self, book: &Book) {
    while self.acts.len() > 0 {
      println!("reduce {}", self.acts.len());
      self.reduce(book);
    }
  }

}
