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
pub struct Ptr(pub Val);

// A node is just a pair of two delta pointers. It uses 64 bits.
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
pub struct Node(pub (Ptr,Ptr));

// A net has:
// - root: a single free wire, used as the entrancy point.
// - acts: a vector of redexes, updated automatically.
// - node: a vector of nodes, with main ports omitted.
// - used: total nodes currently allocated on the graph.
// - rwts: total graph rewrites performed inside this net.
// - next: next pointer to allocate memory (internal).
#[derive(Debug, Clone)]
pub struct Net {
  pub root: Ptr,
  pub acts: Vec<(Ptr, Ptr)>,
  pub node: Vec<Node>,
  pub used: usize,
  pub rwts: usize,
  pub dref: usize,
      next: usize,
}

// A book is just a map of definitions, mapping ids to closed nets.
pub struct Book {
  defs: Vec<Net>,
}

impl Ptr {
  #[inline(always)]
  pub fn new(tag: Tag, val: Val) -> Self {
    Ptr(((tag as u32) << 28) | (val & 0xFFF_FFFF))
  }

  #[inline(always)]
  pub fn data(&self) -> u32 {
    return self.0;
  }

  #[inline(always)]
  pub fn tag(&self) -> Tag {
    (self.data() >> 28) as Tag
  }

  #[inline(always)]
  pub fn val(&self) -> Val {
    (self.data() & 0xFFF_FFFF) as Val
  }

  #[inline(always)]
  pub fn is_nil(&self) -> bool {
    return self.0 == 0x0000_0000;
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
        || self.is_num()
        || self.is_ref();
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
  pub fn adjust(&self, loc: u32) -> Ptr {
    return Ptr::new(self.tag(), if self.has_loc() { self.val() + loc } else { self.val() });
  }
}

impl Node {
  #[inline(always)]
  pub fn new(p1: Ptr, p2: Ptr) -> Self {
    Node((p1, p2))
  }

  #[inline(always)]
  pub fn nil() -> Self {
    Self::new(Ptr::new(NIL,0), Ptr::new(NIL,0))
  }

  #[inline(always)]
  pub fn port(&self, port: Port) -> Ptr {
    return if port == P1 { self.0.0 } else { self.0.1 };
  }

  #[inline(always)]
  pub fn port_mut(&mut self, port: Port) -> &mut Ptr {
    return if port == P1 { &mut self.0.0 } else { &mut self.0.1 };
  }
}

impl Book {
  #[inline(always)]
  pub fn new() -> Self {
    Book { defs: vec![Net::new(0); 1 << 24] }
  }

  #[inline(always)]
  pub fn def(&mut self, id: u32, net: Net) {
    self.defs[id as usize] = net;
  }

  #[inline(always)]
  pub fn get(&self, id: u32) -> Option<&Net> {
    self.defs.get(id as usize)
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
      dref: 0,
    }
  }

  // Creates a net and boots from a REF.
  pub fn boot(&mut self, root_id: u32) {
    self.root = Ptr::new(REF, root_id);
  }

  // Allocates a consecutive chunk of 'size' nodes. Returns the index.
  pub fn alloc(&mut self, size: usize) -> Val {
    let mut space = 0;
    loop {
      if self.next >= self.node.len() {
        space = 0;
        self.next = 0;
      }
      if self.get(self.next as Val, P1).tag() == NIL {
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
    return self.at(index).port(port);
  }

  // Sets the pointer stored on the port 1 or 2 of a node.
  #[inline(always)]
  pub fn set(&mut self, index: Val, port: Port, value: Ptr) {
    *self.at_mut(index).port_mut(port) = value;
  }

  // Links two pointers, forming a new wire.
  // - If one of the pointers is a variable, it will move the other value.
  // - Otherwise, this is an redexes, so we add it to 'acts'.
  pub fn link(&mut self, a: Ptr, b: Ptr) {
    // Substitutes A
    if let Some(target) = a.target(self) {
      *target = b;
    }
    // Substitutes B
    if let Some(target) = b.target(self) {
      *target = a;
    }
    // Creates redex A-B
    if a.is_pri() && b.is_pri() {
      self.acts.push((a, b));
    }
  }

  // Performs an interaction over a redex.
  pub fn interact(&mut self, book: &Book, a: &mut Ptr, b: &mut Ptr) {
    self.rwts += 1;

    // Symmetry
    if !a.is_era() && b.is_ref()
    ||  a.is_num() && b.is_ctr()
    ||  a.is_era() && b.is_ctr() {
      std::mem::swap(a, b);
    }

    // Dereference
    if a.tag() == REF && b.tag() != ERA {
      *a = self.deref(book, *a, Ptr::new(NIL,0));
    }

    // Substitute
    if a.is_var() || b.is_var() {
      self.link(*a, *b);
    }

    // CON-CON
    if a.is_ctr() && b.is_ctr() && a.tag() == b.tag() {
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
      let loc = self.alloc(4);
      self.link(self.get(a.val(), P1), Ptr::new(b.tag(), loc+0));
      self.link(self.get(b.val(), P1), Ptr::new(a.tag(), loc+2));
      self.link(self.get(a.val(), P2), Ptr::new(b.tag(), loc+1));
      self.link(self.get(b.val(), P2), Ptr::new(a.tag(), loc+3));
      *self.at_mut(loc+0) = Node::new(Ptr::new(VR1, loc+2), Ptr::new(VR1, loc+3));
      *self.at_mut(loc+1) = Node::new(Ptr::new(VR2, loc+2), Ptr::new(VR2, loc+3));
      *self.at_mut(loc+2) = Node::new(Ptr::new(VR1, loc+0), Ptr::new(VR1, loc+1));
      *self.at_mut(loc+3) = Node::new(Ptr::new(VR2, loc+0), Ptr::new(VR2, loc+1));
      self.free(a.val());
      self.free(b.val());
    // CTR-NUM
    } else if a.is_ctr() && b.is_num() { // TODO: test
      self.link(self.get(a.val(), P1), Ptr::new(NUM, b.val()));
      self.link(self.get(a.val(), P2), Ptr::new(NUM, b.val()));
      self.free(a.val());
    // CON-ERA
    } else if a.is_ctr() && b.is_era() {
      self.link(self.get(a.val(), P1), Ptr::new(ERA, 0));
      self.link(self.get(a.val(), P2), Ptr::new(ERA, 0));
      self.free(a.val());
    }
  }

  // Expands a REF into its definition (a closed net).
  #[inline(always)]
  pub fn deref(&mut self, book: &Book, ptr: Ptr, parent: Ptr) -> Ptr {
    self.dref += 1;
    let mut ptr = ptr;
    // White ptr is still a REF...
    while ptr.is_ref() {
      // Loads the referenced definition...
      if let Some(got) = book.get(ptr.val()) {
        let loc = self.alloc(got.node.len());
        // Loads nodes, adjusting locations...
        for i in 0 .. got.node.len() as u32 {
          unsafe {
            let got = got.node.get_unchecked(i as usize);
            let p1  = got.port(P1).adjust(loc);
            let p2  = got.port(P2).adjust(loc);
            *self.at_mut(loc + i) = Node::new(p1, p2);
          }
        }
        // Loads redexes, adjusting locations...
        for got in &got.acts {
          let p1 = got.0.adjust(loc);
          let p2 = got.1.adjust(loc);
          self.acts.push((p1, p2));
        }
        // Overwrites 'ptr' with the loaded root pointer, adjusting locations...
        ptr = got.root.adjust(loc);
        // Links root
        if ptr.is_var() {
          if let Some(trg) = ptr.target(self) {
            *trg = parent;
          }
        }
      }
    }
    return ptr;
  }

  pub fn reduce(&mut self, book: &Book) {
    let mut acts : Vec<(Ptr,Ptr)> = vec![];
    // Swaps self.acts with acts
    std::mem::swap(&mut self.acts, &mut acts);
    // While there are redexes...
    while acts.len() > 0 {
      // Apply all redexes
      for (a, b) in &mut acts {
        self.interact(book, a, b);
      }
      // Sets acts's length to 0
      acts.clear();
      // Swaps acts and self.acts
      std::mem::swap(&mut self.acts, &mut acts);
    }
  }

  // Reduce a net to normal form.
  pub fn normal(&mut self, book: &Book) {
    self.expand(book, Ptr::new(VRR, 0));
    while self.acts.len() > 0 {
      self.reduce(book);
      self.expand(book, Ptr::new(VRR, 0));
    }
  }

  // Expands heads.
  pub fn expand(&mut self, book: &Book, dir: Ptr) {
    let ptr = *dir.target(self).unwrap();
    if ptr.is_ctr() {
      self.expand(book, Ptr::new(VR1, ptr.val()));
      self.expand(book, Ptr::new(VR2, ptr.val()));
    } else if ptr.is_ref() {
      *dir.target(self).unwrap() = self.deref(book, ptr, dir);
    }
  }
}
