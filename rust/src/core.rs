// An efficient Interaction Combinator runtime
// ===========================================
// This file implements an efficient interaction combinator runtime. Nodes are represented by 2 aux
// ports (P1, P2), with the main port (P1) omitted. A separate vector, 'rdex', holds main ports,
// and, thus, tracks active pairs that can be reduced in parallel. Pointers are unboxed, meaning
// that ERAs, NUMs and REFs don't use any additional space. REFs lazily expand to closed nets when
// they interact with nodes, and are cleared when they interact with ERAs, allowing for constant
// space evaluation of recursive functions on Scott encoded datatypes.

use std::collections::HashMap;

pub type Tag = u8;
pub type Val = u32;

// Core terms
pub const NIL: Tag = 0x0; // uninitialized
pub const REF: Tag = 0x1; // closed net reference
pub const ERA: Tag = 0x2; // unboxed eraser
pub const VRR: Tag = 0x3; // aux port to root
pub const VR1: Tag = 0x4; // aux port to aux port 1
pub const VR2: Tag = 0x5; // aux port to aux port 2
pub const RDR: Tag = 0x6; // redirect to root
pub const RD1: Tag = 0x7; // redirect to aux port 1
pub const RD2: Tag = 0x8; // redirect to aux port 2
pub const NUM: Tag = 0x9; // unboxed number
pub const CON: Tag = 0xA; // main port of con node
pub const DUP: Tag = 0xB; // main port of dup node

// An auxiliary port.
pub type Port = usize;
pub const P1 : Port = 0;
pub const P2 : Port = 1;

// A tagged pointer.
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
pub struct Ptr(pub Val);

// A node is just a pair of two pointers.
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
pub struct Node(pub (Ptr,Ptr));

// A interaction combinator net.
#[derive(Debug, Clone)]
pub struct Net {
  pub root: Ptr, // entrancy
  pub rdex: Vec<(Ptr,Ptr)>, // redexes
  pub node: Vec<Node>, // nodes
  pub used: usize, // allocated nodes
  pub rwts: usize, // rewrite count
  pub dref: usize, // deref count
  pub next: usize, // next alloc index
}

// A map of id to definitions (closed nets).
pub struct Book {
  defs: Vec<Net>,
}

impl Ptr {
  #[inline(always)]
  pub fn new(tag: Tag, val: Val) -> Self {
    Ptr(((tag as Val) << 28) | (val & 0xFFF_FFFF))
  }

  #[inline(always)]
  pub fn data(&self) -> Val {
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
    return self.tag() == NIL;
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
    return self.is_era() || self.is_ctr() || self.is_num() || self.is_ref();
  }

  #[inline(always)]
  pub fn has_loc(&self) -> bool {
    return self.is_ctr() || self.is_var() && self.tag() != VRR;
  }

  #[inline(always)]
  pub fn adjust(&self, loc: Val) -> Ptr {
    return Ptr::new(self.tag(), self.val() + if self.has_loc() { loc } else { 0 });
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
  pub fn def(&mut self, id: Val, net: Net) {
    self.defs[id as usize] = net;
  }

  #[inline(always)]
  pub fn get(&self, id: Val) -> Option<&Net> {
    self.defs.get(id as usize)
  }
}

impl Net {
  // Creates an empty net with given size.
  pub fn new(size: usize) -> Self {
    Net {
      root: Ptr::new(NIL, 0),
      rdex: vec![],
      node: vec![Node::nil(); size],
      next: 0,
      used: 0,
      rwts: 0,
      dref: 0,
    }
  }

  // Creates a net and boots from a REF.
  pub fn boot(&mut self, root_id: Val) {
    self.root = Ptr::new(REF, root_id);
  }

  // Allocates a consecutive chunk of 'size' nodes. Returns the index.
  pub fn alloc(&mut self, size: usize) -> Val {
    if size == 0 {
      return 0;
    } else {
      let mut space = 0;
      loop {
        if self.next >= self.node.len() {
          space = 0;
          self.next = 0;
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

  // Frees the memory used by a single node.
  #[inline(always)]
  pub fn free(&mut self, val: Val) {
    self.used -= 1;
    self.node[val as usize] = Node::nil();
  }

  // Gets node at given index.
  #[inline(always)]
  pub fn at(&self, index: Val) -> &Node {
    unsafe {
      return self.node.get_unchecked(index as usize);
    }
  }

  // Gets node at given index, mutable.
  #[inline(always)]
  pub fn at_mut(&mut self, index: Val) -> &mut Node {
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

  // Gets a pointer target.
  #[inline(always)]
  pub fn target(&mut self, ptr: Ptr) -> Option<&mut Ptr> {
    match ptr.tag() {
      VRR => { Some(&mut self.root) }
      VR1 => { Some(self.at_mut(ptr.val()).port_mut(P1)) }
      VR2 => { Some(self.at_mut(ptr.val()).port_mut(P2)) }
      _   => { None }
    }
  }

  // Links two pointers, forming a new wire.
  pub fn link(&mut self, a: Ptr, b: Ptr) {
    // Substitutes A
    if a.is_var() {
      *self.target(a).unwrap() = b;
    }
    // Substitutes B
    if b.is_var() {
      *self.target(b).unwrap() = a;
    }
    // Creates redex A-B
    if a.is_pri() && b.is_pri() {
      self.rdex.push((a, b));
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
      self.link(self.get(a.val(), P1), self.get(b.val(), P1));
      self.link(self.get(a.val(), P2), self.get(b.val(), P2));
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

  // Expands a closed net.
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
        for i in 0 .. got.node.len() as Val {
          unsafe {
            let got = got.node.get_unchecked(i as usize);
            let p1  = got.port(P1).adjust(loc);
            let p2  = got.port(P2).adjust(loc);
            *self.at_mut(loc + i) = Node::new(p1, p2);
          }
        }
        // Loads redexes, adjusting locations...
        for got in &got.rdex {
          let p1 = got.0.adjust(loc);
          let p2 = got.1.adjust(loc);
          self.rdex.push((p1, p2));
        }
        // Overwrites 'ptr' with the loaded root pointer, adjusting locations...
        ptr = got.root.adjust(loc);
        // Links root
        if ptr.is_var() {
          *self.target(ptr).unwrap() = parent;
        }
      }
    }
    return ptr;
  }

  // Reduces all redexes.
  pub fn reduce(&mut self, book: &Book) {
    let mut rdex : Vec<(Ptr,Ptr)> = vec![];
    std::mem::swap(&mut self.rdex, &mut rdex);
    while rdex.len() > 0 {
      for (a, b) in &mut rdex {
        self.interact(book, a, b);
      }
      rdex.clear();
      std::mem::swap(&mut self.rdex, &mut rdex);
    }
  }

  // Reduce a net to normal form.
  pub fn normal(&mut self, book: &Book) {
    self.expand(book, Ptr::new(VRR, 0));
    while self.rdex.len() > 0 {
      self.reduce(book);
      self.expand(book, Ptr::new(VRR, 0));
    }
  }

  // Expands heads.
  pub fn expand(&mut self, book: &Book, dir: Ptr) {
    let ptr = *self.target(dir).unwrap();
    if ptr.is_ctr() {
      self.expand(book, Ptr::new(VR1, ptr.val()));
      self.expand(book, Ptr::new(VR2, ptr.val()));
    } else if ptr.is_ref() {
      *self.target(dir).unwrap() = self.deref(book, ptr, dir);
    }
  }
}
