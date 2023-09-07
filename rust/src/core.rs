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

pub type Col = u8;
pub type Tag = u8;
pub type Val = u32;

// Core terms
pub const VR1: Tag = 0; // a P1 variable
pub const VR2: Tag = 1; // a P2 variable
pub const CTR: Tag = 2; // a constructor
pub const REF: Tag = 3; // a reference

pub type Port = Val;
pub const P1 : Port = 0;
pub const P2 : Port = 1;

// A tagged pointer
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
pub struct Ptr {
  pub data: Val,
}

// Special values
pub const NIL: Ptr = Ptr { data: (VR1 as u32) << 30 }; // unitialized
pub const ERA: Ptr = Ptr { data: (CTR as u32) << 30 };

// A node has a tag and two pointers.
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
pub struct Node {
  pub color: Col,
  pub ports: [Ptr; 2],
}

// A net has:
// - acts: a vector of redexes, updated automatically.
// - node: a vector of nodes, with main ports omitted.
// - used: total nodes currently allocated on the graph.
// - rwts: total graph rewrites performed inside this net.
// - next: next pointer to allocate memory (internal).
#[derive(Debug)]
pub struct Net {
  pub acts: Vec<(Ptr, Ptr)>,
  pub node: Vec<Node>,
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
    Ptr { data: ((tag as u32) << 30) | (val & 0x3FFF_FFFF) }
  }

  #[inline(always)]
  pub fn new_vr1(loc: Val) -> Self {
    Self::new(VR1, loc)
  }

  #[inline(always)]
  pub fn new_vr2(loc: Val) -> Self {
    Self::new(VR2, loc)
  }

  #[inline(always)]
  pub fn new_vrr() -> Self {
    Self::new_vr2(0)
  }

  #[inline(always)]
  pub fn new_era() -> Self {
    Self::new(CTR, 0)
  }

  #[inline(always)]
  pub fn new_ctr(loc: Val) -> Self {
    Self::new(CTR, loc)
  }

  #[inline(always)]
  pub fn new_ref(nam: Val) -> Self {
    Self::new(REF, nam)
  }

  //#[inline(always)]
  //pub fn new_num(val: Val) -> Self {
    //Self::new(NUM, val)
  //}

  #[inline(always)]
  pub fn tag(&self) -> Tag {
    (self.data >> 30) as Tag
  }

  #[inline(always)]
  pub fn val(&self) -> Val {
    (self.data & 0x3FFF_FFFF) as Val
  }

  #[inline(always)]
  pub fn is_vr1(&self) -> bool {
    return self.tag() == VR1;
  }

  #[inline(always)]
  pub fn is_vr2(&self) -> bool {
    return self.tag() == VR2;
  }

  #[inline(always)]
  pub fn is_vrr(&self) -> bool {
    return self.tag() == VR2 && self.val() == 0;
  }

  #[inline(always)]
  pub fn is_var(&self) -> bool {
    return self.is_vr1() || self.is_vr2();
  }

  #[inline(always)]
  pub fn is_ctr(&self) -> bool {
    return self.tag() == CTR && self.val() != 0;
  }

  #[inline(always)]
  pub fn is_era(&self) -> bool {
    return self.tag() == CTR && self.val() == 0;
  }

  #[inline(always)]
  pub fn is_ref(&self) -> bool {
    return self.tag() == REF;
  }

  #[inline(always)]
  pub fn target<'a>(&'a self, net: &'a mut Net) -> Option<&mut Ptr> {
    if self.is_vr1() {
      return Some(net.mut_node(self.val()).mut_port(P1));
    } else if self.is_vr2() {
      return Some(net.mut_node(self.val()).mut_port(P2));
    } else {
      return None;
    }
  }

  #[inline(always)]
  pub fn adjust(&self, locs: &[u32]) -> Ptr {
    unsafe {
      if self.is_vr1() || self.is_vr2() || self.is_ctr() {
        Ptr::new(self.tag(), *locs.get_unchecked(self.val() as usize))
      } else {
        Ptr::new(self.tag(), self.val())
      }
    }
  }
}

impl Node {
  #[inline(always)]
  pub fn new(color: Col, p1: Ptr, p2: Ptr) -> Self {
    Node {
      color: color,
      ports: [p1, p2],
    }
  }

  #[inline(always)]
  pub fn nil() -> Self {
    Self::new(0, NIL, NIL)
  }

  #[inline(always)]
  pub fn ref_port(&self, port: Port) -> &Ptr {
    unsafe {
      return self.ports.get_unchecked(port as usize);
    }
  }

  #[inline(always)]
  pub fn mut_port(&mut self, port: Port) -> &mut Ptr {
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
      acts: vec![],
      node: vec![Node::nil(); size],
      next: 1,
      used: 0,
      rwts: 0,
      locs: vec![0; 1 << 16], // FIXME: should be field of Worker, not Net
    }
  }

  // Creates a net and boots from a REF.
  pub fn init(&mut self, root_id: u32) {
    self.link(Ptr::new_vrr(), Ptr::new(REF, root_id));
  }

  // Allocates a node
  #[inline(always)]
  pub fn alloc(&mut self) -> Val {
    loop {
      if self.next >= self.node.len() {
        self.next = 1;
      }
      if self.get_port(self.next as Val, P1).data == NIL.data {
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
  pub fn ref_node(&self, index: u32) -> &Node {
    unsafe {
      return self.node.get_unchecked(index as usize);
    }
  }

  // Gets node at given index, mutable.
  #[inline(always)]
  pub fn mut_node(&mut self, index: u32) -> &mut Node {
    unsafe {
      return self.node.get_unchecked_mut(index as usize);
    }
  }

  // Gets node at given index.
  #[inline(always)]
  pub fn ref_root(&self) -> &Ptr {
    return self.ref_node(0).ref_port(P2);
  }

  // Gets node at given index, mutable.
  #[inline(always)]
  pub fn mut_root(&mut self) -> &mut Ptr {
    return self.mut_node(0).mut_port(P2);
  }

  // Gets the pointer stored on the port 1 or 2 of a node.
  #[inline(always)]
  pub fn get_port(&self, index: Val, port: Port) -> Ptr {
    return *self.ref_node(index).ref_port(port);
  }

  // Sets the pointer stored on the port 1 or 2 of a node.
  #[inline(always)]
  pub fn set_port(&mut self, index: Val, port: Port, value: Ptr) {
    *self.mut_node(index).mut_port(port) = value;
  }
  
  // Sets color of node
  pub fn set_color(&mut self, index: Val, col: Col) {
    self.mut_node(index).color = col;
  }

  // Gets color of node
  pub fn get_color(&self, index: Val) -> Col {
    return self.ref_node(index).color;
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
    if !a.is_var() && !b.is_var() {
      self.acts.push((a, b));
    }
  }

  // Performs an interaction over a redex.
  #[inline(always)]
  pub fn rewrite(&mut self, book: &Book, a: &mut Ptr, b: &mut Ptr) {
    // Dereference
    if  a.is_ref() && !b.is_era() { *a = self.deref(book, *a); }
    if !a.is_era() &&  b.is_ref() { *b = self.deref(book, *b); }
    //println!("rdx {:x} {:x} !", a.data, b.data);
    self.rwts += 1;
    // VAR
    if a.is_var() || b.is_var() {
      //println!("var");
      self.link(*a, *b);
    // CON-CON
    } else if a.is_ctr() && b.is_ctr() && self.get_color(a.val()) == self.get_color(b.val()) {
      //println!("ann");
      let a1 = self.get_port(a.val(), P1);
      let b1 = self.get_port(b.val(), P1);
      self.link(a1, b1);
      let a2 = self.get_port(a.val(), P2);
      let b2 = self.get_port(b.val(), P2);
      self.link(a2, b2);
      self.free(a.val());
      self.free(b.val());
    // CON-DUP
    } else if a.is_ctr() && b.is_ctr() && self.get_color(a.val()) != self.get_color(b.val()) {
      //println!("com {} {}", self.get_color(a.loc()), self.get_color(b.loc()));
      let x1 = self.alloc();
      let x2 = self.alloc();
      let y1 = self.alloc();
      let y2 = self.alloc();
      let ak = self.get_color(a.val());
      let bk = self.get_color(b.val());
      self.set_color(x1, bk);
      self.set_color(x2, bk);
      self.set_color(y1, ak);
      self.set_color(y2, ak);
      self.set_port(x1, P1, Ptr::new_vr1(y1));
      self.set_port(x1, P2, Ptr::new_vr1(y2));
      self.set_port(x2, P1, Ptr::new_vr2(y1));
      self.set_port(x2, P2, Ptr::new_vr2(y2));
      self.set_port(y1, P1, Ptr::new_vr1(x1));
      self.set_port(y1, P2, Ptr::new_vr1(x2));
      self.set_port(y2, P1, Ptr::new_vr2(x1));
      self.set_port(y2, P2, Ptr::new_vr2(x2));
      self.link(self.get_port(a.val(), P1), Ptr::new_ctr(x1));
      self.link(self.get_port(a.val(), P2), Ptr::new_ctr(x2));
      self.link(self.get_port(b.val(), P1), Ptr::new_ctr(y1));
      self.link(self.get_port(b.val(), P2), Ptr::new_ctr(y2));
      self.free(a.val());
      self.free(b.val());
      //println!("col {} {} = {} ? {} {} | {} {} = {} ? {} {}", x1, x2, bc, self.get_color(x1), self.get_color(x2), y1, y2, ac, self.get_color(y1), self.get_color(y2));
    // CON-ERA
    } else if a.is_ctr() && b.is_era() {
      self.link(self.get_port(a.val(), P1), Ptr::new_era());
      self.link(self.get_port(a.val(), P2), Ptr::new_era());
      self.free(a.val());
    // ERA-CON
    } else if a.is_era() && b.is_ctr() {
      self.link(self.get_port(b.val(), P1), Ptr::new_era());
      self.link(self.get_port(b.val(), P2), Ptr::new_era());
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
        //println!("deref: {}", crate::lang::show_net(got));
        // Allocates enough space...
        for i in 1 .. got.node.len() {
          unsafe {
            *self.locs.get_unchecked_mut(i) = self.alloc();
          }
        }
        // Loads nodes, adjusting locations...
        for i in 1 .. got.node.len() {
          unsafe {
            let got = got.node.get_unchecked(i).clone();
            let np1 = got.ref_port(P1).adjust(&self.locs);
            let np2 = got.ref_port(P2).adjust(&self.locs);
            let neo = Node::new(got.color, np1, np2);
            *self.mut_node(*self.locs.get_unchecked(i)) = neo;
          }
        }
        // Loads redexes, adjusting locations...
        for got in &got.acts {
          self.acts.push((got.0.adjust(&self.locs), got.1.adjust(&self.locs)));
        }
        // Overwrites 'ptr' with the loaded root pointer, adjusting locations...
        ptr = *got.ref_root();
        ptr = ptr.adjust(&self.locs);
      }
    }
    return ptr;
  }

  // Reduces all redexes at the same time.
  pub fn reduce(&mut self, book: &Book) -> usize {
    //println!("------------------------ REDUCE {}", self.acts.len());
    //println!("{}", crate::lang::show_net(self));
    //for i in 0 .. 32 {
      //if self.node[i].ports[0].data != 0 {
        //println!("[{:02x}] {:04x} {:08x} | {:08x}", i*2, self.node[i].color, self.node[i].ports[0].data, self.node[i].ports[1].data);
      //}
    //}
    let rwts = self.acts.len();
    // This loop can be parallelized!
    let acts = std::mem::replace(&mut self.acts, vec![]);
    for (mut a, mut b) in acts {
      self.rewrite(book, &mut a, &mut b);
    }
    //if let Some((mut a, mut b)) = self.acts.pop() {
      //self.rewrite(book, &mut a, &mut b);
    //}
    return rwts;
  }

  // Reduces all redexes, until there is none.
  pub fn reduce_all(&mut self, book: &Book) {
    while self.acts.len() > 0 {
      //println!("reducing {}", self.acts.len());
      self.reduce(book);
    }
  }

  // Expands all references in a term.
  pub fn normalize(&mut self, book: &Book) {
    self.reduce_all(book);
    let mut stack = vec![Ptr::new_vrr()];
    while let Some(loc) = stack.pop() {
      let trg = *loc.target(self).unwrap();
      if trg.is_ctr() {
        stack.push(Ptr::new_vr1(trg.val()));
        stack.push(Ptr::new_vr2(trg.val()));
      } else if trg.tag() == REF {
        let res = self.deref(book, trg);
        self.link(res, loc);
        self.reduce_all(book);
        stack.push(loc);
      }
    }
  }

  // Expands all references in a term.
  pub fn expand(&mut self, book: &Book) {
    let mut stack = vec![Ptr::new_vrr()];
    while let Some(loc) = stack.pop() {
      let trg = *loc.target(self).unwrap();
      if trg.is_ctr() {
        stack.push(Ptr::new_vr1(trg.val()));
        stack.push(Ptr::new_vr2(trg.val()));
      } else if trg.tag() == REF {
        let res = self.deref(book, trg);
        self.link(res, loc);
        stack.push(loc);
      }
    }
  }

}
