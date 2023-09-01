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
pub type Val = u64;
//pub type Col = u16;

pub const NIL: Tag = 0x0; // empty node
pub const REF: Tag = 0x1; // reference to a definition (closed net)
pub const NUM: Tag = 0x2; // unboxed number
pub const ERA: Tag = 0x3; // unboxed eraser
pub const VRR: Tag = 0x4; // variable pointing to root
pub const VR1: Tag = 0x5; // variable pointing to aux1 port of node
pub const VR2: Tag = 0x6; // variable pointing to aux2 port of node
pub const RDR: Tag = 0x7; // redirection to root
pub const RD1: Tag = 0x8; // redirection to aux1 port of node
pub const RD2: Tag = 0x9; // redirection to aux2 port of node
pub const CON: Tag = 0xA; // points to main port of con node
pub const DUP: Tag = 0xB; // points to main port of dup node; higher labels also dups

// A node port: 1 or 2. Main ports are omitted.
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
pub enum Port {
  P1,
  P2,
}

// A tagged pointer. When tag >= VR1, it stores an absolute target location (node index).
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
pub struct Ptr {
  pub data: Val,
}

// A node is just a pair of two delta pointers. It uses 64 bits.
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
pub struct Node {
  pub p1: Ptr,
  pub p2: Ptr,
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
}

// A book is just a map of definitions, mapping ids to closed nets.
pub struct Book {
  pub defs: HashMap<u64, Net>,
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
  pub fn mov(&mut self, add: u64) {
    if self.tag() >= VR1 {
      self.data = (self.data & 0xFFFF_0000_0000_0000) | (self.data + add as u64) & 0xFFFF_FFFF_FFFF;
    }
  }
}

impl Node {
  #[inline(always)]
  pub fn nil() -> Self {
    Node {
      //cl: 1,
      p1: Ptr::new(NIL, 0),
      p2: Ptr::new(NIL, 0),
    }
  }
}

impl Book {
  pub fn new() -> Self {
    Book { defs: HashMap::new() }
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
    }
  }

  // Creates a net and boots from a REF.
  pub fn init(size: usize, book: &Book, ref_id: u64) -> Self {
    let mut net = Net::new(size);
    net.root = Ptr::new(REF, ref_id);
    return net;
  }

  // Allocates a consecutive chunk of 'size' nodes. Returns the index.
  #[inline(always)]
  pub fn alloc(&mut self, size: usize) -> Val {
    let mut space = 0;
    loop {
      if self.next >= self.node.len() {
        space = 0;
        self.next = 0;
      }
      if self.get(self.next as Val, Port::P1).tag() == NIL {
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

  // Gets the pointer stored on the port 1 or 2 of a node.
  #[inline(always)]
  pub fn get(&self, val: Val, port: Port) -> Ptr {
    let node = unsafe { self.node.get_unchecked(val as usize) };
    match port {
      Port::P1 => node.p1,
      Port::P2 => node.p2,
    }
  }

  // Sets the pointer stored on the port 1 or 2 of a node.
  #[inline(always)]
  pub fn set(&mut self, val: Val, port: Port, value: Ptr) {
    let node = unsafe { self.node.get_unchecked_mut(val as usize) };
    match port {
      Port::P1 => node.p1 = value,
      Port::P2 => node.p2 = value,
    }
  }

  // Links two pointers, forming a new wire.
  // - If one of the pointers is a variable, it will move the other value.
  // - Otherwise, this is an redexes, so we add it to 'acts'.
  #[inline(always)]
  pub fn link(&mut self, a: Ptr, b: Ptr) {
    let a_tag = a.tag();
    let b_tag = b.tag();
    if a_tag == VRR {
      self.root = b;
    }
    if a_tag == VR1 {
      self.set(a.val(), Port::P1, b);
    }
    if a_tag == VR2 {
      self.set(a.val(), Port::P2, b);
    }
    if b_tag == VRR {
      self.root = a;
    }
    if b_tag == VR1 {
      self.set(b.val(), Port::P1, a);
    }
    if b_tag == VR2 {
      self.set(b.val(), Port::P2, a);
    }
    if a_tag != VRR && a_tag != VR1 && a_tag != VR2
    && b_tag != VRR && b_tag != VR1 && b_tag != VR2 {
      self.acts.push((a, b));
    }
  }

  // Performs an interaction over a redex.
  #[inline(always)]
  pub fn interact(&mut self, book: &Book, a: &mut Ptr, b: &mut Ptr) {
    // Dereference
    if a.tag() == REF && b.tag() != ERA {
      *a = self.deref(book, *a);
    }
    if a.tag() != ERA && b.tag() == REF {
      *b = self.deref(book, *b);
    }
    // Incs rewrites
    self.rwts += 1;
    // Gets tag
    let a_tag = a.tag();
    let b_tag = b.tag();
    // Variable
    if a_tag >= VRR && a_tag <= VR2 || b_tag >= VRR && b_tag <= VR2 {
      self.link(*a, *b);
    // Annihilation
    } else if a_tag >= CON && b_tag >= CON && a_tag == b_tag {
      let a1 = self.get(a.val(), Port::P1);
      let b1 = self.get(b.val(), Port::P1);
      self.link(a1, b1);
      let a2 = self.get(a.val(), Port::P2);
      let b2 = self.get(b.val(), Port::P2);
      self.link(a2, b2);
      self.free(a.val());
      self.free(b.val());
    // Commutation
    } else if a_tag >= CON && b_tag >= CON && a_tag != b_tag {
      let x1 = self.alloc(1);
      let x2 = self.alloc(1);
      let y1 = self.alloc(1);
      let y2 = self.alloc(1);
      self.set(x1, Port::P1, Ptr::new(VR1, y1));
      self.set(x1, Port::P2, Ptr::new(VR1, y2));
      self.set(x2, Port::P1, Ptr::new(VR2, y1));
      self.set(x2, Port::P2, Ptr::new(VR2, y2));
      self.set(y1, Port::P1, Ptr::new(VR1, x1));
      self.set(y1, Port::P2, Ptr::new(VR1, x2));
      self.set(y2, Port::P1, Ptr::new(VR2, x1));
      self.set(y2, Port::P2, Ptr::new(VR2, x2));
      self.link(self.get(a.val(), Port::P1), Ptr::new(b_tag, x1));
      self.link(self.get(a.val(), Port::P2), Ptr::new(b_tag, x2));
      self.link(self.get(b.val(), Port::P1), Ptr::new(a_tag, y1));
      self.link(self.get(b.val(), Port::P2), Ptr::new(a_tag, y2));
      self.free(a.val());
      self.free(b.val());
    // Erasure
    } else if a_tag >= CON && b_tag == ERA {
      self.link(self.get(a.val(), Port::P1), Ptr::new(ERA, 0));
      self.link(self.get(a.val(), Port::P2), Ptr::new(ERA, 0));
      self.free(a.val());
    // Erasure
    } else if a_tag == ERA && b_tag >= CON {
      self.link(self.get(b.val(), Port::P1), Ptr::new(ERA, 0));
      self.link(self.get(b.val(), Port::P2), Ptr::new(ERA, 0));
      self.free(b.val());
    }
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
  pub fn reduce_all(&mut self, book: &Book) -> usize {
    let mut loops = 0;
    while self.acts.len() > 0 {
      //println!("rewrite: {}", self.acts.len());
      loops = loops + 1;
      self.reduce(book);
    }
    return loops;
  }

  // Reduces a term to normal form, expanding references.
  pub fn normal(&mut self, book: &Book) {
    self.reduce_all(book);
    let mut stack = vec![Ptr::new(VRR,0)];
    while let Some(loc) = stack.pop() {
      let trg = match loc.tag() {
        VRR => self.root,
        VR1 => self.get(loc.val(), Port::P1),
        VR2 => self.get(loc.val(), Port::P2),
        _   => panic!(),
      };
      if trg.tag() >= CON {
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

  // Expands a REF into its definition (a closed net).
  #[inline(always)]
  pub fn deref(&mut self, book: &Book, ptr: Ptr) -> Ptr {
    let mut ptr = ptr;
    // White ptr is still a REF...
    while ptr.tag() == REF {
      // Loads the referenced definition...
      if let Some(got) = book.defs.get(&ptr.val()) {
        // Allocates enough space...
        let len = got.node.len();
        let val = self.alloc(len);
        // Loads nodes, adjusting locations...
        for i in 0 .. len as usize {
          let mut node = got.node[i].clone();
          node.p1.mov(val);
          node.p2.mov(val);
          unsafe {
            *self.node.get_unchecked_mut(val as usize + i) = node;
          };
        }
        // Loads redexes, adjusting locations...
        for got in &got.acts {
          let mut node = Node {
            //cl: 1,
            p1: got.0,
            p2: got.1,
          };
          node.p1.mov(val);
          node.p2.mov(val);
          self.acts.push((node.p1, node.p2));
        }
        // Overwrites 'ptr' with the loaded root pointer, adjusting locations...
        ptr = got.root;
        ptr.mov(val);
      }
    }
    return ptr;
  }

}
