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
// to compute tail-recursive functions with constant memory usage. Differential pointers are used,
// to allow REF expansions to be performed with a cheap memcpy.

use std::collections::HashMap;

pub type Tag = u8;
pub type Val = u32;

pub const NIL: Tag = 0; // empty node
pub const REF: Tag = 1; // reference to a definition (closed net)
pub const NUM: Tag = 2; // unboxed number
pub const ERA: Tag = 3; // unboxed eraser
pub const VRT: Tag = 4; // variable pointing to root
pub const VR1: Tag = 5; // variable pointing to aux1 port of node
pub const VR2: Tag = 6; // variable pointing to aux2 port of node
pub const CON: Tag = 7; // points to main port of con node
pub const DUP: Tag = 8; // points to main port of dup node; higher labels also dups

// A node port: 1 or 2. Main ports are omitted.
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
pub enum Port {
  P1,
  P2,
}

// A tagged pointer. When tag >= VR1, it stores an absolute target location (node index).
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
pub struct Ptr {
  pub tag: Tag,
  pub val: Val,
}

// A delta pointer. When tag >= VR1, it stores a relative location instead.
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
pub struct DPtr {
  pub tag: Tag,
  pub val: Val,
}

// A node is just a pair of two delta pointers. It uses 80 bits.
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
pub struct Node {
  pub p1: DPtr,
  pub p2: DPtr,
}

// A net has:
// - root: a single free wire, used as the entrancy point.
// - acts: a vector of active wires, updated automatically.
// - node: a vector of nodes, with main ports omitted.
// - next: next pointer to allocate memory (internal).
// - used: total nodes currently allocated on the graph.
// - rwts: total graph rewrites performed inside this net.
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
  pub defs: HashMap<u32, Net>,
}

impl Ptr {
  // Converts a pointer to a delta pointer.
  #[inline(always)]
  pub fn to_dptr(&self, val: Val) -> DPtr {
    DPtr { 
      tag: self.tag, 
      val: if self.tag >= VR1 { self.val.wrapping_sub(val) } else { self.val } 
    }
  }
}

impl DPtr {
  // Converts a delta pointer to a pointer.
  #[inline(always)]
  pub fn to_ptr(&self, val: Val) -> Ptr {
    Ptr { 
      tag: self.tag, 
      val: if self.tag >= VR1 { val.wrapping_add(self.val) } else { self.val } 
    }
  }
}

impl Node {
  #[inline(always)]
  pub fn nil() -> Self {
    Node {
      p1: DPtr { tag: NIL, val: 0 },
      p2: DPtr { tag: NIL, val: 0 },
    }
  }
}

impl Book {
  pub fn new() -> Self {
    Book { defs: HashMap::new() }
  }

  pub fn def(&mut self, id: u32, net: Net) {
    self.defs.insert(id, net);
  }
}

impl Net {
  // Creates an empty net with given size.
  pub fn new(size: usize) -> Self {
    Net {
      root: Ptr { tag: NIL, val: 0 },
      acts: vec![],
      node: vec![Node::nil(); size],
      next: 0,
      used: 0,
      rwts: 0,
    }
  }

  // Creates a net and boots from a REF.
  pub fn init(size: usize, book: &Book, ref_id: u32) -> Self {
    let mut net = Net::new(size);
    net.boot(book, ref_id);
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
      if self.get(self.next as Val, Port::P1).tag == NIL {
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
      Port::P1 => node.p1.to_ptr(val),
      Port::P2 => node.p2.to_ptr(val),
    }
  }

  // Sets the pointer stored on the port 1 or 2 of a node.
  #[inline(always)]
  pub fn set(&mut self, val: Val, port: Port, value: Ptr) {
    let dptr = value.to_dptr(val);
    let node = unsafe { self.node.get_unchecked_mut(val as usize) };
    match port {
      Port::P1 => node.p1 = dptr,
      Port::P2 => node.p2 = dptr,
    }
  }

  // Links two pointers, forming a new wire.
  // - If one of the pointers is a variable, it will move the other value.
  // - Otherwise, this is an active pair, so we add it to 'acts'.
  #[inline(always)]
  pub fn link(&mut self, a: Ptr, b: Ptr) {
    if a.tag == VRT {
      self.root = b;
    }
    if a.tag == VR1 {
      self.set(a.val, Port::P1, b);
    }
    if a.tag == VR2 {
      self.set(a.val, Port::P2, b);
    }
    if b.tag == VRT {
      self.root = a;
    }
    if b.tag == VR1 {
      self.set(b.val, Port::P1, a);
    }
    if b.tag == VR2 {
      self.set(b.val, Port::P2, a);
    }
    if a.tag != VRT && a.tag != VR1 && a.tag != VR2
    && b.tag != VRT && b.tag != VR1 && b.tag != VR2 {
      self.acts.push((a, b));
    }
  }

  // Reduces all active wires at the same time.
  pub fn reduce(&mut self, book: &Book) {
    let acts = std::mem::replace(&mut self.acts, vec![]);
    // This loop can be parallelized!
    for (mut a, mut b) in acts {
      self.interact(book, &mut a, &mut b);
    }
  }

  // Reduces a net to full normal form.
  pub fn normal(&mut self, book: &Book) -> usize {
    let mut loops = 0;
    while self.acts.len() > 0 {
      self.reduce(book);
      loops = loops + 1;
    }
    return loops;
  }

  // Performs an interaction over an active wire.
  #[inline(always)]
  pub fn interact(&mut self, book: &Book, a: &mut Ptr, b: &mut Ptr) {
    // Collect (for closed nets)
    if a.tag == REF && b.tag == ERA { return; }
    if a.tag == ERA && b.tag == REF { return; }
    // Dereference
    self.load_ref(book, a);
    self.load_ref(book, b);
    // Annihilation
    if a.tag >= CON && b.tag >= CON && a.tag == b.tag {
      let a1 = self.get(a.val, Port::P1);
      let b1 = self.get(b.val, Port::P1);
      self.link(a1, b1);
      let a2 = self.get(a.val, Port::P2);
      let b2 = self.get(b.val, Port::P2);
      self.link(a2, b2);
      self.free(a.val);
      self.free(b.val);
      self.rwts += 1;
    // Commutation
    } else if a.tag >= CON && b.tag >= CON && a.tag != b.tag {
      let x1 = self.alloc(1);
      let x2 = self.alloc(1);
      let y1 = self.alloc(1);
      let y2 = self.alloc(1);
      self.set(x1, Port::P1, Ptr { tag: VR1, val: y1 });
      self.set(x1, Port::P2, Ptr { tag: VR1, val: y2 });
      self.set(x2, Port::P1, Ptr { tag: VR2, val: y1 });
      self.set(x2, Port::P2, Ptr { tag: VR2, val: y2 });
      self.set(y1, Port::P1, Ptr { tag: VR1, val: x1 });
      self.set(y1, Port::P2, Ptr { tag: VR1, val: x2 });
      self.set(y2, Port::P1, Ptr { tag: VR2, val: x1 });
      self.set(y2, Port::P2, Ptr { tag: VR2, val: x2 });
      self.link(self.get(a.val, Port::P1), Ptr { tag: b.tag, val: x1 });
      self.link(self.get(a.val, Port::P2), Ptr { tag: b.tag, val: x2 });
      self.link(self.get(b.val, Port::P1), Ptr { tag: a.tag, val: y1 });
      self.link(self.get(b.val, Port::P2), Ptr { tag: a.tag, val: y2 });
      self.free(a.val);
      self.free(b.val);
      self.rwts += 1;
    // Erasure
    } else if a.tag >= CON && b.tag == ERA {
      self.link(self.get(a.val, Port::P1), Ptr { tag: ERA, val: 0 });
      self.link(self.get(a.val, Port::P2), Ptr { tag: ERA, val: 0 });
      self.free(a.val);
      self.rwts += 1;
    // Erasure
    } else if a.tag == ERA && b.tag >= CON {
      self.link(self.get(b.val, Port::P1), Ptr { tag: ERA, val: 0 });
      self.link(self.get(b.val, Port::P2), Ptr { tag: ERA, val: 0 });
      self.free(b.val);
      self.rwts += 1;
    // Stuck
    } else {
      self.acts.push((*a,*b));
    }
  }

  // Expands a REF into its definition (a closed net).
  #[inline(always)]
  pub fn load_ref(&mut self, book: &Book, ptr: &mut Ptr) {
    // White ptr is still a REF...
    while ptr.tag == REF {
      // Loads the referenced definition...
      if let Some(got) = book.defs.get(&ptr.val) {
        // Allocates enough space...
        let val = self.alloc(got.node.len());
        // Loads nodes with a memcpy (possible because of DPtrs)...
        let rng = (val as usize) .. (val as usize + got.node.len());
        self.node[rng].copy_from_slice(&got.node);
        // Loads active wires, adjusting locations...
        for &(p1, p2) in &got.acts {
          let mut p1 = p1;
          let mut p2 = p2;
          if p1.tag >= VR1 { p1.val += val; }
          if p2.tag >= VR1 { p2.val += val; }
          self.acts.push((p1, p2));
        }
        // Overwrites 'ptr' with the loaded root pointer, adjusting locations...
        *ptr = got.root;
        if got.root.tag >= VR1 { ptr.val += val; }
      }
    }
  }

  // Initializes the net by loading a specific REF
  pub fn boot(&mut self, book: &Book, ref_id: u32) {
    let mut root = Ptr { tag: REF, val: ref_id };
    self.load_ref(&book, &mut root);
    self.root = root;
  }

}
