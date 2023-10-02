// An efficient Interaction Combinator runtime
// ===========================================
// This file implements an efficient interaction combinator runtime. Nodes are represented by 2 aux
// ports (P1, P2), with the main port (P1) omitted. A separate vector, 'rdex', holds main ports,
// and, thus, tracks active pairs that can be reduced in parallel. Pointers are unboxed, meaning
// that ERAs, NUMs and REFs don't use any additional space. REFs lazily expand to closed nets when
// they interact with nodes, and are cleared when they interact with ERAs, allowing for constant
// space evaluation of recursive functions on Scott encoded datatypes.

mod heap;

use self::heap::*;

pub type Tag = u8;
pub type Val = u32;

// Core terms.
//pub const NIL: Tag = 0x0000; // uninitialized
pub const VR1: Tag = 0x0; // aux port to aux port 1
pub const VR2: Tag = 0x1; // aux port to aux port 2
pub const REF: Tag = 0x2; // closed net reference
pub const ERA: Tag = 0x3; // unboxed eraser
pub const CON: Tag = 0x4; // main port of con node
pub const DUP: Tag = 0x5; // main port of dup node

// Root pointer.
pub const ERAS: Ptr = Ptr(0x0000_0000 | ERA as Val);
pub const ROOT: Ptr = Ptr(0x0000_0000 | VR2 as Val);
pub const NULL: Ptr = Ptr(0x0000_0000);

// An auxiliary port.
pub type Port = Val;
pub const P1 : Port = 0;
pub const P2 : Port = 1;

// A tagged pointer.
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
#[repr(transparent)]
pub struct Ptr(pub Val);

// A interaction combinator net.
#[derive(Clone, Debug, Eq, PartialEq, Hash)]
pub struct Net {
  //pub root: Ptr, // entrancy
  pub rdex: Vec<(Ptr,Ptr)>, // redexes
  pub heap: Heap, // nodes
  pub anni: usize, // anni rewrites
  pub comm: usize, // comm rewrites
  pub eras: usize, // eras rewrites
  pub dref: usize, // dref rewrites
}

// A compact closed net, used for dereferences.
#[derive(Clone, Debug, Eq, PartialEq, Hash)]
pub struct Def {
  pub root: Ptr,
  pub rdex: Vec<(Ptr,Ptr)>,
  pub node: Vec<(Ptr,Ptr)>,
}

// A map of id to definitions (closed nets).
pub struct Book {
  defs: Vec<Def>,
}

impl Ptr {
  #[inline(always)]
  pub fn new(tag: Tag, val: Val) -> Self {
    Ptr((val << 4) | (tag as Val))
  }

  #[inline(always)]
  pub fn data(&self) -> Val {
    return self.0;
  }

  #[inline(always)]
  pub fn tag(&self) -> Tag {
    (self.data() & 0xF) as Tag
  }

  #[inline(always)]
  pub fn val(&self) -> Val {
    (self.data() >> 4) as Val
  }

  #[inline(always)]
  pub fn is_nil(&self) -> bool {
    return self.data() == 0;
  }

  #[inline(always)]
  pub fn is_var(&self) -> bool {
    return self.tag() >= VR1 && self.tag() <= VR2;
  }

  #[inline(always)]
  pub fn is_era(&self) -> bool {
    return self.tag() == ERA;
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
    return self.is_era() || self.is_ctr() || self.is_ref();
  }

  #[inline(always)]
  pub fn has_loc(&self) -> bool {
    return self.is_ctr() || self.is_var();
  }

  #[inline(always)]
  pub fn adjust(&self, loc: Val) -> Ptr {
    return Ptr::new(self.tag(), self.val() + if self.has_loc() { loc - 1 } else { 0 });
  }
}

impl Book {
  #[inline(always)]
  pub fn new() -> Self {
    Book { defs: vec![Def::new(); 1 << 24] }
  }

  #[inline(always)]
  pub fn def(&mut self, id: Val, def: Def) {
    self.defs[id as usize] = def;
  }

  #[inline(always)]
  pub fn get(&self, id: Val) -> Option<&Def> {
    self.defs.get(id as usize)
  }
}

impl Def {
  pub fn new() -> Self {
    Def {
      root: NULL,
      rdex: vec![],
      node: vec![],
    }
  }
}

impl Net {
  // Creates an empty net with given size.
  pub fn new(size: usize) -> Self {
    Net {
      rdex: vec![],
      heap: Heap::new(size),
      anni: 0,
      comm: 0,
      eras: 0,
      dref: 0,
    }
  }

  // Creates a net and boots from a REF.
  pub fn boot(&mut self, root_id: Val) {
    self.heap.set_root(Ptr::new(REF, root_id));
  }

  // Converts to a def.
  pub fn to_def(mut self) -> Def {
    let (root,node) = self.heap.compact();
    return Def { root, rdex: self.rdex, node };
  }

  // Gets a pointer's target.
  #[inline(always)]
  pub fn get_target(&self, ptr: Ptr) -> Ptr {
    return self.heap.get(ptr.val(), ptr.0 & 1);
  }

  // Sets a pointer's target.
  #[inline(always)]
  pub fn set_target(&mut self, ptr: Ptr, val: Ptr) {
    self.heap.set(ptr.val(), ptr.0 & 1, val)
  }

  // Links two pointers, forming a new wire.
  pub fn link(&mut self, a: Ptr, b: Ptr) {
    // Creates redex A-B
    if a.is_pri() && b.is_pri() {
      self.rdex.push((a, b));
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

  // Performs an interaction over a redex.
  pub fn interact(&mut self, book: &Book, a: Ptr, b: Ptr) {
    let mut a = a;
    let mut b = b;

    // Dereference A
    if a.is_ref() && b.is_ctr() {
      a = self.deref(book, a, b);
    } else if b.is_ref() && a.is_ctr() {
      b = self.deref(book, b, a);
    }

    // CON-CON
    if a.is_ctr() && b.is_ctr() && a.tag() == b.tag() {
      self.anni(a, b);
    // CON-DUP
    } else if a.is_ctr() && b.is_ctr() && a.tag() != b.tag() {
      self.comm(a, b);
    // ERA-CON
    } else if a.is_era() && b.is_ctr() {
      self.eras(a, b);
    // CON-ERA
    } else if a.is_ctr() && b.is_era() {
      self.eras(b, a);
    // ERA-ERA
    } else if a.is_era() && b.is_era() {
      self.void(a, b);
    }
  }

  #[inline(always)]
  pub fn anni(&mut self, a: Ptr, b: Ptr) {
    self.anni += 1;
    self.link(self.heap.get(a.val(), P1), self.heap.get(b.val(), P1));
    self.link(self.heap.get(a.val(), P2), self.heap.get(b.val(), P2));
    self.heap.free(a.val());
    self.heap.free(b.val());
  }

  #[inline(always)]
  pub fn comm(&mut self, a: Ptr, b: Ptr) {
    self.comm += 1;
    let loc = self.heap.alloc(4);
    self.link(self.heap.get(a.val(), P1), Ptr::new(b.tag(), loc+0));
    self.link(self.heap.get(b.val(), P1), Ptr::new(a.tag(), loc+2));
    self.link(self.heap.get(a.val(), P2), Ptr::new(b.tag(), loc+1));
    self.link(self.heap.get(b.val(), P2), Ptr::new(a.tag(), loc+3));
    self.heap.set(loc+0,P1,Ptr::new(VR1, loc+2));
    self.heap.set(loc+0,P2,Ptr::new(VR1, loc+3));
    self.heap.set(loc+1,P1,Ptr::new(VR2, loc+2));
    self.heap.set(loc+1,P2,Ptr::new(VR2, loc+3));
    self.heap.set(loc+2,P1,Ptr::new(VR1, loc+0));
    self.heap.set(loc+2,P2,Ptr::new(VR1, loc+1));
    self.heap.set(loc+3,P1,Ptr::new(VR2, loc+0));
    self.heap.set(loc+3,P2,Ptr::new(VR2, loc+1));
    self.heap.free(a.val());
    self.heap.free(b.val());
  }

  #[inline(always)]
  pub fn eras(&mut self, a: Ptr, b: Ptr) {
    self.eras += 1;
    self.link(self.heap.get(b.val(), P1), ERAS);
    self.link(self.heap.get(b.val(), P2), ERAS);
    self.heap.free(b.val());
  }

  #[inline(always)]
  pub fn void(&mut self, a: Ptr, b: Ptr) {
    self.eras += 1;
  }

  // Expands a closed net.
  #[inline(always)]
  pub fn deref(&mut self, book: &Book, ptr: Ptr, parent: Ptr) -> Ptr {
    self.dref += 1;
    let mut ptr = ptr;
    // White ptr is still a REF...
    if ptr.is_ref() {
      // Load the closed net.
      if let Some(got) = book.get(ptr.val()) {
        let len = got.node.len();
        let loc = self.heap.alloc(len);
        // Load nodes, adjusted.
        for i in 0 .. len as Val {
          unsafe {
            let p1 = got.node.get_unchecked(i as usize).0.adjust(loc);
            let p2 = got.node.get_unchecked(i as usize).1.adjust(loc);
            self.heap.set(loc+i,P1,p1);
            self.heap.set(loc+i,P2,p2);
          }
        }
        // Load redexes, adjusted.
        for r in &got.rdex {
          let p1 = r.0.adjust(loc);
          let p2 = r.1.adjust(loc);
          self.rdex.push((p1, p2));
        }
        // Load root, adjusted.
        ptr = got.root.adjust(loc);
        // Link root.
        if ptr.is_var() {
          self.set_target(ptr, parent);
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
      for (a, b) in &rdex {
        self.interact(book, *a, *b);
      }
      rdex.clear();
      std::mem::swap(&mut self.rdex, &mut rdex);
    }
  }

  // Reduce a net to normal form.
  pub fn normal(&mut self, book: &Book) {
    self.expand(book, ROOT);
    while self.rdex.len() > 0 {
      self.reduce(book);
      self.expand(book, ROOT);
    }
  }

  // Expands heads.
  pub fn expand(&mut self, book: &Book, dir: Ptr) {
    let ptr = self.get_target(dir);
    if ptr.is_ctr() {
      self.expand(book, Ptr::new(VR1, ptr.val()));
      self.expand(book, Ptr::new(VR2, ptr.val()));
    } else if ptr.is_ref() {
      let exp = self.deref(book, ptr, dir);
      self.set_target(dir, exp);
    }
  }

  // Total rewrite count.
  pub fn rewrites(&mut self) -> usize {
    return self.anni + self.comm + self.eras + self.dref;
  }
}

// The Ptr type on the file above has been refactored. Now, instead of a wrapping struct, it will
// be just a type alias to u32. Refactor the file above to be correct after that change. This will
// require changing all the functions inside `impl Ptr` to independent functions that receive Ptr
// as the first argument, 
