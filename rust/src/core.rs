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

// Core terms.
//pub const NIL: Tag = 0x0000; // uninitialized
pub const VR1: Tag = 0x0; // aux port to aux port 1
pub const VR2: Tag = 0x1; // aux port to aux port 2
pub const REF: Tag = 0x2; // closed net reference
pub const ERA: Tag = 0x3; // unboxed eraser
pub const CON: Tag = 0x4; // main port of con node
pub const DUP: Tag = 0x5; // main port of dup node

// Numeric Operations.
//pub const OPX_ADD: Tag = 0x0100;
//pub const OPY_ADD: Tag = 0x0200;
//pub const OPX_SUB: Tag = 0x0101;
//pub const OPY_SUB: Tag = 0x0201;
//pub const OPX_MUL: Tag = 0x0102;
//pub const OPY_MUL: Tag = 0x0202;
//pub const OPX_DIV: Tag = 0x0103;
//pub const OPY_DIV: Tag = 0x0203;
//pub const OPX_MOD: Tag = 0x0104;
//pub const OPY_MOD: Tag = 0x0204;
//pub const OPX_EQ : Tag = 0x0105;
//pub const OPY_EQ : Tag = 0x0205;
//pub const OPX_NEQ: Tag = 0x0106;
//pub const OPY_NEQ: Tag = 0x0206;
//pub const OPX_LT : Tag = 0x0107;
//pub const OPY_LT : Tag = 0x0207;
//pub const OPX_GT : Tag = 0x0108;
//pub const OPY_GT : Tag = 0x0208;
//pub const OPX_LTE: Tag = 0x0109;
//pub const OPY_LTE: Tag = 0x0209;
//pub const OPX_GTE: Tag = 0x010A;
//pub const OPY_GTE: Tag = 0x020A;
//pub const OPX_AND: Tag = 0x010B;
//pub const OPY_AND: Tag = 0x020B;
//pub const OPX_OR : Tag = 0x010C;
//pub const OPY_OR : Tag = 0x020C;

// Operation ranges.
//pub const MIN_OPX: Tag = 0x0100;
//pub const MAX_OPX: Tag = 0x01FF;
//pub const MIN_OPY: Tag = 0x0200;
//pub const MAX_OPY: Tag = 0x02FF;

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
pub struct Ptr(pub Val);

// Stats object.
pub struct Stat {
}

// A interaction combinator net.
#[derive(Clone, Debug, Eq, PartialEq, Hash)]
pub struct Net {
  //pub root: Ptr, // entrancy
  pub rdex: Vec<(Ptr,Ptr)>, // redexes
  pub node: Vec<Ptr>, // nodes
  pub used: usize, // allocated nodes
  pub anni: usize, // anni rewrites
  pub comm: usize, // comm rewrites
  pub eras: usize, // eras rewrites
  pub dref: usize, // dref rewrites
  pub next: usize, // next alloc index
}

// A map of id to definitions (closed nets).
pub struct Book {
  defs: Vec<Net>,
}

impl Ptr {
  #[inline(always)]
  pub fn new(tag: Tag, val: Val) -> Self {
    Ptr((val << 4) | (tag as Val))
    //Ptr(((tag as Val) << 48) | (val & 0xFFFF_FFFF_FFFF))
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

  //#[inline(always)]
  //pub fn is_u32(&self) -> bool {
    //return self.tag() == U32;
  //}

  //#[inline(always)]
  //pub fn is_i32(&self) -> bool {
    //return self.tag() == I32;
  //}

  //#[inline(always)]
  //pub fn is_num(&self) -> bool {
    //return self.is_u32() || self.is_i32();
  //}

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
    return Ptr::new(self.tag(), self.val() + if self.has_loc() { loc } else { 0 });
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
      //root: Ptr::new(NIL, 0),
      rdex: vec![],
      node: vec![NULL; size * 2],
      next: 1,
      used: 0,
      anni: 0,
      comm: 0,
      eras: 0,
      dref: 0,
    }
  }

  // Creates a net and boots from a REF.
  pub fn boot(&mut self, root_id: Val) {
    self.set_root(Ptr::new(REF, root_id));
  }

  // Allocates a consecutive chunk of 'size' nodes. Returns the index.
  pub fn alloc(&mut self, size: usize) -> Val {
    if size == 0 {
      return 0;
    } else {
      let mut space = 0;
      loop {
        if self.next >= self.node.len() / 2 {
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
    unsafe {
      *self.node.get_unchecked_mut((val * 2 + P1) as usize) = NULL;
      *self.node.get_unchecked_mut((val * 2 + P2) as usize) = NULL;
    }
  }

  // Gets node at given index.
  #[inline(always)]
  pub fn at(&self, index: Val) -> &Ptr {
    unsafe {
      return self.node.get_unchecked(index as usize);
    }
  }

  // Gets node at given index, mutable.
  #[inline(always)]
  pub fn at_mut(&mut self, index: Val) -> &mut Ptr {
    unsafe {
      return self.node.get_unchecked_mut(index as usize);
    }
  }

  // Gets the pointer stored on the port 1 or 2 of a node.
  #[inline(always)]
  pub fn get(&self, index: Val, port: Port) -> Ptr {
    return *self.at(index * 2 + port);
  }

  // Sets the pointer stored on the port 1 or 2 of a node.
  #[inline(always)]
  pub fn set(&mut self, index: Val, port: Port, value: Ptr) {
    *self.at_mut(index * 2 + port) = value;
  }

  // Gets the root node.
  #[inline(always)]
  pub fn get_root(&self) -> Ptr {
    return self.get(0, P2);
  }

  // Sets the root node.
  #[inline(always)]
  pub fn set_root(&mut self, value: Ptr) {
    self.set(0, P2, value);
  }

  // Gets a pointer target.
  #[inline(always)]
  pub fn target(&mut self, ptr: Ptr) -> &mut Ptr {
    return self.at_mut((ptr.val() << 1) | (ptr.0 & 1));
  }

  // Links two pointers, forming a new wire.
  pub fn link(&mut self, a: Ptr, b: Ptr) {
    // Creates redex A-B
    if a.is_pri() && b.is_pri() {
      self.rdex.push((a, b));
    }
    // Substitutes A
    if a.is_var() {
      *self.target(a) = b;
    }
    // Substitutes B
    if b.is_var() {
      *self.target(b) = a;
    }
  }

  // Performs an interaction over a redex.
  pub fn interact(&mut self, book: &Book, a: Ptr, b: Ptr) {
    let mut redex = Some((a, b));

    while let Some((mut a, mut b)) = redex {
      redex = None;

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
  }

  #[inline(always)]
  pub fn anni(&mut self, a: Ptr, b: Ptr) {
    self.anni += 1;
    self.link(self.get(a.val(), P1), self.get(b.val(), P1));
    self.link(self.get(a.val(), P2), self.get(b.val(), P2));
    self.free(a.val());
    self.free(b.val());
  }

  #[inline(always)]
  pub fn comm(&mut self, a: Ptr, b: Ptr) {
    self.comm += 1;
    let loc = self.alloc(4);
    self.link(self.get(a.val(), P1), Ptr::new(b.tag(), loc+0));
    self.link(self.get(b.val(), P1), Ptr::new(a.tag(), loc+2));
    self.link(self.get(a.val(), P2), Ptr::new(b.tag(), loc+1));
    self.link(self.get(b.val(), P2), Ptr::new(a.tag(), loc+3));
    *self.at_mut(loc*2+0) = Ptr::new(VR1, loc+2);
    *self.at_mut(loc*2+1) = Ptr::new(VR1, loc+3);
    *self.at_mut(loc*2+2) = Ptr::new(VR2, loc+2);
    *self.at_mut(loc*2+3) = Ptr::new(VR2, loc+3);
    *self.at_mut(loc*2+4) = Ptr::new(VR1, loc+0);
    *self.at_mut(loc*2+5) = Ptr::new(VR1, loc+1);
    *self.at_mut(loc*2+6) = Ptr::new(VR2, loc+0);
    *self.at_mut(loc*2+7) = Ptr::new(VR2, loc+1);
    self.free(a.val());
    self.free(b.val());
  }

  #[inline(always)]
  pub fn eras(&mut self, a: Ptr, b: Ptr) {
    self.eras += 1;
    self.link(self.get(b.val(), P1), ERAS);
    self.link(self.get(b.val(), P2), ERAS);
    self.free(b.val());
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
      // Loads the referenced definition...
      if let Some(got) = book.get(ptr.val()) {
        let loc = self.alloc(got.node.len() / 2);
        // Loads nodes, adjusting locations...
        for i in 1 .. (got.node.len() / 2) as Val {
          unsafe {
            let p1 = got.node.get_unchecked((i * 2 + P1) as usize).adjust(loc);
            let p2 = got.node.get_unchecked((i * 2 + P2) as usize).adjust(loc);
            *self.at_mut((loc + i) * 2 + P1) = p1;
            *self.at_mut((loc + i) * 2 + P2) = p2;
          }
        }
        // Loads redexes, adjusting locations...
        for got in &got.rdex {
          let p1 = got.0.adjust(loc);
          let p2 = got.1.adjust(loc);
          self.rdex.push((p1, p2));
        }
        // Overwrites 'ptr' with the loaded root pointer, adjusting locations...
        ptr = got.get_root().adjust(loc);
        // Links root
        if ptr.is_var() {
          *self.target(ptr) = parent;
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
    let ptr = *self.target(dir);
    if ptr.is_ctr() {
      self.expand(book, Ptr::new(VR1, ptr.val()));
      self.expand(book, Ptr::new(VR2, ptr.val()));
    } else if ptr.is_ref() {
      *self.target(dir) = self.deref(book, ptr, dir);
    }
  }

  // Total rewrite count.
  pub fn rewrites(&mut self) -> usize {
    return self.anni + self.comm + self.eras + self.dref;
  }
}
