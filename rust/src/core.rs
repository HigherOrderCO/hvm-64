// An efficient Interaction Combinator runtime
// ===========================================
// This file implements an efficient interaction combinator runtime. Nodes are represented by 2 aux
// ports (P1, P2), with the main port (P1) omitted. A separate vector, 'rdex', holds main ports,
// and, thus, tracks active pairs that can be reduced in parallel. Pointers are unboxed, meaning
// that ERAs, NUMs and REFs don't use any additional space. REFs lazily expand to closed nets when
// they interact with nodes, and are cleared when they interact with ERAs, allowing for constant
// space evaluation of recursive functions on Scott encoded datatypes.

use std::collections::HashMap;

pub type Tag = u16;
pub type Val = u64;

// Core terms.
pub const NIL: Tag = 0x0000; // uninitialized
pub const REF: Tag = 0x0001; // closed net reference
pub const ERA: Tag = 0x0002; // unboxed eraser
pub const VRR: Tag = 0x0003; // aux port to root
pub const VR1: Tag = 0x0004; // aux port to aux port 1
pub const VR2: Tag = 0x0005; // aux port to aux port 2
pub const RDR: Tag = 0x0006; // redirect to root
pub const RD1: Tag = 0x0007; // redirect to aux port 1
pub const RD2: Tag = 0x0008; // redirect to aux port 2
pub const U32: Tag = 0x0009; // unboxed u32
pub const I32: Tag = 0x000A; // unboxed i32
pub const CON: Tag = 0x1000; // main port of con node
pub const DUP: Tag = 0x1001; // main port of dup node

// Numeric Operations.
pub const OPX_ADD: Tag = 0x0100;
pub const OPY_ADD: Tag = 0x0200;
pub const OPX_SUB: Tag = 0x0101;
pub const OPY_SUB: Tag = 0x0201;
pub const OPX_MUL: Tag = 0x0102;
pub const OPY_MUL: Tag = 0x0202;
pub const OPX_DIV: Tag = 0x0103;
pub const OPY_DIV: Tag = 0x0203;
pub const OPX_MOD: Tag = 0x0104;
pub const OPY_MOD: Tag = 0x0204;
pub const OPX_EQ : Tag = 0x0105;
pub const OPY_EQ : Tag = 0x0205;
pub const OPX_NEQ: Tag = 0x0106;
pub const OPY_NEQ: Tag = 0x0206;
pub const OPX_LT : Tag = 0x0107;
pub const OPY_LT : Tag = 0x0207;
pub const OPX_GT : Tag = 0x0108;
pub const OPY_GT : Tag = 0x0208;
pub const OPX_LTE: Tag = 0x0109;
pub const OPY_LTE: Tag = 0x0209;
pub const OPX_GTE: Tag = 0x010A;
pub const OPY_GTE: Tag = 0x020A;
pub const OPX_AND: Tag = 0x010B;
pub const OPY_AND: Tag = 0x020B;
pub const OPX_OR : Tag = 0x010C;
pub const OPY_OR : Tag = 0x020C;

// Operation ranges.
pub const MIN_OPX: Tag = 0x0100;
pub const MAX_OPX: Tag = 0x01FF;
pub const MIN_OPY: Tag = 0x0200;
pub const MAX_OPY: Tag = 0x02FF;

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
    Ptr(((tag as Val) << 48) | (val & 0xFFFF_FFFF_FFFF))
  }

  #[inline(always)]
  pub fn data(&self) -> Val {
    return self.0;
  }

  #[inline(always)]
  pub fn tag(&self) -> Tag {
    (self.data() >> 48) as Tag
  }

  #[inline(always)]
  pub fn val(&self) -> Val {
    (self.data() & 0xFFFF_FFFF_FFFF) as Val
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
  pub fn is_u32(&self) -> bool {
    return self.tag() == U32;
  }

  #[inline(always)]
  pub fn is_i32(&self) -> bool {
    return self.tag() == I32;
  }

  #[inline(always)]
  pub fn is_num(&self) -> bool {
    return self.is_u32() || self.is_i32();
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
    return self.is_era() || self.is_ctr() || self.is_num()
        || self.is_ref() || self.is_opX() || self.is_opY();
  }

  #[inline(always)]
  pub fn is_opX(&self) -> bool {
    return self.tag() >= MIN_OPX && self.tag() <= MAX_OPX;
  }

  #[inline(always)]
  pub fn is_opY(&self) -> bool {
    return self.tag() >= MIN_OPY && self.tag() <= MAX_OPY;
  }

  #[inline(always)]
  pub fn has_loc(&self) -> bool {
    return self.is_ctr()
        || self.is_var() && self.tag() != VRR
        || self.is_opX() || self.is_opY();
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
    if a.is_ctr() && b.is_ref()
    || a.is_ctr() && b.is_num()
    || a.is_era() && b.is_ctr()
    || a.is_num() && b.is_opX()
    || a.is_num() && b.is_opY()
    || a.is_ctr() && b.is_opX()
    || a.is_ctr() && b.is_opY()
    || a.is_era() && b.is_opX()
    || a.is_era() && b.is_opY() {
      std::mem::swap(a, b);
    }

    // U32
    if a.is_u32() && b.is_ctr() {
      *a = self.unroll_u32(*a);
    }

    // I32
    if a.is_i32() && b.is_ctr() {
      *a = self.unroll_i32(*a);
    }

    // Dereference
    if a.is_ref() && b.is_ctr() {
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
    // OPX-U32
    } else if a.is_opX() && b.is_u32() {
      let v1 = self.get(a.val(), P1);
      self.set(a.val(), P1, *b);
      self.link(Ptr::new(a.tag() + (MIN_OPY - MIN_OPX), a.val()), v1);
    // OPX-I32
    } else if a.is_opX() && b.is_i32() {
      let v1 = self.get(a.val(), P1);
      self.set(a.val(), P1, *b);
      self.link(Ptr::new(a.tag() + (MIN_OPY - MIN_OPX), a.val()), v1);
    // OPY-U32
    } else if a.is_opY() && b.is_u32() {
      let p1 = self.get(a.val(), P1);
      let p2 = self.get(a.val(), P2);
      let v0 = p1.val() as u32;
      let v1 = b.val() as u32;
      let v2 = match a.tag() {
        OPY_ADD => v0.wrapping_add(v1) as Val,
        OPY_SUB => v0.wrapping_sub(v1) as Val,
        OPY_MUL => v0.wrapping_mul(v1) as Val,
        OPY_DIV => (if v1 != 0 { v0 / v1 } else { 0 }) as Val,
        OPY_MOD => (if v1 != 0 { v0 % v1 } else { 0 }) as Val,
        OPY_EQ  => (v0 == v1) as Val,
        OPY_NEQ => (v0 != v1) as Val,
        OPY_LT  => (v0 < v1) as Val,
        OPY_GT  => (v0 > v1) as Val,
        OPY_LTE => (v0 <= v1) as Val,
        OPY_GTE => (v0 >= v1) as Val,
        OPY_AND => (v0 & v1) as Val,
        OPY_OR  => (v0 | v1) as Val,
        _       => 0,
      };
      self.link(Ptr::new(U32, v2), p2);
      self.free(a.val());
    // OPY-I32
    } else if a.is_opY() && b.is_i32() {
      let p1 = self.get(a.val(), P1);
      let p2 = self.get(a.val(), P2);
      let v0 = p1.val() as i32;
      let v1 = b.val() as i32;
      let v2 = match a.tag() {
        OPY_ADD => v0.wrapping_add(v1) as Val,
        OPY_SUB => v0.wrapping_sub(v1) as Val,
        OPY_MUL => v0.wrapping_mul(v1) as Val,
        OPY_DIV => (if v1 != 0 { v0 / v1 } else { 0 }) as Val,
        OPY_MOD => (if v1 != 0 { v0 % v1 } else { 0 }) as Val,
        OPY_EQ  => (v0 == v1) as Val,
        OPY_NEQ => (v0 != v1) as Val,
        OPY_LT  => (v0 < v1) as Val,
        OPY_GT  => (v0 > v1) as Val,
        OPY_LTE => (v0 <= v1) as Val,
        OPY_GTE => (v0 >= v1) as Val,
        OPY_AND => (v0 & v1) as Val,
        OPY_OR  => (v0 | v1) as Val,
        _       => 0,
      };
      self.link(Ptr::new(U32, v2), p2);
      self.free(a.val());
    // OPX-CTR
    } else if a.is_opX() && b.is_ctr() {
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
    // OPY-CTR
    } else if a.is_opY() && b.is_ctr() {
      let loc = self.alloc(3);
      self.link(self.get(a.val(), P2), Ptr::new(b.tag(), loc+0));
      self.link(self.get(b.val(), P1), Ptr::new(a.tag(), loc+1));
      self.link(self.get(b.val(), P2), Ptr::new(a.tag(), loc+2));
      *self.at_mut(loc+0) = Node::new(Ptr::new(VR2, loc+1), Ptr::new(VR2, loc+2));
      *self.at_mut(loc+1) = Node::new(Ptr::new(VR1, loc+0), Ptr::new(VR2, loc+0));
      *self.at_mut(loc+2) = Node::new(self.get(a.val(),P1), self.get(a.val(),P1));
      self.free(a.val());
      self.free(b.val());
    // OPX-ERA
    } else if a.is_opX() && b.is_era() {
      self.link(self.get(a.val(), P1), Ptr::new(ERA, 0));
      self.link(self.get(a.val(), P2), Ptr::new(ERA, 0));
      self.free(a.val());
    // OPY-ERA
    } else if a.is_opY() && b.is_era() {
      self.link(self.get(a.val(), P2), Ptr::new(ERA, 0));
      self.free(a.val());
    // CON-ERA
    } else if a.is_ctr() && b.is_era() {
      self.link(self.get(a.val(), P1), Ptr::new(ERA, 0));
      self.link(self.get(a.val(), P2), Ptr::new(ERA, 0));
      self.free(a.val());
    }
  }

  // Expands an U32 literal.
  pub fn unroll_u32(&mut self, a: Ptr) -> Ptr {
    // O Case
    if a.val() > 0 && a.val() % 2 == 0 {
      let loc = self.alloc(4);
      let oc1 = Ptr::new(CON, loc + 3);
      let oc2 = Ptr::new(CON, loc + 1);
      let ic1 = Ptr::new(ERA, 0);
      let ic2 = Ptr::new(CON, loc + 2);
      let ec1 = Ptr::new(ERA, 0);
      let ec2 = Ptr::new(VR2, loc + 3);
      let ap1 = Ptr::new(U32, a.val() / 2);
      let ap2 = Ptr::new(VR2, loc + 2);
      *self.at_mut(loc+0) = Node::new(oc1, oc2);
      *self.at_mut(loc+1) = Node::new(ic1, ic2);
      *self.at_mut(loc+2) = Node::new(ec1, ec2);
      *self.at_mut(loc+3) = Node::new(ap1, ap2);
      return Ptr::new(CON, loc + 0);
    }
    // I Case
    else if a.val() > 0 && a.val() % 2 == 1 {
      let loc = self.alloc(4);
      let oc1 = Ptr::new(ERA, 0);
      let oc2 = Ptr::new(CON, loc + 1);
      let ic1 = Ptr::new(CON, loc + 3);
      let ic2 = Ptr::new(CON, loc + 2);
      let ec1 = Ptr::new(ERA, 0);
      let ec2 = Ptr::new(VR2, loc + 3);
      let ap1 = Ptr::new(U32, a.val() / 2);
      let ap2 = Ptr::new(VR2, loc + 2);
      *self.at_mut(loc+0) = Node::new(oc1, oc2);
      *self.at_mut(loc+1) = Node::new(ic1, ic2);
      *self.at_mut(loc+2) = Node::new(ec1, ec2);
      *self.at_mut(loc+3) = Node::new(ap1, ap2);
      return Ptr::new(CON, loc + 0);
    }
    // E Case
    else if a.val() == 0 {
      let loc = self.alloc(3);
      let oc1 = Ptr::new(ERA, 0);
      let oc2 = Ptr::new(CON, loc + 1);
      let ic1 = Ptr::new(ERA, 0);
      let ic2 = Ptr::new(CON, loc + 2);
      let ec1 = Ptr::new(VR2, loc + 2);
      let ec2 = Ptr::new(VR1, loc + 2);
      *self.at_mut(loc+0) = Node::new(oc1, oc2);
      *self.at_mut(loc+1) = Node::new(ic1, ic2);
      *self.at_mut(loc+2) = Node::new(ec1, ec2);
      return Ptr::new(CON, loc + 0);
    }
    unreachable!();
  }

  // Expands an I32 literal.
  // FIXME: currently expanded as U32. Should use balanced ternary instead.
  pub fn unroll_i32(&mut self, a: Ptr) -> Ptr {
    self.unroll_u32(a)
  }

  // Expands a closed net.
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
