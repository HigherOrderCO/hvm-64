// An efficient Interaction Combinator runtime
// ===========================================
// This file implements an efficient interaction combinator runtime. Nodes are represented by 2 aux
// ports (P1, P2), with the main port (P1) omitted. A separate vector, 'rdex', holds main ports,
// and, thus, tracks active pairs that can be reduced in parallel. Pointers are unboxed, meaning
// that ERAs, NUMs and REFs don't use any additional space. REFs lazily expand to closed nets when
// they interact with nodes, and are cleared when they interact with ERAs, allowing for constant
// space evaluation of recursive functions on Scott encoded datatypes.

pub type Tag = u8;
pub type Val = u32;

// Core terms.

pub const VR1: Tag = 0x0; // Variable to aux port 1
pub const VR2: Tag = 0x1; // Variable to aux port 2
pub const RD1: Tag = 0x2; // Redirect to aux port 1
pub const RD2: Tag = 0x3; // Redirect to aux port 2
pub const REF: Tag = 0x4; // Lazy closed net
pub const ERA: Tag = 0x5; // Unboxed eraser
pub const NUM: Tag = 0x6; // Unboxed number
pub const OP2: Tag = 0x7; // Binary numeric operation
pub const OP1: Tag = 0x8; // Unary numeric operation
pub const MAT: Tag = 0x9; // Numeric pattern-matching
pub const CT0: Tag = 0xA; // Main port of con node, label 0
pub const CT1: Tag = 0xB; // Main port of con node, label 1
pub const CT2: Tag = 0xC; // Main port of con node, label 2
pub const CT3: Tag = 0xD; // Main port of con node, label 3
pub const CT4: Tag = 0xE; // Main port of con node, label 4
pub const CT5: Tag = 0xF; // Main port of con node, label 5

// Numeric operations.
pub const USE: Tag = 0x0; // set-next-op
pub const ADD: Tag = 0x1; // addition
pub const SUB: Tag = 0x2; // subtraction
pub const MUL: Tag = 0x3; // multiplication
pub const DIV: Tag = 0x4; // division
pub const MOD: Tag = 0x5; // modulus
pub const EQ : Tag = 0x6; // equal-to
pub const NE : Tag = 0x7; // not-equal-to
pub const LT : Tag = 0x8; // less-than
pub const GT : Tag = 0x9; // greater-than
pub const AND: Tag = 0xA; // logical-and
pub const OR : Tag = 0xB; // logical-or
pub const XOR: Tag = 0xC; // logical-xor
pub const NOT: Tag = 0xD; // logical-not
pub const LSH: Tag = 0xE; // left-shift
pub const RSH: Tag = 0xF; // right-shift

// Root pointer.
pub const ERAS: Ptr = Ptr(0x0000_0000 | ERA as Val);
pub const ROOT: Ptr = Ptr(0x0000_0000 | VR2 as Val);
pub const NULL: Ptr = Ptr(0x0000_0000);

// An auxiliary port.
pub type Port = Val;
pub const P1: Port = 0;
pub const P2: Port = 1;

// A tagged pointer.
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
pub struct Ptr(pub Val);

#[derive(Clone, Debug, Eq, PartialEq, Hash)]
pub struct Heap {
  data: Vec<(Ptr, Ptr)>,
  next: usize,
  used: usize,
  full: bool,
}

// A interaction combinator net.
#[derive(Clone, Debug, Eq, PartialEq, Hash)]
pub struct Net {
  pub rdex: Vec<(Ptr,Ptr)>, // redexes
  pub heap: Heap, // nodes
  pub anni: usize, // anni rewrites
  pub comm: usize, // comm rewrites
  pub eras: usize, // eras rewrites
  pub dref: usize, // dref rewrites
  pub oper: usize, // oper rewrites
}

// A compact closed net, used for dereferences.
#[derive(Clone, Debug, Eq, PartialEq, Hash)]
pub struct Def {
  pub rdex: Vec<(Ptr, Ptr)>,
  pub node: Vec<(Ptr, Ptr)>,
}

// A map of id to definitions (closed nets).
pub struct Book {
  pub defs: Vec<Def>,
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
    return matches!(self.tag(), VR1..=VR2);
  }

  #[inline(always)]
  pub fn is_era(&self) -> bool {
    return matches!(self.tag(), ERA);
  }

  #[inline(always)]
  pub fn is_ctr(&self) -> bool {
    return matches!(self.tag(), CT0..);
  }

  #[inline(always)]
  pub fn is_ref(&self) -> bool {
    return matches!(self.tag(), REF);
  }

  #[inline(always)]
  pub fn is_pri(&self) -> bool {
    return matches!(self.tag(), REF..);
  }

  #[inline(always)]
  pub fn is_num(&self) -> bool {
    return matches!(self.tag(), NUM);
  }

  #[inline(always)]
  pub fn is_op1(&self) -> bool {
    return matches!(self.tag(), OP1);
  }

  #[inline(always)]
  pub fn is_op2(&self) -> bool {
    return matches!(self.tag(), OP2);
  }

  #[inline(always)]
  pub fn is_skp(&self) -> bool {
    return matches!(self.tag(), ERA | NUM | REF);
  }

  #[inline(always)]
  pub fn is_mat(&self) -> bool {
    return matches!(self.tag(), MAT);
  }

  #[inline(always)]
  pub fn has_loc(&self) -> bool {
    return matches!(self.tag(), VR1..=VR2 | OP2..);
  }

  #[inline(always)]
  pub fn adjust(&self, loc: Val) -> Ptr {
    return Ptr::new(self.tag(), self.val() + if self.has_loc() { loc - 1 } else { 0 });
  }

  // Can this redex be skipped (as an optimization)?
  #[inline(always)]
  pub fn can_skip(a: Ptr, b: Ptr) -> bool {
    return matches!(a.tag(), ERA | REF) && matches!(b.tag(), ERA | REF);
  }
}

impl Book {
  #[inline(always)]
  pub fn new() -> Self {
    Book {
      defs: vec![Def::new(); 1 << 24],
    }
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
      rdex: vec![],
      node: vec![],
    }
  }
}

impl Heap {
  pub fn new(size: usize) -> Heap {
    return Heap {
      data: vec![(NULL, NULL); size],
      next: 1,
      used: 0,
      full: false,
    };
  }

  #[inline(always)]
  pub fn alloc(&mut self, size: usize) -> Val {
    if size == 0 {
      return 0;
    } else if !self.full && self.next + size <= self.data.len() {
      self.used += size;
      self.next += size;
      return (self.next - size) as Val;
    } else {
      self.full = true;
      let mut space = 0;
      loop {
        if self.next >= self.data.len() {
          space = 0;
          self.next = 1;
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

  #[inline(always)]
  pub fn free(&mut self, index: Val) {
    self.used -= 1;
    self.set(index, P1, NULL);
    self.set(index, P2, NULL);
  }

  #[inline(always)]
  pub fn lock(&self, index: Val) {
    return;
  }

  #[inline(always)]
  pub fn unlock(&self, index: Val) {
    return;
  }

  #[inline(always)]
  pub fn get(&self, index: Val, port: Port) -> Ptr {
    unsafe {
      let node = self.data.get_unchecked(index as usize);
      if port == P1 {
        return node.0;
      } else {
        return node.1;
      }
    }
  }

  #[inline(always)]
  pub fn set(&mut self, index: Val, port: Port, value: Ptr) {
    unsafe {
      let node = self.data.get_unchecked_mut(index as usize);
      if port == P1 {
        node.0 = value;
      } else {
        node.1 = value;
      }
    }
  }

  #[inline(always)]
  pub fn get_root(&self) -> Ptr {
    return self.get(0, P2);
  }

  #[inline(always)]
  pub fn set_root(&mut self, value: Ptr) {
    self.set(0, P2, value);
  }

  #[inline(always)]
  pub fn compact(&self) -> Vec<(Ptr, Ptr)> {
    let mut node = vec![];
    loop {
      let p1 = self.data[node.len()].0;
      let p2 = self.data[node.len()].1;
      if p1 != NULL || p2 != NULL {
        node.push((p1, p2));
      } else {
        break;
      }
    }
    return node;
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
      oper: 0,
    }
  }

  // Creates a net and boots from a REF.
  pub fn boot(&mut self, root_id: Val) {
    self.heap.set_root(Ptr::new(REF, root_id));
  }

  // Converts to a def.
  pub fn to_def(self) -> Def {
    Def { rdex: self.rdex, node: self.heap.compact() }
  }

  // Reads back from a def.
  pub fn from_def(def: Def) -> Self {
    let mut net = Net::new(def.node.len());
    for (i, &(p1, p2)) in def.node.iter().enumerate() {
      net.heap.set(i as Val, P1, p1);
      net.heap.set(i as Val, P2, p2);
    }
    net.rdex = def.rdex;
    net
  }

  // Gets a pointer's target.
  #[inline(always)]
  pub fn get_target(&self, ptr: Ptr) -> Ptr {
    self.heap.get(ptr.val(), ptr.0 & 1)
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
      if Ptr::can_skip(a, b) {
        self.eras += 1;
      } else {
        self.rdex.push((a, b));
      }
      return;
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
    if a.is_ref() && b.is_pri() && !b.is_skp() {
      a = self.deref(book, a, b);
    } else if b.is_ref() && a.is_pri() && !a.is_skp() {
      b = self.deref(book, b, a);
    }
    match (a.tag(), b.tag()) {
      (CT0.. , CT0..) if a.tag() == b.tag() => self.anni(a, b),
      (CT0.. , CT0..) => self.comm(a, b),
      (CT0.. , ERA  ) => self.era2(a),
      (ERA   , CT0..) => self.era2(b),
      (REF   , ERA  ) => self.eras += 1,
      (ERA   , REF  ) => self.eras += 1,
      (ERA   , ERA  ) => self.eras += 1,
      (VR1   , _    ) => self.link(a, b),
      (VR2   , _    ) => self.link(a, b),
      (_     , VR1  ) => self.link(b, a),
      (_     , VR2  ) => self.link(b, a),
      (CT0.. , NUM  ) => self.copy(a, b),
      (NUM   , CT0..) => self.copy(b, a),
      (NUM   , ERA  ) => self.eras += 1,
      (ERA   , NUM  ) => self.eras += 1,
      (NUM   , NUM  ) => self.eras += 1,
      (OP2   , NUM  ) => self.op2n(a, b),
      (NUM   , OP2  ) => self.op2n(b, a),
      (OP1   , NUM  ) => self.op1n(a, b),
      (NUM   , OP1  ) => self.op1n(b, a),
      (OP2   , CT0..) => self.comm(a, b),
      (CT0.. , OP2  ) => self.comm(b, a),
      (OP1   , CT0..) => self.pass(a, b),
      (CT0.. , OP1  ) => self.pass(b, a),
      (OP2   , ERA  ) => self.era2(a),
      (ERA   , OP2  ) => self.era2(b),
      (OP1   , ERA  ) => self.era1(a),
      (ERA   , OP1  ) => self.era1(b),
      (MAT   , NUM  ) => self.mtch(a, b),
      (NUM   , MAT  ) => self.mtch(b, a),
      (MAT   , CT0..) => self.comm(a, b),
      (CT0.. , MAT  ) => self.comm(b, a),
      (MAT   , ERA  ) => self.era2(a),
      (ERA   , MAT  ) => self.era2(b),
      _               => unreachable!(),
    };
  }

  pub fn conn(&mut self, a: Ptr, b: Ptr) {
    self.anni += 1;
    self.link(self.heap.get(a.val(), P2), self.heap.get(b.val(), P2));
    self.heap.free(a.val());
    self.heap.free(b.val());
  }

  pub fn anni(&mut self, a: Ptr, b: Ptr) {
    self.anni += 1;
    self.link(self.heap.get(a.val(), P1), self.heap.get(b.val(), P1));
    self.link(self.heap.get(a.val(), P2), self.heap.get(b.val(), P2));
    self.heap.free(a.val());
    self.heap.free(b.val());
  }

  pub fn comm(&mut self, a: Ptr, b: Ptr) {
    self.comm += 1;
    let loc = self.heap.alloc(4);
    self.link(self.heap.get(a.val(), P1), Ptr::new(b.tag(), loc + 0));
    self.link(self.heap.get(b.val(), P1), Ptr::new(a.tag(), loc + 2));
    self.link(self.heap.get(a.val(), P2), Ptr::new(b.tag(), loc + 1));
    self.link(self.heap.get(b.val(), P2), Ptr::new(a.tag(), loc + 3));
    self.heap.set(loc + 0, P1, Ptr::new(VR1, loc + 2));
    self.heap.set(loc + 0, P2, Ptr::new(VR1, loc + 3));
    self.heap.set(loc + 1, P1, Ptr::new(VR2, loc + 2));
    self.heap.set(loc + 1, P2, Ptr::new(VR2, loc + 3));
    self.heap.set(loc + 2, P1, Ptr::new(VR1, loc + 0));
    self.heap.set(loc + 2, P2, Ptr::new(VR1, loc + 1));
    self.heap.set(loc + 3, P1, Ptr::new(VR2, loc + 0));
    self.heap.set(loc + 3, P2, Ptr::new(VR2, loc + 1));
    self.heap.free(a.val());
    self.heap.free(b.val());
  }

  pub fn pass(&mut self, a: Ptr, b: Ptr) {
    self.comm += 1;
    let loc = self.heap.alloc(3);
    self.link(self.heap.get(a.val(), P2), Ptr::new(b.tag(), loc+0));
    self.link(self.heap.get(b.val(), P1), Ptr::new(a.tag(), loc+1));
    self.link(self.heap.get(b.val(), P2), Ptr::new(a.tag(), loc+2));
    self.heap.set(loc + 0, P1, Ptr::new(VR2, loc+1));
    self.heap.set(loc + 0, P2, Ptr::new(VR2, loc+2));
    self.heap.set(loc + 1, P1, self.heap.get(a.val(), P1));
    self.heap.set(loc + 1, P2, Ptr::new(VR1, loc+0));
    self.heap.set(loc + 2, P1, self.heap.get(a.val(), P1));
    self.heap.set(loc + 2, P2, Ptr::new(VR2, loc+0));
    self.heap.free(a.val());
    self.heap.free(b.val());
  }

  pub fn copy(&mut self, a: Ptr, b: Ptr) {
    self.comm += 1;
    self.link(self.heap.get(a.val(), P1), b);
    self.link(self.heap.get(a.val(), P2), b);
    self.heap.free(a.val());
  }

  pub fn era2(&mut self, a: Ptr) {
    self.eras += 1;
    self.link(self.heap.get(a.val(), P1), ERAS);
    self.link(self.heap.get(a.val(), P2), ERAS);
    self.heap.free(a.val());
  }

  pub fn era1(&mut self, a: Ptr) {
    self.eras += 1;
    self.link(self.heap.get(a.val(), P2), ERAS);
    self.heap.free(a.val());
  }


  pub fn op2n(&mut self, a: Ptr, b: Ptr) {
    self.oper += 1;
    let mut p1 = self.heap.get(a.val(), P1);
    // Optimization: perform chained ops at once
    if p1.is_num() {
      let mut rt = b.val();
      let mut p2 = self.heap.get(a.val(), P2);
      loop {
        self.oper += 1;
        rt = self.prim(rt, p1.val());
        // If P2 is OP2, keep looping
        if p2.is_op2() {
          p1 = self.heap.get(p2.val(), P1);
          if p1.is_num() {
            p2 = self.heap.get(p2.val(), P2);
            self.oper += 1; // since OP1 is skipped
            continue;
          }
        }
        // If P2 is OP1, flip args and keep looping
        if p2.is_op1() {
          let tmp = rt;
          rt = self.heap.get(p2.val(), P1).val();
          p1 = Ptr::new(NUM, tmp);
          p2 = self.heap.get(p2.val(), P2);
          continue;
        }
        break;
      }
      self.link(Ptr::new(NUM, rt), p2);
      return;
    }
    self.heap.set(a.val(), P1, b);
    self.link(Ptr::new(OP1, a.val()), p1);
  }

  pub fn op1n(&mut self, a: Ptr, b: Ptr) {
    self.oper += 1;
    let p1 = self.heap.get(a.val(), P1);
    let p2 = self.heap.get(a.val(), P2);
    let v0 = p1.val() as Val;
    let v1 = b.val() as Val;
    let v2 = self.prim(v0, v1);
    self.link(Ptr::new(NUM, v2), p2);
    self.heap.free(a.val());
  }

  pub fn prim(&mut self, a: Val, b: Val) -> Val {
    let a_opr = (a >> 24) & 0xF;
    let b_opr = (b >> 24) & 0xF; // not used yet
    let a_val = a & 0xFFFFFF;
    let b_val = b & 0xFFFFFF;
    match a_opr as Tag {
      USE => { ((a_val & 0xF) << 24) | b_val }
      ADD => { (a_val.wrapping_add(b_val)) & 0xFFFFFF }
      SUB => { (a_val.wrapping_sub(b_val)) & 0xFFFFFF }
      MUL => { (a_val.wrapping_mul(b_val)) & 0xFFFFFF }
      DIV => { if b_val == 0 { 0xFFFFFF } else { (a_val.wrapping_div(b_val)) & 0xFFFFFF } }
      MOD => { (a_val.wrapping_rem(b_val)) & 0xFFFFFF }
      EQ  => { ((a_val == b_val) as Val) & 0xFFFFFF }
      NE  => { ((a_val != b_val) as Val) & 0xFFFFFF }
      LT  => { ((a_val < b_val) as Val) & 0xFFFFFF }
      GT  => { ((a_val > b_val) as Val) & 0xFFFFFF }
      AND => { (a_val & b_val) & 0xFFFFFF }
      OR  => { (a_val | b_val) & 0xFFFFFF }
      XOR => { (a_val ^ b_val) & 0xFFFFFF }
      NOT => { (!b_val) & 0xFFFFFF }
      LSH => { (a_val << b_val) & 0xFFFFFF }
      RSH => { (a_val >> b_val) & 0xFFFFFF }
      _   => { unreachable!() }
    }
  }

  pub fn mtch(&mut self, a: Ptr, b: Ptr) {
    self.oper += 1;
    let p1 = self.heap.get(a.val(), P1); // branch
    let p2 = self.heap.get(a.val(), P2); // return
    if b.val() == 0 {
      let loc = self.heap.alloc(1);
      self.heap.set(loc+0, P2, ERAS);
      self.link(p1, Ptr::new(CT0, loc+0));
      self.link(p2, Ptr::new(VR1, loc+0));
      self.heap.free(a.val());
    } else {
      let loc = self.heap.alloc(2);
      self.heap.set(loc+0, P1, ERAS);
      self.heap.set(loc+0, P2, Ptr::new(CT0, loc + 1));
      self.heap.set(loc+1, P1, Ptr::new(NUM, b.val() - 1));
      self.link(p1, Ptr::new(CT0, loc+0));
      self.link(p2, Ptr::new(VR2, loc+1));
      self.heap.free(a.val());
    }
  }

  // Expands a closed net.
  #[inline(always)]
  pub fn deref(&mut self, book: &Book, ptr: Ptr, parent: Ptr) -> Ptr {
    self.dref += 1;
    let mut ptr = ptr;
    // FIXME: change "while" to "if" once lang prevents refs from returning refs
    while ptr.is_ref() {
      // Load the closed net.
      let got = unsafe { book.defs.get_unchecked((ptr.val() as usize) & 0xFFFFFF) };
      if got.node.len() > 0 {
        let len = got.node.len() - 1;
        let loc = self.heap.alloc(len);
        // Load nodes, adjusted.
        for i in 0..len as Val {
          unsafe {
            let p1 = got.node.get_unchecked(1 + i as usize).0.adjust(loc);
            let p2 = got.node.get_unchecked(1 + i as usize).1.adjust(loc);
            self.heap.set(loc + i, P1, p1);
            self.heap.set(loc + i, P2, p2);
          }
        }
        // Load redexes, adjusted.
        for r in &got.rdex {
          let p1 = r.0.adjust(loc);
          let p2 = r.1.adjust(loc);
          self.rdex.push((p1, p2));
        }
        // Load root, adjusted.
        ptr = got.node[0].1.adjust(loc);
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
    let mut rdex: Vec<(Ptr, Ptr)> = vec![];
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
  pub fn rewrites(&self) -> usize {
    return self.anni + self.comm + self.eras + self.dref + self.oper;
  }
}
