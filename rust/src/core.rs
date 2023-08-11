use std::collections::HashMap;

pub type Tag = u8;
pub type Loc = u32;

pub const NIL: Tag = 0; // empty node
pub const REF: Tag = 1; // reference to a definition (closed net)
pub const NUM: Tag = 2; // unboxed number
pub const ERA: Tag = 3; // unboxed eraser
pub const VRT: Tag = 4; // variable pointing to root
pub const VR1: Tag = 5; // variable pointing to aux1 port of node
pub const VR2: Tag = 6; // variable pointing to aux2 port of node
pub const CON: Tag = 7; // points to main port of con node
pub const DUP: Tag = 8; // points to main port of dup node; higher labels also dups

#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
pub enum Port {
  P1,
  P2,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
pub struct Ptr {
  pub tag: Tag,
  pub loc: Loc,
}

#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub struct Node {
  pub p1: Ptr,
  pub p2: Ptr,
}

pub struct Net {
  pub root: Ptr,
  pub acts: Vec<(Ptr, Ptr)>,
  pub node: Vec<Node>,
  pub next: usize,
  pub used: usize,
  pub rwts: usize,
}

pub struct Book {
  pub defs: HashMap<u32, Net>,
}

impl Node {
  #[inline(always)]
  pub fn nil() -> Self {
    Node {
      p1: Ptr { tag: NIL, loc: 0 },
      p2: Ptr { tag: NIL, loc: 0 },
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
  #[inline(always)]
  pub fn new(size: usize) -> Self {
    Net {
      root: Ptr { tag: NIL, loc: 0 },
      acts: vec![],
      node: vec![Node::nil(); size],
      next: 0,
      used: 0,
      rwts: 0,
    }
  }

  #[inline(always)]
  pub fn alloc(&mut self, size: usize) -> Loc {
    let mut space = 0;
    loop {
      if self.next >= self.node.len() {
        space = 0;
        self.next = 0;
      }
      if self.get(self.next as Loc, Port::P1).tag == NIL {
        space += 1;
      } else {
        space = 0;
      }
      self.next += 1;
      if space == size {
        self.used += size;
        return (self.next - space) as Loc;
      }
    }
  }

  #[inline(always)]
  pub fn free(&mut self, loc: Loc) {
    self.used -= 1;
    self.node[loc as usize] = Node::nil();
  }

  #[inline(always)]
  pub fn get(&self, loc: Loc, port: Port) -> Ptr {
    let node = unsafe { self.node.get_unchecked(loc as usize) };
    match port {
      Port::P1 => { node.p1 }
      Port::P2 => { node.p2 }
    }
  }

  #[inline(always)]
  pub fn set(&mut self, loc: Loc, port: Port, value: Ptr) {
    let node = unsafe { self.node.get_unchecked_mut(loc as usize) };
    match port {
      Port::P1 => { node.p1 = value; }
      Port::P2 => { node.p2 = value; }
    }
  }

  #[inline(always)]
  pub fn link(&mut self, a: Ptr, b: Ptr) {
    if a.tag == VRT {
      self.root = b;
    }
    if a.tag == VR1 {
      self.set(a.loc, Port::P1, b);
    }
    if a.tag == VR2 {
      self.set(a.loc, Port::P2, b);
    }
    if b.tag == VRT {
      self.root = a;
    }
    if b.tag == VR1 {
      self.set(b.loc, Port::P1, a);
    }
    if b.tag == VR2 {
      self.set(b.loc, Port::P2, a);
    }
    if a.tag != VRT && a.tag != VR1 && a.tag != VR2
    && b.tag != VRT && b.tag != VR1 && b.tag != VR2 {
      self.acts.push((a, b));
    }
  }

  #[inline(always)]
  pub fn reduce(&mut self, book: &Book) {
    let acts = std::mem::replace(&mut self.acts, vec![]);
    for (mut a, mut b) in acts {
      self.interact(book, &mut a, &mut b);
    }
  }

  pub fn normal(&mut self, book: &Book) -> (usize, usize) {
    let mut rwts = 0;
    let mut iter = 0;
    loop {
      self.reduce(book);
      //println!(">> acts = {} | size = {} | used = {} | rwts = {}", self.acts.len(), self.node.len(), self.used, self.rwts);
      if self.rwts == rwts {
        break;
      }
      rwts = self.rwts;
      iter = iter + 1;
    }
    return (rwts, iter);
  }

  #[inline(always)]
  pub fn interact(&mut self, book: &Book, a: &mut Ptr, b: &mut Ptr) {
    // Dereference
    if a.tag == REF && b.tag == ERA { return; }
    if a.tag == ERA && b.tag == REF { return; }
    self.deref(book, a);
    self.deref(book, b);
    // Annihilation
    if a.tag >= CON && b.tag >= CON && a.tag == b.tag {
      let a1 = self.get(a.loc, Port::P1);
      let b1 = self.get(b.loc, Port::P1);
      self.link(a1, b1);
      let a2 = self.get(a.loc, Port::P2);
      let b2 = self.get(b.loc, Port::P2);
      self.link(a2, b2);
      self.free(a.loc);
      self.free(b.loc);
      self.rwts += 1;
    // Commutation
    } else if a.tag >= CON && b.tag >= CON && a.tag != b.tag {
      let x1 = self.alloc(1);
      let x2 = self.alloc(1);
      let y1 = self.alloc(1);
      let y2 = self.alloc(1);
      self.set(x1, Port::P1, Ptr { tag: VR1, loc: y1 });
      self.set(x1, Port::P2, Ptr { tag: VR1, loc: y2 });
      self.set(x2, Port::P1, Ptr { tag: VR2, loc: y1 });
      self.set(x2, Port::P2, Ptr { tag: VR2, loc: y2 });
      self.set(y1, Port::P1, Ptr { tag: VR1, loc: x1 });
      self.set(y1, Port::P2, Ptr { tag: VR1, loc: x2 });
      self.set(y2, Port::P1, Ptr { tag: VR2, loc: x1 });
      self.set(y2, Port::P2, Ptr { tag: VR2, loc: x2 });
      self.link(self.get(a.loc, Port::P1), Ptr { tag: b.tag, loc: x1 });
      self.link(self.get(a.loc, Port::P2), Ptr { tag: b.tag, loc: x2 });
      self.link(self.get(b.loc, Port::P1), Ptr { tag: a.tag, loc: y1 });
      self.link(self.get(b.loc, Port::P2), Ptr { tag: a.tag, loc: y2 });
      self.free(a.loc);
      self.free(b.loc);
      self.rwts += 1;
    // Erasure
    } else if a.tag >= CON && b.tag == ERA {
      self.link(self.get(a.loc, Port::P1), Ptr { tag: ERA, loc: 0 });
      self.link(self.get(a.loc, Port::P2), Ptr { tag: ERA, loc: 0 });
      self.free(a.loc);
      self.rwts += 1;
    // Erasure
    } else if a.tag == ERA && b.tag >= CON {
      self.link(self.get(b.loc, Port::P1), Ptr { tag: ERA, loc: 0 });
      self.link(self.get(b.loc, Port::P2), Ptr { tag: ERA, loc: 0 });
      self.free(b.loc);
      self.rwts += 1;
    } else {
      self.acts.push((*a,*b));
    }
  }

  // TODO: use delta locs so that this can become a single memcpy
  pub fn deref(&mut self, book: &Book, a: &mut Ptr) {
    while a.tag == REF {
      if let Some(len) = book.defs.get(&a.loc).map(|got| got.node.len()) {
        let loc = self.alloc(len);
        let got = book.defs.get(&a.loc).unwrap();
        for idx in 0 .. len {
          let mask = unsafe { got.node.get_unchecked(idx).clone() };
          let node = unsafe { self.node.get_unchecked_mut((loc as usize) + idx) };
          node.p1 = mask.p1;
          if mask.p1.tag >= VR1 {
            node.p1.loc += loc;
          }
          node.p2 = mask.p2;
          if mask.p2.tag >= VR1 {
            node.p2.loc += loc;
          }
        }
        for acts in &got.acts {
          let mut p1 = acts.0;
          let mut p2 = acts.1;
          if p1.tag >= VR1 {
            p1.loc += loc;
          }
          if p2.tag >= VR1 {
            p2.loc += loc;
          }
          self.acts.push((p1, p2));
        }
        *a = got.root;
        if got.root.tag >= VR1 {
          a.loc += loc;
        }
      }
    }
  }
}
