use std::collections::HashMap;

pub type Tag = u8;
pub type Loc = u32;

pub const NIL: Tag = 0;
pub const REF: Tag = 1;
pub const NUM: Tag = 2;
pub const ERA: Tag = 3;
pub const VRR: Tag = 4;
pub const VR1: Tag = 5;
pub const VR2: Tag = 6;
pub const CON: Tag = 7;
pub const DUP: Tag = 8;

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

#[derive(Debug)]
pub struct Term {
  pub root: Ptr,
  pub acts: Vec<(Ptr, Ptr)>,
  pub node: Vec<Node>,
}

pub struct Net {
  pub defs: HashMap<u32, Term>,
  pub term: Term,
  pub next: usize,
  pub used: usize,
  pub rwts: usize,
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

impl Net {
  #[inline(always)]
  pub fn new(size: usize) -> Self {
    Net {
      defs: HashMap::new(),
      term: Term {
        root: Ptr { tag: NIL, loc: 0 },
        acts: vec![],
        node: vec![Node::nil(); size],
      },
      next: 0,
      used: 0,
      rwts: 0,
    }
  }

  #[inline(always)]
  pub fn alloc(&mut self, size: usize) -> Loc {
    let mut space = 0;
    loop {
      if self.next >= self.term.node.len() {
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
    self.term.node[loc as usize] = Node::nil();
  }

  #[inline(always)]
  pub fn get(&self, loc: Loc, port: Port) -> Ptr {
    let node = unsafe { self.term.node.get_unchecked(loc as usize) };
    match port {
      Port::P1 => { node.p1 }
      Port::P2 => { node.p2 }
    }
  }

  #[inline(always)]
  pub fn set(&mut self, loc: Loc, port: Port, value: Ptr) {
    let node = unsafe { self.term.node.get_unchecked_mut(loc as usize) };
    match port {
      Port::P1 => { node.p1 = value; }
      Port::P2 => { node.p2 = value; }
    }
  }

  #[inline(always)]
  pub fn link(&mut self, a: Ptr, b: Ptr) {
    if a.tag == VRR {
      self.term.root = b;
    }
    if a.tag == VR1 {
      self.set(a.loc, Port::P1, b);
    }
    if a.tag == VR2 {
      self.set(a.loc, Port::P2, b);
    }
    if b.tag == VRR {
      self.term.root = a;
    }
    if b.tag == VR1 {
      self.set(b.loc, Port::P1, a);
    }
    if b.tag == VR2 {
      self.set(b.loc, Port::P2, a);
    }
    if a.tag != VRR && a.tag != VR1 && a.tag != VR2
    && b.tag != VRR && b.tag != VR1 && b.tag != VR2 {
      self.term.acts.push((a, b));
    }
  }

  #[inline(always)]
  pub fn reduce(&mut self) {
    let acts = std::mem::replace(&mut self.term.acts, vec![]);
    for (mut a, mut b) in acts {
      self.interact(&mut a, &mut b);
    }
  }

  pub fn normal(&mut self) -> (usize, usize) {
    let mut rwts = 0;
    let mut iter = 0;
    loop {
      self.reduce();
      //println!(">> acts = {} | size = {} | used = {} | rwts = {}", self.term.acts.len(), self.term.node.len(), self.used, self.rwts);
      if self.rwts == rwts {
        break;
      }
      rwts = self.rwts;
      iter = iter + 1;
    }
    return (rwts, iter);
  }

  #[inline(always)]
  pub fn interact(&mut self, a: &mut Ptr, b: &mut Ptr) {
    // Dereference
    if a.tag == REF && b.tag == ERA { return; }
    if a.tag == ERA && b.tag == REF { return; }
    self.deref(a);
    self.deref(b);
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
      self.term.acts.push((*a,*b));
    }
  }

  pub fn def(&mut self, id: u32, term: Term) {
    self.defs.insert(id, term);
  }

  // TODO: use delta locs so that this can become a single memcpy
  pub fn deref(&mut self, a: &mut Ptr) {
    while a.tag == REF {
      if let Some(len) = self.defs.get(&a.loc).map(|got| got.node.len()) {
        let loc = self.alloc(len);
        let got = self.defs.get(&a.loc).unwrap();
        for idx in 0 .. len {
          let mask = unsafe { got.node.get_unchecked(idx).clone() };
          let node = unsafe { self.term.node.get_unchecked_mut((loc as usize) + idx) };
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
          self.term.acts.push((p1, p2));
        }
        *a = got.root;
        if got.root.tag >= VR1 {
          a.loc += loc;
        }
      }
    }
  }
}
