pub type Tag = u8;
pub type Loc = u32;

pub const NIL: Tag = 0;
pub const NUM: Tag = 1;
pub const VR1: Tag = 2;
pub const VR2: Tag = 3;
pub const ERA: Tag = 4;
pub const CON: Tag = 5;
pub const DUP: Tag = 6;

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
  pub pair: Vec<(Ptr, Ptr)>,
  pub node: Vec<Node>,
  pub free: Vec<Loc>,
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
  pub fn new() -> Self {
    Net {
      pair: vec![],
      node: vec![],
      free: vec![],
      rwts: 0,
    }
  }

  #[inline(always)]
  pub fn alloc(&mut self) -> Loc {
    if let Some(index) = self.free.pop() {
      index
    } else {
      self.node.push(Node::nil());
      (self.node.len() - 1) as Loc
    }
  }

  #[inline(always)]
  pub fn free(&mut self, loc: Loc) {
    self.free.push(loc);
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
    if a.tag == VR1 {
      self.set(a.loc, Port::P1, b);
    }
    if a.tag == VR2 {
      self.set(a.loc, Port::P2, b);
    }
    if b.tag == VR1 {
      self.set(b.loc, Port::P1, a);
    }
    if b.tag == VR2 {
      self.set(b.loc, Port::P2, a);
    }
    if a.tag != VR1 && a.tag != VR2 && b.tag != VR1 && b.tag != VR2 {
      self.pair.push((a, b));
    }
  }

  #[inline(always)]
  pub fn reduce(&mut self) {
    let pair = std::mem::replace(&mut self.pair, vec![]);
    for (a, b) in pair {
      self.interact(a, b);
    }
  }

  #[inline(always)]
  pub fn interact(&mut self, a: Ptr, b: Ptr) {
    // Annihilation
    if a.tag >= CON && b.tag >= CON && a.tag == b.tag {
      //println!(">> anni");
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
      //println!(">> comm");
      //let (x1,x2,y1,y2) = self.alloc4();
      let x1 = self.alloc();
      let x2 = self.alloc();
      let y1 = self.alloc();
      let y2 = self.alloc();
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
      //println!(">> eras");
      self.link(self.get(a.loc, Port::P1), Ptr { tag: ERA, loc: 0 });
      self.link(self.get(a.loc, Port::P2), Ptr { tag: ERA, loc: 0 });
      self.free(a.loc);
      self.rwts += 1;
    // Erasure
    } else if a.tag == ERA && b.tag >= CON {
      //println!(">> eras");
      self.link(self.get(b.loc, Port::P1), Ptr { tag: ERA, loc: 0 });
      self.link(self.get(b.loc, Port::P2), Ptr { tag: ERA, loc: 0 });
      self.free(b.loc);
      self.rwts += 1;
    } else {
      //println!(">> keep");
      self.pair.push((a,b));
    }
  }
}

//pub fn max(net: &Net) -> Loc {
  //let mut max_loc = 0;
  //for node in &net.node {
    //max_loc = max_loc.max(node.p1.loc);
    //max_loc = max_loc.max(node.p2.loc);
  //}
  //return max_loc;
//}

//pub fn histo(net: &Net, chunk_len: usize) -> std::collections::BTreeMap<usize, usize> {
  //let mut histo = std::collections::BTreeMap::new();
  //for node in &net.node {
    //let loc1 = (node.p1.loc as usize) / chunk_len;
    //let count = histo.entry(loc1).or_insert(0);
    //*count += 1;
    //let loc2 = (node.p2.loc as usize) / chunk_len;
    //let count = histo.entry(loc2).or_insert(0);
    //*count += 1;
  //}
  //return histo;
//}
