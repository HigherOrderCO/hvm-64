//Tag on node:
//- big = (tag: 8, p1: 32, p2: 32) // total: 70
//- sml = (tag: 8, p1: 16, p2: 16) // total: 40
//storing 4 uint32:
//- 3 big + 4 * (8+32) = 370 bits
//- 3 sml + 4 * (8+32) = 280 bits

//Tag on ptr:
//- big = (t1: 8, p1: 32, t2: 8, t2: 32) // total: 80
//- sml = (t1: 8, p1: 16, t2: 8, t2: 16) // total: 48
//storing 4 uint32:
//- 3 big + unboxed = 210 bits
//- 3 sml + 4 * 32  = 176 bits

// Tag of the target node
pub type Tag = u8;

// A location in the graph
pub type Loc = u32;

// Tag constants
pub const NIL : Tag = 0; // not allocated
pub const NUM : Tag = 1; // unboxed number
pub const VR1 : Tag = 2; // unboxed variable to port 1
pub const VR2 : Tag = 3; // unboxed variable to port 2
pub const ERA : Tag = 4; // era node
pub const CON : Tag = 5; // con node
pub const DUP : Tag = 6; // dup node

// A port
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
pub enum Port {
  P1,
  P2,
}

// A pointer to some node, or an unboxed value
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
pub struct Ptr {
  pub tag: Tag,
  pub loc: Loc,
}

// An interaction net node
#[derive(Copy, Clone, Debug)]
pub struct Node {
  pub p1: Ptr,
  pub p2: Ptr,
}

pub struct Net {
  pub pair: Vec<(Ptr,Ptr)>, // active pairs
  pub node: Vec<Node>, // graph of nodes
  pub free: Vec<Loc>, // reuse indexes
  pub rwts: usize, // rewrite count
}

impl Node {
  pub fn nil() -> Self {
    Node {
      p1: Ptr { tag: NIL, loc: 0},
      p2: Ptr { tag: NIL, loc: 0},
    }
  }
}

impl Net {
  // Creates a new net
  pub fn new() -> Self {
    Net {
      pair: vec![],
      node: vec![],
      free: vec![],
      rwts: 0,
    }
  }

  pub fn alloc(&mut self) -> Loc {
    if let Some(index) = self.free.pop() {
      return index;
    } else {
      self.node.push(Node::nil());
      return (self.node.len() - 1) as Loc;
    }
  }

  pub fn free(&mut self, loc: Loc) {
    self.free.push(loc);
    self.node[loc as usize] = Node::nil();
  }

  // Gets the nth child of a node.
  pub fn get(&self, loc: Loc, port: Port) -> Ptr {
    match port {
      Port::P1 => self.node[loc as usize].p1,
      Port::P2 => self.node[loc as usize].p2,
    }
  }

  // Sets the nth child of a node.
  pub fn set(&mut self, loc: Loc, port: Port, value: Ptr) {
    match port {
      Port::P1 => self.node[loc as usize].p1 = value,
      Port::P2 => self.node[loc as usize].p2 = value,
    }
  }

  // Links two pointers
  pub fn link(&mut self, a: Ptr, b: Ptr) {
    if a.tag == VR1 {
      //println!("link a vr1");
      self.set(a.loc, Port::P1, b);
    } else if a.tag == VR2 {
      //println!("link a vr2");
      self.set(a.loc, Port::P2, b);
    }
    if b.tag == VR1 {
      //println!("link b vr1");
      self.set(b.loc, Port::P1, a);
    } else if b.tag == VR2 {
      //println!("link b vr2");
      self.set(b.loc, Port::P2, a);
    }
    if a.tag != VR1 && a.tag != VR2 && b.tag != VR1 && b.tag != VR2 {
      //println!("link a-b");
      self.pair.push((a,b));
    }
  }

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

  // Reduces all active pairs in parallel
  pub fn reduce(&mut self) {
    let pair = std::mem::replace(&mut self.pair, vec![]);
    for (a,b) in pair {
      self.interact(a, b);
    }
  }
}
