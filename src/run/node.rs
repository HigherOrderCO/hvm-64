use super::*;

/// See [`Port::traverse_node`].
pub struct TraverseNode {
  pub tag: Tag,
  pub lab: Lab,
  pub p1: Wire,
  pub p2: Wire,
}

/// See [`Port::traverse_op1`].
pub struct TraverseOp1 {
  pub op: Op,
  pub num: Port,
  pub p2: Wire,
}

impl Port {
  #[inline(always)]
  pub fn consume_node(self) -> TraverseNode {
    self.traverse_node()
  }

  #[inline(always)]
  pub fn traverse_node(self) -> TraverseNode {
    TraverseNode {
      tag: self.tag(),
      lab: self.lab(),
      p1: Wire::new(self.addr()),
      p2: Wire::new(self.addr().other_half()),
    }
  }

  #[inline(always)]
  pub fn consume_op1(self) -> TraverseOp1 {
    let op = self.op();
    let s = self.consume_node();
    let num = s.p1.swap_target(Port::FREE);
    TraverseOp1 { op, num, p2: s.p2 }
  }

  #[inline(always)]
  pub fn traverse_op1(self) -> TraverseOp1 {
    let op = self.op();
    let s = self.traverse_node();
    let num = s.p1.load_target();
    TraverseOp1 { op, num, p2: s.p2 }
  }
}

pub struct CreatedNode {
  pub p0: Port,
  pub p1: Port,
  pub p2: Port,
}

impl<'a, M: Mode> Net<'a, M> {
  #[inline(always)]
  pub fn create_node(&mut self, tag: Tag, lab: Lab) -> CreatedNode {
    let addr = self.alloc();
    CreatedNode {
      p0: Port::new(tag, lab, addr.clone()),
      p1: Port::new_var(addr.clone()),
      p2: Port::new_var(addr.other_half()),
    }
  }

  /// Creates a wire an aux port pair.
  #[inline(always)]
  pub fn create_wire(&mut self) -> (Wire, Port) {
    let addr = self.alloc();
    self.half_free(addr.other_half());
    (Wire::new(addr.clone()), Port::new_var(addr))
  }

  /// Creates a wire pointing to a given port; sometimes necessary to avoid
  /// deadlock.
  #[inline(always)]
  pub fn create_wire_to(&mut self, port: Port) -> Wire {
    let addr = self.alloc();
    self.half_free(addr.other_half());
    let wire = Wire::new(addr);
    self.link_port_port(port, Port::new_var(wire.addr()));
    wire
  }
}
