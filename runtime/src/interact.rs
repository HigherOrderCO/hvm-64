use super::*;

impl<'a> Net<'a> {
  /// Performs an interaction between two connected principal ports.
  #[inline(always)]
  pub fn interact(&mut self, a: Port, b: Port) {
    self.tracer.sync();
    trace!(self.tracer, a, b);
    match (a.tag(), b.tag()) {
      // not actually an active pair
      (Var | Red, _) | (_, Var | Red) => unreachable!(),
      // nil-nil
      (Ref, Ref | Num) if !a.is_skippable() => self.call(a, b),
      (Ref | Num, Ref) if !b.is_skippable() => self.call(b, a),
      (Num | Ref, Num | Ref) => self.rwts.eras += 1,
      // comm 2/2
      (Ctr, Switch) if a.lab() != 0 => self.comm22(a, b),
      (Switch, Ctr) if b.lab() != 0 => self.comm22(a, b),
      (Ctr, Op) | (Op, Ctr) => self.comm22(a, b),
      (Ctr, Ctr) if a.lab() != b.lab() => self.comm22(a, b),
      // anni
      (Switch, Switch) | (Op, Op) | (Ctr, Ctr) => self.anni2(a, b),
      // comm 2/0
      (Ref, Ctr) if b.lab() >= a.lab() => self.comm02(a, b),
      (Ctr, Ref) if a.lab() >= b.lab() => self.comm02(b, a),
      (Num, Ctr) => self.comm02(a, b),
      (Ctr, Num) => self.comm02(b, a),
      (Ref, _) if a == Port::ERA => self.comm02(a, b),
      (_, Ref) if b == Port::ERA => self.comm02(b, a),
      // deref
      (Ref, _) => self.call(a, b),
      (_, Ref) => self.call(b, a),
      // native ops
      (Op, Num) => self.op_num(a, b),
      (Num, Op) => self.op_num(b, a),
      (Switch, Num) => self.switch_num(a, b),
      (Num, Switch) => self.switch_num(b, a),
      // todo: what should the semantics of these be?
      (Switch, Ctr) // b.lab() == 0
      | (Ctr, Switch) // a.lab() == 0
      | (Op, Switch)
      | (Switch, Op) => unimplemented!("{:?}-{:?}", a.tag(), b.tag()),
    }
  }

  /// Annihilates two binary agents.
  ///
  /// ```text
  ///         a2 |   | a1
  ///           _|___|_
  ///           \     /
  ///         a  \   /
  ///             \ /
  ///              |
  ///             / \
  ///         b  /   \
  ///           /_____\
  ///            |   |
  ///         b1 |   | b2
  ///
  /// --------------------------- anni2
  ///
  ///         a2 |   | a1
  ///            |   |
  ///             \ /
  ///              X
  ///             / \
  ///            |   |
  ///         b1 |   | b2
  /// ```
  #[inline(never)]
  pub fn anni2(&mut self, a: Port, b: Port) {
    trace!(self.tracer, a, b);
    self.rwts.anni += 1;
    let a = a.consume_node();
    let b = b.consume_node();
    self.link_wire_wire(a.p1, b.p1);
    self.link_wire_wire(a.p2, b.p2);
  }

  /// Commutes two binary agents.
  ///
  /// ```text
  ///         a2 |   | a1
  ///           _|___|_
  ///           \     /
  ///         a  \   /
  ///             \ /
  ///              |
  ///             /#\
  ///         b  /###\
  ///           /#####\
  ///            |   |
  ///         b1 |   | b2
  ///
  /// --------------------------- comm22
  ///
  ///     a2 |         | a1
  ///        |         |
  ///       /#\       /#\
  ///  B2  /###\     /###\  B1
  ///     /#####\   /#####\
  ///      |   \     /   |
  ///   p1 | p2 \   / p1 | p2
  ///      |     \ /     |
  ///      |      X      |
  ///      |     / \     |
  ///   p2 | p1 /   \ p2 | p1
  ///     _|___/_   _\___|_
  ///     \     /   \     /
  ///  A1  \   /     \   /  A2
  ///       \ /       \ /
  ///        |         |
  ///     b1 |         | b2
  /// ```
  #[allow(non_snake_case)]
  #[inline(never)]
  pub fn comm22(&mut self, a: Port, b: Port) {
    trace!(self.tracer, a, b);
    self.rwts.comm += 1;

    let a = a.consume_node();
    let b = b.consume_node();

    let A1 = self.create_node(a.tag, a.lab);
    let A2 = self.create_node(a.tag, a.lab);
    let B1 = self.create_node(b.tag, b.lab);
    let B2 = self.create_node(b.tag, b.lab);

    trace!(self.tracer, A1.p0, A2.p0, B1.p0, B2.p0);
    self.link_port_port(A1.p1, B1.p1);
    self.link_port_port(A1.p2, B2.p1);
    self.link_port_port(A2.p1, B1.p2);
    self.link_port_port(A2.p2, B2.p2);

    trace!(self.tracer);
    self.link_wire_port(a.p1, B1.p0);
    self.link_wire_port(a.p2, B2.p0);
    self.link_wire_port(b.p1, A1.p0);
    self.link_wire_port(b.p2, A2.p0);
  }

  /// Commutes a nilary agent and a binary agent.
  ///
  /// ```text
  ///         a  (---)
  ///              |
  ///              |
  ///             /#\
  ///         b  /###\
  ///           /#####\
  ///            |   |
  ///         b1 |   | b2
  ///
  /// --------------------------- comm02
  ///
  ///     a (---)   (---) a
  ///         |       |
  ///      b1 |       | b2
  /// ```
  #[inline(never)]
  pub fn comm02(&mut self, a: Port, b: Port) {
    trace!(self.tracer, a, b);
    self.rwts.comm += 1;
    let b = b.consume_node();
    self.link_wire_port(b.p1, a.clone());
    self.link_wire_port(b.p2, a);
  }

  /// Interacts a number and a numeric switch node.
  ///
  /// ```text
  ///                             |
  ///         b   (0)             |         b  (n+1)
  ///              |              |              |
  ///              |              |              |
  ///             / \             |             / \
  ///         a  /swi\            |         a  /swi\
  ///           /_____\           |           /_____\
  ///            |   |            |            |   |
  ///         a1 |   | a2         |         a1 |   | a2
  ///                             |
  /// --------------------------- | --------------------------- switch_int
  ///                             |          _ _ _ _ _
  ///                             |        /           \
  ///                             |    y2 |  (n) y1     |
  ///                             |      _|___|_        |
  ///                             |      \     /        |
  ///               _             |    y  \   /         |
  ///             /   \           |        \ /          |
  ///    x2 (*)  | x1  |          |      x2 |  (*) x1   |
  ///       _|___|_    |          |        _|___|_      |
  ///       \     /    |          |        \     /      |
  ///     x  \   /     |          |      x  \   /       |
  ///         \ /      |          |          \ /        |
  ///          |       |          |           |         |
  ///       a1 |       | a2       |        a1 |         | a2
  ///                             |
  /// ```
  #[inline(never)]
  pub fn switch_num(&mut self, a: Port, b: Port) {
    trace!(self.tracer, a, b);
    self.rwts.oper += 1;
    let a = a.consume_node();
    let num = b.num();
    let num = if num.tag() == NumTag::U24 { num.get_u24() } else { 0 };
    if num == 0 {
      let x = self.create_node(Ctr, 0);
      trace!(self.tracer, x.p0);
      self.link_port_port(x.p2, Port::ERA);
      self.link_wire_port(a.p2, x.p1);
      self.link_wire_port(a.p1, x.p0);
    } else {
      let x = self.create_node(Ctr, 0);
      let y = self.create_node(Ctr, 0);
      trace!(self.tracer, x.p0, y.p0);
      self.link_port_port(x.p1, Port::ERA);
      self.link_port_port(x.p2, y.p0);
      self.link_port_port(y.p1, Port::new_num(Num::new_u24(num - 1)));
      self.link_wire_port(a.p2, y.p2);
      self.link_wire_port(a.p1, x.p0);
    }
  }

  /// Interacts a number and a binary numeric operation node.
  ///
  /// ```text
  ///                             |
  ///         b   (n)             |         b   (n)
  ///              |              |              |
  ///              |              |              |
  ///             / \             |             / \
  ///         a  /op \            |         a  /op \
  ///           /_____\           |           /_____\
  ///            |   |            |            |   |
  ///           (m)  | a2         |         a1 |   | a2
  ///                             |
  /// --------------------------- | --------------------------- op_num
  ///                             |           _ _ _
  ///                             |         /       \
  ///                             |        |  (n)    |
  ///                             |       _|___|_    |
  ///                             |       \     /    |
  ///                             |     x  \op$/     |
  ///            (n op m)         |         \ /      |
  ///                |            |          |       |
  ///                | a2         |       a1 |       | a2
  ///                             |
  /// ```
  #[inline(never)]
  pub fn op_num(&mut self, a: Port, b: Port) {
    trace!(self.tracer, a, b);
    let op = a.op();
    let a = a.consume_node();
    let a1 = a.p1.load_target();
    if a1.is_num() {
      self.rwts.oper += 1;
      self.half_free(a.p1.addr());

      let out = Num::operate_binary(b.num(), op, a1.num());

      self.link_wire_port(a.p2, Port::new_num(out));
    } else {
      let x = self.create_node(Op, NumTag::Sym as u16);
      trace!(self.tracer, x.p0);
      self.link_port_port(x.p1, Port::new_num(Num::operate_unary(op, b.num())));
      self.link_wire_port(a.p2, x.p2);
      self.link_wire_port(a.p1, x.p0);
    }
  }
}
