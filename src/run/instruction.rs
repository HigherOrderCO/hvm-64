use super::*;

/// Each instruction corresponds to a fragment of a net that has a native
/// implementation.
///
/// These net fragments may have several free ports, which are each represented
/// with [`TrgId`]s.
///
/// Each `TrgId` of an instruction has an associated polarity -- it can either
/// be an input or an output. Because the underlying interaction net model we're
/// using does not have polarity, we also need instructions for linking out-out
/// or in-in.
///
/// Linking two outputs can be done with [`Instruction::Link`], which creates a
/// "cup" wire.
///
/// Linking two inputs is more complicated, due to the way locking works. It can
/// be done with [`Instruction::Wires`], which creates two "cap" wires. One half
/// of each cap can be used for each input. Once those inputs have been fully
/// unlocked, the other halves of each cap can be linked with
/// [`Instruction::Link`]. For example:
/// ```rust,ignore
/// let (av, aw, bv, bw) = net.do_wires();
/// some_subnet(net, av, bv);
/// net.link(aw, bw);
/// ```
///
/// Each instruction documents both the native implementation and the polarity
/// of each `TrgId`.
///
/// Some instructions take a [`Port`]; these must always be statically-valid
/// ports -- that is, [`Ref`], [`Int`], or [`F32`] ports.
#[derive(Debug, Clone)]
pub enum Instruction {
  /// ```rust,ignore
  /// let trg = Trg::port(port);
  /// ```
  Const { trg: TrgId, port: Port },
  /// ```rust,ignore
  /// net.link_trg(a, b);
  /// ```
  Link { a: TrgId, b: TrgId },
  /// ```rust,ignore
  /// net.link_trg(trg, Trg::port(port));
  /// ```
  LinkConst { trg: TrgId, port: Port },
  /// See [`Net::do_ctr`].
  /// ```rust,ignore
  /// let (lft, rgt) = net.do_ctr(lab, trg);
  /// ```
  Ctr { lab: Lab, trg: TrgId, lft: TrgId, rgt: TrgId },
  /// See [`Net::do_op`].
  /// ```rust,ignore
  /// let (rhs, out) = net.do_op(lab, trg);
  /// ```
  Op { op: Op, trg: TrgId, rhs: TrgId, out: TrgId },
  /// See [`Net::do_op_num`].
  /// ```rust,ignore
  /// let out = net.do_op_num(lab, trg, rhs);
  /// ```
  OpNum { op: Op, trg: TrgId, rhs: Port, out: TrgId },
  /// See [`Net::do_mat`].
  /// ```rust,ignore
  /// let (lft, rgt) = net.do_mat(trg);
  /// ```
  Mat { trg: TrgId, lft: TrgId, rgt: TrgId },
  /// See [`Net::do_wires`].
  /// ```rust,ignore
  /// let (av, aw, bv, bw) = net.do_wires();
  /// ```
  Wires { av: TrgId, aw: TrgId, bv: TrgId, bw: TrgId },
}

/// An index to a [`Trg`] in an [`Instruction`]. These essentially serve the
/// function of registers.
///
/// When compiled, each `TrgId` will be compiled to a variable.
///
/// When interpreted, the `TrgId` serves as an index into a vector.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct TrgId {
  /// Instead of storing the index directly, we store the byte offset, to save a
  /// shift instruction when indexing into the `Trg` vector in interpreted mode.
  ///
  /// This is always `index * size_of::<Trg>()`.
  pub(super) byte_offset: usize,
}

impl TrgId {
  pub fn new(index: usize) -> Self {
    TrgId { byte_offset: index * size_of::<Trg>() }
  }
  pub fn index(&self) -> usize {
    self.byte_offset / size_of::<Trg>()
  }
}

impl fmt::Display for TrgId {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    write!(f, "t{}", self.index())
  }
}

impl fmt::Debug for TrgId {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    write!(f, "TrgId({})", self.index())
  }
}

impl<'a, M: Mode> Net<'a, M> {
  /// `trg ~ {#lab x y}`
  #[inline(always)]
  pub(crate) fn do_ctr(&mut self, lab: Lab, trg: Trg) -> (Trg, Trg) {
    let port = trg.target();
    #[allow(clippy::overly_complex_bool_expr)]
    if !M::LAZY && port.tag() == Ctr && port.lab() == lab {
      trace!(self.tracer, "fast");
      self.free_trg(trg);
      let node = port.consume_node();
      self.rwts.anni += 1;
      (Trg::wire(node.p1), Trg::wire(node.p2))
    // TODO: fast copy?
    } else if false && !M::LAZY && (port.is_num() == Int || port.tag() == Ref && lab >= port.lab()) {
      self.rwts.comm += 1;
      self.free_trg(trg);
      (Trg::port(port.clone()), Trg::port(port))
    } else {
      let n = self.create_node(Ctr, lab);
      self.link_trg_port(trg, n.p0);
      (Trg::port(n.p1), Trg::port(n.p2))
    }
  }

  /// `trg ~ <op x y>`
  #[inline(always)]
  pub(crate) fn do_op(&mut self, op: Op, trg: Trg) -> (Trg, Trg) {
    trace!(self.tracer, op, trg);
    let port = trg.target();
    if !M::LAZY && (port.tag() == Int || port.tag() == F32) {
      self.free_trg(trg);
      let n = self.create_node(Op, op.swap().into());
      n.p1.wire().set_target(port);
      (Trg::port(n.p0), Trg::port(n.p2))
    } else if !M::LAZY && port == Port::ERA {
      self.free_trg(trg);
      (Trg::port(Port::ERA), Trg::port(Port::ERA))
    } else {
      let n = self.create_node(Op, op.into());
      self.link_trg_port(trg, n.p0);
      (Trg::port(n.p1), Trg::port(n.p2))
    }
  }

  /// `trg ~ <op #b x>`
  #[inline(always)]
  pub(crate) fn do_op_num(&mut self, op: Op, trg: Trg, rhs: Port) -> Trg {
    let port = trg.target();
    if !M::LAZY && port.is_num() {
      self.rwts.oper += 1;
      self.free_trg(trg);

      let res = op.op(port.num(), rhs.num());

      if op.is_int() { Trg::port(Port::new_num(Tag::Int, res)) } else { Trg::port(Port::new_num(Tag::F32, res)) }
    } else if !M::LAZY && port == Port::ERA {
      self.free_trg(trg);
      Trg::port(Port::ERA)
    } else {
      let n = self.create_node(Op, op.into());
      self.link_trg_port(trg, n.p0);
      n.p1.wire().set_target(rhs);
      Trg::port(n.p2)
    }
  }

  /// `trg ~ ?<x y>`
  #[inline(always)]
  pub(crate) fn do_mat(&mut self, trg: Trg) -> (Trg, Trg) {
    let port = trg.target();
    if !M::LAZY && port.tag() == Int {
      self.rwts.oper += 1;
      self.free_trg(trg);
      let num = port.int();
      let c1 = self.create_node(Ctr, 0);
      if num == 0 {
        self.link_port_port(c1.p2, Port::ERA);
        (Trg::port(c1.p0), Trg::wire(self.create_wire_to(c1.p1)))
      } else {
        let c2 = self.create_node(Ctr, 0);
        self.link_port_port(c1.p1, Port::ERA);
        self.link_port_port(c1.p2, c2.p0);
        self.link_port_port(c2.p1, Port::new_int(num - 1));
        (Trg::port(c1.p0), Trg::wire(self.create_wire_to(c2.p2)))
      }
    } else if !M::LAZY && port == Port::ERA {
      self.rwts.eras += 1;
      self.free_trg(trg);
      (Trg::port(Port::ERA), Trg::port(Port::ERA))
    } else {
      let m = self.create_node(Mat, 0);
      self.link_trg_port(trg, m.p0);
      (Trg::port(m.p1), Trg::port(m.p2))
    }
  }

  #[inline(always)]
  pub(crate) fn do_wires(&mut self) -> (Trg, Trg, Trg, Trg) {
    let a = self.alloc();
    let b = a.other_half();
    (Trg::port(Port::new_var(a)), Trg::wire(Wire::new(a)), Trg::port(Port::new_var(b)), Trg::wire(Wire::new(b)))
  }

  /// `trg ~ ?<(x (y z)) out>`
  #[inline(always)]
  #[allow(unused)] // TODO: emit this instruction
  pub(crate) fn do_mat_con_con(&mut self, trg: Trg, out: Trg) -> (Trg, Trg, Trg) {
    let port = trg.target();
    if !M::LAZY && trg.target().tag() == Int {
      self.rwts.oper += 1;
      self.free_trg(trg);
      let num = port.int();
      if num == 0 {
        (out, Trg::port(Port::ERA), Trg::port(Port::ERA))
      } else {
        (Trg::port(Port::ERA), Trg::port(Port::new_int(num - 1)), out)
      }
    } else if !M::LAZY && port == Port::ERA {
      self.link_trg_port(out, Port::ERA);
      (Trg::port(Port::ERA), Trg::port(Port::ERA), Trg::port(Port::ERA))
    } else {
      let m = self.create_node(Mat, 0);
      let c1 = self.create_node(Ctr, 0);
      let c2 = self.create_node(Ctr, 0);
      self.link_port_port(m.p1, c1.p0);
      self.link_port_port(c1.p2, c2.p0);
      self.link_trg_port(out, m.p2);
      (Trg::port(c1.p1), Trg::port(c2.p1), Trg::port(c2.p2))
    }
  }

  /// `trg ~ ?<(x y) out>`
  #[inline(always)]
  #[allow(unused)] // TODO: emit this instruction
  pub(crate) fn do_mat_con(&mut self, trg: Trg, out: Trg) -> (Trg, Trg) {
    let port = trg.target();
    if trg.target().tag() == Int {
      self.rwts.oper += 1;
      self.free_trg(trg);
      let num = port.int();
      if num == 0 {
        (out, Trg::port(Port::ERA))
      } else {
        let c2 = self.create_node(Ctr, 0);
        c2.p1.wire().set_target(Port::new_int(num - 1));
        self.link_trg_port(out, c2.p2);
        (Trg::port(Port::ERA), Trg::port(c2.p0))
      }
    } else if port == Port::ERA {
      self.link_trg_port(out, Port::ERA);
      (Trg::port(Port::ERA), Trg::port(Port::ERA))
    } else {
      let m = self.create_node(Mat, 0);
      let c1 = self.create_node(Ctr, 0);
      self.link_port_port(m.p1, c1.p0);
      self.link_trg_port(out, m.p2);
      self.link_trg_port(trg, m.p0);
      (Trg::port(c1.p1), Trg::port(c1.p2))
    }
  }
}
