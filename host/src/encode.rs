use crate::prelude::*;

use crate::Host;
use hvmc_ast::{Lab, Net as AstNet, Tree};
use hvmc_runtime::{Instruction, InterpretedDef, Mode, Net, Port, Trg, TrgId};
use hvmc_util::{maybe_grow, ops::TypedOp as Op};

impl Host {
  /// Converts an ast net to a list of instructions to create the net.
  ///
  /// `get_def` must return the `Port` corresponding to a given `Ref` name.
  pub fn encode_def(&self, net: &AstNet) -> InterpretedDef {
    let mut def = InterpretedDef { instr: Vec::new(), trgs: 1 };
    let mut state = State { host: self, encoder: &mut def, scope: Default::default() };
    state.visit_net(net, TrgId::new(0));
    state.finish();
    def
  }

  /// Encode `tree` directly into `trg`, skipping the intermediate `Def`
  /// representation.
  pub fn encode_tree<M: Mode>(&self, net: &mut Net<M>, trg: Trg, tree: &Tree) {
    let mut state = State { host: self, encoder: net, scope: Default::default() };
    state.visit_tree(tree, trg);
    state.finish();
  }

  /// Encode the root of `ast_net` directly into `trg` and encode its redexes
  /// into `net` redex list.
  pub fn encode_net<M: Mode>(&self, net: &mut Net<M>, trg: Trg, ast_net: &AstNet) {
    let mut state = State { host: self, encoder: net, scope: Default::default() };
    state.visit_net(ast_net, trg);
    state.finish();
  }
}

struct State<'a, E: Encoder> {
  host: &'a Host,
  encoder: &'a mut E,
  scope: Map<&'a str, E::Trg>,
}

impl<'a, E: Encoder> State<'a, E> {
  fn finish(self) {
    assert!(self.scope.is_empty(), "unbound variables: {:?}", self.scope.keys());
  }
  fn visit_net(&mut self, net: &'a AstNet, trg: E::Trg) {
    self.visit_tree(&net.root, trg);
    net.redexes.iter().for_each(|(a, b)| self.visit_redex(a, b));
  }
  fn visit_redex(&mut self, a: &'a Tree, b: &'a Tree) {
    let (port, tree) = match (a, b) {
      (Tree::Era, t) | (t, Tree::Era) => (Port::ERA, t),
      (Tree::Ref { nam }, t) | (t, Tree::Ref { nam }) => (Port::new_ref(&self.host.defs[nam]), t),
      (Tree::Int { val }, t) | (t, Tree::Int { val }) => (Port::new_int(*val), t),
      (t, u) => {
        let (av, aw, bv, bw) = self.encoder.wires();
        self.visit_tree(t, av);
        self.visit_tree(u, bv);
        self.encoder.link(aw, bw);
        return;
      }
    };
    let trg = self.encoder.make_const(port);
    self.visit_tree(tree, trg);
  }
  fn visit_tree(&mut self, tree: &'a Tree, trg: E::Trg) {
    static ERA: Tree = Tree::Era;
    maybe_grow(move || match tree {
      Tree::Era => self.encoder.link_const(trg, Port::ERA),
      Tree::Int { val } => self.encoder.link_const(trg, Port::new_int(*val)),
      Tree::F32 { val } => self.encoder.link_const(trg, Port::new_float(val.0)),
      Tree::Ref { nam } => self.encoder.link_const(trg, Port::new_ref(&self.host.defs[nam])),
      Tree::Ctr { lab, ports } => {
        if ports.is_empty() {
          return self.visit_tree(&ERA, trg);
        }
        let mut trg = trg;
        for port in &ports[0 .. ports.len() - 1] {
          let (l, r) = self.encoder.ctr(*lab, trg);
          self.visit_tree(port, l);
          trg = r;
        }
        self.visit_tree(ports.last().unwrap(), trg);
      }
      Tree::Adt { lab, variant_index, variant_count, fields } => {
        let mut trg = trg;
        for _ in 0 .. *variant_index {
          let (l, r) = self.encoder.ctr(*lab, trg);
          self.visit_tree(&ERA, l);
          trg = r;
        }
        let (mut l, mut r) = self.encoder.ctr(*lab, trg);
        for field in fields {
          let (x, y) = self.encoder.ctr(*lab, l);
          self.visit_tree(field, x);
          l = y;
        }
        for _ in 0 .. (*variant_count - *variant_index - 1) {
          let (x, y) = self.encoder.ctr(*lab, r);
          self.visit_tree(&ERA, x);
          r = y;
        }
        self.encoder.link(l, r);
      }
      Tree::Op { op, rhs: lft, out: rgt } => match &**lft {
        Tree::Int { val } => {
          let o = self.encoder.op_num(*op, trg, Port::new_int(*val));
          self.visit_tree(rgt, o);
        }
        Tree::F32 { val } => {
          let o = self.encoder.op_num(*op, trg, Port::new_float(val.0));
          self.visit_tree(rgt, o);
        }
        _ => {
          let (l, r) = self.encoder.op(*op, trg);
          self.visit_tree(lft, l);
          self.visit_tree(rgt, r);
        }
      },
      Tree::Mat { zero, succ, out } => {
        let (a, o) = self.encoder.mat(trg);
        let (z, s) = self.encoder.ctr(0, a);
        self.visit_tree(zero, z);
        self.visit_tree(succ, s);
        self.visit_tree(out, o);
      }
      Tree::Var { nam } => match self.scope.entry(nam) {
        Entry::Occupied(e) => self.encoder.link(e.remove(), trg),
        Entry::Vacant(e) => {
          e.insert(trg);
        }
      },
    })
  }
}

trait Encoder {
  type Trg;
  fn link_const(&mut self, trg: Self::Trg, port: Port);
  fn link(&mut self, a: Self::Trg, b: Self::Trg);
  fn make_const(&mut self, port: Port) -> Self::Trg;
  fn ctr(&mut self, lab: Lab, trg: Self::Trg) -> (Self::Trg, Self::Trg);
  fn op(&mut self, op: Op, trg: Self::Trg) -> (Self::Trg, Self::Trg);
  fn op_num(&mut self, op: Op, trg: Self::Trg, rhs: Port) -> Self::Trg;
  fn mat(&mut self, trg: Self::Trg) -> (Self::Trg, Self::Trg);
  fn wires(&mut self) -> (Self::Trg, Self::Trg, Self::Trg, Self::Trg);
}

impl Encoder for InterpretedDef {
  type Trg = TrgId;
  fn link_const(&mut self, trg: Self::Trg, port: Port) {
    self.instr.push(Instruction::LinkConst { trg, port });
  }
  fn link(&mut self, a: Self::Trg, b: Self::Trg) {
    self.instr.push(Instruction::Link { a, b });
  }
  fn make_const(&mut self, port: Port) -> Self::Trg {
    let trg = self.new_trg_id();
    self.instr.push(Instruction::Const { trg, port });
    trg
  }
  fn ctr(&mut self, lab: Lab, trg: Self::Trg) -> (Self::Trg, Self::Trg) {
    let lft = self.new_trg_id();
    let rgt = self.new_trg_id();
    self.instr.push(Instruction::Ctr { lab, trg, lft, rgt });
    (lft, rgt)
  }
  fn op(&mut self, op: Op, trg: Self::Trg) -> (Self::Trg, Self::Trg) {
    let rhs = self.new_trg_id();
    let out = self.new_trg_id();
    self.instr.push(Instruction::Op { op, trg, rhs, out });
    (rhs, out)
  }
  fn op_num(&mut self, op: Op, trg: Self::Trg, rhs: Port) -> Self::Trg {
    let out = self.new_trg_id();
    self.instr.push(Instruction::OpNum { op, trg, rhs, out });
    out
  }
  fn mat(&mut self, trg: Self::Trg) -> (Self::Trg, Self::Trg) {
    let lft = self.new_trg_id();
    let rgt = self.new_trg_id();
    self.instr.push(Instruction::Mat { trg, lft, rgt });
    (lft, rgt)
  }
  fn wires(&mut self) -> (Self::Trg, Self::Trg, Self::Trg, Self::Trg) {
    let av = self.new_trg_id();
    let aw = self.new_trg_id();
    let bv = self.new_trg_id();
    let bw = self.new_trg_id();
    self.instr.push(Instruction::Wires { av, aw, bv, bw });
    (av, aw, bv, bw)
  }
}

impl<'a, M: Mode> Encoder for Net<'a, M> {
  type Trg = Trg;

  fn link_const(&mut self, trg: Self::Trg, port: Port) {
    self.link_trg_port(trg, port)
  }
  fn link(&mut self, a: Self::Trg, b: Self::Trg) {
    self.link_trg(a, b)
  }
  fn make_const(&mut self, port: Port) -> Self::Trg {
    Trg::port(port)
  }
  fn ctr(&mut self, lab: Lab, trg: Self::Trg) -> (Self::Trg, Self::Trg) {
    self.do_ctr(lab, trg)
  }
  fn op(&mut self, op: Op, trg: Self::Trg) -> (Self::Trg, Self::Trg) {
    self.do_op(op, trg)
  }
  fn op_num(&mut self, op: Op, trg: Self::Trg, rhs: Port) -> Self::Trg {
    self.do_op_num(op, trg, rhs)
  }
  fn mat(&mut self, trg: Self::Trg) -> (Self::Trg, Self::Trg) {
    self.do_mat(trg)
  }
  fn wires(&mut self) -> (Self::Trg, Self::Trg, Self::Trg, Self::Trg) {
    self.do_wires()
  }
}
