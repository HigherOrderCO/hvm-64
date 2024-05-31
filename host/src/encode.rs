use hvm64_util::prelude::*;

use crate::Host;
use hvm64_ast::{Lab, Net as AstNet, Tree};
use hvm64_num::{Num, NumTag};
use hvm64_runtime::{Instruction, InterpretedDef, Net, Port, Trg, TrgId};
use hvm64_util::maybe_grow;

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
  pub fn encode_tree(&self, net: &mut Net, trg: Trg, tree: &Tree) {
    let mut state = State { host: self, encoder: net, scope: Default::default() };
    state.visit_tree(tree, trg);
    state.finish();
  }

  /// Encode the root of `ast_net` directly into `trg` and encode its redexes
  /// into `net` redex list.
  pub fn encode_net(&self, net: &mut Net, trg: Trg, ast_net: &AstNet) {
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
      (Tree::Ref(name), t) | (t, Tree::Ref(name)) => (Port::new_ref(&self.host.defs[name]), t),
      (Tree::Num(num), t) | (t, Tree::Num(num)) => (Port::new_num(*num), t),
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
    maybe_grow(move || match tree {
      Tree::Era => self.encoder.link_const(trg, Port::ERA),
      Tree::Num(num) => self.encoder.link_const(trg, Port::new_num(*num)),
      Tree::Ref(name) => self.encoder.link_const(trg, Port::new_ref(&self.host.defs[name])),
      Tree::Ctr { lab, p1, p2 } => {
        let (l, r) = self.encoder.ctr(*lab, trg);
        self.visit_tree(p1, l);
        self.visit_tree(p2, r);
      }
      Tree::Op { rhs, out } => {
        let (op, rhs, out) = match (&**rhs, &**out) {
          (Tree::Num(op), Tree::Op { rhs, out }) if op.tag() == NumTag::Sym => (unsafe { op.get_sym() }, rhs, out),
          _ => (NumTag::Sym, rhs, out),
        };
        if let Tree::Num(num) = **rhs {
          let o = self.encoder.op_num(op, trg, num);
          self.visit_tree(out, o);
        } else {
          let (r, o) = self.encoder.op(op, trg);
          self.visit_tree(rhs, r);
          self.visit_tree(out, o);
        }
      }
      Tree::Switch { arms, out } => {
        let (a, o) = self.encoder.switch(trg);
        self.visit_tree(arms, a);
        self.visit_tree(out, o);
      }
      Tree::Var(name) => match self.scope.entry(name) {
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
  fn op(&mut self, op: NumTag, trg: Self::Trg) -> (Self::Trg, Self::Trg);
  fn op_num(&mut self, op: NumTag, trg: Self::Trg, rhs: Num) -> Self::Trg;
  fn switch(&mut self, trg: Self::Trg) -> (Self::Trg, Self::Trg);
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
    let p1 = self.new_trg_id();
    let p2 = self.new_trg_id();
    self.instr.push(Instruction::Ctr { lab, trg, p1, p2 });
    (p1, p2)
  }
  fn op(&mut self, op: NumTag, trg: Self::Trg) -> (Self::Trg, Self::Trg) {
    let rhs = self.new_trg_id();
    let out = self.new_trg_id();
    self.instr.push(Instruction::Op { op, trg, rhs, out });
    (rhs, out)
  }
  fn op_num(&mut self, op: NumTag, trg: Self::Trg, rhs: Num) -> Self::Trg {
    let out = self.new_trg_id();
    self.instr.push(Instruction::OpNum { op, trg, rhs, out });
    out
  }
  fn switch(&mut self, trg: Self::Trg) -> (Self::Trg, Self::Trg) {
    let arms = self.new_trg_id();
    let out = self.new_trg_id();
    self.instr.push(Instruction::Switch { trg, arms, out });
    (arms, out)
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

impl<'a> Encoder for Net<'a> {
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
  fn op(&mut self, op: NumTag, trg: Self::Trg) -> (Self::Trg, Self::Trg) {
    self.do_op(op, trg)
  }
  fn op_num(&mut self, op: NumTag, trg: Self::Trg, rhs: Num) -> Self::Trg {
    self.do_op_num(op, trg, rhs)
  }
  fn switch(&mut self, trg: Self::Trg) -> (Self::Trg, Self::Trg) {
    self.do_switch(trg)
  }
  fn wires(&mut self) -> (Self::Trg, Self::Trg, Self::Trg, Self::Trg) {
    self.do_wires()
  }
}
