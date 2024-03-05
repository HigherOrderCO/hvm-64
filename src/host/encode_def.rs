use crate::util::maybe_grow;

use super::*;

/// Converts an ast net to a list of instructions to create the net.
///
/// `get_def` must return the `Port` corresponding to a given `Ref` name.
pub(crate) fn encode_def<F: FnMut(&str) -> Port>(net: &Net, get_def: F) -> InterpretedDef {
  let mut state =
    State { get_def, scope: Default::default(), instr: Default::default(), end: Default::default(), next_index: 1 };

  state.visit_tree(&net.root, TrgId::new(0));

  net.redexes.iter().for_each(|(a, b)| state.visit_redex(a, b));

  assert!(state.scope.is_empty(), "unbound variables: {:?}", state.scope.keys());

  state.instr.append(&mut state.end);

  InterpretedDef { instr: state.instr, trgs: state.next_index }
}

struct State<'a, F: FnMut(&str) -> Port> {
  get_def: F,
  scope: HashMap<&'a str, TrgId>,
  instr: Vec<Instruction>,
  end: Vec<Instruction>,
  next_index: usize,
}

impl<'a, F: FnMut(&str) -> Port> State<'a, F> {
  fn id(&mut self) -> TrgId {
    let i = self.next_index;
    self.next_index += 1;
    TrgId::new(i)
  }
  fn visit_redex(&mut self, a: &'a Tree, b: &'a Tree) {
    let (port, tree) = match (a, b) {
      (Tree::Era, t) | (t, Tree::Era) => (Port::ERA, t),
      (Tree::Ref { nam }, t) | (t, Tree::Ref { nam }) => ((self.get_def)(&nam), t),
      (Tree::Num { val }, t) | (t, Tree::Num { val }) => (Port::new_num(*val), t),
      (t, u) => {
        let av = self.id();
        let aw = self.id();
        let bv = self.id();
        let bw = self.id();
        self.instr.push(Instruction::Wires { av, aw, bv, bw });
        self.end.push(Instruction::Link { a: aw, b: bw });
        self.visit_tree(t, av);
        self.visit_tree(u, bv);
        return;
      }
    };
    let trg = self.id();
    self.instr.push(Instruction::Const { port, trg });
    self.visit_tree(tree, trg);
  }
  fn visit_tree(&mut self, tree: &'a Tree, trg: TrgId) {
    maybe_grow(move || match tree {
      Tree::Era => {
        self.instr.push(Instruction::LinkConst { trg, port: Port::ERA });
      }
      Tree::Ref { nam } => {
        self.instr.push(Instruction::LinkConst { trg, port: (self.get_def)(&nam) });
      }
      Tree::Num { val } => {
        self.instr.push(Instruction::LinkConst { trg, port: Port::new_num(*val) });
      }
      Tree::Var { nam } => match self.scope.entry(nam) {
        Entry::Occupied(e) => {
          let other = e.remove();
          self.instr.push(Instruction::Link { a: other, b: trg });
        }
<<<<<<< HEAD
        Entry::Vacant(e) => {
          e.insert(trg);
=======
      };
      let trg = self.id();
      self.instr.push(Instruction::Const { port, trg });
      self.visit_tree(tree, trg);
    }
    fn visit_tree(&mut self, tree: &'a Tree, trg: TrgId) {
      maybe_grow(move || match tree {
        Tree::Era => {
          self.instr.push(Instruction::LinkConst { trg, port: Port::ERA });
>>>>>>> cc29e4c ([sc-484] Optimize pre-reduce pass)
        }
      },
      Tree::Ctr { lab, lft, rgt } => {
        let l = self.id();
        let r = self.id();
        self.instr.push(Instruction::Ctr { lab: *lab, trg, lft: l, rgt: r });
        self.visit_tree(lft, l);
        self.visit_tree(rgt, r);
      }
      Tree::Op { op, rhs, out } => {
        if let Tree::Num { val } = &**rhs {
          let o = self.id();
          self.instr.push(Instruction::OpNum { op: *op, rhs: *val, trg, out: o });
          self.visit_tree(out, o);
        } else {
          let r = self.id();
          let o = self.id();
          self.instr.push(Instruction::Op { op: *op, trg, rhs: r, out: o });
          self.visit_tree(rhs, r);
          self.visit_tree(out, o);
        }
<<<<<<< HEAD
      }
      Tree::Mat { sel, ret } => {
        let l = self.id();
        let r = self.id();
        self.instr.push(Instruction::Mat { trg, lft: l, rgt: r });
        self.visit_tree(sel, l);
        self.visit_tree(ret, r);
      }
    })
=======
        Tree::Op { op, rhs, out } => {
          if let Tree::Num { val } = &**rhs {
            let o = self.id();
            self.instr.push(Instruction::OpNum { op: *op, rhs: *val, trg, out: o });
            self.visit_tree(out, o);
          } else {
            let r = self.id();
            let o = self.id();
            self.instr.push(Instruction::Op { op: *op, trg, rhs: r, out: o });
            self.visit_tree(rhs, r);
            self.visit_tree(out, o);
          }
        }
        Tree::Mat { sel, ret } => {
          let l = self.id();
          let r = self.id();
          self.instr.push(Instruction::Mat { trg, lft: l, rgt: r });
          self.visit_tree(sel, l);
          self.visit_tree(ret, r);
        }
      })
    }
>>>>>>> cc29e4c ([sc-484] Optimize pre-reduce pass)
  }
}
