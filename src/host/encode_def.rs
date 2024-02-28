use super::*;

/// Converts an ast net to a list of instructions to create the net.
///
/// `defs` must be populated with every `Ref` node that may appear in the net.
pub(super) fn ast_net_to_instructions(defs: &HashMap<String, DefRef>, net: &Net) -> Vec<Instruction> {
  let mut state =
    State { defs, scope: Default::default(), instr: Default::default(), end: Default::default(), next_index: 1 };

  state.visit_tree(&net.root, TrgId::new(0));

  net.redexes.iter().for_each(|(a, b)| state.visit_redex(a, b));

  assert!(state.scope.is_empty(), "unbound variables: {:?}", state.scope.keys());

  state.instr.append(&mut state.end);

  return state.instr;

  struct State<'a> {
    defs: &'a HashMap<String, DefRef>,
    scope: HashMap<&'a str, TrgId>,
    instr: Vec<Instruction>,
    end: Vec<Instruction>,
    next_index: usize,
  }

  impl<'a> State<'a> {
    fn id(&mut self) -> TrgId {
      let i = self.next_index;
      self.next_index += 1;
      TrgId::new(i)
    }
    fn visit_redex(&mut self, a: &'a Tree, b: &'a Tree) {
      let (port, tree) = match (a, b) {
        (Tree::Era, t) | (t, Tree::Era) => (Port::ERA, t),
        (Tree::Ref { nam }, t) | (t, Tree::Ref { nam }) => (Port::new_ref(&self.defs[nam]), t),
        (Tree::Num { val }, t) | (t, Tree::Num { val }) => (Port::new_num(*val), t),
        (t, u) => {
          let av = self.id();
          let aw = self.id();
          let bv = self.id();
          let bw = self.id();
          self.next_index += 4;
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
      match tree {
        Tree::Era => {
          self.instr.push(Instruction::LinkConst { trg, port: Port::ERA });
        }
        Tree::Ref { nam } => {
          self.instr.push(Instruction::LinkConst { trg, port: Port::new_ref(&self.defs[nam]) });
        }
        Tree::Num { val } => {
          self.instr.push(Instruction::LinkConst { trg, port: Port::new_num(*val) });
        }
        Tree::Var { nam } => match self.scope.entry(nam) {
          Entry::Occupied(e) => {
            let other = e.remove();
            self.instr.push(Instruction::Link { a: other, b: trg });
          }
          Entry::Vacant(e) => {
            e.insert(trg);
          }
        },
        Tree::Ctr { lab, lft, rgt } => {
          let l = self.id();
          let r = self.id();
          self.instr.push(Instruction::Ctr { lab: *lab, trg, lft: l, rgt: r });
          self.visit_tree(lft, l);
          self.visit_tree(rgt, r);
        }
        Tree::Op2 { opr, lft, rgt } => {
          let l = self.id();
          let r = self.id();
          self.instr.push(Instruction::Op2 { op: *opr, trg, lft: l, rgt: r });
          self.visit_tree(lft, l);
          self.visit_tree(rgt, r);
        }
        Tree::Op1 { opr, lft, rgt } => {
          let r = self.id();
          self.instr.push(Instruction::Op1 { op: *opr, num: *lft, trg, rgt: r });
          self.visit_tree(rgt, r);
        }
        Tree::Mat { sel, ret } => {
          let l = self.id();
          let r = self.id();
          self.instr.push(Instruction::Mat { trg, lft: l, rgt: r });
          self.visit_tree(sel, l);
          self.visit_tree(ret, r);
        }
      }
    }
  }
}
