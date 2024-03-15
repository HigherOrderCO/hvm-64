use crate::ast::{Net, Tree};

use super::{create_var, var_to_num};

impl Net {
  /// Transforms the net `x & ...` into `y & x ~ (arg y) & ...`
  ///
  /// The result is equivalent a λ-calculus application. Thus,
  /// if the net is a λ-calculus term, then this function will
  /// apply an argument to it.
  pub fn apply_tree(&mut self, arg: Tree) {
    let mut fresh = 0usize;
    self.ensure_no_conflicts(&mut fresh);
    arg.ensure_no_conflicts(&mut fresh);

    let fresh_str = create_var(fresh + 1);

    let fun = core::mem::take(&mut self.root);
    let app = Tree::Ctr { lab: 0, ports: vec![arg, Tree::Var { nam: fresh_str.clone() }] };
    self.root = Tree::Var { nam: fresh_str };
    self.redexes.push((fun, app));
  }

  pub(crate) fn ensure_no_conflicts(&self, fresh: &mut usize) {
    self.root.ensure_no_conflicts(fresh);
    for (a, b) in &self.redexes {
      a.ensure_no_conflicts(fresh);
      b.ensure_no_conflicts(fresh);
    }
  }
}

impl Tree {
  /// Increases `fresh` until `create_var(*fresh)` does not conflict
  /// with a [`Tree::Var`]  in `tree`
  ///
  /// This function can be called multiple times with many trees to
  /// ensure that `fresh` does not conflict with any of them.
  pub(crate) fn ensure_no_conflicts(&self, fresh: &mut usize) {
    if let Tree::Var { nam } = self {
      if let Some(var_num) = var_to_num(nam) {
        *fresh = (*fresh).max(var_num);
      }
    }
    self.children().for_each(|child| child.ensure_no_conflicts(fresh));
  }
}
