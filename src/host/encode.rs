//! Code for directly encoding a [`hvmc::ast::Net`] or a [`hvmc::ast::Tree`]
//! into a [`hvmc::run::Net`]

use std::collections::{hash_map::Entry, HashMap};

use crate::{
  ast::{Net, Tree},
  host::Host,
  run::{self, Mode, Port, Trg},
};

impl Host {
  /// Encode `tree` directly into `trg`, skipping the intermediate `Def`
  /// representation.
  pub fn encode_tree<M: Mode>(&self, net: &mut run::Net<M>, trg: Trg, tree: &Tree) {
    EncodeState { host: self, net, vars: Default::default() }.encode(trg, tree);
  }
  /// Encode the root of `ast_net` directly into `trg` and encode its redexes
  /// into `net` redex list.
  pub fn encode_net<M: Mode>(&self, net: &mut run::Net<M>, trg: Trg, ast_net: &Net) {
    let mut state = EncodeState { host: self, net, vars: Default::default() };
    for (l, r) in &ast_net.rdex {
      let (ap, a, bp, b) = state.net.do_wires();
      state.encode(ap, l);
      state.encode(bp, r);
      state.net.link_trg(a, b);
    }
    state.encode(trg, &ast_net.root);
  }
}

struct EncodeState<'c, 'n, M: Mode> {
  host: &'c Host,
  net: &'c mut run::Net<'n, M>,
  vars: HashMap<&'c str, Trg>,
}

impl<'c, 'n, M: Mode> EncodeState<'c, 'n, M> {
  fn encode(&mut self, trg: Trg, tree: &'c Tree) {
    match tree {
      Tree::Era => self.net.link_trg_port(trg, Port::ERA),
      Tree::Num { val } => self.net.link_trg_port(trg, Port::new_num(*val)),
      Tree::Ref { nam } => self.net.link_trg_port(trg, Port::new_ref(&self.host.defs[nam])),
      Tree::Ctr { lab, lft, rgt } => {
        let (l, r) = self.net.do_ctr(*lab, trg);
        self.encode(l, lft);
        self.encode(r, rgt);
      }
      Tree::Op2 { opr, lft, rgt } => {
        let (l, r) = self.net.do_op2(*opr, trg);
        self.encode(l, lft);
        self.encode(r, rgt);
      }
      Tree::Op1 { opr, lft, rgt } => {
        let r = self.net.do_op1(*opr, *lft, trg);
        self.encode(r, rgt);
      }
      Tree::Mat { sel, ret } => {
        let (s, r) = self.net.do_mat(trg);
        self.encode(s, sel);
        self.encode(r, ret);
      }
      Tree::Var { nam } => match self.vars.entry(nam) {
        Entry::Occupied(e) => self.net.link_trg(e.remove(), trg),
        Entry::Vacant(e) => {
          e.insert(trg);
        }
      },
    }
  }
}
