//! Reduce the solving any annihilations and commutations.

use std::{
  collections::HashMap,
  sync::{Arc, Mutex},
};

use crate::{
  ast::{Book, Net, Tree},
  host::{ast_net_to_instructions, DefRef, Host},
  run::{self, Def, Heap, InterpretedDef, LabSet, Rewrites},
  util::maybe_grow,
};

impl Book {
  /// Reduces the definitions in the book individually, except for the skipped
  /// ones.
  ///
  /// Defs that are not in the book are treated as inert defs.
  ///
  /// `max_memory` is measured in words.
  pub fn pre_reduce(&mut self, skip: &dyn Fn(&str) -> bool, max_memory: usize, max_rwts: u64) -> PreReduceStats {
    let mut host = Host::default();
    let captured_redexes = Arc::new(Mutex::new(Vec::new()));
    // When a ref is not found in the `Host`, then
    // put an inert def in its place
    host.insert_book_with_default(self, &mut |_| {
      DefRef::Owned(Box::new(Def::new(LabSet::ALL, InertDef(captured_redexes.clone()))))
    });
    let area = run::Heap::new_words(max_memory);
    let (seen, rewrites) = {
      let mut state = State {
        book: self,
        skip,
        captured_redexes,
        max_rwts,
        host,
        area: &area,
        seen: HashMap::new(),
        total_rewrites: Rewrites::default(),
      };
      for nam in self.nets.keys() {
        state.pre_reduce(&nam)
      }
      (
        std::mem::take(&mut state.seen).into_iter().map(|(k, v)| (k.to_owned(), v)).collect::<Vec<_>>(),
        state.total_rewrites,
      )
    };

    for (nam, state) in seen.into_iter() {
      if let SeenState::Reduced(net) = state {
        self.nets.insert(nam.to_owned(), net);
      }
    }

    PreReduceStats { rewrites, errors: vec![] }
  }
}

pub struct PreReduceStats {
  pub rewrites: Rewrites,
  pub errors: Vec<String>,
}

enum SeenState {
  Cycled,
  Reduced(Net),
}

/// A Def that pushes all interactions to its inner Vec.
#[derive(Default)]
struct InertDef(Arc<Mutex<Vec<(run::Port, run::Port)>>>);

impl run::AsDef for InertDef {
  unsafe fn call<M: run::Mode>(def: *const run::Def<Self>, _: &mut run::Net<M>, port: run::Port) {
    let def = unsafe { &*def };
    def.data.0.lock().unwrap().push((run::Port::new_ref(def), port));
  }
}

/// State of the pre-reduction algorithm.
///
/// Here's how it works:
/// - Each definition is visited in topological order (dependencies before
///   dependents). In the case of cycles, one will be arbitrarily selected to be
///   first.
/// - The definition is reduced in a [`run::Net`]
/// - The reduced [`run::Net`] is readback into an [`ast::Net`]
/// - The [`ast::Net`] is encoded into a [`Vec<Instruction>`]
/// - The [`ast::Net`] is stored in the [`State`], as it will be used later.
/// - The [`InterpretedDef`] corresponding to the definition is mutated in-place
///   and its instructions are replaced with the generated [`Vec<Instruction>`]
///
/// At the end, each mutated [`ast::Net`] is placed into the [`Book`],
/// overriding the previous one.
struct State<'a> {
  book: &'a Book,

  host: Host,
  max_rwts: u64,

  area: &'a Heap,
  captured_redexes: Arc<Mutex<Vec<(run::Port, run::Port)>>>,

  skip: &'a dyn Fn(&str) -> bool,
  seen: HashMap<&'a str, SeenState>,

  total_rewrites: Rewrites<u64>,
}

impl<'a> State<'a> {
  fn visit_tree(&mut self, tree: &'a Tree) {
    maybe_grow(move || match tree {
      Tree::Era | Tree::Num { .. } | Tree::Var { .. } => (),
      Tree::Ref { nam } => {
        self.pre_reduce(nam);
      }
      Tree::Ctr { lft, rgt, .. } | Tree::Op2 { lft, rgt, .. } | Tree::Mat { sel: lft, ret: rgt } => {
        self.visit_tree(lft);
        self.visit_tree(rgt);
      }
      Tree::Op1 { rgt, .. } => self.visit_tree(rgt),
    })
  }
  fn visit_net(&mut self, net: &'a Net) {
    self.visit_tree(&net.root);
    for (a, b) in &net.redexes {
      self.visit_tree(a);
      self.visit_tree(b);
    }
  }
  fn pre_reduce(&mut self, nam: &'a str) {
    if self.seen.contains_key(nam) || (self.skip)(nam) || self.book.get(nam).is_none() {
      return;
    }
    self.seen.insert(nam, SeenState::Cycled);
    // First, pre-reduce all nets referenced by this net by walking the tree
    self.visit_net(self.book.get(nam).unwrap());

    let mut rt = run::Net::<run::Strict>::new(self.area);
    rt.boot(self.host.defs.get(nam).expect("No function."));
    rt.expand();
    rt.reduce(self.max_rwts as usize);

    self.total_rewrites += rt.rwts;

    // Move interactions with inert defs back into the net redexes array
    rt.redexes.extend(core::mem::take::<Vec<_>>(self.captured_redexes.lock().unwrap().as_mut()).into_iter());

    let net = self.host.readback(&mut rt);

    // Mutate the host in-place with the pre-reduced net.
    let instr = ast_net_to_instructions(&net, |nam| run::Port::new_ref(&self.host.defs[nam]));
    if let DefRef::Owned(def_box) = self.host.defs.get_mut(nam).unwrap() {
      let interpreted_def: &mut crate::run::Def<InterpretedDef> = def_box.downcast_mut().unwrap();
      interpreted_def.data.instr = instr;
    };

    // Replace the "Cycled" state with the "Reduced" state
    self.seen.insert(nam, SeenState::Reduced(net));
  }
}
