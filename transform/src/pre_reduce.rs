//! Reduce the nets in the book, to avoid repeating work at runtime.
//!
//! Here's how it works:
//! - Each definition is visited in topological order (dependencies before
//!   dependents). In the case of cycles, one will be arbitrarily selected to be
//!   first.
//! - The definition is reduced in a [`hvmc_runtime::Net`]
//! - The reduced [`hvmc_runtime::Net`] is readback into an [`ast::Net`]
//! - The [`ast::Net`] is encoded into a [`Vec<Instruction>`]
//! - The [`ast::Net`] is stored in the [`State`], as it will be used later.
//! - The [`InterpretedDef`] corresponding to the definition is mutated in-place
//!   and its instructions are replaced with the generated [`Vec<Instruction>`]
//!
//! At the end, each mutated [`ast::Net`] is placed into the [`Book`],
//! overriding the previous one.

use crate::prelude::*;
use hvmc_ast::{Book, Tree};
use hvmc_host::{
  stdlib::{AsHostedDef, HostedDef},
  DefRef, Host,
};
use hvmc_runtime::{Def, Heap, InterpretedDef, LabSet, Mode, Port, Rewrites, Strict};
use hvmc_util::maybe_grow;

use alloc::sync::Arc;
use parking_lot::Mutex;

pub trait PreReduce {
  fn pre_reduce(&mut self, skip: &dyn Fn(&str) -> bool, max_memory: Option<usize>, max_rwts: u64) -> PreReduceStats;
}

impl PreReduce for Book {
  /// Reduces the definitions in the book individually, except for the skipped
  /// ones.
  ///
  /// Defs that are not in the book are treated as inert defs.
  ///
  /// `max_memory` is measured in bytes.
  fn pre_reduce(&mut self, skip: &dyn Fn(&str) -> bool, max_memory: Option<usize>, max_rwts: u64) -> PreReduceStats {
    let mut host = Host::default();
    let captured_redexes = Arc::new(Mutex::new(Vec::new()));
    // When a ref is not found in the `Host`, put an inert def in its place.
    host.insert_book_with_default(self, &mut |_| unsafe {
      HostedDef::new_hosted(LabSet::ALL, InertDef(captured_redexes.clone()))
    });
    let area = Heap::new(max_memory).expect("pre-reduce memory allocation failed");

    let mut state = State {
      book: self,
      skip,
      captured_redexes,
      max_rwts,
      host,
      area: &area,
      seen: Map::new(),
      rewrites: Rewrites::default(),
    };

    for nam in self.nets.keys() {
      state.pre_reduce(nam)
    }

    let State { seen, rewrites, .. } = state;

    let mut not_normal = vec![];
    for (nam, state) in seen {
      if let SeenState::Reduced { net, normal } = state {
        if !normal {
          not_normal.push(nam.clone());
        }
        self.nets.insert(nam, net);
      }
    }

    PreReduceStats { rewrites, not_normal, errors: vec![] }
  }
}

pub struct PreReduceStats {
  pub rewrites: Rewrites,
  pub not_normal: Vec<String>,
  pub errors: Vec<String>,
}

enum SeenState {
  Cycled,
  Reduced { net: hvmc_ast::Net, normal: bool },
}

/// A Def that pushes all interactions to its inner Vec.
#[derive(Default)]
struct InertDef(Arc<Mutex<Vec<(Port, Port)>>>);

impl AsHostedDef for InertDef {
  fn call<M: Mode>(def: &Def<Self>, _: &mut hvmc_runtime::Net<M>, port: Port) {
    def.data.0.lock().push((Port::new_ref(def), port));
  }
}

/// State of the pre-reduction algorithm.
struct State<'a> {
  book: &'a Book,

  host: Host,
  max_rwts: u64,

  area: &'a Heap,
  captured_redexes: Arc<Mutex<Vec<(Port, Port)>>>,

  skip: &'a dyn Fn(&str) -> bool,
  seen: Map<String, SeenState>,

  rewrites: Rewrites<u64>,
}

impl<'a> State<'a> {
  fn visit_tree(&mut self, tree: &Tree) {
    maybe_grow(move || {
      if let Tree::Ref { nam } = tree {
        self.pre_reduce(nam);
      }
      tree.children().for_each(|child| self.visit_tree(child))
    })
  }
  fn visit_net(&mut self, net: &hvmc_ast::Net) {
    self.visit_tree(&net.root);
    for (a, b) in &net.redexes {
      self.visit_tree(a);
      self.visit_tree(b);
    }
  }
  fn pre_reduce(&mut self, nam: &str) {
    if self.seen.contains_key(nam) || (self.skip)(nam) || self.book.get(nam).is_none() {
      return;
    }

    self.seen.insert(nam.to_owned(), SeenState::Cycled);
    // First, pre-reduce all nets referenced by this net by walking the tree
    self.visit_net(self.book.get(nam).unwrap());

    let mut rt = hvmc_runtime::Net::<Strict>::new(self.area);
    rt.boot(self.host.defs.get(nam).expect("No function."));
    let n_reduced = rt.reduce(self.max_rwts as usize);

    self.rewrites += rt.rwts;

    // Move interactions with inert defs back into the net redexes array
    self.captured_redexes.lock().drain(..).for_each(|r| rt.redux(r.0, r.1));

    let net = self.host.readback(&rt);

    // Mutate the host in-place with the pre-reduced net.
    let instr = self.host.encode_def(&net);
    if let DefRef::Owned(def_box) = self.host.defs.get_mut(nam).unwrap() {
      let interpreted_def: &mut Def<InterpretedDef> = def_box.downcast_mut().unwrap();
      interpreted_def.data = instr;
    };

    // Replace the "Cycled" state with the "Reduced" state
    *self.seen.get_mut(nam).unwrap() = SeenState::Reduced { net, normal: n_reduced.is_some() };
  }
}
