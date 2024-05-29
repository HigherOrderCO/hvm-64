use hvm64_util::prelude::*;
use hvm64_ast::{Book, Tree};
use hvm64_util::maybe_grow;

pub trait Prune {
  fn prune(&mut self, entrypoints: &[String]);
}

impl Prune for Book {
  fn prune(&mut self, entrypoints: &[String]) {
    let mut state = PruneState { book: self, unvisited: self.keys().map(|x| x.to_owned()).collect() };
    for name in entrypoints {
      state.visit_def(name);
    }
    let unvisited = state.unvisited;
    for name in unvisited {
      self.remove(&name);
    }
  }
}

#[derive(Debug)]
struct PruneState<'a> {
  book: &'a Book,
  unvisited: Set<String>,
}

impl<'a> PruneState<'a> {
  fn visit_def(&mut self, name: &str) {
    if self.unvisited.remove(name) {
      for tree in self.book[name].trees() {
        self.visit_tree(tree);
      }
    }
  }
  fn visit_tree(&mut self, tree: &Tree) {
    maybe_grow(|| {
      if let Tree::Ref { nam } = tree {
        self.visit_def(nam);
      } else {
        tree.children().for_each(|t| self.visit_tree(t));
      }
    })
  }
}
