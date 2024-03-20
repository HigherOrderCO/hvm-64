use std::{
  collections::{HashMap, HashSet},
  ops::BitOr,
};

use crate::{
  ast::{Book, Net, Tree},
  util::maybe_grow,
};

impl Book {
  pub fn inline(&mut self) -> HashSet<String> {
    let mut state = InlineState::default();
    state.populate_inlinees(self);
    let mut all_changed = HashSet::new();
    for (name, net) in &mut self.nets {
      let mut inlined = false;
      for tree in net.trees_mut() {
        inlined |= state.inline_into(tree);
      }
      if inlined {
        all_changed.insert(name.to_owned());
      }
    }
    all_changed
  }
}

#[derive(Debug, Default)]
struct InlineState {
  inlinees: HashMap<String, Tree>,
}

impl InlineState {
  fn populate_inlinees(&mut self, book: &Book) {
    for (name, net) in &book.nets {
      if net.should_inline() {
        let mut node = &net.root;
        while let Tree::Ref { nam } = node {
          let net = &book.nets[nam];
          if net.should_inline() {
            node = &net.root;
          } else {
            break;
          }
        }
        self.inlinees.insert(name.to_owned(), node.clone());
      }
    }
  }
  fn inline_into(&self, tree: &mut Tree) -> bool {
    maybe_grow(|| {
      let Tree::Ref { nam } = &*tree else {
        return tree.children_mut().map(|t| self.inline_into(t)).fold(false, bool::bitor);
      };
      if let Some(inlined) = self.inlinees.get(nam) {
        *tree = inlined.clone();
        true
      } else {
        false
      }
    })
  }
}

impl Net {
  fn should_inline(&self) -> bool {
    self.redexes.is_empty() && self.root.children().next().is_none()
  }
}
