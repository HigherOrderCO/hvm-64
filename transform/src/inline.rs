use core::ops::BitOr;

use crate::prelude::*;
use hvmc_ast::{Book, Net, Tree};
use hvmc_util::maybe_grow;

use super::TransformError;

pub trait Inline {
  fn inline(&mut self) -> Result<Set<String>, TransformError>;
}

impl Inline for Book {
  fn inline(&mut self) -> Result<Set<String>, TransformError> {
    let mut state = InlineState::default();
    state.populate_inlinees(self)?;
    let mut all_changed = Set::new();
    for (name, net) in &mut self.nets {
      let mut inlined = false;
      for tree in net.trees_mut() {
        inlined |= state.inline_into(tree);
      }
      if inlined {
        all_changed.insert(name.to_owned());
      }
    }
    Ok(all_changed)
  }
}

#[derive(Debug, Default)]
struct InlineState {
  inlinees: Map<String, Tree>,
}

impl InlineState {
  fn populate_inlinees(&mut self, book: &Book) -> Result<(), TransformError> {
    for (name, net) in &book.nets {
      if net.should_inline() {
        // Detect cycles with tortoise and hare algorithm
        let mut hare = &net.root;
        let mut tortoise = &net.root;
        // Whether or not the tortoise should take a step
        let mut parity = false;
        while let Tree::Ref { nam } = hare {
          let Some(net) = &book.nets.get(nam) else { break };
          if net.should_inline() {
            hare = &net.root;
          } else {
            break;
          }
          if parity {
            let Tree::Ref { nam: tortoise_nam } = tortoise else { unreachable!() };
            if tortoise_nam == nam {
              Err(TransformError::InfiniteRefCycle(nam.to_owned()))?;
            }
            tortoise = &book.nets[tortoise_nam].root;
          }
          parity = !parity;
        }
        self.inlinees.insert(name.to_owned(), hare.clone());
      }
    }
    Ok(())
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

trait ShouldInline {
  fn should_inline(&self) -> bool;
}

impl ShouldInline for Net {
  fn should_inline(&self) -> bool {
    self.redexes.is_empty() && self.root.children().next().is_none()
  }
}
