use crate::prelude::*;
use hvm64_ast::{Tree, MAX_ARITY};
use hvm64_util::maybe_grow;

pub trait CoalesceCtrs {
  fn coalesce_constructors(&mut self);
}

impl CoalesceCtrs for Tree {
  /// Join chains of CTR nodes, such as `(a (b (c d)))` into n-ary nodes `(a b c
  /// d)`
  fn coalesce_constructors(&mut self) {
    maybe_grow(|| match self {
      Tree::Ctr { lab, ports } => {
        ports.iter_mut().for_each(Tree::coalesce_constructors);
        match &mut ports.pop() {
          Some(Tree::Ctr { lab: inner_lab, ports: inner_ports })
            if inner_lab == lab && ports.len() + inner_ports.len() < MAX_ARITY =>
          {
            ports.append(inner_ports);
          }
          Some(other) => ports.push(mem::take(other)),
          None => (),
        }
      }
      other => other.children_mut().for_each(Tree::coalesce_constructors),
    })
  }
}
