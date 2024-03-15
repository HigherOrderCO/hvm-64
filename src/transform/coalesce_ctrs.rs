use crate::{
  ast::{Tree, MAX_ARITY},
  util::maybe_grow,
};

impl Tree {
  /// Join chains of CTR nodes, such as `(a (b (c d)))` into n-ary nodes `(a b c d)`
  pub fn coalesce_constructors(&mut self) {
    maybe_grow(|| match self {
      Tree::Ctr { lab, ports } => {
        ports.iter_mut().for_each(Tree::coalesce_constructors);
        match ports.pop() {
          Some(Tree::Ctr { lab: inner_lab, ports: mut inner_ports })
            if inner_lab == *lab && ports.len() + inner_ports.len() < MAX_ARITY =>
          {
            ports.extend(inner_ports.drain(..));
          }
          Some(other) => ports.push(other),
          None => (),
        }
      }
      other => other.children_mut().for_each(Tree::coalesce_constructors),
    })
  }
}
