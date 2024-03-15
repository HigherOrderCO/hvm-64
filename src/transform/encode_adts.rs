use crate::{ast::Tree, util::maybe_grow};

impl Tree {
  /// Encode scott-encoded ADTs into optimized compact ADT nodes
  pub fn encode_scott_adts(&mut self) {
    maybe_grow(|| match self {
      &mut Tree::Ctr { lab, ref mut ports } => {
        fn get_variant_index(lab: u16, ports: &[Tree]) -> Option<usize> {
          let Some(Tree::Var { nam: ret_var }) = ports.last() else { return None };

          let mut variant_index = None;
          for (idx, i) in ports[0 .. ports.len() - 1].iter().enumerate() {
            match i {
              Tree::Era => {}
              Tree::Ctr { lab: inner_lab, ports } if *inner_lab == lab && variant_index.is_none() => {
                // Ensure that the last port is the return variable
                let Some(Tree::Var { nam }) = ports.last() else { return None };
                if nam != ret_var {
                  return None;
                }
                variant_index = Some(idx);
              }
              // Nilary variant
              Tree::Var { nam } if nam == ret_var && variant_index.is_none() => {
                variant_index = Some(idx);
              }
              // Does not encode an ADT.
              _ => return None,
            }
          }
          variant_index
        }

        if let Some(variant_index) = get_variant_index(lab, ports) {
          let fields = match ports.swap_remove(variant_index) {
            Tree::Ctr { ports: mut fields, .. } => {
              fields.pop();
              fields
            }
            Tree::Var { .. } => vec![],
            _ => unreachable!(),
          };
          *self = Tree::Adt { lab, variant_index, variant_count: ports.len(), fields };
        }

        self.children_mut().for_each(Tree::encode_scott_adts);
      }
      other => other.children_mut().for_each(Tree::encode_scott_adts),
    })
  }
}
