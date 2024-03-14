use crate::{ast::Tree, util::maybe_grow};

impl Tree {
  /// Encode scott-encoded ADTs into optimized compact ADT nodes
  pub fn encode_scott_adts(&mut self) {
    maybe_grow(|| match self {
      ctr @ Tree::Ctr { .. } => {
        let Tree::Ctr { lab, mut ports } = std::mem::take(ctr) else { unreachable!() };
        fn get_field_and_ret(lab: &u16, ports: &[Tree]) -> Option<(usize, String)> {
          let mut fields_idx = None;
          let ret_var = match ports.last() {
            Some(Tree::Var { nam: ret_var }) => ret_var.clone(),
            _ => {
              return None;
            }
          };

          for (idx, i) in ports.iter().take(ports.len() - 1).enumerate() {
            match i {
              Tree::Era => (),
              Tree::Ctr { lab: inner_lab, ports } if lab == inner_lab && fields_idx.is_none() => {
                // Ensure that the last port
                // is the return variable
                if match ports.last() {
                  Some(Tree::Var { nam }) => *nam != ret_var,
                  _ => true,
                } {
                  return None;
                }
                fields_idx = Some(idx);
              }
              // Nilary field.
              Tree::Var { nam } if ret_var == *nam && fields_idx.is_none() => {
                fields_idx = Some(idx);
              }
              _ => {
                // Does not encode an ADT.
                return None;
              }
            }
          }
          fields_idx.map(move |x| (x, ret_var))
        }

        if let Some((fields_idx, ret_var)) = get_field_and_ret(&lab, &ports) {
          let fields = match ports.swap_remove(fields_idx) {
            Tree::Ctr { ports: mut fields, .. } => {
              fields.pop();
              fields
            }
            Tree::Var { nam } if nam == ret_var => {
              vec![]
            }
            _ => unreachable!()
          };
          let new = Tree::Adt { lab, variant_index: fields_idx, variant_count: ports.len(), fields };
          *ctr = new;
        } else {
          *ctr = Tree::Ctr { lab, ports };
          ctr.children_mut().for_each(Tree::encode_scott_adts);
        }
      }
      other => other.children_mut().for_each(Tree::encode_scott_adts),
    })
  }
}
