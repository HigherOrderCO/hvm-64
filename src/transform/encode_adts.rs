use crate::{
  ast::{Tree, MAX_ADT_FIELDS, MAX_ADT_VARIANTS},
  util::maybe_grow,
};

impl Tree {
  /// Encode scott-encoded ADTs into optimized compact ADT nodes
  pub fn encode_scott_adts(&mut self) {
    maybe_grow(|| match self {
      ctr @ Tree::Ctr { .. } => {
        let old = format!("{}", ctr);
        let Tree::Ctr { lab, mut ports } = std::mem::take(ctr) else { unreachable!() };
        let mut fields_idx = None;
        let mut ret_var = None;
        for (idx, i) in ports.iter().enumerate() {
          if idx >= MAX_ADT_VARIANTS {
            continue; // not an ADT variant
          };
          match i {
            Tree::Era => (),
            Tree::Ctr { lab: inner_lab, ports } if ports.len() <= MAX_ADT_FIELDS + 1 && lab == *inner_lab => {
              if let Some(Tree::Var { nam }) = ports.last() {
                if let Some(old_nam) = ret_var.replace(nam.clone()) {
                  if old_nam != *nam {
                    fields_idx = None;
                    break;
                  }
                }
              };
              fields_idx.replace(idx);
            }
            Tree::Var { nam } if idx == ports.len() - 1 => {
              if let Some(old_nam) = ret_var.replace(nam.clone()) {
                if old_nam != *nam {
                  fields_idx = None;
                  break;
                }
              }
            }
            _ => {
              // Does not encode an ADT.
              fields_idx = None;
              break;
            }
          }
        }

        if let (Some(fields_idx), Some(ret_var)) = (fields_idx, ret_var) {
          let Tree::Ctr { ports: mut fields, .. } = ports.remove(fields_idx) else { unreachable!() };
          // Remove the output
          match fields.pop() {
            Some(Tree::Var { nam }) if ret_var.as_ref() == nam => {
              let new = Tree::Adt { lab, variant_index: fields_idx, variant_count: ports.len(), fields };
              println!("-- {old}\n -> {new}");
              *ctr = new;
            }
            _ => {
              unreachable!()
            }
          }
        } else {
          *ctr = Tree::Ctr { lab, ports };
          ctr.children_mut().for_each(Tree::encode_scott_adts);
        }
      }
      other => other.children_mut().for_each(Tree::encode_scott_adts),
    })
  }
}
