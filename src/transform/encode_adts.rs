use crate::{
  ast::{Tree, MAX_ADT_VARIANTS},
  util::maybe_grow,
};

impl Tree {
  /// Encode scott-encoded ADTs into optimized compact ADT nodes
  pub fn encode_scott_adts(&mut self) {
    maybe_grow(|| match self {
      &mut Tree::Ctr { lab, ref mut ports } => {
        fn get_adt_info(lab: u16, ports: &[Tree]) -> Option<(usize, usize)> {
          let Some(Tree::Var { nam: ret_var }) = ports.last() else { return None };

          let mut variant_index = None;
          for (idx, i) in ports.iter().enumerate().rev().skip(1) {
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
              _ => return Some((idx + 1, variant_index?)),
            }
          }
          Some((0, variant_index?))
        }

        if let Some((start, idx)) = get_adt_info(lab, ports) {
          let fields = match ports.swap_remove(idx) {
            Tree::Ctr { ports: mut fields, .. } => {
              fields.pop();
              fields
            }
            Tree::Var { .. } => vec![],
            _ => unreachable!(),
          };
          let adt = Tree::Adt { lab, variant_index: idx - start, variant_count: ports.len() - start, fields };
          if start == 0 {
            *self = adt;
          } else {
            ports.truncate(start);
            ports.push(adt);
          }
        } else {
          while let Some(&Tree::Adt { lab: inner_lab, variant_count, .. }) = ports.last() {
            if inner_lab == lab
              && variant_count < MAX_ADT_VARIANTS
              && matches!(ports.get(ports.len() - 2), Some(&Tree::Era))
            {
              ports.swap_remove(ports.len() - 2);
              let Some(Tree::Adt { variant_index, variant_count, .. }) = ports.last_mut() else { unreachable!() };
              *variant_index += 1;
              *variant_count += 1;
              if ports.len() == 1 {
                *self = ports.pop().unwrap();
                break;
              }
            } else {
              break;
            }
          }
        }

        self.children_mut().for_each(Tree::encode_scott_adts);
      }
      other => other.children_mut().for_each(Tree::encode_scott_adts),
    })
  }
}
