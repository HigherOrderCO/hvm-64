use crate::run;
use std::collections::{BTreeMap, HashMap};

// TODO: Factor out duplication with `gen_cuda_book` in main.rs

// Generate data to pass to CUDA runtime.
// Returns (book_data, jump_data, function_ids)
pub fn gen_cuda_book_data(book: &run::Book) -> (Vec<u32>, Vec<u32>, HashMap<String, u32>) {
  // Sort the book.defs by key
  let mut defs = BTreeMap::new();
  for i in 0 .. book.defs.len() {
    if book.defs[i].node.len() > 0 {
      defs.insert(i as u32, book.defs[i].clone());
    }
  }

  // Generate map from function name to id
  let function_ids = defs.keys().enumerate().map(|(i, &id)|  {
    (crate::ast::val_to_name(id), id)
  }).collect::<HashMap<_, _>>();

  // Generate book data
  let mut book_data = vec![];
  for (i, (id, net)) in defs.iter().enumerate() {
    // Collect all pointers from root, nodes and rdex into a single buffer
    book_data.push(net.node.len() as u32);
    book_data.push(net.rdex.len() as u32);
    // .node
    for (i, node) in net.node.iter().enumerate() {
      book_data.push(node.0.data());
      book_data.push(node.1.data());
    }
    // .rdex
    for (i, (a, b)) in net.rdex.iter().enumerate() {
      book_data.push(a.data());
      book_data.push(b.data());
    }
  }

  let mut jump_data = vec![];
  let mut index = 0;
  for (i, id) in defs.keys().enumerate() {
    jump_data.push(*id);
    jump_data.push(index);
    index += 2 + 2 * defs[id].node.len() as u32 + 2 * defs[id].rdex.len() as u32;
  }

  (book_data, jump_data, function_ids)
}
