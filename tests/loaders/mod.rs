#![allow(dead_code)]

use hvmc::{ast::*, host::Host, run};
use std::fs;

pub fn load_file(file: &str) -> String {
  let path = format!("{}/tests/programs/{}", env!("CARGO_MANIFEST_DIR"), file);
  fs::read_to_string(path).unwrap()
}

pub fn core_apply(net: &mut Net, arg: Tree) {
  let fun = core::mem::take(&mut net.root);
  let var = format!("$_TEST_APP{:p}", net);
  let oth = Tree::Ctr { lab: 0, lft: Box::new(arg), rgt: Box::new(Tree::Var { nam: var.clone() }) };
  net.root = Tree::Var { nam: var };
  net.rdex.push((fun, oth));
}

// Parses code and generate Book from hvm-core syntax
pub fn parse_core(code: &str) -> Book {
  code.parse().unwrap()
}

// For every pair in the map, replaces all matches of a string with the other
// string
pub fn replace_template(mut code: String, map: &[(&str, &str)]) -> String {
  for (from, to) in map {
    code = code.replace(from, to);
  }
  code
}

#[allow(unused_variables)]
pub fn normal(book: Book, size: usize) -> (hvmc::run::Rewrites, Net) {
  let area = run::Net::<run::Strict>::init_heap(size);
  let host = Host::new(&book);

  let mut rnet = run::Net::<run::Strict>::new(&area);
  rnet.boot(&host.defs["main"]);
  rnet.normal();

  let net = host.readback(&rnet);
  (rnet.rwts, net)
}
