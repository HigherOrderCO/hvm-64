#![allow(dead_code)]

// use hvmc::{ast::*, run, stdlib::create_host};
use hvmc_ast::{Book, Net};
use hvmc_host::stdlib::create_host;
use hvmc_runtime as run;
use std::fs;

pub fn load_file(file: &str) -> String {
  let path = format!("{}/tests/programs/{}", env!("CARGO_MANIFEST_DIR"), file);
  fs::read_to_string(path).unwrap()
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

pub fn normal_with(book: Book, mem: Option<usize>, entry_point: &str) -> (run::Rewrites, Net) {
  let area = run::Heap::new(mem).unwrap();
  let host = create_host(&book);

  let mut rnet = run::Net::<run::Strict>::new(&area);
  rnet.boot(&host.lock().defs[entry_point]);
  rnet.normal();

  let net = host.lock().readback(&rnet);
  (rnet.rwts, net)
}

pub fn normal(book: Book, mem: Option<usize>) -> (run::Rewrites, Net) {
  normal_with(book, mem, "main")
}
