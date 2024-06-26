#![allow(dead_code)]

// use hvm64::{ast::*, run, stdlib::create_host};
use hvm64_ast::{Book, Net};
use hvm64_host::Host;
use hvm64_runtime as run;
use std::fs;

pub fn load_file(file: &str) -> String {
  let path = format!("{}/tests/programs/{}", env!("CARGO_MANIFEST_DIR"), file);
  fs::read_to_string(path).unwrap()
}

// Parses code and generate Book from hvm-64 syntax
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
  let host = Host::new(&book);

  let mut rnet = run::Net::new(&area);
  rnet.boot(&host.defs[entry_point]);
  rnet.normal();

  let net = host.readback(&rnet);
  (rnet.rwts, net)
}

pub fn normal(book: Book, mem: Option<usize>) -> (run::Rewrites, Net) {
  normal_with(book, mem, "main")
}
