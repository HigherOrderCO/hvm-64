#![allow(dead_code)]

use hvm_lang::term::{parser, DefNames, DefinitionBook, Term};
use hvmc::{ast::*, run};
use std::fs;

pub fn load_file(file: &str) -> String {
  let path = format!("{}/tests/programs/{}", env!("CARGO_MANIFEST_DIR"), file);
  fs::read_to_string(path).unwrap()
}

// Parses code and generate Book from hvm-core syntax
pub fn parse_core(code: &str) -> Book {
  do_parse_book(code)
}

// Parses code and generate DefinitionBook from hvm-lang syntax
pub fn parse_lang(code: &str) -> DefinitionBook {
  parser::parse_definition_book(code).unwrap()
}

// Loads file and generate DefinitionBook from hvm-lang syntax
pub fn load_lang(file: &str) -> DefinitionBook {
  let code = load_file(file);
  parse_lang(&code)
}

// For every pair in the map, replaces all matches of a string the other string
pub fn replace_template(mut code: String, map: &[(&str, &str)]) -> String {
  for (from, to) in map {
    code = code.replace(from, to);
  }
  code
}

pub fn normal(book: Book, size: usize) -> (run::Net, Net) {
  let book = book_to_runtime(&book);
  let mut rnet = run::Net::new(size);
  rnet.boot(name_to_val("main"));
  rnet.normal(&book);

  let net = net_from_runtime(&rnet);
  (rnet, net)
}

pub fn hvm_lang_normal(book: DefinitionBook, size: usize) -> (Term, DefNames, hvm_lang::RunInfo) {
  hvm_lang::run_book(book, size).unwrap()
}

#[cfg(feature = "cuda")]
pub fn normal_gpu(book: Book) -> hvmc::cuda::host::HostNet {
  let book = book_to_runtime(&book);
  hvmc::cuda::host::run_on_gpu(&book, "main").unwrap()
}
