#![allow(dead_code)]

use hvmc::{ast::*, host::Host, run};
use hvml::term::{parser, term_to_net::Labels, Book as LangBook};
use std::fs;

pub fn load_file(file: &str) -> String {
  let path = format!("{}/tests/programs/{}", env!("CARGO_MANIFEST_DIR"), file);
  fs::read_to_string(path).unwrap()
}

// Parses code and generate Book from hvm-core syntax
pub fn parse_core(code: &str) -> Book {
  code.parse().unwrap()
}

// Parses code and generate LangBook from hvm-lang syntax
pub fn parse_lang(code: &str) -> LangBook {
  parser::parse_book(code, LangBook::default, false).unwrap()
}

// Loads file and generate LangBook from hvm-lang syntax
pub fn load_lang(file: &str) -> LangBook {
  let code = load_file(file);
  parse_lang(&code)
}

// For every pair in the map, replaces all matches of a string with the other
// string
pub fn replace_template(mut code: String, map: &[(&str, &str)]) -> String {
  for (from, to) in map {
    code = code.replace(from, to);
  }
  code
}

pub fn hvm_lang_readback(net: &Net, book: &LangBook) -> (String, bool) {
  let net = hvml::net::hvmc_to_net::hvmc_to_net(net);
  let (res_term, readback_errors) = hvml::term::net_to_term::net_to_term(&net, book, &Labels::default(), true);
  (format!("{}", res_term), readback_errors.is_empty())
}

pub fn hvm_lang_normal(book: &mut LangBook, size: usize) -> (hvmc::run::Rewrites, Net) {
  let compiled = hvml::compile_book(book, hvml::CompileOpts::light(), book.entrypoint.clone()).unwrap();
  let (root, res_lnet) = normal(compiled.core_book, size);
  (root, res_lnet)
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
