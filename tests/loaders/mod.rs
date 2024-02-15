#![allow(dead_code)]

use hvmc::{
  ast::*,
  host::Host,
  run::{self, Node},
};
use hvml::term::{parser, term_to_net::Labels, Book as DefinitionBook, DefNames};
use std::fs;

pub fn load_file(file: &str) -> String {
  let path = format!("{}/tests/programs/{}", env!("CARGO_MANIFEST_DIR"), file);
  fs::read_to_string(path).unwrap()
}

// Parses code and generate Book from hvm-core syntax
pub fn parse_core(code: &str) -> Book {
  code.parse().unwrap()
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

// For every pair in the map, replaces all matches of a string with the other
// string
pub fn replace_template(mut code: String, map: &[(&str, &str)]) -> String {
  for (from, to) in map {
    code = code.replace(from, to);
  }
  code
}

pub fn hvm_lang_readback(net: &Net, book: &DefinitionBook) -> (String, bool) {
  let net = hvml::net::hvmc_to_net::hvmc_to_net(net);
  let (res_term, readback_errors) = hvml::term::net_to_term::net_to_term(&net, book, &Labels::default(), true);
  dbg!("done");
  (format!("{}", res_term.display(&book.def_names)), readback_errors.0.is_empty())
}

pub fn hvm_lang_normal(book: &mut DefinitionBook, size: usize) -> (hvmc::run::Rewrites, Net) {
  let compiled = hvml::compile_book(book, hvml::Opts::light()).unwrap();
  let (root, res_lnet) = normal(compiled.core_book, size);
  (root, res_lnet)
}

#[allow(unused_variables)]
pub fn normal(book: Book, size: usize) -> (hvmc::run::Rewrites, Net) {
  fn normal_cpu<'area>(host: &Host, area: &'area [Node]) -> run::Net<'area> {
    let mut rnet = run::Net::new(area);
    rnet.boot(host.defs.get(DefNames::ENTRY_POINT).unwrap());
    rnet.normal();
    rnet
  }

  let area = run::Net::init_heap(size);
  let host = Host::new(&book);

  let rnet = normal_cpu(&host, &area);

  let net = host.readback(&rnet);
  (rnet.rwts, net)
}
