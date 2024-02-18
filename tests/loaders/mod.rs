#![allow(dead_code)]

use hvml::term::{parser, Book as DefinitionBook, Name};
use hvmc::{ast::*, run};
use std::{collections::HashMap, fs};

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
  parser::parse_book(code, DefinitionBook::default, false).unwrap()
}

// Loads file and generate DefinitionBook from hvm-lang syntax
pub fn load_lang(file: &str) -> DefinitionBook {
  let code = load_file(file);
  parse_lang(&code)
}

// For every pair in the map, replaces all matches of a string with the other string
pub fn replace_template(mut code: String, map: &[(&str, &str)]) -> String {
  for (from, to) in map {
    code = code.replace(from, to);
  }
  code
}

pub fn hvm_lang_readback(net: &Net, book: &DefinitionBook, id_map: HashMap<run::Val, Name>) -> (String, bool) {
  let net = hvml::net::hvmc_to_net::hvmc_to_net(net, &id_map);
  let (res_term, readback_errs) = hvml::term::net_to_term::net_to_term(&net, book, &Default::default(), false);
  let term_display = res_term.to_string();

  (term_display, readback_errs.is_empty())
}

pub fn hvm_lang_normal(book: &mut DefinitionBook, size: usize) -> (run::Net, Net, HashMap<run::Val, Name>) {
  let result = hvml::compile_book(book, hvml::CompileOpts::light(), None).unwrap();
  let (root, res_lnet) = normal(result.core_book, size);
  (root, res_lnet, result.hvmc_names.hvmc_to_hvml)
}

#[allow(unused_variables)]
pub fn normal(book: Book, size: usize) -> (run::Net, Net) {
  fn normal_cpu(book: run::Book, size: usize) -> run::Net {
    let mut rnet = run::Net::new(size, false);
    rnet.normal(&book);
    rnet
  }

  let book = book_to_runtime(&book);

  let rnet = normal_cpu(book, size);

  let net = rnet.net_from_runtime();
  (rnet, net)
}
