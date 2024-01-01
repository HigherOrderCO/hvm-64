#![allow(dead_code)]

use hvml::term::{parser, DefId, Book as DefinitionBook};
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
  parser::parse_definition_book(code).unwrap()
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

pub fn hvm_lang_readback(net: &Net, book: &DefinitionBook, id_map: HashMap<run::Val, DefId>) -> (String, bool) {
  let net = hvml::net::hvmc_to_net::hvmc_to_net(net, &|val| id_map[&val]);
  let (res_term, readback_errs) = hvml::term::net_to_term::net_to_term(&net, book, &Default::default(), false);

  (res_term.clone().display(&book.def_names).to_string(), readback_errs.is_empty())
}

pub fn hvm_lang_normal<'a>(book: &mut DefinitionBook, size: usize) -> (run::Net<'a>, Net, HashMap<run::Val, DefId>) {
  let result = hvml::compile_book(book, hvml::OptimizationLevel::Heavy).unwrap();
  let (root, res_lnet) = normal(result.core_book, size);
  (root, res_lnet, result.hvmc_names.hvmc_name_to_id)
}

#[allow(unused_variables)]
pub fn normal(book: Book, size: usize) -> (run::Net<'static>, Net) {
  fn normal_cpu(book: run::Book, size: usize) -> run::Net<'static> {
    let data = Box::leak(run::Heap::init(size));
    let mut rnet = run::Net::new(data);
    rnet.boot(name_to_val("main"));
    rnet.normal(&book);
    rnet
  }

  #[cfg(feature = "cuda")]
  fn normal_gpu(book: run::Book) -> run::Net {
    let (_, host_net) = hvmc::cuda::host::run_on_gpu(&book, "main").unwrap();
    host_net.to_runtime_net()
  }

  let book = book_to_runtime(&book);

  let rnet = {
    #[cfg(not(feature = "cuda"))]
    {
      normal_cpu(book, size)
    }
    #[cfg(feature = "cuda")]
    {
      normal_gpu(book)
    }
  };

  let net = net_from_runtime(&rnet);
  (rnet, net)
}
