use hvm_lang::term::{parser, DefinitionBook};
use hvmc::{ast::*, *};
use std::fs;

// Loads file and generate net from hvm-core syntax
fn load_from_core(file: &str, size: usize) -> (run::Book, run::Net) {
  let code = fs::read_to_string(file).unwrap();
  let book = ast::book_to_runtime(&do_parse_book(&code));
  let mut net = run::Net::new(size);
  net.boot(name_to_val("main"));
  (book, net)
}

// Loads file and generate net from hvm-lang syntax
fn load_from_lang(file: &str, size: usize) -> (run::Book, run::Net) {
  let code = fs::read_to_string(file).unwrap();

  let mut book = parser::parse_definition_book(&code).unwrap();
  let (book, _) = hvm_lang::compile_book(&mut book).unwrap();
  let book = book_to_runtime(&book);

  let mut net = run::Net::new(size);
  net.boot(name_to_val("main"));
  (book, net)
}

fn result_net(code: &str) -> Net {
  Net { root: parse_tree(&mut code.chars().peekable()).unwrap(), rdex: vec![] }
}

trait Normal {
  fn normalize(self, size: usize) -> (run::Book, Net);
}

impl Normal for Net {
  fn normalize(self, size: usize) -> (run::Book, Net) {
    let mut rnet = run::Net::new(size);
    net_to_runtime(&mut rnet, &self);

    let book = book_to_runtime(&Book::default());
    rnet.normal(&book);

    let net = net_from_runtime(&rnet);
    (book, net)
  }
}

impl Normal for DefinitionBook {
  fn normalize(mut self, size: usize) -> (run::Book, Net) {
    let (book, _) = hvm_lang::compile_book(&mut self).unwrap();
    let book = book_to_runtime(&book);

    let mut net = run::Net::new(size);
    net.boot(name_to_val("main"));
    net.normal(&book);

    let net = net_from_runtime(&net);
    (book, net)
  }
}

#[test]
fn test_era_era() {
  let net = Net { root: Tree::Era, rdex: vec![(Tree::Era, ast::Tree::Era)] };
  let (_, net) = net.normalize(16);
  assert_eq!(net, result_net("*"));
}

#[test]
fn test_con_dup() {
  let net = do_parse_net(&"root & (x x) ~ {2 * root}");
  let (_, net) = net.normalize(16);
  assert_eq!(net, result_net("(b b)"));
}
