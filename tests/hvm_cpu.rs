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

fn result_net(code: &str) -> ast::Net {
  ast::Net {
    root: ast::parse_tree(&mut code.chars().peekable()).unwrap(),
    rdex: vec![],
  }
}

#[test]
fn test_era_era() {
  let net = ast::Net {
    root: ast::Tree::Era,
    rdex: vec![(ast::Tree::Era, ast::Tree::Era)],
  };
  let mut rnet = run::Net::new(1 << 16);

  ast::net_to_runtime(&mut rnet, &net);
  let book = ast::Book::default();
  let book = ast::book_to_runtime(&book);

  rnet.normal(&book);

  let net = ast::net_from_runtime(&rnet);
  assert_eq!(net, result_net("*"));
}

#[test]
fn test_con_dup() {
  let net = ast::Net {
    root: ast::Tree::Var { nam: "root".to_string() },
    rdex: vec![(
      ast::Tree::Ctr {
        lab: 0,
        lft: ast::Tree::Var { nam: "foo".to_string() }.into(),
        rgt: ast::Tree::Var { nam: "foo".to_string() }.into(),
      },
      ast::Tree::Ctr {
        lab: 2,
        lft: ast::Tree::Era.into(),
        rgt: ast::Tree::Var { nam: "root".to_string() }.into(),
      },
    )],
  };
  let book = ast::book_to_runtime(&ast::Book::default());

  let mut rnet = run::Net::new(10);
  ast::net_to_runtime(&mut rnet, &net);

  rnet.normal(&book);

  let net = ast::net_from_runtime(&rnet);
  assert_eq!(net, result_net("(b b)"));
}
