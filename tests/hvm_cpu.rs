use hvm_lang::term::{parser, DefinitionBook};
use hvmc::{ast::*, *};
use insta::assert_snapshot;
use std::fs;

// Loads file and generate net from hvm-core syntax
fn load_from_core(file: &str, size: usize) -> (run::Book, run::Net) {
  let code = fs::read_to_string(file).unwrap();
  let book = book_to_runtime(&do_parse_book(&code));
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

impl Normal for Book {
  fn normalize(self, size: usize) -> (run::Book, Net) {
    let book = book_to_runtime(&self);
    let mut net = run::Net::new(size);
    net.boot(name_to_val("main"));
    net.normal(&book);

    let net = net_from_runtime(&net);
    (book, net)
  }
}

impl Normal for DefinitionBook {
  fn normalize(mut self, size: usize) -> (run::Book, Net) {
    let (book, _) = hvm_lang::compile_book(&mut self).unwrap();
    book.normalize(size)
  }
}

#[test]
fn test_era_era() {
  let net = do_parse_net(&"* & * ~ *");
  let (_, net) = net.normalize(16);
  assert_snapshot!(show_net(&net), @"*");
}

#[test]
fn test_con_dup() {
  let net = do_parse_net(&"root & (x x) ~ {2 * root}");
  let (_, net) = net.normalize(16);
  assert_snapshot!(show_net(&net), @"(b b)");
}

#[test]
fn test_church_mul() {
  let (term, defs, info) = hvm_lang::run_book(
    parser::parse_definition_book(
      &"
      C_2 = λa λb (a (a b))
      C_3 = λa λb (a (a (a b)))
      Mult = λm λn λs λz (m (n s) z)
      main = (Mult C_2 C_3)",
    )
    .unwrap(),
    64,
  )
  .unwrap();

  assert_snapshot!(show_net(&info.net), @"([([b c] d) {2 (d e) (e [c f])}] (b f))");
  assert_snapshot!(term.to_string(&defs), @"λa λb (a (a (a (a (a (a b))))))");
}

#[test]
fn test_bool_and() {
  let book = do_parse_book(
    &"
    @true = (b (c b))
    @fals = (b (c c))
    @and  = ((b (@fals c)) (b c))
    @main = root & @and ~ (@true (@fals root))
  ",
  );

  let (_, net) = book.normalize(64);
  assert_snapshot!(show_net(&net), @"(b (c c))");
}
