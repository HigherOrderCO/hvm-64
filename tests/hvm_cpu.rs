use hvm_lang::term::{parser, DefNames, DefinitionBook, Term};
use hvmc::{ast::*, *};
use insta::assert_snapshot;
use std::fs;

// Loads file and generate net from hvm-core syntax
fn load_from_core(file: &str, size: usize) -> (run::Book, run::Net) {
  let path = format!("{}/tests/programs/{}", env!("CARGO_MANIFEST_DIR"), file);
  let code = fs::read_to_string(path).unwrap();

  let book = book_to_runtime(&do_parse_book(&code));
  let mut net = run::Net::new(size);
  net.boot(name_to_val("main"));
  (book, net)
}

// Loads file and generate net from hvm-lang syntax
fn load_from_lang(file: &str, size: usize) -> (Term, DefNames, hvm_lang::RunInfo) {
  let path = format!("{}/tests/programs/{}", env!("CARGO_MANIFEST_DIR"), file);
  let code = fs::read_to_string(path).unwrap();

  let book = parser::parse_definition_book(&code).unwrap();
  hvm_lang::run_book(book, size).unwrap()
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
  let net = do_parse_net("* & * ~ *");
  let (_, net) = net.normalize(16);
  assert_snapshot!(show_net(&net), @"*");
}

#[test]
fn test_con_dup() {
  let net = do_parse_net("root & (x x) ~ {2 * root}");
  let (_, net) = net.normalize(16);
  assert_snapshot!(show_net(&net), @"(b b)");
}

#[test]
fn test_church_mul() {
  let (term, defs, info) = hvm_lang::run_book(
    parser::parse_definition_book(
      "
      C_2 = λa λb (a (a b))
      C_3 = λa λb (a (a (a b)))
      Mult = λm λn λs λz (m (n s) z)
      main = (Mult C_2 C_3)",
    )
    .unwrap(),
    64,
  )
  .unwrap();

  assert_snapshot!(show_net(&info.net), @"({2 ({2 b c} d) {3 (d e) (e {2 c f})}} (b f))");
  assert_snapshot!(term.to_string(&defs), @"λa λb (a (a (a (a (a (a b))))))");
}

#[test]
fn test_bool_and() {
  let book = do_parse_book(
    "
    @true = (b (c b))
    @fals = (b (c c))
    @and  = ((b (@fals c)) (b c))
    @main = root & @and ~ (@true (@fals root))
  ",
  );

  let (_, net) = book.normalize(64);
  assert_snapshot!(show_net(&net), @"(b (c c))");
}

#[test]
fn test_neg_fusion() {
  let (_, _, info) = load_from_lang("neg_fusion.hvm", 516);

  assert_snapshot!(show_net(&info.net), @"(b (* b))");
  assert_snapshot!(info.stats.rewrites.total_rewrites().to_string(), @"153");
}

#[test]
fn test_tree_alloc() {
  let (_, _, info) = load_from_lang("tree_alloc.hvm", 516);

  assert_snapshot!(show_net(&info.net), @"(b (* b))");
  assert_snapshot!(info.stats.rewrites.total_rewrites().to_string(), @"104");
}

const QUEUE: &'static str = include_str!("./programs/queue.hvm");

fn make_queue(lenght: u32) -> String {
  let mut body = String::new();

  for i in 1 ..= lenght {
    body += &format!("let q = (Qadd {i} q)\n")
  }

  for i in 0 .. lenght {
    body += &format!("(Qrem q λv{i} λq\n");
  }

  for i in 1 ..= lenght {
    body += &format!("(Cons {i} ");
  }

  format!("{QUEUE}\nmain = let q = Qnew\n{body} Nil{}", ")".repeat(lenght as usize * 2))
}

fn run_queue(lenght: u32, mem_size: usize) -> (Term, DefNames, hvm_lang::RunInfo) {
  let a = make_queue(lenght);
  hvm_lang::run_book(parser::parse_definition_book(&a).unwrap(), mem_size).unwrap()
}

#[test]
fn test_queues() {
  let (term, defs, info_3) = run_queue(3, 512);
  assert_snapshot!(info_3.stats.rewrites.total_rewrites().to_string(), @"62");
  let (_, _, info) = run_queue(4, 512);
  assert_snapshot!(info.stats.rewrites.total_rewrites().to_string(), @"81");
  let (_, _, info) = run_queue(5, 512);
  assert_snapshot!(info.stats.rewrites.total_rewrites().to_string(), @"100");
  let (_, _, info) = run_queue(10, 512);
  assert_snapshot!(info.stats.rewrites.total_rewrites().to_string(), @"195");
  let (_, _, info) = run_queue(20, 512);
  assert_snapshot!(info.stats.rewrites.total_rewrites().to_string(), @"385");

  assert_snapshot!(show_net(&info_3.net), @"((#1 (((#2 (((#3 ((* @7) b)) (* b)) c)) (* c)) d)) (* d))");
  assert_snapshot!(term.to_string(&defs), @"λa λ* ((a 1) λb λ* ((b 2) λc λ* ((c 3) λ* λd d)))");
}

// Numeric Operations test

fn op_net(lnum: u32, op: u8, rnum: u32) -> Net {
  do_parse_net(&format!("root & <#{lnum} <#{rnum} root>> ~ #{op}"))
}

#[test]
fn test_add() {
  let net = op_net(10, run::ADD, 2);
  let (_, net) = net.normalize(16);
  assert_snapshot!(show_net(&net), @"#12");
}

#[test]
fn test_sub() {
  let net = op_net(10, run::SUB, 2);
  let (_, net) = net.normalize(16);
  assert_snapshot!(show_net(&net), @"#8");
}

#[test]
fn test_mul() {
  let net = op_net(10, run::MUL, 2);
  let (_, net) = net.normalize(16);
  assert_snapshot!(show_net(&net), @"#20");
}

#[test]
fn test_div() {
  let net = op_net(10, run::DIV, 2);
  let (_, net) = net.normalize(16);
  assert_snapshot!(show_net(&net), @"#5");
}

#[test]
fn test_mod() {
  let net = op_net(10, run::MOD, 2);
  let (_, net) = net.normalize(16);
  assert_snapshot!(show_net(&net), @"#0");
}

#[test]
fn test_eq() {
  let net = op_net(10, run::EQ, 2);
  let (_, net) = net.normalize(16);
  assert_snapshot!(show_net(&net), @"#0");
}

#[test]
fn test_ne() {
  let net = op_net(10, run::NE, 2);
  let (_, net) = net.normalize(16);
  assert_snapshot!(show_net(&net), @"#1");
}

#[test]
fn test_lt() {
  let net = op_net(10, run::LT, 2);
  let (_, net) = net.normalize(16);
  assert_snapshot!(show_net(&net), @"#0");
}

#[test]
fn test_gt() {
  let net = op_net(10, run::GT, 2);
  let (_, net) = net.normalize(16);
  assert_snapshot!(show_net(&net), @"#1");
}

#[test]
fn test_and() {
  let net = op_net(10, run::AND, 2);
  let (_, net) = net.normalize(16);
  assert_snapshot!(show_net(&net), @"#2");
}

#[test]
fn test_or() {
  let net = op_net(10, run::OR, 2);
  let (_, net) = net.normalize(16);
  assert_snapshot!(show_net(&net), @"#10");
}

#[test]
fn test_xor() {
  let net = op_net(10, run::XOR, 2);
  let (_, net) = net.normalize(16);
  assert_snapshot!(show_net(&net), @"#8");
}

#[test]
fn test_lsh() {
  let net = op_net(10, run::LSH, 2);
  let (_, net) = net.normalize(16);
  assert_snapshot!(show_net(&net), @"#40");
}

#[test]
fn test_rsh() {
  let net = op_net(10, run::RSH, 2);
  let (_, net) = net.normalize(16);
  assert_snapshot!(show_net(&net), @"#2");
}

#[test]
fn test_div_by_0() {
  let net = op_net(9, run::DIV, 0);
  let (_, net) = net.normalize(16);
  assert_snapshot!(show_net(&net), @"#16777215");
}
