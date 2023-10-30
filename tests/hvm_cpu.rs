use hvm_lang::term::{parser, DefNames, DefinitionBook, Term};
use hvmc::{ast::*, *};
use insta::{assert_debug_snapshot, assert_snapshot};
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

fn total_rewrites(net: &run::Net) -> usize {
  net.anni + net.comm + net.eras + net.dref + net.oper
}

trait Normal {
  fn normalize(self, size: usize) -> (run::Net, Net);
}

impl Normal for Net {
  fn normalize(self, size: usize) -> (run::Net, Net) {
    let mut rnet = run::Net::new(size);
    net_to_runtime(&mut rnet, &self);

    let book = book_to_runtime(&Book::default());
    rnet.normal(&book);

    let net = net_from_runtime(&rnet);
    (rnet, net)
  }
}

impl Normal for Book {
  fn normalize(self, size: usize) -> (run::Net, Net) {
    let book = book_to_runtime(&self);
    let mut rnet = run::Net::new(size);
    rnet.boot(name_to_val("main"));
    rnet.normal(&book);

    let net = net_from_runtime(&rnet);
    (rnet, net)
  }
}

impl Normal for DefinitionBook {
  fn normalize(mut self, size: usize) -> (run::Net, Net) {
    let (rnet, _) = hvm_lang::compile_book(&mut self).unwrap();
    rnet.normalize(size)
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
  assert_debug_snapshot!(info.stats.rewrites.total_rewrites(), @"153");
}

#[test]
fn test_tree_alloc() {
  let (_, _, info) = load_from_lang("tree_alloc.hvm", 516);

  assert_snapshot!(show_net(&info.net), @"(b (* b))");
  assert_debug_snapshot!(info.stats.rewrites.total_rewrites(), @"104");
}

const QUEUE: &'static str = include_str!("./programs/queue.hvm");

fn make_queue(len: u32) -> String {
  let mut body = String::new();

  for i in 1 ..= len {
    body += &format!("let q = (Qadd {i} q)\n")
  }

  for i in 0 .. len {
    body += &format!("(Qrem q λv{i} λq\n");
  }

  for i in 1 ..= len {
    body += &format!("(Cons {i} ");
  }

  format!("{QUEUE}\nmain = let q = Qnew\n{body} Nil{}", ")".repeat(len as usize * 2))
}

fn run_queue_of_size(len: u32, mem_size: usize) -> (Term, DefNames, hvm_lang::RunInfo) {
  let queue = make_queue(len);
  hvm_lang::run_book(parser::parse_definition_book(&queue).unwrap(), mem_size).unwrap()
}

#[test]
fn test_queues() {
  let info = [
    run_queue_of_size(3, 512),
    run_queue_of_size(4, 512),
    run_queue_of_size(5, 512),
    run_queue_of_size(10, 512),
    run_queue_of_size(20, 512),
  ]
  .map(|(term, defs, info)| (term, defs, info.net, info.stats.rewrites.total_rewrites()));

  assert_debug_snapshot!(info[0].3, @"62");
  assert_debug_snapshot!(info[1].3, @"81");
  assert_debug_snapshot!(info[2].3, @"100");
  assert_debug_snapshot!(info[3].3, @"195");
  assert_debug_snapshot!(info[4].3, @"385");

  let (term, defs, net, _) = &info[0];
  assert_snapshot!(show_net(&net), @"((#1 (((#2 (((#3 ((* @7) b)) (* b)) c)) (* c)) d)) (* d))");
  assert_snapshot!(term.to_string(&defs), @"λa λ* ((a 1) λb λ* ((b 2) λc λ* ((c 3) λ* λd d)))");
}

const LIST: &'static str = include_str!("./programs/list_put_get.hvm");

fn run_list_fn(list_fun: &str, args: &str, mem_size: usize) -> (Term, DefNames, hvm_lang::RunInfo) {
  let list = format!("{LIST}\nlet (got, list) = (List.{list_fun} list {args}); got");
  hvm_lang::run_book(parser::parse_definition_book(&list).unwrap(), mem_size).unwrap()
}

#[test]
fn test_list_got() {
  let info = [
    run_list_fn("got", "0", 2048),
    run_list_fn("got", "1", 2048),
    run_list_fn("got", "3", 2048),
    run_list_fn("got", "7", 2048),
    run_list_fn("got", "15", 2048),
    run_list_fn("got", "31", 2048),
  ]
  .map(|(_, _, info)| info.stats.rewrites.total_rewrites());

  assert_debug_snapshot!(info[0], @"573");
  assert_debug_snapshot!(info[1], @"595");
  assert_debug_snapshot!(info[2], @"639");
  assert_debug_snapshot!(info[3], @"727");
  assert_debug_snapshot!(info[4], @"903");
  assert_debug_snapshot!(info[5], @"1255");

  //Tests the linearity of the function
  let delta = info[1] - info[0];
  assert_eq!(info[1] + delta * 2, info[2]);
  assert_eq!(info[2] + delta * 4, info[3]);
  assert_eq!(info[3] + delta * 8, info[4]);
  assert_eq!(info[4] + delta * 16, info[5]);
}

#[test]
fn test_list_put() {
  let info = [
    run_list_fn("put", "0 2", 2048),
    run_list_fn("put", "1 4", 2048),
    run_list_fn("put", "3 8", 2048),
    run_list_fn("put", "7 16", 2048),
    run_list_fn("put", "15 32", 2048),
    run_list_fn("put", "31 64", 2048),
  ]
  .map(|(_, _, info)| info.stats.rewrites.total_rewrites());

  assert_debug_snapshot!(info[0], @"563");
  assert_debug_snapshot!(info[1], @"586");
  assert_debug_snapshot!(info[2], @"632");
  assert_debug_snapshot!(info[3], @"724");
  assert_debug_snapshot!(info[4], @"908");
  assert_debug_snapshot!(info[5], @"1276");

  //Tests the linearity of the function
  let delta = info[1] - info[0];
  assert_eq!(info[1] + delta * 2, info[2]);
  assert_eq!(info[2] + delta * 4, info[3]);
  assert_eq!(info[3] + delta * 8, info[4]);
  assert_eq!(info[4] + delta * 16, info[5]);
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
fn test_not() {
  let net = op_net(0, run::NOT, 256);
  let (_, net) = net.normalize(16);
  assert_snapshot!(show_net(&net), @"#16776959");
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
/// Division by zero always return the value of 0xFFFFFF,
/// that is read as the unsigned integer `16777215`
fn test_div_by_0() {
  let net = op_net(9, run::DIV, 0);
  let (_, net) = net.normalize(16);
  assert_snapshot!(show_net(&net), @"#16777215");
}

#[test]
fn test_chained_ops() {
  let net = do_parse_net(
    "a
      & (#70 (#50 a)) ~ ({2 b {3 c d}} ({4 e {5 f g}} h))
      & <d <i j>> ~ #3
      & <j <k l>> ~ #2
      & <l <m h>> ~ #3
      & <n <o m>> ~ #1
      & <#80 <p o>> ~ #1
      & <#70 <b p>> ~ #3
      & <#10 <q n>> ~ #3
      & <#80 <r q>> ~ #1
      & <#70 <e r>> ~ #3
      & <#10 <s k>> ~ #3
      & <#80 <t s>> ~ #1
      & <#70 <f t>> ~ #3
      & <#178 <u i>> ~ #1
      & <v <#20 u>> ~ #2
      & <c <w v>> ~ #3
      & <#178 <x w>> ~ #1
      & <y <#20 x>> ~ #2
      & <#10 <z y>> ~ #3
      & <#80 <ab z>> ~ #1
      & <#70 <g ab>> ~ #3",
  );
  let (rnet, net) = net.normalize(256);

  assert_debug_snapshot!(total_rewrites(&rnet), @"86");
  assert_snapshot!(show_net(&net), @"#2138224");
}
