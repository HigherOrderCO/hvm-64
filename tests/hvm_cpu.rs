use hvm_lang::term::{parser, DefNames, DefinitionBook, Term};
use hvmc::{ast::*, run::NumericOp, *};
use insta::{assert_debug_snapshot, assert_snapshot};
use std::fs;

fn load_file(file: &str) -> String {
  let path = format!("{}/tests/programs/{}", env!("CARGO_MANIFEST_DIR"), file);
  fs::read_to_string(path).unwrap()
}

// Parses code and generate Book from hvm-core syntax
fn parse_core(code: &str) -> Book {
  do_parse_book(code)
}

// Parses code and generate DefinitionBook from hvm-lang syntax
fn parse_lang(code: &str) -> DefinitionBook {
  parser::parse_definition_book(code).unwrap()
}

// Loads file and generate DefinitionBook from hvm-lang syntax
fn load_lang(file: &str) -> DefinitionBook {
  let code = load_file(file);
  parse_lang(&code)
}

// For every pair in the map, replaces all matches of a string the other string
fn replace_template(mut code: String, map: &[(&str, &str)]) -> String {
  for (from, to) in map {
    code = code.replace(from, to);
  }
  code
}

trait Normal {
  type Result;
  fn normal(self, size: usize) -> Self::Result;
}

impl Normal for Book {
  type Result = (run::Net, Net);

  fn normal(self, size: usize) -> Self::Result {
    let book = book_to_runtime(&self);
    let mut rnet = run::Net::new(size);
    rnet.boot(name_to_val("main"));
    rnet.normal(&book);

    let net = net_from_runtime(&rnet);
    (rnet, net)
  }
}

impl Normal for DefinitionBook {
  type Result = (Term, DefNames, hvm_lang::RunInfo);

  fn normal(self, size: usize) -> Self::Result {
    hvm_lang::run_book(self, size).unwrap()
  }
}

#[test]
fn test_era_era() {
  let net = parse_core("@main = * & * ~ *");
  let (rnet, net) = net.normal(16);
  assert_snapshot!(show_net(&net), @"*");
  assert_debug_snapshot!(rnet.rewrites(), @"2");
}

#[test]
fn test_con_dup() {
  let net = parse_core("@main = root & (x x) ~ {2 * root}");
  let (rnet, net) = net.normal(16);
  assert_snapshot!(show_net(&net), @"(b b)");
  assert_debug_snapshot!(rnet.rewrites(), @"5");
}

#[test]
fn test_bool_and() {
  let book = parse_core(
    "
    @true = (b (c b))
    @fals = (b (c c))
    @and  = ((b (@fals c)) (b c))
    @main = root & @and ~ (@true (@fals root))
  ",
  );
  let (rnet, net) = book.normal(64);

  assert_snapshot!(show_net(&net), @"(b (c c))");
  assert_debug_snapshot!(rnet.rewrites(), @"8");
}

#[test]
fn test_church_mul() {
  let book = load_lang("church_mul.hvm");
  let (term, defs, info) = book.normal(64);

  assert_snapshot!(show_net(&info.net), @"({2 ({2 b c} d) {3 (d e) (e {2 c f})}} (b f))");
  assert_snapshot!(term.to_string(&defs), @"λa λb (a (a (a (a (a (a b))))))");
}

#[test]
fn test_neg_fusion() {
  let book = load_lang("neg_fusion.hvm");
  let (_, _, info) = book.normal(512);

  assert_snapshot!(show_net(&info.net), @"(b (* b))");
  assert_debug_snapshot!(info.stats.rewrites.total_rewrites(), @"153");
}

#[test]
fn test_tree_alloc() {
  let book = load_lang("tree_alloc.hvm");
  let (_, _, info) = book.normal(512);

  assert_snapshot!(show_net(&info.net), @"(b (* b))");
  assert_debug_snapshot!(info.stats.rewrites.total_rewrites(), @"104");
}

fn make_queue(len: u32) -> DefinitionBook {
  let template = load_file("queue.hvm");
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

  body += &format!(" Nil{}", ")".repeat(len as usize * 2));

  let code = replace_template(template, &[("{main_body}", &body)]);
  parse_lang(&code)
}

#[test]
fn test_queues() {
  let info = [
    make_queue(3).normal(512),
    make_queue(4).normal(512),
    make_queue(5).normal(512),
    make_queue(10).normal(512),
    make_queue(20).normal(512),
  ]
  .map(|(term, defs, info)| (term, defs, info.net, info.stats.rewrites.total_rewrites()));

  assert_debug_snapshot!(info[0].3, @"62");
  assert_debug_snapshot!(info[1].3, @"81");
  assert_debug_snapshot!(info[2].3, @"100");
  assert_debug_snapshot!(info[3].3, @"195");
  assert_debug_snapshot!(info[4].3, @"385");

  let (term, defs, net, _) = &info[0];
  assert_snapshot!(show_net(net), @"((#1 (((#2 (((#3 ((* @7) b)) (* b)) c)) (* c)) d)) (* d))");
  assert_snapshot!(term.to_string(defs), @"λa λ* ((a 1) λb λ* ((b 2) λc λ* ((c 3) λ* λd d)))");
}

fn list_got(index: u32) -> DefinitionBook {
  let template = load_file("list_put_get.hvm");
  let code = replace_template(template, &[("{fun}", "got"), ("{args}", &index.to_string())]);
  parse_lang(&code)
}

#[test]
fn test_list_got() {
  let rwts = [
    list_got(0).normal(2048),
    list_got(1).normal(2048),
    list_got(3).normal(2048),
    list_got(7).normal(2048),
    list_got(15).normal(2048),
    list_got(31).normal(2048),
  ]
  .map(|(_, _, info)| info.stats.rewrites.total_rewrites());

  assert_debug_snapshot!(rwts[0], @"573");
  assert_debug_snapshot!(rwts[1], @"595");
  assert_debug_snapshot!(rwts[2], @"639");
  assert_debug_snapshot!(rwts[3], @"727");
  assert_debug_snapshot!(rwts[4], @"903");
  assert_debug_snapshot!(rwts[5], @"1255");

  //Tests the linearity of the function
  let delta = rwts[1] - rwts[0];
  assert_eq!(rwts[1] + delta * 2, rwts[2]);
  assert_eq!(rwts[2] + delta * 4, rwts[3]);
  assert_eq!(rwts[3] + delta * 8, rwts[4]);
  assert_eq!(rwts[4] + delta * 16, rwts[5]);
}

fn list_put(index: u32, value: u32) -> DefinitionBook {
  let code = load_file("list_put_get.hvm");
  let list = code.replace("{fun}", "put").replace("{args}", &format!("{index} {value}"));
  parse_lang(&list)
}

#[test]
fn test_list_put() {
  let rwts = [
    list_put(0, 2).normal(2048),
    list_put(1, 4).normal(2048),
    list_put(3, 8).normal(2048),
    list_put(7, 16).normal(2048),
    list_put(15, 32).normal(2048),
    list_put(31, 64).normal(2048),
  ]
  .map(|(_, _, info)| info.stats.rewrites.total_rewrites());

  assert_debug_snapshot!(rwts[0], @"563");
  assert_debug_snapshot!(rwts[1], @"586");
  assert_debug_snapshot!(rwts[2], @"632");
  assert_debug_snapshot!(rwts[3], @"724");
  assert_debug_snapshot!(rwts[4], @"908");
  assert_debug_snapshot!(rwts[5], @"1276");

  //Tests the linearity of the function
  let delta = rwts[1] - rwts[0];
  assert_eq!(rwts[1] + delta * 2, rwts[2]);
  assert_eq!(rwts[2] + delta * 4, rwts[3]);
  assert_eq!(rwts[3] + delta * 8, rwts[4]);
  assert_eq!(rwts[4] + delta * 16, rwts[5]);
}

// Numeric Operations test

fn op_net(lnum: u32, op: NumericOp, rnum: u32) -> Book {
  parse_core(&format!("@main = root & <#{lnum} <#{rnum} root>> ~ #{op}"))
}

#[test]
fn test_add() {
  let net = op_net(10, run::ADD, 2);
  let (rnet, net) = net.normal(16);
  assert_snapshot!(show_net(&net), @"#12");
  assert_debug_snapshot!(rnet.rewrites(), @"5");
}

#[test]
fn test_sub() {
  let net = op_net(10, run::SUB, 2);
  let (_, net) = net.normal(16);
  assert_snapshot!(show_net(&net), @"#8");
}

#[test]
fn test_mul() {
  let net = op_net(10, run::MUL, 2);
  let (_, net) = net.normal(16);
  assert_snapshot!(show_net(&net), @"#20");
}

#[test]
fn test_div() {
  let net = op_net(10, run::DIV, 2);
  let (_, net) = net.normal(16);
  assert_snapshot!(show_net(&net), @"#5");
}

#[test]
fn test_mod() {
  let net = op_net(10, run::MOD, 2);
  let (_, net) = net.normal(16);
  assert_snapshot!(show_net(&net), @"#0");
}

#[test]
fn test_eq() {
  let net = op_net(10, run::EQ, 2);
  let (_, net) = net.normal(16);
  assert_snapshot!(show_net(&net), @"#0");
}

#[test]
fn test_ne() {
  let net = op_net(10, run::NE, 2);
  let (_, net) = net.normal(16);
  assert_snapshot!(show_net(&net), @"#1");
}

#[test]
fn test_lt() {
  let net = op_net(10, run::LT, 2);
  let (_, net) = net.normal(16);
  assert_snapshot!(show_net(&net), @"#0");
}

#[test]
fn test_gt() {
  let net = op_net(10, run::GT, 2);
  let (_, net) = net.normal(16);
  assert_snapshot!(show_net(&net), @"#1");
}

#[test]
fn test_and() {
  let net = op_net(10, run::AND, 2);
  let (_, net) = net.normal(16);
  assert_snapshot!(show_net(&net), @"#2");
}

#[test]
fn test_or() {
  let net = op_net(10, run::OR, 2);
  let (_, net) = net.normal(16);
  assert_snapshot!(show_net(&net), @"#10");
}

#[test]
fn test_xor() {
  let net = op_net(10, run::XOR, 2);
  let (_, net) = net.normal(16);
  assert_snapshot!(show_net(&net), @"#8");
}

#[test]
fn test_not() {
  let net = op_net(0, run::NOT, 256);
  let (rnet, net) = net.normal(16);
  assert_snapshot!(show_net(&net), @"#16776959");
  assert_debug_snapshot!(rnet.rewrites(), @"4");
}

#[test]
fn test_lsh() {
  let net = op_net(10, run::LSH, 2);
  let (_, net) = net.normal(16);
  assert_snapshot!(show_net(&net), @"#40");
}

#[test]
fn test_rsh() {
  let net = op_net(10, run::RSH, 2);
  let (_, net) = net.normal(16);
  assert_snapshot!(show_net(&net), @"#2");
}

#[test]
/// Division by zero always return the value of 0xFFFFFF,
/// that is read as the unsigned integer `16777215`
fn test_div_by_0() {
  let net = op_net(9, run::DIV, 0);
  let (rnet, net) = net.normal(16);
  assert_snapshot!(show_net(&net), @"#16777215");
  assert_debug_snapshot!(rnet.rewrites(), @"5");
}

#[test]
fn test_chained_ops() {
  let net = parse_core(
    "@main = a
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
  let (rnet, net) = net.normal(256);

  assert_debug_snapshot!(rnet.rewrites(), @"87");
  assert_snapshot!(show_net(&net), @"#2138224");
}
