use hvmc::{
  ast::{show_net, Book},
  run,
  run::NumericOp,
};
use insta::{assert_debug_snapshot, assert_snapshot};
use loaders::*;

mod loaders;

fn op_net(lnum: u32, op: NumericOp, rnum: u32) -> Book {
  parse_core(&format!("@main = root & <#{lnum} <#{rnum} root>> ~ #{op}"))
}

#[test]
fn test_add() {
  let net = op_net(10, run::ADD, 2);
  let (rnet, net) = normal(net, 16);
  assert_snapshot!(show_net(&net), @"#12");
  assert_debug_snapshot!(rnet.rewrites(), @"5");
}

#[test]
fn test_sub() {
  let net = op_net(10, run::SUB, 2);
  let (_, net) = normal(net, 16);
  assert_snapshot!(show_net(&net), @"#8");
}

#[test]
fn test_mul() {
  let net = op_net(10, run::MUL, 2);
  let (_, net) = normal(net, 16);
  assert_snapshot!(show_net(&net), @"#20");
}

#[test]
fn test_div() {
  let net = op_net(10, run::DIV, 2);
  let (_, net) = normal(net, 16);
  assert_snapshot!(show_net(&net), @"#5");
}

#[test]
fn test_mod() {
  let net = op_net(10, run::MOD, 2);
  let (_, net) = normal(net, 16);
  assert_snapshot!(show_net(&net), @"#0");
}

#[test]
fn test_eq() {
  let net = op_net(10, run::EQ, 2);
  let (_, net) = normal(net, 16);
  assert_snapshot!(show_net(&net), @"#0");
}

#[test]
fn test_ne() {
  let net = op_net(10, run::NE, 2);
  let (_, net) = normal(net, 16);
  assert_snapshot!(show_net(&net), @"#1");
}

#[test]
fn test_lt() {
  let net = op_net(10, run::LT, 2);
  let (_, net) = normal(net, 16);
  assert_snapshot!(show_net(&net), @"#0");
}

#[test]
fn test_gt() {
  let net = op_net(10, run::GT, 2);
  let (_, net) = normal(net, 16);
  assert_snapshot!(show_net(&net), @"#1");
}

#[test]
fn test_and() {
  let net = op_net(10, run::AND, 2);
  let (_, net) = normal(net, 16);
  assert_snapshot!(show_net(&net), @"#2");
}

#[test]
fn test_or() {
  let net = op_net(10, run::OR, 2);
  let (_, net) = normal(net, 16);
  assert_snapshot!(show_net(&net), @"#10");
}

#[test]
fn test_xor() {
  let net = op_net(10, run::XOR, 2);
  let (_, net) = normal(net, 16);
  assert_snapshot!(show_net(&net), @"#8");
}

#[test]
fn test_not() {
  let net = op_net(0, run::NOT, 256);
  let (rnet, net) = normal(net, 16);
  assert_snapshot!(show_net(&net), @"#16776959");
  assert_debug_snapshot!(rnet.rewrites(), @"4");
}

#[test]
fn test_lsh() {
  let net = op_net(10, run::LSH, 2);
  let (_, net) = normal(net, 16);
  assert_snapshot!(show_net(&net), @"#40");
}

#[test]
fn test_rsh() {
  let net = op_net(10, run::RSH, 2);
  let (_, net) = normal(net, 16);
  assert_snapshot!(show_net(&net), @"#2");
}

#[test]
/// Division by zero always return the value of 0xFFFFFF,
/// that is read as the unsigned integer `16777215`
fn test_div_by_0() {
  let net = op_net(9, run::DIV, 0);
  let (rnet, net) = normal(net, 16);
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
  let (rnet, net) = normal(net, 256);

  assert_debug_snapshot!(rnet.rewrites(), @"87");
  assert_snapshot!(show_net(&net), @"#2138224");
}
