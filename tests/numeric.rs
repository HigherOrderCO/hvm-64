use hvmc::{
  ast::{Book, Net},
  ops::Op,
};
use insta::{assert_debug_snapshot, assert_snapshot};
use loaders::*;

mod loaders;

fn op_net(lnum: u32, op: Op, rnum: u32) -> Book {
  let code = format!("@main = root & #{lnum} ~ <{op} #{rnum} root>");
  println!("Code: {code:?}");
  parse_core(&code)
}

#[test]
fn test_add() {
  let net = op_net(10, Op::Add, 2);
  let (rwts, net) = normal(net, 16);
  assert_snapshot!(Net::to_string(&net), @"#12");
  assert_debug_snapshot!(rwts.total(), @"3");
}

#[test]
fn test_sub() {
  let net = op_net(10, Op::Sub, 2);
  let (_rwts, net) = normal(net, 16);
  assert_snapshot!(Net::to_string(&net), @"#8");
}

#[test]
fn test_mul() {
  let net = op_net(10, Op::Mul, 2);
  let (_rwts, net) = normal(net, 16);
  assert_snapshot!(Net::to_string(&net), @"#20");
}

#[test]
fn test_div() {
  let net = op_net(10, Op::Div, 2);
  let (_rwts, net) = normal(net, 16);
  assert_snapshot!(Net::to_string(&net), @"#5");
}

#[test]
fn test_mod() {
  let net = op_net(10, Op::Mod, 2);
  let (_rwts, net) = normal(net, 16);
  assert_snapshot!(Net::to_string(&net), @"#0");
}

#[test]
fn test_eq() {
  let net = op_net(10, Op::Eq, 2);
  let (_rwts, net) = normal(net, 16);
  assert_snapshot!(Net::to_string(&net), @"#0");
}

#[test]
fn test_ne() {
  let net = op_net(10, Op::Ne, 2);
  let (_rwts, net) = normal(net, 16);
  assert_snapshot!(Net::to_string(&net), @"#1");
}

#[test]
fn test_lt() {
  let net = op_net(10, Op::Lt, 2);
  let (_rwts, net) = normal(net, 16);
  assert_snapshot!(Net::to_string(&net), @"#0");
}

#[test]
fn test_gt() {
  let net = op_net(10, Op::Gt, 2);
  let (_rwts, net) = normal(net, 16);
  assert_snapshot!(Net::to_string(&net), @"#1");
}

#[test]
fn test_and() {
  let net = op_net(10, Op::And, 2);
  let (_rwts, net) = normal(net, 16);
  assert_snapshot!(Net::to_string(&net), @"#2");
}

#[test]
fn test_or() {
  let net = op_net(10, Op::Or, 2);
  let (_rwts, net) = normal(net, 16);
  assert_snapshot!(Net::to_string(&net), @"#10");
}

#[test]
fn test_xor() {
  let net = op_net(10, Op::Xor, 2);
  let (_rwts, net) = normal(net, 16);
  assert_snapshot!(Net::to_string(&net), @"#8");
}

#[test]
fn test_not() {
  let net = op_net(0, Op::Not, 256);
  let (rwts, net) = normal(net, 16);
  assert_snapshot!(Net::to_string(&net), @"#1152921504606846975");
  assert_debug_snapshot!(rwts.total(), @"3");
}

#[test]
fn test_lsh() {
  let net = op_net(10, Op::Lsh, 2);
  let (_rwts, net) = normal(net, 16);
  assert_snapshot!(Net::to_string(&net), @"#40");
}

#[test]
fn test_rsh() {
  let net = op_net(10, Op::Rsh, 2);
  let (_rwts, net) = normal(net, 16);
  assert_snapshot!(Net::to_string(&net), @"#2");
}

#[test]
fn test_div_by_0() {
  let net = op_net(9, Op::Div, 0);
  let (rwts, net) = normal(net, 16);
  assert_snapshot!(Net::to_string(&net), @"#0");
  assert_debug_snapshot!(rwts.total(), @"3");
}
