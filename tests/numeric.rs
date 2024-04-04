use hvmc::{
  ast::{Book, Net},
  ops::IntOp,
};
use insta::{assert_debug_snapshot, assert_snapshot};
use loaders::*;

mod loaders;

fn op_net_u32(lnum: u32, op: IntOp, rnum: u32) -> Book {
  let code = format!("@main = root & #{lnum} ~ <u32.{op} #{rnum} root>");
  println!("Code: {code:?}");
  parse_core(&code)
}

use IntOp::*;

#[test]
fn test_add() {
  let net = op_net_u32(10, Add, 2);
  let (rwts, net) = normal(net, Some(128));
  assert_snapshot!(Net::to_string(&net), @"#12");
  assert_debug_snapshot!(rwts.total(), @"3");
}

#[test]
fn test_sub() {
  let net = op_net_u32(10, Sub, 2);
  let (_rwts, net) = normal(net, Some(128));
  assert_snapshot!(Net::to_string(&net), @"#8");
}

#[test]
fn test_mul() {
  let net = op_net_u32(10, Mul, 2);
  let (_rwts, net) = normal(net, Some(128));
  assert_snapshot!(Net::to_string(&net), @"#20");
}

#[test]
fn test_div() {
  let net = op_net_u32(10, Div, 2);
  let (_rwts, net) = normal(net, Some(128));
  assert_snapshot!(Net::to_string(&net), @"#5");
}

#[test]
fn test_rem() {
  let net = op_net_u32(10, Rem, 2);
  let (_rwts, net) = normal(net, Some(128));
  assert_snapshot!(Net::to_string(&net), @"#0");
}

#[test]
fn test_eq() {
  let net = op_net_u32(10, Eq, 2);
  let (_rwts, net) = normal(net, Some(128));
  assert_snapshot!(Net::to_string(&net), @"#0");
}

#[test]
fn test_ne() {
  let net = op_net_u32(10, Ne, 2);
  let (_rwts, net) = normal(net, Some(128));
  assert_snapshot!(Net::to_string(&net), @"#1");
}

#[test]
fn test_lt() {
  let net = op_net_u32(10, Lt, 2);
  let (_rwts, net) = normal(net, Some(128));
  assert_snapshot!(Net::to_string(&net), @"#0");
}

#[test]
fn test_gt() {
  let net = op_net_u32(10, Gt, 2);
  let (_rwts, net) = normal(net, Some(128));
  assert_snapshot!(Net::to_string(&net), @"#1");
}

#[test]
fn test_and() {
  let net = op_net_u32(10, And, 2);
  let (_rwts, net) = normal(net, Some(128));
  assert_snapshot!(Net::to_string(&net), @"#2");
}

#[test]
fn test_or() {
  let net = op_net_u32(10, Or, 2);
  let (_rwts, net) = normal(net, Some(128));
  assert_snapshot!(Net::to_string(&net), @"#10");
}

#[test]
fn test_xor() {
  let net = op_net_u32(10, Xor, 2);
  let (_rwts, net) = normal(net, Some(128));
  assert_snapshot!(Net::to_string(&net), @"#8");
}

#[test]
fn test_lsh() {
  let net = op_net_u32(10, Shl, 2);
  let (_rwts, net) = normal(net, Some(128));
  assert_snapshot!(Net::to_string(&net), @"#40");
}

#[test]
fn test_rsh() {
  let net = op_net_u32(10, Shr, 2);
  let (_rwts, net) = normal(net, Some(128));
  assert_snapshot!(Net::to_string(&net), @"#2");
}

#[test]
fn test_div_by_0() {
  let net = op_net_u32(9, Div, 0);
  let (rwts, net) = normal(net, Some(128));
  assert_snapshot!(Net::to_string(&net), @"#0");
  assert_debug_snapshot!(rwts.total(), @"3");
}
