use hvmc::ast::Net;
use insta::{assert_debug_snapshot, assert_snapshot};
use loaders::*;

mod loaders;

#[test]
fn test_era_era() {
  let net = parse_core("@main = * & * ~ *");
  let (rwts, net) = normal(net, 16);
  assert_snapshot!(Net::to_string(&net), @"*");
  assert_debug_snapshot!(rwts.total(), @"2");
}

#[test]
fn test_era_era2() {
  let net = parse_core("@main = (* *) & * ~ *");
  let (rwts, net) = normal(net, 16);
  assert_snapshot!(Net::to_string(&net), @"(* *)");
  assert_debug_snapshot!(rwts.total(), @"2");
}

#[test]
fn test_commutation() {
  let net = parse_core("@main = root & (x x) ~ [* root]");
  let (rwts, net) = normal(net, 16);
  assert_snapshot!(Net::to_string(&net), @"(a a)");
  assert_debug_snapshot!(rwts.total(), @"5");
}

#[test]
fn test_bool_and() {
  let book = parse_core(
    "
    @true = (b (* b))
    @fals = (* (b b))
    @and  = ((b (@fals c)) (b c))
    @main = root & @and ~ (@true (@fals root))
  ",
  );
  let (rwts, net) = normal(book, 64);

  assert_snapshot!(Net::to_string(&net), @"(* (a a))");
  assert_debug_snapshot!(rwts.total(), @"9");
}

#[test]
fn test_church_mul() {
  let mut book = load_lang("church_mul.hvm");
  let (rwts, net) = hvm_lang_normal(&mut book, 64);
  let (readback, valid_readback) = hvm_lang_readback(&net, &book);

  assert!(valid_readback);
  assert_snapshot!(Net::to_string(&net), @"({5 (a {3 b c}) {7 (d a) ({3 c e} d)}} (e b))");
  assert_snapshot!(readback, @"λa λb let #0{c g} = (let #1{d h} = a; d let #2{e f} = h; (e (f #0{g b}))); c");
  assert_debug_snapshot!(rwts.total(), @"12");
}

#[test]
fn test_tree_alloc() {
  let mut book = load_lang("tree_alloc.hvm");
  println!("{:?}", book);
  println!("{:?}", "Hi");
  let (rwts, net) = hvm_lang_normal(&mut book, 512);
  let (readback, valid_readback) = hvm_lang_readback(&net, &book);

  assert!(valid_readback);
  assert_snapshot!(Net::to_string(&net), @"(a (* a))");
  assert_snapshot!(readback, @"λa λ* a");
  assert_debug_snapshot!(rwts.total(), @"99");
}

#[test]
fn test_queue() {
  let mut book = load_lang("queue.hvm");
  let (rwts, net) = hvm_lang_normal(&mut book, 512);
  let (readback, valid_readback) = hvm_lang_readback(&net, &book);

  assert!(valid_readback);
  assert_snapshot!(Net::to_string(&net), @"(((* (a a)) (((((b c) (b c)) (((({3 (d e) (f d)} (f e)) ((* (g g)) h)) (* h)) i)) (* i)) j)) (* j))");
  assert_snapshot!(readback, @"λa λ* (a λ* λb b λc λ* (c λd λe (d e) λf λ* (f λg λh let #0{i j} = g; (i (j h)) λ* λk k)))");
  assert_debug_snapshot!(rwts.total(), @"59");
}
