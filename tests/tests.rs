use hvmc::ast::show_net;
use insta::{assert_debug_snapshot, assert_snapshot};
use loaders::*;

mod loaders;

#[test]
fn test_era_era() {
  let net = parse_core("@main = * & * ~ *");
  let (rnet, net) = normal(net, 16);
  assert_snapshot!(show_net(&net), @"*");
  assert_debug_snapshot!(rnet.get_rewrites().total(), @"2");
}

#[test]
fn test_era_era2() {
  let net = parse_core("@main = (* *) & * ~ *");
  let (rnet, net) = normal(net, 16);
  assert_snapshot!(show_net(&net), @"(* *)");
  assert_debug_snapshot!(rnet.get_rewrites().total(), @"2");
}

#[test]
fn test_commutation() {
  let net = parse_core("@main = root & (x x) ~ [* root]");
  let (rnet, net) = normal(net, 16);
  assert_snapshot!(show_net(&net), @"(a a)"); 
  assert_debug_snapshot!(rnet.get_rewrites().total(), @"5");
}

#[test]
fn test_bool_and() {
  let book = parse_core(
    "
    @true = (a (* a))
    @fals = (* (a a))
    @and  = ((a (@fals b)) (a b))
    @main = root & @and ~ (@true (@fals root))
  ",
  );
  let (rnet, net) = normal(book, 64);

  assert_snapshot!(show_net(&net), @"(* (a a))");
  assert_debug_snapshot!(rnet.get_rewrites().total(), @"9");
}

#[test]
fn test_church_mul() {
  let mut book = load_lang("church_mul.hvm");
  let (rnet, net, id_map) = hvm_lang_normal(&mut book, 64);
  let (readback, valid_readback) = hvm_lang_readback(&net, &book, id_map);

  assert!(valid_readback);
  assert_snapshot!(show_net(&net), @"({5 (a {3 b c}) {7 (d a) ({3 c e} d)}} (e b))");
  assert_snapshot!(readback, @"λa λb (a (a (a (a (a (a b))))))");
  assert_debug_snapshot!(rnet.get_rewrites().total(), @"12");
}

#[test]
fn test_tree_alloc() {
  let mut book = load_lang("tree_alloc.hvm");
  let (rnet, net, id_map) = hvm_lang_normal(&mut book, 512);
  let (readback, valid_readback) = hvm_lang_readback(&net, &book, id_map);

  assert!(valid_readback);
  assert_snapshot!(show_net(&net), @"(a (* a))");
  assert_snapshot!(readback, @"λa λ* a");
  assert_debug_snapshot!(rnet.get_rewrites().total(), @"100");
}

#[test]
fn test_queue() {
  // TODO: Is this file/readback correct?
  let mut book = load_lang("queue.hvm");
  let (rnet, net, id_map) = hvm_lang_normal(&mut book, 512);
  let (readback, valid_readback) = hvm_lang_readback(&net, &book, id_map);

  assert!(valid_readback);
  assert_snapshot!(show_net(&net), @"(((* (a a)) (((((b c) (b c)) (((({3 (d e) (f d)} (f e)) ((* (g g)) h)) (* h)) i)) (* i)) j)) (* j))");
  assert_snapshot!(readback, @"λa λ* (a λ* λb b λc λ* (c λd λe (d e) λf λ* (f λg λh (g (g h)) λ* λi i)))");
  assert_debug_snapshot!(rnet.get_rewrites().total(), @"59");
}
