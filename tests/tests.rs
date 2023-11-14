use hvmc::ast::show_net;
use insta::{assert_debug_snapshot, assert_snapshot};
use loaders::*;

mod loaders;

#[test]
#[cfg(not(feature = "cuda"))] // FIXME: gpu runtime errors on nets with `*` on the root
fn test_era_era() {
  let net = parse_core("@main = * & * ~ *");
  let (rnet, net) = normal(net, 16);
  assert_snapshot!(show_net(&net), @"*");
  assert_debug_snapshot!(rnet.rewrites(), @"2");
}

#[test]
fn test_era_era2() {
  let net = parse_core("@main = (* *) & * ~ *");
  let (rnet, net) = normal(net, 16);
  assert_snapshot!(show_net(&net), @"(* *)");
  assert_debug_snapshot!(rnet.rewrites(), @"2");
}

#[test]
fn test_commutation() {
  let net = parse_core("@main = root & (x x) ~ [* root]");
  let (rnet, net) = normal(net, 16);
  assert_snapshot!(show_net(&net), @"(a a)");
  assert_debug_snapshot!(rnet.rewrites(), @"5");
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
  assert_debug_snapshot!(rnet.rewrites(), @"9");
}

#[test]
fn test_church_mul() {
  let mut book = load_lang("church_mul.hvm");
  let (rnet, net, id_map) = hvm_lang_normal(&mut book, 64);
  let (readback, valid_readback) = hvm_lang_readback(&net, &book, id_map);

  assert!(valid_readback);
  assert_snapshot!(show_net(&net), @"({2 ({2 a b} c) {3 (c d) (d {2 b e})}} (a e))");
  assert_snapshot!(readback, @"λa λb (a (a (a (a (a (a b))))))");
  assert_debug_snapshot!(rnet.rewrites(), @"12");
}

#[test]
fn test_tree_alloc() {
  let mut book = load_lang("tree_alloc.hvm");
  let (rnet, net, id_map) = hvm_lang_normal(&mut book, 512);
  let (readback, valid_readback) = hvm_lang_readback(&net, &book, id_map);

  assert!(valid_readback);
  assert_snapshot!(show_net(&net), @"(a (* a))");
  assert_snapshot!(readback, @"λa λ* a");
  assert_debug_snapshot!(rnet.rewrites(), @"104");
}

#[test]
fn test_queue() {
  let mut book = load_lang("queue.hvm");
  let (rnet, net, id_map) = hvm_lang_normal(&mut book, 512);
  let (readback, valid_readback) = hvm_lang_readback(&net, &book, id_map);

  assert!(valid_readback);
  assert_snapshot!(show_net(&net), @"(((* @B) (((((a b) (a b)) (((({2 (c d) (d e)} (c e)) ((* @A) f)) (* f)) g)) (* g)) h)) (* h))");
  assert_snapshot!(readback, @"λa λ* ((a λ* λb b) λc λ* ((c λd λe (d e)) λf λ* ((f λg λh (g (g h))) λ* λi i)))");
  assert_debug_snapshot!(rnet.rewrites(), @"65");
}
