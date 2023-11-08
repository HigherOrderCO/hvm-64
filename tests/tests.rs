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
  assert_snapshot!(show_net(&net), @"(b b)");
  assert_debug_snapshot!(rnet.rewrites(), @"5");
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
  let (rnet, net) = normal(book, 64);

  assert_snapshot!(show_net(&net), @"(* (b b))");
  assert_debug_snapshot!(rnet.rewrites(), @"9");
}

#[test]
fn test_church_mul() {
  let mut book = load_lang("church_mul.hvm");
  let (rnet, net, id_map) = hvm_lang_normal(&mut book, 64);
  let readback = hvm_lang_readback(&net, &book, id_map);

  assert_snapshot!(show_net(&net), @"({2 ({2 b c} d) {3 (d e) (e {2 c f})}} (b f))");
  assert_snapshot!(readback, @"λa λb (a (a (a (a (a (a b))))))");
  assert_debug_snapshot!(rnet.rewrites(), @"12");
}

#[test]
fn test_neg_fusion() {
  let mut book = load_lang("neg_fusion.hvm");
  let (rnet, net, id_map) = hvm_lang_normal(&mut book, 512);
  let readback = hvm_lang_readback(&net, &book, id_map);

  assert_snapshot!(show_net(&net), @"(b (* b))");
  assert_snapshot!(readback, @"λa λ* a");

  // TODO: investigate why this difference exists
  if cfg!(feature = "cuda") {
    assert_debug_snapshot!(rnet.rewrites(), @"160");
  } else {
    assert_debug_snapshot!(rnet.rewrites(), @"153");
  }
}

#[test]
fn test_tree_alloc() {
  let mut book = load_lang("tree_alloc.hvm");
  let (rnet, net, id_map) = hvm_lang_normal(&mut book, 512);
  let readback = hvm_lang_readback(&net, &book, id_map);

  assert_snapshot!(show_net(&net), @"(b (* b))");
  assert_snapshot!(readback, @"λa λ* a");
  assert_debug_snapshot!(rnet.rewrites(), @"104");
}

#[test]
fn test_queue() {
  let mut book = load_lang("queue.hvm");
  let (rnet, net, id_map) = hvm_lang_normal(&mut book, 512);
  let readback = hvm_lang_readback(&net, &book, id_map);

  assert_snapshot!(show_net(&net), @"(((* @B) (((((b c) (b c)) (((({2 (d e) (e f)} (d f)) ((* @A) g)) (* g)) h)) (* h)) i)) (* i))");
  assert_snapshot!(readback, @"λa λ* ((a λ* λb b) λc λ* ((c λd λe (d e)) λf λ* ((f λg λh (g (g h))) λ* λi i)))");
  assert_debug_snapshot!(rnet.rewrites(), @"65");
}
