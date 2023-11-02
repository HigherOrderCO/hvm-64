use hvmc::ast::show_net;
use insta::{assert_debug_snapshot, assert_snapshot};
use loaders::*;

mod loaders;

#[test]
fn test_era_era() {
  let net = parse_core("@main = * & * ~ *");
  let (rnet, net) = normal(net, 16);
  assert_snapshot!(show_net(&net), @"*");
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
  let book = load_lang("church_mul.hvm");
  let (term, defs, info) = hvm_lang_normal(book, 64);

  assert_snapshot!(show_net(&info.net), @"({2 ({2 b c} d) {3 (d e) (e {2 c f})}} (b f))");
  assert_snapshot!(term.to_string(&defs), @"λa λb (a (a (a (a (a (a b))))))");
  assert_debug_snapshot!(info.stats.rewrites.total_rewrites(), @"12");
}

#[test]
fn test_neg_fusion() {
  let book = load_lang("neg_fusion.hvm");
  let (term, defs, info) = hvm_lang_normal(book, 512);

  assert_snapshot!(show_net(&info.net), @"(b (* b))");
  assert_snapshot!(term.to_string(&defs), @"λa λ* a");
  assert_debug_snapshot!(info.stats.rewrites.total_rewrites(), @"153");
}

#[test]
fn test_tree_alloc() {
  let book = load_lang("tree_alloc.hvm");
  let (term, defs, info) = hvm_lang_normal(book, 512);

  assert_snapshot!(show_net(&info.net), @"(b (* b))");
  assert_snapshot!(term.to_string(&defs), @"λa λ* a");
  assert_debug_snapshot!(info.stats.rewrites.total_rewrites(), @"104");
}

#[test]
fn test_queue() {
  let book = load_lang("queue.hvm");
  let (term, defs, info) = hvm_lang_normal(book, 512);

  assert_snapshot!(show_net(&info.net), @"((#1 (((#2 (((#3 ((* @7) b)) (* b)) c)) (* c)) d)) (* d))");
  assert_snapshot!(term.to_string(&defs), @"λa λ* ((a 1) λb λ* ((b 2) λc λ* ((c 3) λ* λd d)))");
  assert_debug_snapshot!(info.stats.rewrites.total_rewrites(), @"62");
}
