use hvmc::ast::show_net;
use insta::{assert_debug_snapshot, assert_snapshot};
use loaders::*;

mod loaders;

pub fn load_bench(file: &str) -> String {
  let path = format!("{}/benches/programs/{}", env!("CARGO_MANIFEST_DIR"), file);
  std::fs::read_to_string(path).unwrap()
}

// Loads file and generate Book from hvm-core syntax
fn load_core(file: &str) -> hvmc::ast::Book {
  let code = load_bench(file);
  parse_core(&code)
}

// Loads file and generate DefinitionBook from hvm-lang syntax
pub fn load_lang(file: &str) -> hvml::term::Book {
  let code = load_bench(file);
  parse_lang(&code)
}

#[test]
fn dec_bits() {
  let book = load_core("binary-counter/dec_bits.hvmc");
  let (rnet, net) = normal(book, 1 << 16);

  assert_snapshot!(show_net(&net), @"(* (* (a a)))");
  assert_debug_snapshot!(rnet.get_rewrites().total(), @"180113");
}

#[test]
fn dec_bits_tree() {
  let book = load_core("binary-counter/dec_bits_tree.hvmc");
  let (rnet, net) = normal(book, 1 << 13);

  assert_snapshot!(show_net(&net), @"(((((((* (* (a a))) (* (* (b b)))) ((* (* (c c))) (* (* (d d))))) (((* (* (e e))) (* (* (f f)))) ((* (* (g g))) (* (* (h h)))))) ((((* (* (i i))) (* (* (j j)))) ((* (* (k k))) (* (* (l l))))) (((* (* (m m))) (* (* (n n)))) ((* (* (o o))) (* (* (p p))))))) (((((* (* (q q))) (* (* (r r)))) ((* (* (s s))) (* (* (t t))))) (((* (* (u u))) (* (* (v v)))) ((* (* (w w))) (* (* (x x)))))) ((((* (* (y y))) (* (* (z z)))) ((* (* (aa aa))) (* (* (ab ab))))) (((* (* (ac ac))) (* (* (ad ad)))) ((* (* (ae ae))) (* (* (af af)))))))) ((((((* (* (ag ag))) (* (* (ah ah)))) ((* (* (ai ai))) (* (* (aj aj))))) (((* (* (ak ak))) (* (* (al al)))) ((* (* (am am))) (* (* (an an)))))) ((((* (* (ao ao))) (* (* (ap ap)))) ((* (* (aq aq))) (* (* (ar ar))))) (((* (* (as as))) (* (* (at at)))) ((* (* (au au))) (* (* (av av))))))) (((((* (* (aw aw))) (* (* (ax ax)))) ((* (* (ay ay))) (* (* (az az))))) (((* (* (ba ba))) (* (* (bb bb)))) ((* (* (bc bc))) (* (* (bd bd)))))) ((((* (* (be be))) (* (* (bf bf)))) ((* (* (bg bg))) (* (* (bh bh))))) (((* (* (bi bi))) (* (* (bj bj)))) ((* (* (bk bk))) (* (* (bl bl)))))))))");
  assert_debug_snapshot!(rnet.get_rewrites().total(), @"2878593");
}

#[test]
#[ignore] // FIXME: panics at src/run.rs::expand with `attempt to multiply with overflow``
fn test_church_exp() {
  let book = load_core("church/church_exp.hvmc");
  let (rnet, net) = normal(book, 1 << 12);

  assert_snapshot!(show_net(&net));
  assert_debug_snapshot!(rnet.get_rewrites().total(), @"1943");
}

#[test]
#[ignore] // FIXME: church numbers bigger than C_16 causes stack overflow
fn test_church_mul() {
  let mut book = load_lang("church/church_mul.hvm");
  let (rnet, net, id_map) = hvm_lang_normal(&mut book, 512);
  let (readback, valid_readback) = hvm_lang_readback(&net, &book, id_map);

  assert_debug_snapshot!(valid_readback, @"false"); // invalid because of dup labels 

  // TODO: investigate why this difference exists
  assert_snapshot!(show_net(&net), @"({2 ({2 b {3 c {4 d {5 e f}}}} g) {3 (g h) {4 (h i) {5 (i j) k}}}} (b l))");
  assert_snapshot!(readback, @"λa λb ((* {λc d {λd e {λe f {λf (a (a (a (a {b {c {c {c c}}}})))) a}}}}) c)");
  assert_debug_snapshot!(rnet.get_rewrites().total(), @"17");
}

#[test]
fn alloc_big_tree() {
  let book = load_core("tree/alloc_big_tree.hvmc");
  let (rnet, net) = normal(book, 1 << 16);

  assert_snapshot!(show_net(&net)); // file snapshot
  assert_debug_snapshot!(rnet.get_rewrites().total(), @"28723");
}

#[test]
fn test_neg_fusion() {
  let mut book = load_lang("fusion/neg_fusion.hvm");
  let (rnet, net, id_map) = hvm_lang_normal(&mut book, 512);
  let (readback, valid_readback) = hvm_lang_readback(&net, &book, id_map);

  assert!(valid_readback);
  assert_snapshot!(show_net(&net), @"(a (* a))");
  assert_snapshot!(readback, @"λa λ* a");

  assert_debug_snapshot!(rnet.get_rewrites().total(), @"148");
}
