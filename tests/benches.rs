use hvmc::ast::Net;
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
#[cfg(not(feature = "cuda"))] // FIXME: hangs indefinitely
fn dec_bits() {
  let book = load_core("binary-counter/dec_bits.hvmc");
  let (rwts, net) = normal(book, 1 << 18);

  assert_snapshot!(Net::to_string(&net), @"(* (* (a a)))");
  assert_debug_snapshot!(rwts.total(), @"81846");
}

#[test]
fn dec_bits_tree() {
  let book = load_core("binary-counter/dec_bits_tree.hvmc");
  let (rwts, net) = normal(book, 1 << 21);

  // FIXME: The refs to the rule `@E` are not expanded
  if cfg!(feature = "cuda") {
    assert_snapshot!(Net::to_string(&net), @"((((((@E @E) (@E @E)) ((@E @E) (@E @E))) (((@E @E) (@E @E)) ((@E @E) (@E @E)))) ((((@E @E) (@E @E)) ((@E @E) (@E @E))) (((@E @E) (@E @E)) ((@E @E) (@E @E))))) (((((@E @E) (@E @E)) ((@E @E) (@E @E))) (((@E @E) (@E @E)) ((@E @E) (@E @E)))) ((((@E @E) (@E @E)) ((@E @E) (@E @E))) (((@E @E) (@E @E)) ((@E @E) (@E @E))))))");
    assert_debug_snapshot!(rwts.total(), @"2878529");
  } else {
    assert_snapshot!(Net::to_string(&net), @"(((((((* (* (a a))) (* (* (b b)))) ((* (* (c c))) (* (* (d d))))) (((* (* (e e))) (* (* (f f)))) ((* (* (g g))) (* (* (h h)))))) ((((* (* (i i))) (* (* (j j)))) ((* (* (k k))) (* (* (l l))))) (((* (* (m m))) (* (* (n n)))) ((* (* (o o))) (* (* (p p))))))) (((((* (* (q q))) (* (* (r r)))) ((* (* (s s))) (* (* (t t))))) (((* (* (u u))) (* (* (v v)))) ((* (* (w w))) (* (* (x x)))))) ((((* (* (y y))) (* (* (z z)))) ((* (* (aa aa))) (* (* (ab ab))))) (((* (* (ac ac))) (* (* (ad ad)))) ((* (* (ae ae))) (* (* (af af)))))))) ((((((* (* (ag ag))) (* (* (ah ah)))) ((* (* (ai ai))) (* (* (aj aj))))) (((* (* (ak ak))) (* (* (al al)))) ((* (* (am am))) (* (* (an an)))))) ((((* (* (ao ao))) (* (* (ap ap)))) ((* (* (aq aq))) (* (* (ar ar))))) (((* (* (as as))) (* (* (at at)))) ((* (* (au au))) (* (* (av av))))))) (((((* (* (aw aw))) (* (* (ax ax)))) ((* (* (ay ay))) (* (* (az az))))) (((* (* (ba ba))) (* (* (bb bb)))) ((* (* (bc bc))) (* (* (bd bd)))))) ((((* (* (be be))) (* (* (bf bf)))) ((* (* (bg bg))) (* (* (bh bh))))) (((* (* (bi bi))) (* (* (bj bj)))) ((* (* (bk bk))) (* (* (bl bl)))))))))");
    assert_debug_snapshot!(rwts.total(), @"1307490");
  }
}

#[test]
#[cfg(not(feature = "cuda"))] // FIXME: gpu runtime panics with `CUDA_ERROR_ILLEGAL_ADDRESS`
fn test_church_exp() {
  let book = load_core("church/church_exp.hvmc");
  let (rwts, net) = normal(book, 1 << 16);

  assert_snapshot!(Net::to_string(&net));
  assert_debug_snapshot!(rwts.total(), @"1942");
}

#[test]
fn test_church_mul() {
  let mut book = load_lang("church/church_mul.hvm");
  let (rwts, net) = hvm_lang_normal(&mut book, 512);
  let (readback, valid_readback) = hvm_lang_readback(&net, &book);

  assert!(valid_readback);

  // TODO: investigate why this difference exists
  if cfg!(feature = "cuda") {
    assert_snapshot!(Net::to_string(&net), @"({2 ({2 b {3 c {4 d e}}} f) {3 (f g) {4 (g h) {5 i j}}}} (b k))");
    assert_snapshot!(readback, @"λa λb ((* {λc d {λd e {λe (a (a (a {b {c {c c}}}))) {a a}}}}) c)");
    assert_debug_snapshot!(rwts.total(), @"15");
  } else {
    assert_snapshot!(Net::to_string(&net), @"({3 (a {3 b {5 c {7 d {9 e {11 f {13 g {15 h {17 i {19 j {21 k {23 l {25 m {27 n {29 o {31 p {33 q {35 r {37 s {39 t u}}}}}}}}}}}}}}}}}}}) {5 (v a) {7 (w v) {9 (x w) {11 (y x) {13 (z y) {15 (aa z) {17 (ab aa) {19 (ac ab) {21 (ad ac) {23 (ae ad) {25 (af ae) {27 (ag af) {29 (ah ag) {31 (ai ah) {33 (aj ai) {35 (ak aj) {37 (al ak) {39 (am al) ({3 c {5 d {7 e {9 f {11 g {13 h {15 i {17 j {19 k {21 l {23 m {25 n {27 o {29 p {31 q {33 r {35 s {37 t {39 u an}}}}}}}}}}}}}}}}}}} am)}}}}}}}}}}}}}}}}}}} (an b))");
    assert_snapshot!(readback, @"λa λb let#0 {c aq} = (d (e (f (g (h (i (j (k (l (m (n (o (p (q (r (s (t (u (v (w #0 {x #1 {y #2 {z #3 {aa #4 {ab #5 {ac #6 {ad #7 {ae #8 {af #9 {ag #10 {ah #11 {ai #12 {aj #13 {ak #14 {al #15 {am #16 {an #17 {ao #18 {ap b}}}}}}}}}}}}}}}}}}})))))))))))))))))))); c");
    assert_debug_snapshot!(rwts.total(), @"43");
  }
}

#[test]
#[cfg(not(feature = "cuda"))] // FIXME: hangs indefinitely
fn alloc_big_tree() {
  let book = load_core("tree/alloc_big_tree.hvmc");
  let (rwts, net) = normal(book, 1 << 16);

  assert_snapshot!(Net::to_string(&net)); // file snapshot
  assert_debug_snapshot!(rwts.total(), @"24626");
}

#[test]
fn test_neg_fusion() {
  let mut book = load_lang("fusion/neg_fusion.hvm");
  let (rwts, net) = hvm_lang_normal(&mut book, 512);
  let (readback, valid_readback) = hvm_lang_readback(&net, &book);

  assert!(valid_readback);
  assert_snapshot!(Net::to_string(&net), @"(a (* a))");
  assert_snapshot!(readback, @"λa λ* a");

  // TODO: investigate why this difference exists
  if cfg!(feature = "cuda") {
    assert_debug_snapshot!(rwts.total(), @"160");
  } else {
    assert_debug_snapshot!(rwts.total(), @"112");
  }
}
