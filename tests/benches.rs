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
fn dec_bits() {
  let book = load_core("binary-counter/dec_bits.hvmc");
  let (rwts, net) = normal(book, 1 << 18);

  assert_snapshot!(Net::to_string(&net), @"(* (* (a a)))");
  assert_debug_snapshot!(rwts.total(), @"180036");
}

#[test]
fn dec_bits_tree() {
  let book = load_core("binary-counter/dec_bits_tree.hvmc");
  let (rwts, net) = normal(book, 1 << 21);

  // FIXME: The refs to the rule `@E` are not expanded
  assert_snapshot!(Net::to_string(&net), @"(((((((* (* (a a))) (* (* (b b)))) ((* (* (c c))) (* (* (d d))))) (((* (* (e e))) (* (* (f f)))) ((* (* (g g))) (* (* (h h)))))) ((((* (* (i i))) (* (* (j j)))) ((* (* (k k))) (* (* (l l))))) (((* (* (m m))) (* (* (n n)))) ((* (* (o o))) (* (* (p p))))))) (((((* (* (q q))) (* (* (r r)))) ((* (* (s s))) (* (* (t t))))) (((* (* (u u))) (* (* (v v)))) ((* (* (w w))) (* (* (x x)))))) ((((* (* (y y))) (* (* (z z)))) ((* (* (aa aa))) (* (* (ab ab))))) (((* (* (ac ac))) (* (* (ad ad)))) ((* (* (ae ae))) (* (* (af af)))))))) ((((((* (* (ag ag))) (* (* (ah ah)))) ((* (* (ai ai))) (* (* (aj aj))))) (((* (* (ak ak))) (* (* (al al)))) ((* (* (am am))) (* (* (an an)))))) ((((* (* (ao ao))) (* (* (ap ap)))) ((* (* (aq aq))) (* (* (ar ar))))) (((* (* (as as))) (* (* (at at)))) ((* (* (au au))) (* (* (av av))))))) (((((* (* (aw aw))) (* (* (ax ax)))) ((* (* (ay ay))) (* (* (az az))))) (((* (* (ba ba))) (* (* (bb bb)))) ((* (* (bc bc))) (* (* (bd bd)))))) ((((* (* (be be))) (* (* (bf bf)))) ((* (* (bg bg))) (* (* (bh bh))))) (((* (* (bi bi))) (* (* (bj bj)))) ((* (* (bk bk))) (* (* (bl bl)))))))))");
  assert_debug_snapshot!(rwts.total(), @"2874410");
}

#[test]
fn test_church_exp() {
  let book = load_core("church/church_exp.hvmc");
  let (rwts, net) = normal(book, 1 << 16);

  assert_snapshot!(Net::to_string(&net));
  assert_debug_snapshot!(rwts.total(), @"803");
}

#[test]
fn test_church_mul() {
  let mut book = load_lang("church/church_mul.hvm");
  let (rwts, net) = hvm_lang_normal(&mut book, 512);
  let (readback, valid_readback) = hvm_lang_readback(&net, &book);

  assert!(valid_readback);
  assert_snapshot!(Net::to_string(&net), @"({3 (a {3 b {5 c {7 d {9 e {11 f {13 g {15 h {17 i {19 j {21 k {23 l {25 m {27 n {29 o {31 p {33 q {35 r {37 s {39 t u}}}}}}}}}}}}}}}}}}}) {5 (v a) {7 (w v) {9 (x w) {11 (y x) {13 (z y) {15 (aa z) {17 (ab aa) {19 (ac ab) {21 (ad ac) {23 (ae ad) {25 (af ae) {27 (ag af) {29 (ah ag) {31 (ai ah) {33 (aj ai) {35 (ak aj) {37 (al ak) {39 (am al) ({3 c {5 d {7 e {9 f {11 g {13 h {15 i {17 j {19 k {21 l {23 m {25 n {27 o {29 p {31 q {33 r {35 s {37 t {39 u an}}}}}}}}}}}}}}}}}}} am)}}}}}}}}}}}}}}}}}}} (an b))");
  assert_snapshot!(readback, @"λa λb let #0{c qb} = (let #0{d rb} = a; d (let #1{e sb} = rb; e (let #2{f tb} = sb; f (let #3{g ub} = tb; g (let #4{h vb} = ub; h (let #5{i wb} = vb; i (let #6{j xb} = wb; j (let #7{k yb} = xb; k (let #8{l zb} = yb; l (let #9{m ac} = zb; m (let #10{n bc} = ac; n (let #11{o cc} = bc; o (let #12{p dc} = cc; p (let #13{q ec} = dc; q (let #14{r fc} = ec; r (let #15{s gc} = fc; s (let #16{t hc} = gc; t (let #17{u ic} = hc; u let #18{v w} = ic; (v (w #0{let #1{x jc} = qb; x #1{let #2{y kc} = jc; y #2{let #3{z lc} = kc; z #3{let #4{ab mc} = lc; ab #4{let #5{bb nc} = mc; bb #5{let #6{cb oc} = nc; cb #6{let #7{db pc} = oc; db #7{let #8{eb qc} = pc; eb #8{let #9{fb rc} = qc; fb #9{let #10{gb sc} = rc; gb #10{let #11{hb tc} = sc; hb #11{let #12{ib uc} = tc; ib #12{let #13{jb vc} = uc; jb #13{let #14{kb wc} = vc; kb #14{let #15{lb xc} = wc; lb #15{let #16{mb yc} = xc; mb #16{let #17{nb zc} = yc; nb let #18{ob pb} = zc; #17{ob #18{pb b}}}}}}}}}}}}}}}}}}})))))))))))))))))))); c");
  assert_debug_snapshot!(rwts.total(), @"48");
}

#[test]
fn alloc_big_tree() {
  let book = load_core("tree/alloc_big_tree.hvmc");
  let (rwts, net) = normal(book, 1 << 16);

  assert_snapshot!(Net::to_string(&net)); // file snapshot
  assert_debug_snapshot!(rwts.total(), @"24562");
}

#[test]
fn test_neg_fusion() {
  let mut book = load_lang("fusion/neg_fusion.hvm");
  let (rwts, net) = hvm_lang_normal(&mut book, 512);
  let (readback, valid_readback) = hvm_lang_readback(&net, &book);

  assert!(valid_readback);
  assert_snapshot!(Net::to_string(&net), @"(a (* a))");
  assert_snapshot!(readback, @"λa λ* a");

  assert_debug_snapshot!(rwts.total(), @"147");
}
