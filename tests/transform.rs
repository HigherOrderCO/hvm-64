#![cfg(feature = "std")]

//! Tests for transformation passes

use hvm64_ast::{Book, Net, Tree};
use hvm64_transform::{
  coalesce_ctrs::CoalesceCtrs, encode_adts::EncodeAdts, eta_reduce::EtaReduce, inline::Inline, pre_reduce::PreReduce,
  prune::Prune, TransformError,
};

use insta::assert_snapshot;
use std::str::FromStr;

mod loaders;
use loaders::*;

#[test]
/// Test that ensures that pre_reduce only reduces repeated refs once.
pub fn test_fast_pre_reduce() {
  let book = parse_core(&load_file("heavy_pre_reduction.hvm"));
  let (mut book_1, mut book_2) = (book.clone(), book);

  let rwts_1 = book_1.pre_reduce(&|x| !["expensive", "main_fast"].contains(&x), None, u64::MAX).rewrites;
  let rwts_2 =
    book_2.pre_reduce(&|x| !["expensive_1", "expensive_2", "main_slow"].contains(&x), None, u64::MAX).rewrites;

  let rwts_1 = rwts_1 + normal_with(book_1, None, "main_fast").0;
  let rwts_2 = rwts_2 + normal_with(book_2, None, "main_slow").0;

  assert_snapshot!(format!("Fast:\n{rwts_1}Slow:\n{rwts_2}"), @r###"
  Fast:
  RWTS   :          33_236
  - ANNI :           4_385
  - COMM :          11_725
  - ERAS :           1_598
  - DREF :          15_528
  - OPER :               0
  Slow:
  RWTS   :          50_951
  - ANNI :           8_763
  - COMM :          23_450
  - ERAS :           3_196
  - DREF :          15_542
  - OPER :               0
  "###)
}

#[test]
pub fn test_adt_encoding() {
  pub fn parse_and_encode(net: &str) -> String {
    let mut net = Net::from_str(net).unwrap();
    net.trees_mut().for_each(Tree::coalesce_constructors);
    net.trees_mut().for_each(Tree::encode_scott_adts);
    format!("{net}")
  }
  assert_snapshot!(parse_and_encode("(a (b (c d)))"), @"(a b c d)");
  assert_snapshot!(parse_and_encode("(a (b c (d e)))"), @"(a b c d e)");
  assert_snapshot!(parse_and_encode("(a b c d e f g h)"), @"(a b c d e f g h)");
  assert_snapshot!(parse_and_encode("(a b c d (e f g h (i j k l)))"), @"(a b c d (e f g h (i j k l)))");

  assert_snapshot!(parse_and_encode("(* ((a R) R))"), @"(:1:2 a)");
  assert_snapshot!(parse_and_encode("((a R) (* R))"), @"(:0:2 a)");
  assert_snapshot!(parse_and_encode("(* (* ((a R) R)))"), @"(:2:3 a)");
  assert_snapshot!(parse_and_encode("(* ((a R) (* R)))"), @"(:1:3 a)");
  assert_snapshot!(parse_and_encode("((a (b R)) R)"), @"(:0:1 a b)");
  assert_snapshot!(parse_and_encode("((a (b (c R))) R)"), @"(:0:1 a b c)");
  assert_snapshot!(parse_and_encode("(* ((a (b (c R))) R))"), @"(:1:2 a b c)");
  assert_snapshot!(parse_and_encode("{4 * {4 {4 a {4 b {4 c R}}} R}}"), @"{4:1:2 a b c}");
  assert_snapshot!(parse_and_encode("(* x x)"), @"(:1:2)");
  assert_snapshot!(parse_and_encode("(((((* x x) x) * x) x) * x)"), @"(:0:2 (:0:2 (:1:2)))");
  assert_snapshot!(parse_and_encode("(a b * (a b c) * c)"), @"(a b (:1:3 a b))");
  assert_snapshot!(parse_and_encode("(* (:0:1))"), @"(:1:2)");
  assert_snapshot!(parse_and_encode("(a * (:0:1))"), @"(a (:1:2))");
}

#[test]
pub fn test_eta() {
  pub fn parse_and_reduce(net: &str) -> String {
    let mut net = Net::from_str(net).unwrap();
    net.eta_reduce();
    format!("{net}")
  }
  assert_snapshot!(parse_and_reduce("((x y) (x y))"), @"(x x)");
  assert_snapshot!(parse_and_reduce("((a b c d e f) (a b c d e f))"), @"(a a)");
  assert_snapshot!(parse_and_reduce("<+ (a b) (a b)>"), @"<+ a a>");
  assert_snapshot!(parse_and_reduce("(a b) & ((a b) (c d)) ~ (c d) "), @r###"
  a
    & (a c) ~ c
  "###);
  assert_snapshot!(parse_and_reduce("((a b) [a b])"), @"((a b) [a b])");
  assert_snapshot!(parse_and_reduce("((a b c) b c)"), @"((a b) b)");
  assert_snapshot!(parse_and_reduce("([(a b) (c d)] [(a b) (c d)])"), @"(a a)");
  assert_snapshot!(parse_and_reduce("(* *)"), @"*");
  assert_snapshot!(parse_and_reduce("([(#0 #0) (#12345 #12345)] [(* *) (a a)])"), @"([#0 #12345] [* (a a)])");
}

#[test]
pub fn test_inline() {
  pub fn parse_and_inline(net: &str) -> Result<String, TransformError> {
    let mut net = Book::from_str(net).unwrap();
    net.inline().map(|_| format!("{net}"))
  }
  assert_snapshot!(parse_and_inline("
    @era = *
    @num = #123
    @abab = (a b a b)
    @ref = @abab
    @def = @ref
    @eff = @def
    @I = (:0:1)
    @K = (:0:2)
    @1234 = (:1:2 (:3:4))
    @foo = @bar
    @bar = @baz
    @baz = @unbound
    @into = ((@era @num @abab @ref @def @eff @I @K) (@into @1234 @foo @bar @baz @unbound))
  ").unwrap(), @r###"
  @1234 = (:1:2 (:3:4))

  @I = (:0:1)

  @K = (:0:2)

  @abab = (a b a b)

  @bar = @unbound

  @baz = @unbound

  @def = @abab

  @eff = @abab

  @era = *

  @foo = @unbound

  @into = ((* #123 @abab @abab @abab @abab (:0:1) (:0:2)) (@into @1234 @unbound @unbound @unbound @unbound))

  @num = #123

  @ref = @abab
  "###);

  for net in ["@a = @a", "@a = @b  @b = @c  @c = @d  @d = @e  @e = @f  @f = @c"] {
    assert!(matches!(parse_and_inline(net), Err(TransformError::InfiniteRefCycle(_))));
  }
}

#[test]
pub fn test_prune() {
  pub fn parse_and_prune(net: &str) -> String {
    let mut net = Book::from_str(net).unwrap();
    net.prune(&["main".to_owned()]);
    format!("{net}")
  }
  assert_snapshot!(parse_and_prune("
    @self = (* @self)
    @main = (@main @a @b)
    @a = (@b @c @d)
    @b = (@c @c)
    @c = @d
    @d = @at
    @idk = (@e @f)
  "), @r###"
  @a = (@b @c @d)

  @b = (@c @c)

  @c = @d

  @d = @at

  @main = (@main @a @b)
  "###);
}
