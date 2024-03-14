//! Tests for transformation passes

pub mod loaders;

use hvmc::util::show_rewrites;
use insta::{assert_display_snapshot, assert_snapshot};
use loaders::*;

#[test]
/// Test that ensures that pre_reduce only reduces repeated refs once.
pub fn test_fast_pre_reduce() {
  let book = parse_core(&load_file("heavy_pre_reduction.hvmc"));
  let (mut book_1, mut book_2) = (book.clone(), book);

  let rwts_1 = book_1.pre_reduce(&|x| !["expensive", "main_fast"].contains(&x), 1 << 29, u64::MAX).rewrites;
  let rwts_2 =
    book_2.pre_reduce(&|x| !["expensive_1", "expensive_2", "main_slow"].contains(&x), 1 << 29, u64::MAX).rewrites;

  let rwts_1 = show_rewrites(&(rwts_1 + normal_with(book_1, 1 << 29, "main_fast").0));
  let rwts_2 = show_rewrites(&(rwts_2 + normal_with(book_2, 1 << 29, "main_slow").0));

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
  use hvmc::ast::{Net, Tree};
  use std::str::FromStr;
  pub fn parse_and_encode(net: &str) -> String {
    let mut net = Net::from_str(net).unwrap();
    println!("{net}");
    net.trees_mut().for_each(Tree::coalesce_constructors);
    println!("{net}");
    net.trees_mut().for_each(Tree::encode_scott_adts);
    format!("{net}")
  }
  assert_display_snapshot!(parse_and_encode("(a (b (c d)))"), @"(a b c d)");
  assert_display_snapshot!(parse_and_encode("(a (b c (d e)))"), @"(a b c d e)");
  assert_display_snapshot!(parse_and_encode("(a b c d e f g h)"), @"(a b c d e f g h)");
  assert_display_snapshot!(parse_and_encode("(a b c d (e f g h (i j k l)))"), @"(a b c d (e f g h (i j k l)))");

  assert_display_snapshot!(parse_and_encode("(* ((a R) R))"), @"(:1:2 a)");
  assert_display_snapshot!(parse_and_encode("((a R) (* R))"), @"(:0:2 a)");
  assert_display_snapshot!(parse_and_encode("(* (* ((a R) R)))"), @"(:2:3 a)");
  assert_display_snapshot!(parse_and_encode("(* ((a R) (* R)))"), @"(:1:3 a)");
  assert_display_snapshot!(parse_and_encode("((a (b R)) R)"), @"(:0:1 a b)");
  assert_display_snapshot!(parse_and_encode("((a (b (c R))) R)"), @"(:0:1 a b c)");
  assert_display_snapshot!(parse_and_encode("(* ((a (b (c R))) R))"), @"(:1:2 a b c)");
  assert_display_snapshot!(parse_and_encode("{4 * {4 {4 a {4 b {4 c R}}} R}}"), @"{4:1:2 a b c}");
  assert_display_snapshot!(parse_and_encode("(* x x)"), @"(:1:2)");
}
