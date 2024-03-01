//! Tests for transformation passes

pub mod loaders;

use hvmc::util::show_rewrites;
use insta::assert_snapshot;
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
  RWTS   :          17_730
  - ANNI :           4_385
  - COMM :          11_725
  - ERAS :           1_598
  - DREF :              22
  - OPER :               0
  Slow:
  RWTS   :          35_445
  - ANNI :           8_763
  - COMM :          23_450
  - ERAS :           3_196
  - DREF :              36
  - OPER :               0
  "###)
}
