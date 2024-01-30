use crate::loaders::*;
use hvml::term::Book;
use insta::assert_debug_snapshot;

mod loaders;

fn list_got(index: u32) -> Book {
  let template = load_file("list_put_got.hvm");
  let code = replace_template(template, &[("{fun}", "Got"), ("{args}", &format!("S{}", index))]);
  parse_lang(&code)
}

#[test]
fn test_list_got() {
  let mut rwts_list = Vec::new();

  for index in [
    0,
    1,
    3,
    7,
    // FIXME: Gpu runtime panics with `CUDA_ERROR_ILLEGAL_ADDRESS` when index is >= 14
    // if the list inside `list_put_got.hvm` is [31..0] instead of [0..31], it fails when <= 17
    #[cfg(not(feature = "cuda"))]
    15,
    #[cfg(not(feature = "cuda"))]
    31,
  ] {
    let mut book = list_got(index);
    let (rwts, _) = hvm_lang_normal(&mut book, 2048);
    rwts_list.push(rwts.total())
  }

  assert_debug_snapshot!(rwts_list[0], @"467");
  assert_debug_snapshot!(rwts_list[1], @"477");
  assert_debug_snapshot!(rwts_list[2], @"497");
  assert_debug_snapshot!(rwts_list[3], @"537");
  #[cfg(not(feature = "cuda"))]
  assert_debug_snapshot!(rwts_list[4], @"617");
  #[cfg(not(feature = "cuda"))]
  assert_debug_snapshot!(rwts_list[5], @"777");

  // Tests the linearity of the function
  let delta = rwts_list[1] - rwts_list[0];
  assert_eq!(rwts_list[1] + delta * 2, rwts_list[2]);
  assert_eq!(rwts_list[2] + delta * 4, rwts_list[3]);
  #[cfg(not(feature = "cuda"))]
  assert_eq!(rwts_list[3] + delta * 8, rwts_list[4]);
  #[cfg(not(feature = "cuda"))]
  assert_eq!(rwts_list[4] + delta * 16, rwts_list[5]);
}

fn list_put(index: u32, value: u32) -> Book {
  let template = load_file("list_put_got.hvm");
  let code = replace_template(template, &[("{fun}", "Put"), ("{args}", &format!("S{index} S{value}"))]);
  parse_lang(&code)
}

#[test]
fn test_list_put() {
  let mut rwts_list = Vec::new();

  for (index, value) in [(0, 2), (1, 4), (3, 8), (7, 16), (15, 32), (31, 0)] {
    let mut book = list_put(index, value);
    let (rwts, _) = hvm_lang_normal(&mut book, 2048);
    rwts_list.push(rwts.total())
  }

  assert_debug_snapshot!(rwts_list[0], @"456");
  assert_debug_snapshot!(rwts_list[1], @"466");
  assert_debug_snapshot!(rwts_list[2], @"486");
  assert_debug_snapshot!(rwts_list[3], @"526");
  assert_debug_snapshot!(rwts_list[4], @"606");
  assert_debug_snapshot!(rwts_list[5], @"766");

  //Tests the linearity of the function
  let delta = rwts_list[1] - rwts_list[0];
  assert_eq!(rwts_list[1] + delta * 2, rwts_list[2]);
  assert_eq!(rwts_list[2] + delta * 4, rwts_list[3]);
  assert_eq!(rwts_list[3] + delta * 8, rwts_list[4]);
  assert_eq!(rwts_list[4] + delta * 16, rwts_list[5]);
}
