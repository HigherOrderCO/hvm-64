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

  for index in [0, 1, 3, 7, 15, 31] {
    let mut book = list_got(index);
    let (rwts, _) = hvm_lang_normal(&mut book, 2048);
    rwts_list.push(rwts.total())
  }

  assert_debug_snapshot!(rwts_list[0], @"577");
  assert_debug_snapshot!(rwts_list[1], @"601");
  assert_debug_snapshot!(rwts_list[2], @"649");
  assert_debug_snapshot!(rwts_list[3], @"745");
  assert_debug_snapshot!(rwts_list[4], @"937");
  assert_debug_snapshot!(rwts_list[5], @"1321");

  // Tests the linearity of the function
  let delta = rwts_list[1] - rwts_list[0];
  assert_eq!(rwts_list[1] + delta * 2, rwts_list[2]);
  assert_eq!(rwts_list[2] + delta * 4, rwts_list[3]);
  assert_eq!(rwts_list[3] + delta * 8, rwts_list[4]);
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

  assert_debug_snapshot!(rwts_list[0], @"567");
  assert_debug_snapshot!(rwts_list[1], @"592");
  assert_debug_snapshot!(rwts_list[2], @"642");
  assert_debug_snapshot!(rwts_list[3], @"742");
  assert_debug_snapshot!(rwts_list[4], @"942");
  assert_debug_snapshot!(rwts_list[5], @"1342");

  //Tests the linearity of the function
  let delta = rwts_list[1] - rwts_list[0];
  assert_eq!(rwts_list[1] + delta * 2, rwts_list[2]);
  assert_eq!(rwts_list[2] + delta * 4, rwts_list[3]);
  assert_eq!(rwts_list[3] + delta * 8, rwts_list[4]);
  assert_eq!(rwts_list[4] + delta * 16, rwts_list[5]);
}
