use crate::loaders::*;
use hvmc::ast::Book;
use insta::assert_debug_snapshot;
mod loaders;

fn list_got(index: u32) -> Book {
  let code = load_file("list_put_got.hvmc");
  let mut book = parse_core(&code);
  let def = book.get_mut("GenGotIndex").unwrap();
  def.apply_tree(hvmc::ast::Tree::Ref { nam: format!("S{index}") });
  let def = book.get_mut("main").unwrap();
  def.apply_tree(hvmc::ast::Tree::Ref { nam: format!("GenGotIndex") });
  book
}

fn list_put(index: u32, value: u32) -> Book {
  let code = load_file("list_put_got.hvmc");
  let mut book = parse_core(&code);
  let def = book.get_mut("GenPutIndexValue").unwrap();
  def.apply_tree(hvmc::ast::Tree::Ref { nam: format!("S{index}") });
  def.apply_tree(hvmc::ast::Tree::Ref { nam: format!("S{value}") });
  let def = book.get_mut("main").unwrap();
  def.apply_tree(hvmc::ast::Tree::Ref { nam: format!("GenPutIndexValue") });
  book
}

#[test]
fn test_list_got() {
  let mut rwts_list = Vec::new();

  for index in [0, 1, 3, 7, 15, 31] {
    let book = list_got(index);
    let (rwts, _) = normal(book, None);
    rwts_list.push(rwts.total())
  }

  assert_debug_snapshot!(rwts_list[0], @"594");
  assert_debug_snapshot!(rwts_list[1], @"623");
  assert_debug_snapshot!(rwts_list[2], @"681");
  assert_debug_snapshot!(rwts_list[3], @"797");
  assert_debug_snapshot!(rwts_list[4], @"1029");
  assert_debug_snapshot!(rwts_list[5], @"1493");

  // Tests the linearity of the function
  let delta = rwts_list[1] - rwts_list[0];
  assert_eq!(rwts_list[1] + delta * 2, rwts_list[2]);
  assert_eq!(rwts_list[2] + delta * 4, rwts_list[3]);
  assert_eq!(rwts_list[3] + delta * 8, rwts_list[4]);
  assert_eq!(rwts_list[4] + delta * 16, rwts_list[5]);
}

#[test]
fn test_list_put() {
  let mut rwts_list = Vec::new();

  for (index, value) in [(0, 2), (1, 4), (3, 8), (7, 16), (15, 32), (31, 0)] {
    let book = list_put(index, value);
    let (rwts, _) = normal(book, None);
    rwts_list.push(rwts.total())
  }

  assert_debug_snapshot!(rwts_list[0], @"585");
  assert_debug_snapshot!(rwts_list[1], @"615");
  assert_debug_snapshot!(rwts_list[2], @"675");
  assert_debug_snapshot!(rwts_list[3], @"795");
  assert_debug_snapshot!(rwts_list[4], @"1035");
  assert_debug_snapshot!(rwts_list[5], @"1515");

  //Tests the linearity of the function
  let delta = rwts_list[1] - rwts_list[0];
  assert_eq!(rwts_list[1] + delta * 2, rwts_list[2]);
  assert_eq!(rwts_list[2] + delta * 4, rwts_list[3]);
  assert_eq!(rwts_list[3] + delta * 8, rwts_list[4]);
  assert_eq!(rwts_list[4] + delta * 16, rwts_list[5]);
}
