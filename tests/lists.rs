use crate::loaders::*;
use hvmc::ast::Book;
use insta::assert_debug_snapshot;
mod loaders;

fn list_got(index: u32) -> Book {
  let code = load_file("list_put_got.hvmc");
  let mut book = parse_core(&code);
  println!("{:#?}", book.keys().collect::<Vec<_>>());
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
  println!("{:?}", def);
  let def = book.get_mut("main").unwrap();
  def.apply_tree(hvmc::ast::Tree::Ref { nam: format!("GenPutIndexValue") });
  book
}

#[test]
fn test_list_got() {
  let mut rwts_list = Vec::new();

  for index in [0, 1, 3, 7, 15, 31] {
    let book = list_got(index);
    let (rwts, _) = normal(book, 2048);
    rwts_list.push(rwts.total())
  }

  assert_debug_snapshot!(rwts_list[0], @"579");
  assert_debug_snapshot!(rwts_list[1], @"601");
  assert_debug_snapshot!(rwts_list[2], @"645");
  assert_debug_snapshot!(rwts_list[3], @"733");
  assert_debug_snapshot!(rwts_list[4], @"909");
  assert_debug_snapshot!(rwts_list[5], @"1261");

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
    let (rwts, _) = normal(book, 2048);
    rwts_list.push(rwts.total())
  }

  assert_debug_snapshot!(rwts_list[0], @"570");
  assert_debug_snapshot!(rwts_list[1], @"593");
  assert_debug_snapshot!(rwts_list[2], @"639");
  assert_debug_snapshot!(rwts_list[3], @"731");
  assert_debug_snapshot!(rwts_list[4], @"915");
  assert_debug_snapshot!(rwts_list[5], @"1283");

  //Tests the linearity of the function
  let delta = rwts_list[1] - rwts_list[0];
  assert_eq!(rwts_list[1] + delta * 2, rwts_list[2]);
  assert_eq!(rwts_list[2] + delta * 4, rwts_list[3]);
  assert_eq!(rwts_list[3] + delta * 8, rwts_list[4]);
  assert_eq!(rwts_list[4] + delta * 16, rwts_list[5]);
}
