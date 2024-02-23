
use crate::loaders::*;
use hvmc::ast::Book;
use insta::assert_debug_snapshot;
mod loaders;



fn list_got(index: u32) -> Book {
  let code = load_file("list_put_got.hvmc");
  let mut book = parse_core(&code);
  println!("{:#?}", book.keys().collect::<Vec<_>>());
  let mut def = book.get_mut("GenGotIndex").unwrap();
  core_apply(&mut def, hvmc::ast::Tree::Ref { nam: format!("S{index}") });
  let mut def = book.get_mut("main").unwrap();
  core_apply(&mut def, hvmc::ast::Tree::Ref { nam: format!("GenGotIndex") });
  book
}

fn list_put(index: u32, value: u32) -> Book {
  let code = load_file("list_put_got.hvmc");
  let mut book = parse_core(&code);
  let mut def = book.get_mut("GenPutIndexValue").unwrap();
  def.with_argument(hvmc::ast::Tree::Ref { nam: format!("S{index}") });
  def.with_argument(hvmc::ast::Tree::Ref { nam: format!("S{value}") });
  println!("{:?}", def);
  let mut def = book.get_mut("main").unwrap();
  def.with_argument(hvmc::ast::Tree::Ref { nam: format!("GenPutIndexValue") });
  book
}

#[test]
fn test_list_got() {
  let mut rwts_list = Vec::new();

  for index in [0, 1, 3, 7, 15, 31] {
    let mut book = list_got(index);
    let (rwts, _) = normal(book, 2048);
    rwts_list.push(rwts.total())
  }

  assert_debug_snapshot!(rwts_list[0], @"581");
  assert_debug_snapshot!(rwts_list[1], @"605");
  assert_debug_snapshot!(rwts_list[2], @"653");
  assert_debug_snapshot!(rwts_list[3], @"749");
  assert_debug_snapshot!(rwts_list[4], @"941");
  assert_debug_snapshot!(rwts_list[5], @"1325");

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
    let mut book = list_put(index, value);
    let (rwts, _) = normal(book, 2048);
    rwts_list.push(rwts.total())
  }

  assert_debug_snapshot!(rwts_list[0], @"572");
  assert_debug_snapshot!(rwts_list[1], @"597");
  assert_debug_snapshot!(rwts_list[2], @"647");
  assert_debug_snapshot!(rwts_list[3], @"747");
  assert_debug_snapshot!(rwts_list[4], @"947");
  assert_debug_snapshot!(rwts_list[5], @"1347");

  //Tests the linearity of the function
  let delta = rwts_list[1] - rwts_list[0];
  assert_eq!(rwts_list[1] + delta * 2, rwts_list[2]);
  assert_eq!(rwts_list[2] + delta * 4, rwts_list[3]);
  assert_eq!(rwts_list[3] + delta * 8, rwts_list[4]);
  assert_eq!(rwts_list[4] + delta * 16, rwts_list[5]);
}
