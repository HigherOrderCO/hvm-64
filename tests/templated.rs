use crate::loaders::*;
use hvm_lang::term::DefinitionBook;
use insta::assert_debug_snapshot;

mod loaders;

fn list_got(index: u32) -> DefinitionBook {
  let template = load_file("list_put_get.hvm");
  let code = replace_template(template, &[("{fun}", "got"), ("{args}", &index.to_string())]);
  parse_lang(&code)
}

#[test]
fn test_list_got() {
  let rwts = [
  list_got(0),
  list_got(1),
  list_got(3),
  list_got(7),
  list_got(15),
  list_got(31)
  ]
  .map(|book| hvm_lang_normal(book, 2048))
  .map(|(_, _, info)| info.stats.rewrites.total_rewrites());

  assert_debug_snapshot!(rwts[0], @"573");
  assert_debug_snapshot!(rwts[1], @"595");
  assert_debug_snapshot!(rwts[2], @"639");
  assert_debug_snapshot!(rwts[3], @"727");
  assert_debug_snapshot!(rwts[4], @"903");
  assert_debug_snapshot!(rwts[5], @"1255");

  //Tests the linearity of the function
  let delta = rwts[1] - rwts[0];
  assert_eq!(rwts[1] + delta * 2, rwts[2]);
  assert_eq!(rwts[2] + delta * 4, rwts[3]);
  assert_eq!(rwts[3] + delta * 8, rwts[4]);
  assert_eq!(rwts[4] + delta * 16, rwts[5]);
}

fn list_put(index: u32, value: u32) -> DefinitionBook {
  let template = load_file("list_put_get.hvm");
  let code = replace_template(template, &[("{fun}", "put"), ("{args}", &format!("{index} {value}"))]);
  parse_lang(&code)
}

#[test]
fn test_list_put() {
  let rwts = [
  list_put(0, 2),
  list_put(1, 4),
  list_put(3, 8),
  list_put(7, 16),
  list_put(15, 32),
  list_put(31, 64)
  ]
  .map(|book| hvm_lang_normal(book, 2048))
  .map(|(_, _, info)| info.stats.rewrites.total_rewrites());

  assert_debug_snapshot!(rwts[0], @"563");
  assert_debug_snapshot!(rwts[1], @"586");
  assert_debug_snapshot!(rwts[2], @"632");
  assert_debug_snapshot!(rwts[3], @"724");
  assert_debug_snapshot!(rwts[4], @"908");
  assert_debug_snapshot!(rwts[5], @"1276");

  //Tests the linearity of the function
  let delta = rwts[1] - rwts[0];
  assert_eq!(rwts[1] + delta * 2, rwts[2]);
  assert_eq!(rwts[2] + delta * 4, rwts[3]);
  assert_eq!(rwts[3] + delta * 8, rwts[4]);
  assert_eq!(rwts[4] + delta * 16, rwts[5]);
}
