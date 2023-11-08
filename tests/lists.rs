use crate::loaders::*;
use hvm_lang::term::DefinitionBook;
use insta::assert_debug_snapshot;

mod loaders;

fn list_got(index: u32) -> DefinitionBook {
  let template = load_file("list_put_got.hvm");
  let code = replace_template(template, &[("{fun}", "Got"), ("{args}", &format!("S{}", index))]);
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
    list_got(31),
  ]
  .map(|mut book| hvm_lang_normal(&mut book, 2048))
  .map(|(rnet, _, _)| rnet.rewrites());

  assert_debug_snapshot!(rwts[0], @"583");
  assert_debug_snapshot!(rwts[1], @"619");
  assert_debug_snapshot!(rwts[2], @"691");
  assert_debug_snapshot!(rwts[3], @"835");
  assert_debug_snapshot!(rwts[4], @"1123");
  assert_debug_snapshot!(rwts[5], @"1699");

  //Tests the linearity of the function
  let delta = rwts[1] - rwts[0];
  assert_eq!(rwts[1] + delta * 2, rwts[2]);
  assert_eq!(rwts[2] + delta * 4, rwts[3]);
  assert_eq!(rwts[3] + delta * 8, rwts[4]);
  assert_eq!(rwts[4] + delta * 16, rwts[5]);
}

fn list_put(index: u32, value: u32) -> DefinitionBook {
  let template = load_file("list_put_got.hvm");
  let code = replace_template(template, &[("{fun}", "Put"), ("{args}", &format!("S{index} S{value}"))]);
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
    list_put(31, 0),
  ]
  .map(|mut book| hvm_lang_normal(&mut book, 2048))
  .map(|(rnet, _, _)| rnet.rewrites());

  assert_debug_snapshot!(rwts[0], @"566");
  assert_debug_snapshot!(rwts[1], @"593");
  assert_debug_snapshot!(rwts[2], @"647");
  assert_debug_snapshot!(rwts[3], @"755");
  assert_debug_snapshot!(rwts[4], @"971");
  assert_debug_snapshot!(rwts[5], @"1403");

  //Tests the linearity of the function
  let delta = rwts[1] - rwts[0];
  assert_eq!(rwts[1] + delta * 2, rwts[2]);
  assert_eq!(rwts[2] + delta * 4, rwts[3]);
  assert_eq!(rwts[3] + delta * 8, rwts[4]);
  assert_eq!(rwts[4] + delta * 16, rwts[5]);
}
