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
  let mut rwts = Vec::new();

  for i in [0, 1, 3, 7, 15, 31] {
    let mut book = list_got(i);
    let (rnet, _, _) = hvm_lang_normal(&mut book, 2048);
    rwts.push(rnet.rewrites())
  }

  assert_debug_snapshot!(rwts[0], @"583");
  assert_debug_snapshot!(rwts[1], @"615");
  assert_debug_snapshot!(rwts[2], @"679");
  assert_debug_snapshot!(rwts[3], @"807");
  assert_debug_snapshot!(rwts[4], @"1063");
  assert_debug_snapshot!(rwts[5], @"1575");

  // Tests the linearity of the function
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
  let mut rwts = Vec::new();

  for (i, value) in [(0, 2), (1, 4), (3, 8), (7, 16), (15, 32), (31, 0)] {
    let mut book = list_put(i, value);
    let (rnet, _, _) = hvm_lang_normal(&mut book, 2048);
    rwts.push(rnet.rewrites())
  }

  assert_debug_snapshot!(rwts[0], @"566");
  assert_debug_snapshot!(rwts[1], @"588");
  assert_debug_snapshot!(rwts[2], @"632");
  assert_debug_snapshot!(rwts[3], @"720");
  assert_debug_snapshot!(rwts[4], @"896");
  assert_debug_snapshot!(rwts[5], @"1248");

  //Tests the linearity of the function
  let delta = rwts[1] - rwts[0];
  assert_eq!(rwts[1] + delta * 2, rwts[2]);
  assert_eq!(rwts[2] + delta * 4, rwts[3]);
  assert_eq!(rwts[3] + delta * 8, rwts[4]);
  assert_eq!(rwts[4] + delta * 16, rwts[5]);
}
