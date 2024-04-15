use crate::prelude::*;

/// Creates a variable uniquely identified by `id`.
pub(crate) fn create_var(mut id: usize) -> String {
  let mut txt = Vec::new();
  id += 1;
  while id > 0 {
    id -= 1;
    txt.push((id % 26) as u8 + b'a');
    id /= 26;
  }
  txt.reverse();
  String::from_utf8(txt).unwrap()
}

/// Inverse function of [`create_var`].
///
/// Returns None when the provided string is not an output of
/// `create_var`.
pub(crate) fn var_to_num(s: &str) -> Option<usize> {
  let mut n = 0usize;
  for i in s.chars() {
    let i = (i as u32).checked_sub('a' as u32)? as usize;
    if i > 'z' as usize {
      return None;
    }
    n *= 26;
    n += i;
    n += 1;
  }
  n.checked_sub(1) // if it's none, then it means the initial string was ''
}

#[test]
fn test_create_var() {
  assert_eq!(create_var(0), "a");
  assert_eq!(create_var(1), "b");
  assert_eq!(create_var(25), "z");
  assert_eq!(create_var(26), "aa");
  assert_eq!(create_var(27), "ab");
  assert_eq!(create_var(51), "az");
  assert_eq!(create_var(52), "ba");
  assert_eq!(create_var(676), "za");
  assert_eq!(create_var(701), "zz");
  assert_eq!(create_var(702), "aaa");
  assert_eq!(create_var(703), "aab");
  assert_eq!(create_var(728), "aba");
  assert_eq!(create_var(1351), "ayz");
  assert_eq!(create_var(1352), "aza");
  assert_eq!(create_var(1378), "baa");
}

#[test]
fn test_var_to_num() {
  for i in [0, 1, 2, 3, 10, 26, 27, 30, 50, 70] {
    assert_eq!(Some(i), var_to_num(&create_var(i)));
  }
}
