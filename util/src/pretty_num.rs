use crate::prelude::*;
use core::str::from_utf8;

pub fn pretty_num(n: u64) -> String {
  n.to_string().as_bytes().rchunks(3).rev().map(|x| from_utf8(x).unwrap()).flat_map(|x| ["_", x]).skip(1).collect()
}
