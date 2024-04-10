use crate::prelude::*;

use core::time::Duration;

use crate::run::Rewrites;

pub fn show_rewrites(rwts: &Rewrites) -> String {
  format!(
    "{}{}{}{}{}{}",
    format_args!("RWTS   : {:>15}\n", pretty_num(rwts.total())),
    format_args!("- ANNI : {:>15}\n", pretty_num(rwts.anni)),
    format_args!("- COMM : {:>15}\n", pretty_num(rwts.comm)),
    format_args!("- ERAS : {:>15}\n", pretty_num(rwts.eras)),
    format_args!("- DREF : {:>15}\n", pretty_num(rwts.dref)),
    format_args!("- OPER : {:>15}\n", pretty_num(rwts.oper)),
  )
}

pub fn show_stats(rwts: &Rewrites, elapsed: Duration) -> String {
  format!(
    "{}{}{}",
    show_rewrites(rwts),
    format_args!("TIME   : {:.3?}\n", elapsed),
    format_args!("RPS    : {:.3} M\n", (rwts.total() as f64) / (elapsed.as_millis() as f64) / 1000.0),
  )
}

fn pretty_num(n: u64) -> String {
  n.to_string()
    .as_bytes()
    .rchunks(3)
    .rev()
    .map(|x| core::str::from_utf8(x).unwrap())
    .flat_map(|x| ["_", x])
    .skip(1)
    .collect()
}
