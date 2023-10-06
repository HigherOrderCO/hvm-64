#![allow(dead_code)]
#![allow(unused_variables)]
#![allow(unused_imports)]
#![allow(non_snake_case)]
#![allow(non_upper_case_globals)]

mod core;
mod lang;
#[rustfmt::skip]
mod book;
mod comp;

use crate::core::*;
use crate::lang::*;

fn main() {
  let book = &book::setup_book();

  // Initializes the net
  let net = &mut Net::new(1 << 28);
  net.boot(name_to_val("ex3"));

  // Marks initial time
  let start = std::time::Instant::now();

  // Computes its normal form
  net.normal(book);

  //Shows results and stats
  //println!("[net]\n{}", show_net(&net));
  println!("ANNI: {}", net.anni);
  println!("COMM: {}", net.comm);
  println!("ERAS: {}", net.eras);
  println!("DREF: {}", net.dref);
  println!("-----");
  println!("INTR: {}", net.anni + net.comm + net.eras);
  println!("RWTS: {}", net.anni + net.comm + net.eras + net.dref);
  println!("TIME: {:.3} s", (start.elapsed().as_millis() as f64) / 1000.0);
  println!("RPS : {:.3} million", (net.rewrites() as f64) / (start.elapsed().as_millis() as f64) / 1000.0);

  //println!("{}", &comp::compile_cuda_book(&book));
}

//ANNI: 1605959706
//COMM: 3276807
//ERAS: 536608764
//DREF: 801832962
//-----
//INTR: 2145845277
//RWTS: 2947678239
//TIME: 12.193 s
//RPS : 241.752 million
