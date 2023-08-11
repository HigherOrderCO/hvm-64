#![allow(dead_code)]
#![allow(unused_variables)]
#![allow(unused_imports)]
#![allow(non_snake_case)]

mod core;
mod lang;

use crate::core::*;
use crate::lang::*;

// Syntax
// ------

fn rf(nam: u32) -> Ptr {
  return Ptr { tag: REF, loc: nam };
}

fn main() {
  let fresh = &mut 0;

  let net = &mut Net::new(256);

  // Church Nats
  let c_s = define(net, 101, "$(0 (0 s (0 z k)) (0 (1 (0 k r) s) (0 z r)))");
  let c_z = define(net, 102, "$(0 * (0 a a))");

  // Constants
  let c0  = define(net,   0, "$(0 * (0 a a))");
  let c1  = define(net,   1, "$(0 (0 a R) (0 a R))");
  let c2  = define(net,   2, "$(0 (2 (0 b a) (0 a R)) (0 b R))");
  let c3  = define(net,   3, "$(0 (3 (3 (0 c b) (0 b a)) (0 a R)) (0 c R))");
  let c4  = define(net,   4, "$(0 (4 (4 (4 (0 d c) (0 c b)) (0 b a)) (0 a R)) (0 d R))");
  let c5  = define(net,   5, "$(0 (5 (5 (5 (5 (0 e d) (0 d c)) (0 c b)) (0 b a)) (0 a R)) (0 e R))");
  let c6  = define(net,   6, "$(0 (6 (6 (6 (6 (6 (0 f e) (0 e d)) (0 d c)) (0 c b)) (0 b a)) (0 a R)) (0 f R))");
  let c7  = define(net,   7, "$(0 (7 (7 (7 (7 (7 (7 (0 g f) (0 f e)) (0 e d)) (0 d c)) (0 c b)) (0 b a)) (0 a R)) (0 g R))");
  let c8  = define(net,   8, "$(0 (8 (8 (8 (8 (8 (8 (8 (0 h g) (0 g f)) (0 f e)) (0 e d)) (0 d c)) (0 c b)) (0 b a)) (0 a R)) (0 h R))");
  let c9  = define(net,   9, "$(0 (9 (9 (9 (9 (9 (9 (9 (9 (0 i h) (0 h g)) (0 g f)) (0 f e)) (0 e d)) (0 d c)) (0 c b)) (0 b a)) (0 a R)) (0 i R))");
  let c10 = define(net,  10, "$(0 (10 (10 (10 (10 (10 (10 (10 (10 (10 (0 j i) (0 i h)) (0 h g)) (0 g f)) (0 f e)) (0 e d)) (0 d c)) (0 c b)) (0 b a)) (0 a R)) (0 j R))");
  let c11 = define(net,  11, "$(0 (11 (11 (11 (11 (11 (11 (11 (11 (11 (11 (0 k j) (0 j i)) (0 i h)) (0 h g)) (0 g f)) (0 f e)) (0 e d)) (0 d c)) (0 c b)) (0 b a)) (0 a R)) (0 k R))");
  let c12 = define(net,  12, "$(0 (12 (12 (12 (12 (12 (12 (12 (12 (12 (12 (12 (0 l k) (0 k j)) (0 j i)) (0 i h)) (0 h g)) (0 g f)) (0 f e)) (0 e d)) (0 d c)) (0 c b)) (0 b a)) (0 a R)) (0 l R))");
  let c13 = define(net,  13, "$(0 (13 (13 (13 (13 (13 (13 (13 (13 (13 (13 (13 (13 (0 m l) (0 l k)) (0 k j)) (0 j i)) (0 i h)) (0 h g)) (0 g f)) (0 f e)) (0 e d)) (0 d c)) (0 c b)) (0 b a)) (0 a R)) (0 m R))");
  let c14 = define(net,  14, "$(0 (14 (14 (14 (14 (14 (14 (14 (14 (14 (14 (14 (14 (14 (0 n m) (0 m l)) (0 l k)) (0 k j)) (0 j i)) (0 i h)) (0 h g)) (0 g f)) (0 f e)) (0 e d)) (0 d c)) (0 c b)) (0 b a)) (0 a R)) (0 n R))");
  let c15 = define(net,  15, "$(0 (15 (15 (15 (15 (15 (15 (15 (15 (15 (15 (15 (15 (15 (15 (0 o n) (0 n m)) (0 m l)) (0 l k)) (0 k j)) (0 j i)) (0 i h)) (0 h g)) (0 g f)) (0 f e)) (0 e d)) (0 d c)) (0 c b)) (0 b a)) (0 a R)) (0 o R))");
  let c16 = define(net,  16, "$(0 (16 (16 (16 (16 (16 (16 (16 (16 (16 (16 (16 (16 (16 (16 (16 (0 p o) (0 o n)) (0 n m)) (0 m l)) (0 l k)) (0 k j)) (0 j i)) (0 i h)) (0 h g)) (0 g f)) (0 f e)) (0 e d)) (0 d c)) (0 c b)) (0 b a)) (0 a R)) (0 p R))");
  let c17 = define(net,  17, "$(0 (17 (17 (17 (17 (17 (17 (17 (17 (17 (17 (17 (17 (17 (17 (17 (17 (0 q p) (0 p o)) (0 o n)) (0 n m)) (0 m l)) (0 l k)) (0 k j)) (0 j i)) (0 i h)) (0 h g)) (0 g f)) (0 f e)) (0 e d)) (0 d c)) (0 c b)) (0 b a)) (0 a R)) (0 q R))");
  let c18 = define(net,  18, "$(0 (18 (18 (18 (18 (18 (18 (18 (18 (18 (18 (18 (18 (18 (18 (18 (18 (18 (0 r q) (0 q p)) (0 p o)) (0 o n)) (0 n m)) (0 m l)) (0 l k)) (0 k j)) (0 j i)) (0 i h)) (0 h g)) (0 g f)) (0 f e)) (0 e d)) (0 d c)) (0 c b)) (0 b a)) (0 a R)) (0 r R))");
  let c19 = define(net,  19, "$(0 (19 (19 (19 (19 (19 (19 (19 (19 (19 (19 (19 (19 (19 (19 (19 (19 (19 (19 (0 s r) (0 r q)) (0 q p)) (0 p o)) (0 o n)) (0 n m)) (0 m l)) (0 l k)) (0 k j)) (0 j i)) (0 i h)) (0 h g)) (0 g f)) (0 f e)) (0 e d)) (0 d c)) (0 c b)) (0 b a)) (0 a R)) (0 s R))");
  let c20 = define(net,  20, "$(0 (20 (20 (20 (20 (20 (20 (20 (20 (20 (20 (20 (20 (20 (20 (20 (20 (20 (20 (20 (0 t s) (0 s r)) (0 r q)) (0 q p)) (0 p o)) (0 o n)) (0 n m)) (0 m l)) (0 l k)) (0 k j)) (0 j i)) (0 i h)) (0 h g)) (0 g f)) (0 f e)) (0 e d)) (0 d c)) (0 c b)) (0 b a)) (0 a R)) (0 t R))");
  let c21 = define(net,  21, "$(0 (21 (21 (21 (21 (21 (21 (21 (21 (21 (21 (21 (21 (21 (21 (21 (21 (21 (21 (21 (21 (0 u t) (0 t s)) (0 s r)) (0 r q)) (0 q p)) (0 p o)) (0 o n)) (0 n m)) (0 m l)) (0 l k)) (0 k j)) (0 j i)) (0 i h)) (0 h g)) (0 g f)) (0 f e)) (0 e d)) (0 d c)) (0 c b)) (0 b a)) (0 a R)) (0 u R))");
  let c22 = define(net,  22, "$(0 (22 (22 (22 (22 (22 (22 (22 (22 (22 (22 (22 (22 (22 (22 (22 (22 (22 (22 (22 (22 (22 (0 v u) (0 u t)) (0 t s)) (0 s r)) (0 r q)) (0 q p)) (0 p o)) (0 o n)) (0 n m)) (0 m l)) (0 l k)) (0 k j)) (0 j i)) (0 i h)) (0 h g)) (0 g f)) (0 f e)) (0 e d)) (0 d c)) (0 c b)) (0 b a)) (0 a R)) (0 v R))");
  let c23 = define(net,  23, "$(0 (23 (23 (23 (23 (23 (23 (23 (23 (23 (23 (23 (23 (23 (23 (23 (23 (23 (23 (23 (23 (23 (23 (0 w v) (0 v u)) (0 u t)) (0 t s)) (0 s r)) (0 r q)) (0 q p)) (0 p o)) (0 o n)) (0 n m)) (0 m l)) (0 l k)) (0 k j)) (0 j i)) (0 i h)) (0 h g)) (0 g f)) (0 f e)) (0 e d)) (0 d c)) (0 c b)) (0 b a)) (0 a R)) (0 w R))");
  let c24 = define(net,  24, "$(0 (24 (24 (24 (24 (24 (24 (24 (24 (24 (24 (24 (24 (24 (24 (24 (24 (24 (24 (24 (24 (24 (24 (24 (0 x w) (0 w v)) (0 v u)) (0 u t)) (0 t s)) (0 s r)) (0 r q)) (0 q p)) (0 p o)) (0 o n)) (0 n m)) (0 m l)) (0 l k)) (0 k j)) (0 j i)) (0 i h)) (0 h g)) (0 g f)) (0 f e)) (0 e d)) (0 d c)) (0 c b)) (0 b a)) (0 a R)) (0 x R))");

  // Utils
  let id  = define(net, 100, "$(0 x x)");

  // Scott Nats
  let suc = define(net, 103, "$(0 a (0 (0 a b) (0 * b)))");
  let zer = define(net, 104, "$(0 * (0 a a))");

  // Gen
  let g_s = define(net, 105, "$(0 (100 a b) (0 (0 a (0 b c)) c))");
  let g_z = define(net, 106, "$(0 x x)");

  // O = λxs λo λi λe (o xs)
  // I = λxs λo λi λe (i xs)
  // E =     λo λi λe e
  let O   = define(net, 200, "$(0 xs (0 (0 xs r) (0 * (0 * r))))");
  let I   = define(net, 201, "$(0 xs (0 * (0 (0 xs r) (0 * r))))");
  let E   = define(net, 202, "$(0 * (0 * (0 e e)))");

  //decO = λp(I (dec p))
  //decI = λp(low p)
  //dec  = λx(((x decO) decI) E)
  let decO = define(net, 203, "$(0 p idecp) & @201 ~ (0 decp idecp) & @205 ~ (0 p decp)");
  let decI = define(net, 204, "$(0 p lowp) & @208 ~ (0 p lowp)");
  let dec  = define(net, 205, "$(0 (0 @203 (0 @204 (0 @202 ret))) ret)");

  // lowO = λp(O (O p))
  // lowI = λp(O (I p))
  // low  = λx(((x lowO) lowI) E)
  let lowO = define(net, 206, "$(0 p oop) & @200 ~ (0 p op) & @200 ~ (0 op oop)");
  let lowI = define(net, 207, "$(0 p oip) & @201 ~ (0 p ip) & @200 ~ (0 ip oip)");
  let low  = define(net, 208, "$(0 (0 @206 (0 @207 (0 @202 ret))) ret)");

  // runO = λp(run (dec (O p)))
  // runI = λp(run (dec (I p)))
  // run  = λx(((x runO) runI) E)
  let runO = define(net, 209, "$(0 p rundecop) & @211 ~ (0 decop rundecop) & @205 ~ (0 op decop) & @200 ~ (0 p op)");
  let runI = define(net, 210, "$(0 p rundecip) & @211 ~ (0 decip rundecip) & @205 ~ (0 ip decip) & @201 ~ (0 p ip)");
  let low  = define(net, 211, "$(0 (0 @209 (0 @210 (0 @202 ret))) ret)");

  // This example decreases a binary counter until it reaches 0. It uses recursion, based on
  // supercombinators. This, coupled with the REF-ERA rule, allows recursive calls to be collected
  // before they get the chance of expanding forever. This is key for efficient memory usage. Here,
  // `run` is the looping decrementer, `((n I) E)` is a BitString with `n` bits 1. If this test is
  // a success, our program will never need more than ~256 nodes of space; which is the case, as
  // we allocated only 256 nodes for `net` above.
  // main = (run ((n I) E))
  let main = define(net, 1000, "
    $ main
    & @211 ~ (0 nie main)
    & @20  ~ (0 @201 (0 @202 nie))
  "); 

  let mut root = rf(main);
  net.deref(&mut root);
  net.term.root = root;

  println!("[net]\n{}", show_net(&net));

  let (rwts, iter) = net.normal();

  println!("[net]\n{}", show_net(&net));
  println!("size: {}", net.term.node.len());
  println!("used: {}", net.used);
  println!("rwts: {}", net.rwts);
  println!("iter: {}", iter);

}

