#![allow(dead_code)]
#![allow(unused_variables)]
#![allow(unused_imports)]
#![allow(non_snake_case)]

mod core;
mod lang;

use crate::core::*;
use crate::lang::*;

fn main() {
  // Initializes the book
  let book = &mut Book::new();

  // Church Nats
  let c_s = define(book, 101, "$(0 (0 s (0 z k)) (0 (1 (0 k r) s) (0 z r)))");
  let c_z = define(book, 102, "$(0 * (0 a a))");

  // Constants
  let c0  = define(book,   0, "$(0 * (0 a a))");
  let c1  = define(book,   1, "$(0 (0 a R) (0 a R))");
  let c2  = define(book,   2, "$(0 (2 (0 b a) (0 a R)) (0 b R))");
  let c3  = define(book,   3, "$(0 (3 (3 (0 c b) (0 b a)) (0 a R)) (0 c R))");
  let c4  = define(book,   4, "$(0 (4 (4 (4 (0 d c) (0 c b)) (0 b a)) (0 a R)) (0 d R))");
  let c5  = define(book,   5, "$(0 (5 (5 (5 (5 (0 e d) (0 d c)) (0 c b)) (0 b a)) (0 a R)) (0 e R))");
  let c6  = define(book,   6, "$(0 (6 (6 (6 (6 (6 (0 f e) (0 e d)) (0 d c)) (0 c b)) (0 b a)) (0 a R)) (0 f R))");
  let c7  = define(book,   7, "$(0 (7 (7 (7 (7 (7 (7 (0 g f) (0 f e)) (0 e d)) (0 d c)) (0 c b)) (0 b a)) (0 a R)) (0 g R))");
  let c8  = define(book,   8, "$(0 (8 (8 (8 (8 (8 (8 (8 (0 h g) (0 g f)) (0 f e)) (0 e d)) (0 d c)) (0 c b)) (0 b a)) (0 a R)) (0 h R))");
  let c9  = define(book,   9, "$(0 (9 (9 (9 (9 (9 (9 (9 (9 (0 i h) (0 h g)) (0 g f)) (0 f e)) (0 e d)) (0 d c)) (0 c b)) (0 b a)) (0 a R)) (0 i R))");
  let c10 = define(book,  10, "$(0 (10 (10 (10 (10 (10 (10 (10 (10 (10 (0 j i) (0 i h)) (0 h g)) (0 g f)) (0 f e)) (0 e d)) (0 d c)) (0 c b)) (0 b a)) (0 a R)) (0 j R))");
  let c11 = define(book,  11, "$(0 (11 (11 (11 (11 (11 (11 (11 (11 (11 (11 (0 k j) (0 j i)) (0 i h)) (0 h g)) (0 g f)) (0 f e)) (0 e d)) (0 d c)) (0 c b)) (0 b a)) (0 a R)) (0 k R))");
  let c12 = define(book,  12, "$(0 (12 (12 (12 (12 (12 (12 (12 (12 (12 (12 (12 (0 l k) (0 k j)) (0 j i)) (0 i h)) (0 h g)) (0 g f)) (0 f e)) (0 e d)) (0 d c)) (0 c b)) (0 b a)) (0 a R)) (0 l R))");
  let c13 = define(book,  13, "$(0 (13 (13 (13 (13 (13 (13 (13 (13 (13 (13 (13 (13 (0 m l) (0 l k)) (0 k j)) (0 j i)) (0 i h)) (0 h g)) (0 g f)) (0 f e)) (0 e d)) (0 d c)) (0 c b)) (0 b a)) (0 a R)) (0 m R))");
  let c14 = define(book,  14, "$(0 (14 (14 (14 (14 (14 (14 (14 (14 (14 (14 (14 (14 (14 (0 n m) (0 m l)) (0 l k)) (0 k j)) (0 j i)) (0 i h)) (0 h g)) (0 g f)) (0 f e)) (0 e d)) (0 d c)) (0 c b)) (0 b a)) (0 a R)) (0 n R))");
  let c15 = define(book,  15, "$(0 (15 (15 (15 (15 (15 (15 (15 (15 (15 (15 (15 (15 (15 (15 (0 o n) (0 n m)) (0 m l)) (0 l k)) (0 k j)) (0 j i)) (0 i h)) (0 h g)) (0 g f)) (0 f e)) (0 e d)) (0 d c)) (0 c b)) (0 b a)) (0 a R)) (0 o R))");
  let c16 = define(book,  16, "$(0 (16 (16 (16 (16 (16 (16 (16 (16 (16 (16 (16 (16 (16 (16 (16 (0 p o) (0 o n)) (0 n m)) (0 m l)) (0 l k)) (0 k j)) (0 j i)) (0 i h)) (0 h g)) (0 g f)) (0 f e)) (0 e d)) (0 d c)) (0 c b)) (0 b a)) (0 a R)) (0 p R))");
  let c17 = define(book,  17, "$(0 (17 (17 (17 (17 (17 (17 (17 (17 (17 (17 (17 (17 (17 (17 (17 (17 (0 q p) (0 p o)) (0 o n)) (0 n m)) (0 m l)) (0 l k)) (0 k j)) (0 j i)) (0 i h)) (0 h g)) (0 g f)) (0 f e)) (0 e d)) (0 d c)) (0 c b)) (0 b a)) (0 a R)) (0 q R))");
  let c18 = define(book,  18, "$(0 (18 (18 (18 (18 (18 (18 (18 (18 (18 (18 (18 (18 (18 (18 (18 (18 (18 (0 r q) (0 q p)) (0 p o)) (0 o n)) (0 n m)) (0 m l)) (0 l k)) (0 k j)) (0 j i)) (0 i h)) (0 h g)) (0 g f)) (0 f e)) (0 e d)) (0 d c)) (0 c b)) (0 b a)) (0 a R)) (0 r R))");
  let c19 = define(book,  19, "$(0 (19 (19 (19 (19 (19 (19 (19 (19 (19 (19 (19 (19 (19 (19 (19 (19 (19 (19 (0 s r) (0 r q)) (0 q p)) (0 p o)) (0 o n)) (0 n m)) (0 m l)) (0 l k)) (0 k j)) (0 j i)) (0 i h)) (0 h g)) (0 g f)) (0 f e)) (0 e d)) (0 d c)) (0 c b)) (0 b a)) (0 a R)) (0 s R))");
  let c20 = define(book,  20, "$(0 (20 (20 (20 (20 (20 (20 (20 (20 (20 (20 (20 (20 (20 (20 (20 (20 (20 (20 (20 (0 t s) (0 s r)) (0 r q)) (0 q p)) (0 p o)) (0 o n)) (0 n m)) (0 m l)) (0 l k)) (0 k j)) (0 j i)) (0 i h)) (0 h g)) (0 g f)) (0 f e)) (0 e d)) (0 d c)) (0 c b)) (0 b a)) (0 a R)) (0 t R))");
  let c21 = define(book,  21, "$(0 (21 (21 (21 (21 (21 (21 (21 (21 (21 (21 (21 (21 (21 (21 (21 (21 (21 (21 (21 (21 (0 u t) (0 t s)) (0 s r)) (0 r q)) (0 q p)) (0 p o)) (0 o n)) (0 n m)) (0 m l)) (0 l k)) (0 k j)) (0 j i)) (0 i h)) (0 h g)) (0 g f)) (0 f e)) (0 e d)) (0 d c)) (0 c b)) (0 b a)) (0 a R)) (0 u R))");
  let c22 = define(book,  22, "$(0 (22 (22 (22 (22 (22 (22 (22 (22 (22 (22 (22 (22 (22 (22 (22 (22 (22 (22 (22 (22 (22 (0 v u) (0 u t)) (0 t s)) (0 s r)) (0 r q)) (0 q p)) (0 p o)) (0 o n)) (0 n m)) (0 m l)) (0 l k)) (0 k j)) (0 j i)) (0 i h)) (0 h g)) (0 g f)) (0 f e)) (0 e d)) (0 d c)) (0 c b)) (0 b a)) (0 a R)) (0 v R))");
  let c23 = define(book,  23, "$(0 (23 (23 (23 (23 (23 (23 (23 (23 (23 (23 (23 (23 (23 (23 (23 (23 (23 (23 (23 (23 (23 (23 (0 w v) (0 v u)) (0 u t)) (0 t s)) (0 s r)) (0 r q)) (0 q p)) (0 p o)) (0 o n)) (0 n m)) (0 m l)) (0 l k)) (0 k j)) (0 j i)) (0 i h)) (0 h g)) (0 g f)) (0 f e)) (0 e d)) (0 d c)) (0 c b)) (0 b a)) (0 a R)) (0 w R))");
  let c24 = define(book,  24, "$(0 (24 (24 (24 (24 (24 (24 (24 (24 (24 (24 (24 (24 (24 (24 (24 (24 (24 (24 (24 (24 (24 (24 (24 (0 x w) (0 w v)) (0 v u)) (0 u t)) (0 t s)) (0 s r)) (0 r q)) (0 q p)) (0 p o)) (0 o n)) (0 n m)) (0 m l)) (0 l k)) (0 k j)) (0 j i)) (0 i h)) (0 h g)) (0 g f)) (0 f e)) (0 e d)) (0 d c)) (0 c b)) (0 b a)) (0 a R)) (0 x R))");

  // Utils
  let id  = define(book, 100, "$(0 x x)");

  // Scott Nats
  let suc = define(book, 103, "$(0 a (0 (0 a b) (0 * b)))");
  let zer = define(book, 104, "$(0 * (0 a a))");

  // Gen
  let g_s = define(book, 105, "$(0 (100 a b) (0 (0 a (0 b c)) c))");
  let g_z = define(book, 106, "$(0 x x)");

  // O = λxs λo λi λe (o xs)
  // I = λxs λo λi λe (i xs)
  // E =     λo λi λe e
  let O   = define(book, 200, "$(0 xs (0 (0 xs r) (0 * (0 * r))))");
  let I   = define(book, 201, "$(0 xs (0 * (0 (0 xs r) (0 * r))))");
  let E   = define(book, 202, "$(0 * (0 * (0 e e)))");

  //decO = λp(I (dec p))
  //decI = λp(low p)
  //dec  = λx(((x decO) decI) E)
  let decO = define(book, 203, "$(0 p idecp) & @201 ~ (0 decp idecp) & @205 ~ (0 p decp)");
  let decI = define(book, 204, "$(0 p lowp) & @208 ~ (0 p lowp)");
  let dec  = define(book, 205, "$(0 (0 @203 (0 @204 (0 @202 ret))) ret)");

  // lowO = λp(O (O p))
  // lowI = λp(O (I p))
  // low  = λx(((x lowO) lowI) E)
  let lowO = define(book, 206, "$(0 p oop) & @200 ~ (0 p op) & @200 ~ (0 op oop)");
  let lowI = define(book, 207, "$(0 p oip) & @201 ~ (0 p ip) & @200 ~ (0 ip oip)");
  let low  = define(book, 208, "$(0 (0 @206 (0 @207 (0 @202 ret))) ret)");

  // runO = λp(run (dec (O p)))
  // runI = λp(run (dec (I p)))
  // run  = λx(((x runO) runI) E)
  let runO = define(book, 209, "$(0 p rundecop) & @211 ~ (0 decop rundecop) & @205 ~ (0 op decop) & @200 ~ (0 p op)");
  let runI = define(book, 210, "$(0 p rundecip) & @211 ~ (0 decip rundecip) & @205 ~ (0 ip decip) & @201 ~ (0 p ip)");
  let low  = define(book, 211, "$(0 (0 @209 (0 @210 (0 @202 ret))) ret)");

  // Creates a nat, for testing
  // example_0 = ((n suc) zer)
  let example_0 = define(book, 1000, "
    $ root
    & @5 ~ (0 @103 (0 @104 root))
  ");

  // Allocates a big tree
  // example_1 = ((n g_s) g_z)
  let example_1 = define(book, 1000, "
    $ root
    & @23 ~ (0 @105 (0 @106 root))
  ");

  // This example decreases a binary counter until it reaches 0. It uses recursion, based on
  // supercombinators. This, coupled with the REF-ERA rule, allows recursive calls to be collected
  // before they get the chance of expanding forever. This is key for efficient memory usage. Here,
  // `run` is the looping decrementer, `((n I) E)` is a BitString with `n` bits 1. If this test is
  // a success, our program will never need more than ~256 nodes of space; which is the case, as
  // we allocated only 256 nodes for `net` above.
  // main = (run ((n I) E))
  let example_2 = define(book, 1000, "
    $ main
    & @211 ~ (0 nie main)
    & @20  ~ (0 @201 (0 @202 nie))
  "); 

  // Initializes the net
  let net = &mut Net::init(1 << 26, &book, example_2);
  //println!("[net]\n{}", show_net(&net));

  // Computes its normal form
  let iter = net.normal(book);

  // Shows results and stats
  //println!("[net]\n{}", show_net(&net));
  println!("size: {}", net.node.len());
  println!("used: {}", net.used);
  println!("rwts: {}", net.rwts);
  println!("iter: {}", iter);

}
