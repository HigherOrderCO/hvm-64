#![allow(dead_code)]
#![allow(unused_variables)]

mod core;
mod lang;

use crate::core::*;
use crate::lang::*;

// Syntax
// ------

fn main() {
  let fresh = &mut 0;

  let id   = do_parse("(0 x x)");
  let c2   = do_parse("(0 (2 (0 b a) (0 a R)) (0 b R))");
  let c3   = do_parse("(0 (3 (3 (0 c b) (0 b a)) (0 a R)) (0 c R))");
  let c4   = do_parse("(0 (4 (4 (4 (0 d c) (0 c b)) (0 b a)) (0 a R)) (0 d R))");
  let c5   = do_parse("(0 (5 (5 (5 (5 (0 e d) (0 d c)) (0 c b)) (0 b a)) (0 a R)) (0 e R))");
  // continue, from c6 to c24
  let c6   = do_parse("(0 (6 (6 (6 (6 (6 (0 f e) (0 e d)) (0 d c)) (0 c b)) (0 b a)) (0 a R)) (0 f R))");
  let c7   = do_parse("(0 (7 (7 (7 (7 (7 (7 (0 g f) (0 f e)) (0 e d)) (0 d c)) (0 c b)) (0 b a)) (0 a R)) (0 g R))");
  let c8   = do_parse("(0 (8 (8 (8 (8 (8 (8 (8 (0 h g) (0 g f)) (0 f e)) (0 e d)) (0 d c)) (0 c b)) (0 b a)) (0 a R)) (0 h R))");
  let c9   = do_parse("(0 (9 (9 (9 (9 (9 (9 (9 (9 (0 i h) (0 h g)) (0 g f)) (0 f e)) (0 e d)) (0 d c)) (0 c b)) (0 b a)) (0 a R)) (0 i R))");
  let c10  = do_parse("(0 (10 (10 (10 (10 (10 (10 (10 (10 (10 (0 j i) (0 i h)) (0 h g)) (0 g f)) (0 f e)) (0 e d)) (0 d c)) (0 c b)) (0 b a)) (0 a R)) (0 j R))");
  let c11  = do_parse("(0 (11 (11 (11 (11 (11 (11 (11 (11 (11 (11 (0 k j) (0 j i)) (0 i h)) (0 h g)) (0 g f)) (0 f e)) (0 e d)) (0 d c)) (0 c b)) (0 b a)) (0 a R)) (0 k R))");
  let c12  = do_parse("(0 (12 (12 (12 (12 (12 (12 (12 (12 (12 (12 (12 (0 l k) (0 k j)) (0 j i)) (0 i h)) (0 h g)) (0 g f)) (0 f e)) (0 e d)) (0 d c)) (0 c b)) (0 b a)) (0 a R)) (0 l R))");
  let c13  = do_parse("(0 (13 (13 (13 (13 (13 (13 (13 (13 (13 (13 (13 (13 (0 m l) (0 l k)) (0 k j)) (0 j i)) (0 i h)) (0 h g)) (0 g f)) (0 f e)) (0 e d)) (0 d c)) (0 c b)) (0 b a)) (0 a R)) (0 m R))");
  let c14  = do_parse("(0 (14 (14 (14 (14 (14 (14 (14 (14 (14 (14 (14 (14 (14 (0 n m) (0 m l)) (0 l k)) (0 k j)) (0 j i)) (0 i h)) (0 h g)) (0 g f)) (0 f e)) (0 e d)) (0 d c)) (0 c b)) (0 b a)) (0 a R)) (0 n R))");
  let c15  = do_parse("(0 (15 (15 (15 (15 (15 (15 (15 (15 (15 (15 (15 (15 (15 (15 (0 o n) (0 n m)) (0 m l)) (0 l k)) (0 k j)) (0 j i)) (0 i h)) (0 h g)) (0 g f)) (0 f e)) (0 e d)) (0 d c)) (0 c b)) (0 b a)) (0 a R)) (0 o R))");
  let c16  = do_parse("(0 (16 (16 (16 (16 (16 (16 (16 (16 (16 (16 (16 (16 (16 (16 (16 (0 p o) (0 o n)) (0 n m)) (0 m l)) (0 l k)) (0 k j)) (0 j i)) (0 i h)) (0 h g)) (0 g f)) (0 f e)) (0 e d)) (0 d c)) (0 c b)) (0 b a)) (0 a R)) (0 p R))");
  let c17  = do_parse("(0 (17 (17 (17 (17 (17 (17 (17 (17 (17 (17 (17 (17 (17 (17 (17 (17 (0 q p) (0 p o)) (0 o n)) (0 n m)) (0 m l)) (0 l k)) (0 k j)) (0 j i)) (0 i h)) (0 h g)) (0 g f)) (0 f e)) (0 e d)) (0 d c)) (0 c b)) (0 b a)) (0 a R)) (0 q R))");
  let c18  = do_parse("(0 (18 (18 (18 (18 (18 (18 (18 (18 (18 (18 (18 (18 (18 (18 (18 (18 (18 (0 r q) (0 q p)) (0 p o)) (0 o n)) (0 n m)) (0 m l)) (0 l k)) (0 k j)) (0 j i)) (0 i h)) (0 h g)) (0 g f)) (0 f e)) (0 e d)) (0 d c)) (0 c b)) (0 b a)) (0 a R)) (0 r R))");
  let c19  = do_parse("(0 (19 (19 (19 (19 (19 (19 (19 (19 (19 (19 (19 (19 (19 (19 (19 (19 (19 (19 (0 s r) (0 r q)) (0 q p)) (0 p o)) (0 o n)) (0 n m)) (0 m l)) (0 l k)) (0 k j)) (0 j i)) (0 i h)) (0 h g)) (0 g f)) (0 f e)) (0 e d)) (0 d c)) (0 c b)) (0 b a)) (0 a R)) (0 s R))");
  let c20  = do_parse("(0 (20 (20 (20 (20 (20 (20 (20 (20 (20 (20 (20 (20 (20 (20 (20 (20 (20 (20 (20 (0 t s) (0 s r)) (0 r q)) (0 q p)) (0 p o)) (0 o n)) (0 n m)) (0 m l)) (0 l k)) (0 k j)) (0 j i)) (0 i h)) (0 h g)) (0 g f)) (0 f e)) (0 e d)) (0 d c)) (0 c b)) (0 b a)) (0 a R)) (0 t R))");
  let c21  = do_parse("(0 (21 (21 (21 (21 (21 (21 (21 (21 (21 (21 (21 (21 (21 (21 (21 (21 (21 (21 (21 (21 (0 u t) (0 t s)) (0 s r)) (0 r q)) (0 q p)) (0 p o)) (0 o n)) (0 n m)) (0 m l)) (0 l k)) (0 k j)) (0 j i)) (0 i h)) (0 h g)) (0 g f)) (0 f e)) (0 e d)) (0 d c)) (0 c b)) (0 b a)) (0 a R)) (0 u R))");
  let c22  = do_parse("(0 (22 (22 (22 (22 (22 (22 (22 (22 (22 (22 (22 (22 (22 (22 (22 (22 (22 (22 (22 (22 (22 (0 v u) (0 u t)) (0 t s)) (0 s r)) (0 r q)) (0 q p)) (0 p o)) (0 o n)) (0 n m)) (0 m l)) (0 l k)) (0 k j)) (0 j i)) (0 i h)) (0 h g)) (0 g f)) (0 f e)) (0 e d)) (0 d c)) (0 c b)) (0 b a)) (0 a R)) (0 v R))");
  let c23  = do_parse("(0 (23 (23 (23 (23 (23 (23 (23 (23 (23 (23 (23 (23 (23 (23 (23 (23 (23 (23 (23 (23 (23 (23 (0 w v) (0 v u)) (0 u t)) (0 t s)) (0 s r)) (0 r q)) (0 q p)) (0 p o)) (0 o n)) (0 n m)) (0 m l)) (0 l k)) (0 k j)) (0 j i)) (0 i h)) (0 h g)) (0 g f)) (0 f e)) (0 e d)) (0 d c)) (0 c b)) (0 b a)) (0 a R)) (0 w R))");
  let c24  = do_parse("(0 (24 (24 (24 (24 (24 (24 (24 (24 (24 (24 (24 (24 (24 (24 (24 (24 (24 (24 (24 (24 (24 (24 (24 (0 x w) (0 w v)) (0 v u)) (0 u t)) (0 t s)) (0 s r)) (0 r q)) (0 q p)) (0 p o)) (0 o n)) (0 n m)) (0 m l)) (0 l k)) (0 k j)) (0 j i)) (0 i h)) (0 h g)) (0 g f)) (0 f e)) (0 e d)) (0 d c)) (0 c b)) (0 b a)) (0 a R)) (0 x R))");

  //let csuc = do_parse("(0 (0 s (0 z k)) (0 s(1 (0 k r) s) (0 z r)))"); // λn λs λz (s ((n s) z))
  let succ = do_parse("(0 a (0 (0 a b) (0 * b)))");
  let zero = do_parse("(0 * (0 a a))");
  let gens = do_parse("(0 (100 a b) (0 (0 a (0 b c)) c))");
  let genz = do_parse("(0 x x)");

  let net  = &mut Net::new();

  // (n gens genz)
  let func = do_make_tree(net, &do_copy_tree(&c22, fresh));
  let args = do_make_tree(net, &arg(do_copy_tree(&gens, fresh), arg(do_copy_tree(&genz, fresh), num(42))));

  net.pair.push((func, args));
  println!("net:\n{}", show_net(&net));

  let mut rwts = 0;
  let mut iter = 0;
  loop {
    net.reduce();
    println!("... {}", net.pair.len());
    if net.rwts == rwts {
      break;
    }
    rwts = net.rwts;
    iter = iter + 1;
  }

  //println!("net:\n{}", show_net(&net));
  println!("size: {}", net.node.len());
  println!("rwts: {}", net.rwts);
  println!("iter: {}", iter);

}
