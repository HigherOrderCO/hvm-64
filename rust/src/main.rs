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

  // Church Nat constructors
  let c_s = define(book, "c_s", "$ (0 (0 s (0 z k)) (0 (1 (0 k r) s) (0 z r)))");
  let c_z = define(book, "c_z", "$ (0 * (0 a a))");

  // Church Nat constants
  define(book, "c0" , "$ (0 * (0 a a))");
  define(book, "c1" , "$ (0 (0 a R) (0 a R))");
  define(book, "c2" , "$ (0 (1 (0 b a) (0 a R)) (0 b R))");
  define(book, "c3" , "$ (0 (1 (1 (0 c b) (0 b a)) (0 a R)) (0 c R))");
  define(book, "c4" , "$ (0 (1 (1 (1 (0 d c) (0 c b)) (0 b a)) (0 a R)) (0 d R))");
  define(book, "c5" , "$ (0 (1 (1 (1 (1 (0 e d) (0 d c)) (0 c b)) (0 b a)) (0 a R)) (0 e R))");
  define(book, "c6" , "$ (0 (1 (1 (1 (1 (1 (0 f e) (0 e d)) (0 d c)) (0 c b)) (0 b a)) (0 a R)) (0 f R))");
  define(book, "c7" , "$ (0 (1 (1 (1 (1 (1 (1 (0 g f) (0 f e)) (0 e d)) (0 d c)) (0 c b)) (0 b a)) (0 a R)) (0 g R))");
  define(book, "c8" , "$ (0 (1 (1 (1 (1 (1 (1 (1 (0 h g) (0 g f)) (0 f e)) (0 e d)) (0 d c)) (0 c b)) (0 b a)) (0 a R)) (0 h R))");
  define(book, "c9" , "$ (0 (1 (1 (1 (1 (1 (1 (1 (1 (0 i h) (0 h g)) (0 g f)) (0 f e)) (0 e d)) (0 d c)) (0 c b)) (0 b a)) (0 a R)) (0 i R))");
  define(book, "c10", "$ (0 (1 (1 (1 (1 (1 (1 (1 (1 (1 (0 j i) (0 i h)) (0 h g)) (0 g f)) (0 f e)) (0 e d)) (0 d c)) (0 c b)) (0 b a)) (0 a R)) (0 j R))");
  define(book, "c11", "$ (0 (1 (1 (1 (1 (1 (1 (1 (1 (1 (1 (0 k j) (0 j i)) (0 i h)) (0 h g)) (0 g f)) (0 f e)) (0 e d)) (0 d c)) (0 c b)) (0 b a)) (0 a R)) (0 k R))");
  define(book, "c12", "$ (0 (1 (1 (1 (1 (1 (1 (1 (1 (1 (1 (1 (0 l k) (0 k j)) (0 j i)) (0 i h)) (0 h g)) (0 g f)) (0 f e)) (0 e d)) (0 d c)) (0 c b)) (0 b a)) (0 a R)) (0 l R))");
  define(book, "c13", "$ (0 (1 (1 (1 (1 (1 (1 (1 (1 (1 (1 (1 (1 (0 m l) (0 l k)) (0 k j)) (0 j i)) (0 i h)) (0 h g)) (0 g f)) (0 f e)) (0 e d)) (0 d c)) (0 c b)) (0 b a)) (0 a R)) (0 m R))");
  define(book, "c14", "$ (0 (1 (1 (1 (1 (1 (1 (1 (1 (1 (1 (1 (1 (1 (0 n m) (0 m l)) (0 l k)) (0 k j)) (0 j i)) (0 i h)) (0 h g)) (0 g f)) (0 f e)) (0 e d)) (0 d c)) (0 c b)) (0 b a)) (0 a R)) (0 n R))");
  define(book, "c15", "$ (0 (1 (1 (1 (1 (1 (1 (1 (1 (1 (1 (1 (1 (1 (1 (0 o n) (0 n m)) (0 m l)) (0 l k)) (0 k j)) (0 j i)) (0 i h)) (0 h g)) (0 g f)) (0 f e)) (0 e d)) (0 d c)) (0 c b)) (0 b a)) (0 a R)) (0 o R))");
  define(book, "c16", "$ (0 (1 (1 (1 (1 (1 (1 (1 (1 (1 (1 (1 (1 (1 (1 (1 (0 p o) (0 o n)) (0 n m)) (0 m l)) (0 l k)) (0 k j)) (0 j i)) (0 i h)) (0 h g)) (0 g f)) (0 f e)) (0 e d)) (0 d c)) (0 c b)) (0 b a)) (0 a R)) (0 p R))");
  define(book, "c17", "$ (0 (1 (1 (1 (1 (1 (1 (1 (1 (1 (1 (1 (1 (1 (1 (1 (1 (0 q p) (0 p o)) (0 o n)) (0 n m)) (0 m l)) (0 l k)) (0 k j)) (0 j i)) (0 i h)) (0 h g)) (0 g f)) (0 f e)) (0 e d)) (0 d c)) (0 c b)) (0 b a)) (0 a R)) (0 q R))");
  define(book, "c18", "$ (0 (1 (1 (1 (1 (1 (1 (1 (1 (1 (1 (1 (1 (1 (1 (1 (1 (1 (0 r q) (0 q p)) (0 p o)) (0 o n)) (0 n m)) (0 m l)) (0 l k)) (0 k j)) (0 j i)) (0 i h)) (0 h g)) (0 g f)) (0 f e)) (0 e d)) (0 d c)) (0 c b)) (0 b a)) (0 a R)) (0 r R))");
  define(book, "c19", "$ (0 (1 (1 (1 (1 (1 (1 (1 (1 (1 (1 (1 (1 (1 (1 (1 (1 (1 (1 (0 s r) (0 r q)) (0 q p)) (0 p o)) (0 o n)) (0 n m)) (0 m l)) (0 l k)) (0 k j)) (0 j i)) (0 i h)) (0 h g)) (0 g f)) (0 f e)) (0 e d)) (0 d c)) (0 c b)) (0 b a)) (0 a R)) (0 s R))");
  define(book, "c20", "$ (0 (1 (1 (1 (1 (1 (1 (1 (1 (1 (1 (1 (1 (1 (1 (1 (1 (1 (1 (1 (0 t s) (0 s r)) (0 r q)) (0 q p)) (0 p o)) (0 o n)) (0 n m)) (0 m l)) (0 l k)) (0 k j)) (0 j i)) (0 i h)) (0 h g)) (0 g f)) (0 f e)) (0 e d)) (0 d c)) (0 c b)) (0 b a)) (0 a R)) (0 t R))");
  define(book, "c21", "$ (0 (1 (1 (1 (1 (1 (1 (1 (1 (1 (1 (1 (1 (1 (1 (1 (1 (1 (1 (1 (1 (0 u t) (0 t s)) (0 s r)) (0 r q)) (0 q p)) (0 p o)) (0 o n)) (0 n m)) (0 m l)) (0 l k)) (0 k j)) (0 j i)) (0 i h)) (0 h g)) (0 g f)) (0 f e)) (0 e d)) (0 d c)) (0 c b)) (0 b a)) (0 a R)) (0 u R))");
  define(book, "c22", "$ (0 (1 (1 (1 (1 (1 (1 (1 (1 (1 (1 (1 (1 (1 (1 (1 (1 (1 (1 (1 (1 (1 (0 v u) (0 u t)) (0 t s)) (0 s r)) (0 r q)) (0 q p)) (0 p o)) (0 o n)) (0 n m)) (0 m l)) (0 l k)) (0 k j)) (0 j i)) (0 i h)) (0 h g)) (0 g f)) (0 f e)) (0 e d)) (0 d c)) (0 c b)) (0 b a)) (0 a R)) (0 v R))");
  define(book, "c23", "$ (0 (1 (1 (1 (1 (1 (1 (1 (1 (1 (1 (1 (1 (1 (1 (1 (1 (1 (1 (1 (1 (1 (1 (0 w v) (0 v u)) (0 u t)) (0 t s)) (0 s r)) (0 r q)) (0 q p)) (0 p o)) (0 o n)) (0 n m)) (0 m l)) (0 l k)) (0 k j)) (0 j i)) (0 i h)) (0 h g)) (0 g f)) (0 f e)) (0 e d)) (0 d c)) (0 c b)) (0 b a)) (0 a R)) (0 w R))");
  define(book, "c24", "$ (0 (1 (1 (1 (1 (1 (1 (1 (1 (1 (1 (1 (1 (1 (1 (1 (1 (1 (1 (1 (1 (1 (1 (1 (0 x w) (0 w v)) (0 v u)) (0 u t)) (0 t s)) (0 s r)) (0 r q)) (0 q p)) (0 p o)) (0 o n)) (0 n m)) (0 m l)) (0 l k)) (0 k j)) (0 j i)) (0 i h)) (0 h g)) (0 g f)) (0 f e)) (0 e d)) (0 d c)) (0 c b)) (0 b a)) (0 a R)) (0 x R))");

  // Utils
  define(book, "id", "$ (0 x x)");

  // Scott Nats
  define(book, "suc", "$ (0 a (0 (0 a b) (0 * b)))");
  define(book, "zer", "$ (0 * (0 a a))");

  // Generators for a big binary tree
  define(book, "g_s", "$ (0 (100 a b) (0 (0 a (0 b c)) c))");
  define(book, "g_z", "$ (0 x x)");

  // BitString constructors
  // O = λxs λo λi λe (o xs)
  // I = λxs λo λi λe (i xs)
  // E =     λo λi λe e
  define(book, "O", "$ (0 xs (0 (0 xs r) (0 * (0 * r))))");
  define(book, "I", "$ (0 xs (0 * (0 (0 xs r) (0 * r))))");
  define(book, "E", "$ (0 * (0 * (0 e e)))");

  // Decrements a BitString
  // decO = λp(I (dec p))
  // decI = λp(low p)
  // dec  = λx(((x decO) decI) E)
  define(book, "decO", "
    $ (0 p idecp)
    & @I   ~ (0 decp idecp)
    & @dec ~ (0 p decp)
  ");
  define(book, "decI", "
    $ (0 p lowp)
    & @low ~ (0 p lowp)
  ");
  define(book, "dec" , "
    $ (0 (0 @decO (0 @decI (0 @E ret))) ret)
  ");

  // Auxiliary function
  // lowO = λp(O (O p))
  // lowI = λp(O (I p))
  // low  = λx(((x lowO) lowI) E)
  define(book, "lowO", "
    $ (0 p oop)
    & @O ~ (0 p op)
    & @O ~ (0 op oop)
  ");
  define(book, "lowI", "
    $ (0 p oip)
    & @I ~ (0 p ip)
    & @O ~ (0 ip oip)
  ");
  define(book, "low" , "
    $ (0 (0 @lowO (0 @lowI (0 @E ret))) ret)
  ");

  // Decrements a BitString until it is zero
  // runO = λp(run (dec (O p)))
  // runI = λp(run (dec (I p)))
  // run  = λx(((x runO) runI) E)
  define(book, "runO", "
    $ (0 p ret)
    & @run ~ (0 decop ret)
    & @dec ~ (0 op decop)
    & @O   ~ (0 p op)
  ");
  define(book, "runI", "
    $ (0 p ret)
    & @run ~ (0 decip ret)
    & @dec ~ (0 ip decip)
    & @I   ~ (0 p ip)
  ");
  define(book, "run" , "
    $ (0 (0 @runO (0 @runI (0 @E ret))) ret)
  ");

  // Creates a nat, for testing
  // ex0 = ((n suc) zer)
  define(book, "ex0", "
    $ root
    & @c5 ~ (0 @suc (0 @zer root))
  ");

  // Allocates a big tree
  // ex1 = ((n g_s) g_z)
  define(book, "ex1", "
    $ root
    & @c22 ~ (0 @g_s (0 @g_z root))
  ");

  // This example decreases a binary counter until it reaches 0. It uses recursion, based on
  // supercombinators. This, coupled with the REF-ERA rule, allows recursive calls to be collected
  // before they get the chance of expanding forever. This is key for efficient memory usage. Here,
  // `run` is the looping decrementer, `((n I) E)` is a BitString with `n` bits 1. If this test is
  // a success, our program will never need more than ~256 nodes of space; which is the case, as
  // we allocated only 256 nodes for `net` above.
  // main = (run ((n I) E))
  define(book, "ex2", "
    $ main
    & @run ~ (0 nie main)
    & @c20 ~ (0 @I (0 @E nie))
  "); 

  //define(book, "term", "
    //$ main
    //& (0 (1 (0 b a) (0 a R)) (0 b R))
    //~ (0 (0 x x) main)
  //");

  // (λfλx(f x) λgλy(g (g y)))
  //define(book, "term", "
    //$ main
    //&    (0 (1 (0 xb xa) (0 xa xR)) (0 xb xR))
    //~ (0 (0 (2 (0 yb ya) (0 ya yR)) (0 yb yR)) main)
  //");

  // ((N @g_s) @g_z)
  // c2  = (0 (1 (0 b a) (0 a R)) (0 b R))
  // c3  = (0 (1 (1 (0 c b) (0 b a)) (0 a R)) (0 c R))
  // c5  = (0 (1 (1 (1 (1 (0 e d) (0 d c)) (0 c b)) (0 b a)) (0 a R)) (0 e R))
  // c12 = (0 (1 (1 (1 (1 (1 (1 (1 (1 (1 (1 (1 (0 l k) (0 k j)) (0 j i)) (0 i h)) (0 h g)) (0 g f)) (0 f e)) (0 e d)) (0 d c)) (0 c b)) (0 b a)) (0 a R)) (0 l R))
  // c16 = (0 (1 (1 (1 (1 (1 (1 (1 (1 (1 (1 (1 (1 (1 (1 (1 (0 p o) (0 o n)) (0 n m)) (0 m l)) (0 l k)) (0 k j)) (0 j i)) (0 i h)) (0 h g)) (0 g f)) (0 f e)) (0 e d)) (0 d c)) (0 c b)) (0 b a)) (0 a R)) (0 p R))
  // c18 = (0 (1 (1 (1 (1 (1 (1 (1 (1 (1 (1 (1 (1 (1 (1 (1 (1 (1 (0 r q) (0 q p)) (0 p o)) (0 o n)) (0 n m)) (0 m l)) (0 l k)) (0 k j)) (0 j i)) (0 i h)) (0 h g)) (0 g f)) (0 f e)) (0 e d)) (0 d c)) (0 c b)) (0 b a)) (0 a R)) (0 r R))
  define(book, "term", "
    $ root
    & (0 (1 (1 (1 (1 (1 (1 (1 (1 (1 (1 (1 (0 l k) (0 k j)) (0 j i)) (0 i h)) (0 h g)) (0 g f)) (0 f e)) (0 e d)) (0 d c)) (0 c b)) (0 b a)) (0 a R)) (0 l R))
    ~ (0 (0 (2 ga gb) (0 (0 ga (0 gb gc)) gc)) (0 (0 gx gx) root))
  ");

  let got = book.defs.get(&name_to_u32("term")).unwrap();
  println!("  net->root          = 0x{:08x};", got.root.data);
  for i in 0 .. got.acts.len() {
    println!("  net->anni_data[{:2}] = (Wire) {{0x{:08x},0x{:08x}}};", i, got.acts[i].0.data, got.acts[i].1.data);
  }
  for i in 0 .. got.node.len() {
    println!("  net->node_data[{:2}] = (Node) {{0x{:08x},0x{:08x}}};", i, got.node[i].p1.data, got.node[i].p2.data);
  }

  // Initializes the net
  let net = &mut Net::init(1 << 26, &book, name_to_u32("term"));
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
