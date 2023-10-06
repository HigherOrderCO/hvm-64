#![cfg_attr(rustfmt, rustfmt_skip)]

use crate::{core::Book, lang::define};

pub fn setup_book() -> Book {
  // Initializes the book
  let mut book = Book::new();

  // Church Nat constructors
  let c_z = define(&mut book, "c_z", "$ (0 * (0 a a))");
  let c_s = define(&mut book, "c_s", "$ (0 (0 s (0 z k)) (0 (1 (0 k r) s) (0 z r)))");

  // Utils
  define(&mut book, "id", "$ (0 x x)");

  // Church Nat constants
  {
    define(&mut book, "c0" , "$ (0 * (0 a a))");
    define(&mut book, "c1" , "$ (0 (0 a R) (0 a R))");
    define(&mut book, "c2" , "$ (0 (1 (0 b a) (0 a R)) (0 b R))");
    define(&mut book, "c3" , "$ (0 (1 (1 (0 c b) (0 b a)) (0 a R)) (0 c R))");
    define(&mut book, "c4" , "$ (0 (1 (1 (1 (0 d c) (0 c b)) (0 b a)) (0 a R)) (0 d R))");
    define(&mut book, "c5" , "$ (0 (1 (1 (1 (1 (0 e d) (0 d c)) (0 c b)) (0 b a)) (0 a R)) (0 e R))");
    define(&mut book, "c6" , "$ (0 (1 (1 (1 (1 (1 (0 f e) (0 e d)) (0 d c)) (0 c b)) (0 b a)) (0 a R)) (0 f R))");
    define(&mut book, "c7" , "$ (0 (1 (1 (1 (1 (1 (1 (0 g f) (0 f e)) (0 e d)) (0 d c)) (0 c b)) (0 b a)) (0 a R)) (0 g R))");
    define(&mut book, "c8" , "$ (0 (1 (1 (1 (1 (1 (1 (1 (0 h g) (0 g f)) (0 f e)) (0 e d)) (0 d c)) (0 c b)) (0 b a)) (0 a R)) (0 h R))");
    define(&mut book, "c9" , "$ (0 (1 (1 (1 (1 (1 (1 (1 (1 (0 i h) (0 h g)) (0 g f)) (0 f e)) (0 e d)) (0 d c)) (0 c b)) (0 b a)) (0 a R)) (0 i R))");
    define(&mut book, "c10", "$ (0 (1 (1 (1 (1 (1 (1 (1 (1 (1 (0 j i) (0 i h)) (0 h g)) (0 g f)) (0 f e)) (0 e d)) (0 d c)) (0 c b)) (0 b a)) (0 a R)) (0 j R))");
    define(&mut book, "c11", "$ (0 (1 (1 (1 (1 (1 (1 (1 (1 (1 (1 (0 k j) (0 j i)) (0 i h)) (0 h g)) (0 g f)) (0 f e)) (0 e d)) (0 d c)) (0 c b)) (0 b a)) (0 a R)) (0 k R))");
    define(&mut book, "c12", "$ (0 (1 (1 (1 (1 (1 (1 (1 (1 (1 (1 (1 (0 l k) (0 k j)) (0 j i)) (0 i h)) (0 h g)) (0 g f)) (0 f e)) (0 e d)) (0 d c)) (0 c b)) (0 b a)) (0 a R)) (0 l R))");
    define(&mut book, "c13", "$ (0 (1 (1 (1 (1 (1 (1 (1 (1 (1 (1 (1 (1 (0 m l) (0 l k)) (0 k j)) (0 j i)) (0 i h)) (0 h g)) (0 g f)) (0 f e)) (0 e d)) (0 d c)) (0 c b)) (0 b a)) (0 a R)) (0 m R))");
    define(&mut book, "c14", "$ (0 (1 (1 (1 (1 (1 (1 (1 (1 (1 (1 (1 (1 (1 (0 n m) (0 m l)) (0 l k)) (0 k j)) (0 j i)) (0 i h)) (0 h g)) (0 g f)) (0 f e)) (0 e d)) (0 d c)) (0 c b)) (0 b a)) (0 a R)) (0 n R))");
    define(&mut book, "c15", "$ (0 (1 (1 (1 (1 (1 (1 (1 (1 (1 (1 (1 (1 (1 (1 (0 o n) (0 n m)) (0 m l)) (0 l k)) (0 k j)) (0 j i)) (0 i h)) (0 h g)) (0 g f)) (0 f e)) (0 e d)) (0 d c)) (0 c b)) (0 b a)) (0 a R)) (0 o R))");
    define(&mut book, "c16", "$ (0 (1 (1 (1 (1 (1 (1 (1 (1 (1 (1 (1 (1 (1 (1 (1 (0 p o) (0 o n)) (0 n m)) (0 m l)) (0 l k)) (0 k j)) (0 j i)) (0 i h)) (0 h g)) (0 g f)) (0 f e)) (0 e d)) (0 d c)) (0 c b)) (0 b a)) (0 a R)) (0 p R))");
    define(&mut book, "c17", "$ (0 (1 (1 (1 (1 (1 (1 (1 (1 (1 (1 (1 (1 (1 (1 (1 (1 (0 q p) (0 p o)) (0 o n)) (0 n m)) (0 m l)) (0 l k)) (0 k j)) (0 j i)) (0 i h)) (0 h g)) (0 g f)) (0 f e)) (0 e d)) (0 d c)) (0 c b)) (0 b a)) (0 a R)) (0 q R))");
    define(&mut book, "c18", "$ (0 (1 (1 (1 (1 (1 (1 (1 (1 (1 (1 (1 (1 (1 (1 (1 (1 (1 (0 r q) (0 q p)) (0 p o)) (0 o n)) (0 n m)) (0 m l)) (0 l k)) (0 k j)) (0 j i)) (0 i h)) (0 h g)) (0 g f)) (0 f e)) (0 e d)) (0 d c)) (0 c b)) (0 b a)) (0 a R)) (0 r R))");
    define(&mut book, "c19", "$ (0 (1 (1 (1 (1 (1 (1 (1 (1 (1 (1 (1 (1 (1 (1 (1 (1 (1 (1 (0 s r) (0 r q)) (0 q p)) (0 p o)) (0 o n)) (0 n m)) (0 m l)) (0 l k)) (0 k j)) (0 j i)) (0 i h)) (0 h g)) (0 g f)) (0 f e)) (0 e d)) (0 d c)) (0 c b)) (0 b a)) (0 a R)) (0 s R))");
    define(&mut book, "c20", "$ (0 (1 (1 (1 (1 (1 (1 (1 (1 (1 (1 (1 (1 (1 (1 (1 (1 (1 (1 (1 (0 t s) (0 s r)) (0 r q)) (0 q p)) (0 p o)) (0 o n)) (0 n m)) (0 m l)) (0 l k)) (0 k j)) (0 j i)) (0 i h)) (0 h g)) (0 g f)) (0 f e)) (0 e d)) (0 d c)) (0 c b)) (0 b a)) (0 a R)) (0 t R))");
    define(&mut book, "c21", "$ (0 (1 (1 (1 (1 (1 (1 (1 (1 (1 (1 (1 (1 (1 (1 (1 (1 (1 (1 (1 (1 (0 u t) (0 t s)) (0 s r)) (0 r q)) (0 q p)) (0 p o)) (0 o n)) (0 n m)) (0 m l)) (0 l k)) (0 k j)) (0 j i)) (0 i h)) (0 h g)) (0 g f)) (0 f e)) (0 e d)) (0 d c)) (0 c b)) (0 b a)) (0 a R)) (0 u R))");
    define(&mut book, "c22", "$ (0 (1 (1 (1 (1 (1 (1 (1 (1 (1 (1 (1 (1 (1 (1 (1 (1 (1 (1 (1 (1 (1 (0 v u) (0 u t)) (0 t s)) (0 s r)) (0 r q)) (0 q p)) (0 p o)) (0 o n)) (0 n m)) (0 m l)) (0 l k)) (0 k j)) (0 j i)) (0 i h)) (0 h g)) (0 g f)) (0 f e)) (0 e d)) (0 d c)) (0 c b)) (0 b a)) (0 a R)) (0 v R))");
    define(&mut book, "c23", "$ (0 (1 (1 (1 (1 (1 (1 (1 (1 (1 (1 (1 (1 (1 (1 (1 (1 (1 (1 (1 (1 (1 (1 (0 w v) (0 v u)) (0 u t)) (0 t s)) (0 s r)) (0 r q)) (0 q p)) (0 p o)) (0 o n)) (0 n m)) (0 m l)) (0 l k)) (0 k j)) (0 j i)) (0 i h)) (0 h g)) (0 g f)) (0 f e)) (0 e d)) (0 d c)) (0 c b)) (0 b a)) (0 a R)) (0 w R))");
    define(&mut book, "c24", "$ (0 (1 (1 (1 (1 (1 (1 (1 (1 (1 (1 (1 (1 (1 (1 (1 (1 (1 (1 (1 (1 (1 (1 (1 (0 x w) (0 w v)) (0 v u)) (0 u t)) (0 t s)) (0 s r)) (0 r q)) (0 q p)) (0 p o)) (0 o n)) (0 n m)) (0 m l)) (0 l k)) (0 k j)) (0 j i)) (0 i h)) (0 h g)) (0 g f)) (0 f e)) (0 e d)) (0 d c)) (0 c b)) (0 b a)) (0 a R)) (0 x R))");
    define(&mut book, "c25", "$ (0 (1 (1 (1 (1 (1 (1 (1 (1 (1 (1 (1 (1 (1 (1 (1 (1 (1 (1 (1 (1 (1 (1 (1 (1 (0 y x) (0 x w)) (0 w v)) (0 v u)) (0 u t)) (0 t s)) (0 s r)) (0 r q)) (0 q p)) (0 p o)) (0 o n)) (0 n m)) (0 m l)) (0 l k)) (0 k j)) (0 j i)) (0 i h)) (0 h g)) (0 g f)) (0 f e)) (0 e d)) (0 d c)) (0 c b)) (0 b a)) (0 a R)) (0 x R))");
    define(&mut book, "c26", "$ (0 (1 (1 (1 (1 (1 (1 (1 (1 (1 (1 (1 (1 (1 (1 (1 (1 (1 (1 (1 (1 (1 (1 (1 (1 (1 (0 z y) (0 y x)) (0 x w)) (0 w v)) (0 v u)) (0 u t)) (0 t s)) (0 s r)) (0 r q)) (0 q p)) (0 p o)) (0 o n)) (0 n m)) (0 m l)) (0 l k)) (0 k j)) (0 j i)) (0 i h)) (0 h g)) (0 g f)) (0 f e)) (0 e d)) (0 d c)) (0 c b)) (0 b a)) (0 a R)) (0 x R))");

    define(&mut book, "k0" , "$ (0 * (0 a a))");
    define(&mut book, "k1" , "$ (0 (0 a R) (0 a R))");
    define(&mut book, "k2" , "$ (0 (2 (0 b a) (0 a R)) (0 b R))");
    define(&mut book, "k3" , "$ (0 (2 (2 (0 c b) (0 b a)) (0 a R)) (0 c R))");
    define(&mut book, "k4" , "$ (0 (2 (2 (2 (0 d c) (0 c b)) (0 b a)) (0 a R)) (0 d R))");
    define(&mut book, "k5" , "$ (0 (2 (2 (2 (2 (0 e d) (0 d c)) (0 c b)) (0 b a)) (0 a R)) (0 e R))");
    define(&mut book, "k6" , "$ (0 (2 (2 (2 (2 (2 (0 f e) (0 e d)) (0 d c)) (0 c b)) (0 b a)) (0 a R)) (0 f R))");
    define(&mut book, "k7" , "$ (0 (2 (2 (2 (2 (2 (2 (0 g f) (0 f e)) (0 e d)) (0 d c)) (0 c b)) (0 b a)) (0 a R)) (0 g R))");
    define(&mut book, "k8" , "$ (0 (2 (2 (2 (2 (2 (2 (2 (0 h g) (0 g f)) (0 f e)) (0 e d)) (0 d c)) (0 c b)) (0 b a)) (0 a R)) (0 h R))");
    define(&mut book, "k9" , "$ (0 (2 (2 (2 (2 (2 (2 (2 (2 (0 i h) (0 h g)) (0 g f)) (0 f e)) (0 e d)) (0 d c)) (0 c b)) (0 b a)) (0 a R)) (0 i R))");
    define(&mut book, "k10", "$ (0 (2 (2 (2 (2 (2 (2 (2 (2 (2 (0 j i) (0 i h)) (0 h g)) (0 g f)) (0 f e)) (0 e d)) (0 d c)) (0 c b)) (0 b a)) (0 a R)) (0 j R))");
    define(&mut book, "k11", "$ (0 (2 (2 (2 (2 (2 (2 (2 (2 (2 (2 (0 k j) (0 j i)) (0 i h)) (0 h g)) (0 g f)) (0 f e)) (0 e d)) (0 d c)) (0 c b)) (0 b a)) (0 a R)) (0 k R))");
    define(&mut book, "k12", "$ (0 (2 (2 (2 (2 (2 (2 (2 (2 (2 (2 (2 (0 l k) (0 k j)) (0 j i)) (0 i h)) (0 h g)) (0 g f)) (0 f e)) (0 e d)) (0 d c)) (0 c b)) (0 b a)) (0 a R)) (0 l R))");
    define(&mut book, "k13", "$ (0 (2 (2 (2 (2 (2 (2 (2 (2 (2 (2 (2 (2 (0 m l) (0 l k)) (0 k j)) (0 j i)) (0 i h)) (0 h g)) (0 g f)) (0 f e)) (0 e d)) (0 d c)) (0 c b)) (0 b a)) (0 a R)) (0 m R))");
    define(&mut book, "k14", "$ (0 (2 (2 (2 (2 (2 (2 (2 (2 (2 (2 (2 (2 (2 (0 n m) (0 m l)) (0 l k)) (0 k j)) (0 j i)) (0 i h)) (0 h g)) (0 g f)) (0 f e)) (0 e d)) (0 d c)) (0 c b)) (0 b a)) (0 a R)) (0 n R))");
    define(&mut book, "k15", "$ (0 (2 (2 (2 (2 (2 (2 (2 (2 (2 (2 (2 (2 (2 (2 (0 o n) (0 n m)) (0 m l)) (0 l k)) (0 k j)) (0 j i)) (0 i h)) (0 h g)) (0 g f)) (0 f e)) (0 e d)) (0 d c)) (0 c b)) (0 b a)) (0 a R)) (0 o R))");
    define(&mut book, "k16", "$ (0 (2 (2 (2 (2 (2 (2 (2 (2 (2 (2 (2 (2 (2 (2 (2 (0 p o) (0 o n)) (0 n m)) (0 m l)) (0 l k)) (0 k j)) (0 j i)) (0 i h)) (0 h g)) (0 g f)) (0 f e)) (0 e d)) (0 d c)) (0 c b)) (0 b a)) (0 a R)) (0 p R))");
    define(&mut book, "k17", "$ (0 (2 (2 (2 (2 (2 (2 (2 (2 (2 (2 (2 (2 (2 (2 (2 (2 (0 q p) (0 p o)) (0 o n)) (0 n m)) (0 m l)) (0 l k)) (0 k j)) (0 j i)) (0 i h)) (0 h g)) (0 g f)) (0 f e)) (0 e d)) (0 d c)) (0 c b)) (0 b a)) (0 a R)) (0 q R))");
    define(&mut book, "k18", "$ (0 (2 (2 (2 (2 (2 (2 (2 (2 (2 (2 (2 (2 (2 (2 (2 (2 (2 (0 r q) (0 q p)) (0 p o)) (0 o n)) (0 n m)) (0 m l)) (0 l k)) (0 k j)) (0 j i)) (0 i h)) (0 h g)) (0 g f)) (0 f e)) (0 e d)) (0 d c)) (0 c b)) (0 b a)) (0 a R)) (0 r R))");
    define(&mut book, "k19", "$ (0 (2 (2 (2 (2 (2 (2 (2 (2 (2 (2 (2 (2 (2 (2 (2 (2 (2 (2 (0 s r) (0 r q)) (0 q p)) (0 p o)) (0 o n)) (0 n m)) (0 m l)) (0 l k)) (0 k j)) (0 j i)) (0 i h)) (0 h g)) (0 g f)) (0 f e)) (0 e d)) (0 d c)) (0 c b)) (0 b a)) (0 a R)) (0 s R))");
    define(&mut book, "k20", "$ (0 (2 (2 (2 (2 (2 (2 (2 (2 (2 (2 (2 (2 (2 (2 (2 (2 (2 (2 (2 (0 t s) (0 s r)) (0 r q)) (0 q p)) (0 p o)) (0 o n)) (0 n m)) (0 m l)) (0 l k)) (0 k j)) (0 j i)) (0 i h)) (0 h g)) (0 g f)) (0 f e)) (0 e d)) (0 d c)) (0 c b)) (0 b a)) (0 a R)) (0 t R))");
    define(&mut book, "k21", "$ (0 (2 (2 (2 (2 (2 (2 (2 (2 (2 (2 (2 (2 (2 (2 (2 (2 (2 (2 (2 (2 (0 u t) (0 t s)) (0 s r)) (0 r q)) (0 q p)) (0 p o)) (0 o n)) (0 n m)) (0 m l)) (0 l k)) (0 k j)) (0 j i)) (0 i h)) (0 h g)) (0 g f)) (0 f e)) (0 e d)) (0 d c)) (0 c b)) (0 b a)) (0 a R)) (0 u R))");
    define(&mut book, "k22", "$ (0 (2 (2 (2 (2 (2 (2 (2 (2 (2 (2 (2 (2 (2 (2 (2 (2 (2 (2 (2 (2 (2 (0 v u) (0 u t)) (0 t s)) (0 s r)) (0 r q)) (0 q p)) (0 p o)) (0 o n)) (0 n m)) (0 m l)) (0 l k)) (0 k j)) (0 j i)) (0 i h)) (0 h g)) (0 g f)) (0 f e)) (0 e d)) (0 d c)) (0 c b)) (0 b a)) (0 a R)) (0 v R))");
    define(&mut book, "k23", "$ (0 (2 (2 (2 (2 (2 (2 (2 (2 (2 (2 (2 (2 (2 (2 (2 (2 (2 (2 (2 (2 (2 (2 (0 w v) (0 v u)) (0 u t)) (0 t s)) (0 s r)) (0 r q)) (0 q p)) (0 p o)) (0 o n)) (0 n m)) (0 m l)) (0 l k)) (0 k j)) (0 j i)) (0 i h)) (0 h g)) (0 g f)) (0 f e)) (0 e d)) (0 d c)) (0 c b)) (0 b a)) (0 a R)) (0 w R))");
    define(&mut book, "k24", "$ (0 (2 (2 (2 (2 (2 (2 (2 (2 (2 (2 (2 (2 (2 (2 (2 (2 (2 (2 (2 (2 (2 (2 (2 (0 x w) (0 w v)) (0 v u)) (0 u t)) (0 t s)) (0 s r)) (0 r q)) (0 q p)) (0 p o)) (0 o n)) (0 n m)) (0 m l)) (0 l k)) (0 k j)) (0 j i)) (0 i h)) (0 h g)) (0 g f)) (0 f e)) (0 e d)) (0 d c)) (0 c b)) (0 b a)) (0 a R)) (0 x R))");
  }

  // Church Nats
  define(&mut book, "mul", "$ (0 (0 a b) (0 (0 c a) (0 c b)))");

  // Bools
  define(&mut book, "T", "$ (0 t (0 * t))");
  define(&mut book, "F", "$ (0 * (0 f f))");
  define(&mut book, "not", "$ (0 (0 f (0 t r)) (0 t (0 f r)))");
  define(&mut book, "and", "$ (0 (0 (0 (0 @T (0 @F a)) a) (0 (0 (0 @F (0 @F b)) b) c)) c)");

  // Scott Nats
  define(&mut book, "S", "$ (0 a (0 (0 a b) (0 * b)))");
  define(&mut book, "Z", "$ (0 * (0 a a))");

  // Generators for a big binary tree
  // λr. λt. ((t r) r)
  //define(&mut book, "g_s", "$ (0 (2 (0 b c) b) c)");
  define(&mut book, "g_s", "$ (0 (2 r0 r1) (0 (0 r0 (0 r1 r)) r))");
  define(&mut book, "g_z", "$ (0 x x)");

  // BitString constructors
  // O = λxs λo λi λe (o xs)
  // I = λxs λo λi λe (i xs)
  // E =     λo λi λe e
  define(&mut book, "O", "$ (0 xs (0 (0 xs r) (0 * (0 * r))))");
  define(&mut book, "I", "$ (0 xs (0 * (0 (0 xs r) (0 * r))))");
  define(&mut book, "E", "$ (0 * (0 * (0 e e)))");

  // Double
  define(&mut book, "nidS", "
    $ (0 p ret)
    & @S   ~ (0 nidp ret)
    & @nid ~ (0 p nidp)
  ");
  define(&mut book, "nid" , "
    $ (0 (0 @nidS (0 @Z ret)) ret)
  ");

  // Decrements a BitString
  // decO = λp(I (dec p))
  // decI = λp(low p)
  // dec  = λx(((x decO) decI) E)
  define(&mut book, "decO", "
    $ (0 p idecp)
    & @I   ~ (0 decp idecp)
    & @dec ~ (0 p decp)
  ");
  define(&mut book, "decI", "
    $ (0 p lowp)
    & @low ~ (0 p lowp)
  ");
  define(&mut book, "dec" , "
    $ (0 (0 @decO (0 @decI (0 @E ret))) ret)
  ");

  // Auxiliary function
  // lowO = λp(O (O p))
  // lowI = λp(O (I p))
  // low  = λx(((x lowO) lowI) E)
  define(&mut book, "lowO", "
    $ (0 p oop)
    & @O ~ (0 p op)
    & @O ~ (0 op oop)
  ");
  define(&mut book, "lowI", "
    $ (0 p oip)
    & @I ~ (0 p ip)
    & @O ~ (0 ip oip)
  ");
  define(&mut book, "low" , "
    $ (0 (0 @lowO (0 @lowI (0 @E ret))) ret)
  ");

  // Decrements a BitString until it is zero
  // runO = λp(run (dec (O p)))
  // runI = λp(run (dec (I p)))
  // run  = λx(((x runO) runI) E)
  define(&mut book, "runO", "
    $ (0 p ret)
    & @run ~ (0 decop ret)
    & @dec ~ (0 op decop)
    & @O   ~ (0 p op)
  ");
  define(&mut book, "runI", "
    $ (0 p ret)
    & @run ~ (0 decip ret)
    & @dec ~ (0 ip decip)
    & @I   ~ (0 p ip)
  ");
  define(&mut book, "run" , "
    $ (0 (0 @runO (0 @runI (0 @E ret))) ret)
  ");

  // Decrements 2^N BitStrings until they reach zero
  // brnZ = (run (c8 S Z))
  // brnS = λp {(brn p) (brn p)}
  // brn  = λn ((n brnS) brnZ)
  define(&mut book, "brnZ", "
    $ ret
    & @run ~ (0 val ret)
    & @c11 ~ (0 @I (0 @E val))
  ");
  define(&mut book, "brnS", "
    $ (0 (1 p0 p1) (0 r0 r1))
    & @brn ~ (0 p0 r0)
    & @brn ~ (0 p1 r1)
  ");
  define(&mut book, "brn", "
    $ (0 (0 @brnS (0 @brnZ r)) r)
  ");

  // af  = λx (x afS afZ)
  // afS = λp (and (af p) (af p))
  // afZ = T
  define(&mut book, "af", "
    $ (0 (0 @afS (0 @afZ a)) a)
  ");
  define(&mut book, "afS", "
    $ (0 (1 a b) c)
    & (0 b d) ~ @af
    & (0 e (0 d c)) ~ @and
    & (0 a e) ~ @af
  ");
  define(&mut book, "afZ", "
    $ @T
  ");

  // Church multiplication.
  define(&mut book, "ex0", "
    $ root
    & @c2 ~ (0 @k2 root)
  ");

  // Allocates a big tree.
  define(&mut book, "ex1", "
    $ root
    & @c24 ~ (0 @g_s (0 @g_z root))
  ");

  // Decrease a binary counter.
  define(&mut book, "ex2", "
    $ main
    & @c26 ~ (0 @I (0 @E nie))
    & @run ~ (0 nie main)
  "); 

  // Decreases many binary counters.
  // NOTE: takes about 25s on Apple M1 (at 250m RPS)
  define(&mut book, "ex3", "
    $ res
    & @c16 ~ (0 @S (0 @Z dep))
    & @brn ~ (0 dep res)
  ");

  // Performs '123 * 321'. ('3' is used for MUL.)
  define(&mut book, "ex4", "
    $ ret
    & 3 ~ {123 {321 ret}}
  ");

  // Conditional: 'if 1 { 123 } else { 321 }'.
  define(&mut book, "ex5", "
    $ ret
    & 1 ~ ? (0 123 321) ret
  ");

  book
}
