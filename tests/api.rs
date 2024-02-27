//! Tests for front-facing APIs and interfaces

use hvmc::{
  ast::{Book, Net, Tree},
  host::Host,
};
use insta::assert_display_snapshot;

#[test]
fn test_with_argument() {
  use hvmc::run;
  fn eval_with_args(fun: &str, args: &[&str]) -> Net {
    let area = run::Net::<run::Strict>::init_heap(1 << 10);

    let mut fun: Net = fun.parse().unwrap();
    for arg in args {
      let arg: Tree = arg.parse().unwrap();
      fun.with_argument(arg)
    }
    // TODO: When feature/sc-472/argument-passing, use encode_net instead.
    let mut book = Book::default();
    book.nets.insert("main".into(), fun);
    let host = Host::new(&book);

    let mut rnet = run::Net::<run::Strict>::new(&area);
    rnet.boot(&host.defs["main"]);
    rnet.normal();
    let got_result = host.readback(&rnet);
    got_result
  }
  assert_display_snapshot!(
    eval_with_args("(a a)", &vec!["(a a)"]),
    @"(a a)"
  );
  assert_display_snapshot!(
    eval_with_args("b & (a b) ~ a", &vec!["(a a)"]),
    @"a"
  );
  assert_display_snapshot!(
    eval_with_args("(z0 z0)", &vec!["(z1 z1)"]),
    @"(a a)"
  );
  assert_display_snapshot!(
    eval_with_args("(* #1)", &vec!["(a a)"]),
    @"#1"
  );
  assert_display_snapshot!(
    eval_with_args("(<+ a b> (a b))", &vec!["#1", "#2"]),
    @"#3"
  );
  assert_display_snapshot!(
    eval_with_args("(<* a b> (a b))", &vec!["#2", "#3"]),
    @"#6"
  );
  assert_display_snapshot!(
    eval_with_args("(<* a b> (a b))", &vec!["#2"]),
    @"(<2* a> a)"
  );
}
