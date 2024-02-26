use std::{
  fs,
  path::{Path, PathBuf},
  str::FromStr,
};

use hvmc::{
  ast::{self, Net},
  host::Host,
  run::{self, Strict},
};
use insta::{assert_debug_snapshot, assert_display_snapshot, assert_snapshot};
use loaders::*;

mod loaders;

#[test]
fn test_era_era() {
  let net = parse_core("@main = * & * ~ *");
  let (rwts, net) = normal(net, 16);
  assert_snapshot!(Net::to_string(&net), @"*");
  assert_debug_snapshot!(rwts.total(), @"2");
}

#[test]
fn test_era_era2() {
  let net = parse_core("@main = (* *) & * ~ *");
  let (rwts, net) = normal(net, 16);
  assert_snapshot!(Net::to_string(&net), @"(* *)");
  assert_debug_snapshot!(rwts.total(), @"2");
}

#[test]
fn test_commutation() {
  let net = parse_core("@main = root & (x x) ~ [* root]");
  let (rwts, net) = normal(net, 16);
  assert_snapshot!(Net::to_string(&net), @"(a a)");
  assert_debug_snapshot!(rwts.total(), @"5");
}

#[test]
fn test_bool_and() {
  let book = parse_core(
    "
    @true = (b (* b))
    @fals = (* (b b))
    @and  = ((b (@fals c)) (b c))
    @main = root & @and ~ (@true (@fals root))
  ",
  );
  let (rwts, net) = normal(book, 64);

  assert_snapshot!(Net::to_string(&net), @"(* (a a))");
  assert_debug_snapshot!(rwts.total(), @"9");
}

#[derive(Default)]
pub struct TestOpts {
  _lazy: bool,
}

fn test_host(name: &str, host: Host, _opts: TestOpts) {
  println!("{name}");
  let Some(entrypoint) = host.defs.get("main") else { return };
  let heap = run::Net::<Strict>::init_heap(1 << 28);
  let mut net = run::Net::<Strict>::new(&heap);
  net.boot(entrypoint);
  net.normal();

  assert_display_snapshot!(format!("{name}/strict"), host.readback(&net));
  assert_display_snapshot!(format!("{name}/strict_rwts"), net.rwts.total());

  // Now, test lazy mode and
  // ensure the outputs are equal.
}

fn test_path(path: &Path, opts: TestOpts) {
  println!("Exec {path:?}");
  let code = fs::read_to_string(&path).unwrap();
  let book = ast::Book::from_str(&code).unwrap();
  let host = Host::new(&book);

  let path = path.strip_prefix(env!("CARGO_MANIFEST_DIR")).unwrap();

  test_host(path.to_str().unwrap(), host, opts);
}

fn test_dir(dir: &Path, filter: &dyn Fn(&Path) -> bool) {
  insta::glob!(dir, "**/*.hvmc", |p| {
    if filter(p) {
      test_path(p, Default::default());
    }
  })
}

fn is_slow(p: &Path) -> bool {
  p.components().any(|x| x.as_os_str().to_str().unwrap() == "slow")
}

fn manifest_relative(sub: &str) -> PathBuf {
  format!("{}/{}", env!("CARGO_MANIFEST_DIR"), sub).into()
}

#[test]
fn test_fast_programs() {
  test_dir(&manifest_relative("tests/programs/"), &|p| !is_slow(p))
}

#[test]
#[ignore = "very slow"]
fn test_slow_programs() {
  test_dir(&manifest_relative("tests/programs/"), &|p| is_slow(p))
}

#[test]
fn test_fast_examples() {
  test_dir(&manifest_relative("examples/"), &|p| !is_slow(p));
}

#[test]
#[ignore = "very slow"]
fn test_slow_examples() {
  test_dir(&manifest_relative("examples/"), &|p| is_slow(p))
}
