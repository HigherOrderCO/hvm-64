use std::{
  fs,
  io::{self, Write},
  path::{Path, PathBuf},
  str::FromStr,
  time::Instant,
};

use hvmc::{
  ast::{self, Net},
  host::Host,
  run::{self, Strict},
  util::show_rewrites,
};
use insta::{assert_debug_snapshot, assert_snapshot};
use loaders::*;

mod loaders;

use serial_test::serial;

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

fn test_run(name: &str, host: Host) {
  print!("{name}...");
  io::stdout().flush().unwrap();
  let Some(entrypoint) = host.defs.get("main") else {
    println!(" skipping");
    return;
  };
  let heap = run::Net::<Strict>::init_heap(1 << 32);
  let mut net = run::Net::<Strict>::new(&heap);
  net.boot(entrypoint);
  let start = Instant::now();
  net.parallel_normal();
  println!(" {:.3?}", start.elapsed());

  let output = format!("{}\n{}", host.readback(&net), show_rewrites(&net.rwts));
  assert_snapshot!(output);
}

fn test_path(path: &Path) {
  let code = fs::read_to_string(&path).unwrap();
  let book = ast::Book::from_str(&code).unwrap();
  let host = Host::new(&book);

  let path = path.strip_prefix(env!("CARGO_MANIFEST_DIR")).unwrap();

  test_run(path.to_str().unwrap(), host);
}

fn test_dir(dir: &Path, filter: impl Fn(&Path) -> bool) {
  insta::glob!(dir, "**/*.hvmc", |p| {
    if filter(p) {
      test_path(p);
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
#[serial]
fn test_fast_programs() {
  test_dir(&manifest_relative("tests/programs/"), |p| !is_slow(p))
}

#[test]
#[ignore = "slow"]
#[serial]
fn test_slow_programs() {
  test_dir(&manifest_relative("tests/programs/"), |p| is_slow(p))
}

#[test]
#[serial]
fn test_fast_examples() {
  test_dir(&manifest_relative("examples/"), |p| !is_slow(p));
}

#[test]
#[serial]
#[ignore = "slow"]
fn test_slow_examples() {
  test_dir(&manifest_relative("examples/"), |p| is_slow(p))
}
