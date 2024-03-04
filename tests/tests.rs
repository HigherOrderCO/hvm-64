use std::{
  fs,
  io::{self, Write},
  path::{Path, PathBuf},
  str::FromStr,
  sync::{Arc, Mutex},
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
    @false = (* (b b))
    @and  = ((b (@false c)) (b c))
    @main = root & @and ~ (@true (@false root))
  ",
  );
  let (rwts, net) = normal(book, 64);

  assert_snapshot!(Net::to_string(&net), @"(* (a a))");
  assert_debug_snapshot!(rwts.total(), @"9");
}

fn test_run(name: &str, host: Arc<Mutex<Host>>) {
  print!("{name}...");
  io::stdout().flush().unwrap();
  let heap = run::Heap::new_words(1 << 29);
  let mut net = run::Net::<Strict>::new(&heap);
  // The host is locked inside this block.
  {
    let lock = host.lock().unwrap();
    let Some(entrypoint) = lock.defs.get("main") else {
      println!(" skipping");
      return;
    };
    net.boot(entrypoint);
  }
  let start = Instant::now();
  net.parallel_normal();
  println!(" {:.3?}", start.elapsed());

  let output = format!("{}\n{}", host.lock().unwrap().readback(&net), show_rewrites(&net.rwts));
  assert_snapshot!(output);
}

fn test_path(path: &Path) {
  let code = fs::read_to_string(&path).unwrap();
  let book = ast::Book::from_str(&code).unwrap();
  let host = hvmc::stdlib::create_host(&book);

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

fn manifest_relative(sub: &str) -> PathBuf {
  format!("{}/{}", env!("CARGO_MANIFEST_DIR"), sub).into()
}

#[test]
#[serial]
fn test_programs() {
  test_dir(&manifest_relative("tests/programs/"), |_| true)
}

#[test]
#[serial]
fn test_examples() {
  test_dir(&manifest_relative("examples/"), |_| true);
}
