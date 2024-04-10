#![cfg(feature = "std")]

use parking_lot::Mutex;
use std::{
  fs,
  io::{self, Write},
  path::{Path, PathBuf},
  str::FromStr,
  sync::Arc,
  time::Instant,
};

use hvmc::{
  ast::{self, Book, Net},
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
  let (rwts, net) = normal(net, Some(128));
  assert_snapshot!(Net::to_string(&net), @"*");
  assert_debug_snapshot!(rwts.total(), @"3");
}

#[test]
fn test_era_era2() {
  let net = parse_core("@main = (* *) & * ~ *");
  let (rwts, net) = normal(net, Some(128));
  assert_snapshot!(Net::to_string(&net), @"(* *)");
  assert_debug_snapshot!(rwts.total(), @"5");
}

#[test]
fn test_commutation() {
  let net = parse_core("@main = root & (x x) ~ [* root]");
  let (rwts, net) = normal(net, Some(128));
  assert_snapshot!(Net::to_string(&net), @"(a a)");
  assert_debug_snapshot!(rwts.total(), @"7");
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
  let (rwts, net) = normal(book, Some(128));

  assert_snapshot!(Net::to_string(&net), @"(* (a a))");
  assert_debug_snapshot!(rwts.total(), @"14");
}

fn execute_host(host: Arc<Mutex<Host>>) -> Option<(run::Rewrites, Net)> {
  let heap = run::Heap::new(None).unwrap();
  let mut net = run::Net::<Strict>::new(&heap);
  // The host is locked inside this block.
  {
    let lock = host.lock();
    let Some(entrypoint) = lock.defs.get("main") else {
      println!(" skipping");
      return None;
    };
    net.boot(entrypoint);
  }
  let start = Instant::now();
  net.parallel_normal();
  println!(" {:.3?}", start.elapsed());
  Some((net.rwts, host.lock().readback(&net)))
}

fn test_run(name: &str, host: Arc<Mutex<Host>>) {
  print!("{name}...");
  io::stdout().flush().unwrap();

  let Some((rwts, net)) = execute_host(host) else { return };

  let output = format!("{}\n{}", net, show_rewrites(&rwts));
  assert_snapshot!(output);
}

fn test_pre_reduce_run(path: &str, mut book: Book) {
  print!("{path}...");
  print!(" pre-reduce");
  io::stdout().flush().unwrap();

  let start = Instant::now();
  let pre_stats = book.pre_reduce(&|x| x == "main", None, u64::MAX);
  print!(" {:.3?}...", start.elapsed());
  io::stdout().flush().unwrap();

  let host = hvmc::stdlib::create_host(&book);
  let Some((rwts, net)) = execute_host(host) else {
    assert_snapshot!(show_rewrites(&pre_stats.rewrites));
    return;
  };

  let output = format!("{}\npre-reduce:\n{}run:\n{}", net, show_rewrites(&pre_stats.rewrites), show_rewrites(&rwts));
  assert_snapshot!(output);
}

fn test_path(path: &Path) {
  let code = fs::read_to_string(&path).unwrap();
  let book = ast::Book::from_str(&code).unwrap();
  let host = hvmc::stdlib::create_host(&book);

  let path = path.strip_prefix(env!("CARGO_MANIFEST_DIR")).unwrap();
  let path = path.to_str().unwrap();

  test_pre_reduce_run(path, book.clone());
  test_run(path, host);
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
