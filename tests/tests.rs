#![cfg(feature = "std")]

use hvm64_transform::pre_reduce::PreReduce;
use parking_lot::Mutex;
use std::{
  fs,
  io::{self, Write},
  path::{Path, PathBuf},
  str::FromStr,
  sync::Arc,
  time::Instant,
};

use hvm64_ast::{self as ast, Book, Net};
use hvm64_host::{stdlib::create_host, Host};
use hvm64_runtime as run;

use insta::assert_snapshot;

use serial_test::serial;

fn execute_host(host: Arc<Mutex<Host>>) -> Option<(run::Rewrites, Net)> {
  let heap = run::Heap::new(None).unwrap();
  let mut net = run::Net::new(&heap);
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

  let output = format!("{}\n{}", net, &rwts);
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

  let host = create_host(&book);
  let Some((rwts, net)) = execute_host(host) else {
    assert_snapshot!(&pre_stats.rewrites);
    return;
  };

  let output = format!("{}\npre-reduce:\n{}run:\n{}", net, &pre_stats.rewrites, &rwts);
  assert_snapshot!(output);
}

fn test_path(path: &Path) {
  let code = fs::read_to_string(path).unwrap();
  let book = ast::Book::from_str(&code).unwrap();
  let host = create_host(&book);

  let path = path.strip_prefix(env!("CARGO_MANIFEST_DIR")).unwrap();
  let path = path.to_str().unwrap();

  test_pre_reduce_run(path, book.clone());
  test_run(path, host);
}

fn test_dir(dir: &Path, filter: impl Fn(&Path) -> bool) {
  insta::glob!(dir, "**/*.hvm", |p| {
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
