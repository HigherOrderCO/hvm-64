use criterion::{black_box, criterion_group, criterion_main, Criterion};
use hvmc::{ast::*, *};
use std::{
  fs,
  path::{Path, PathBuf},
  time::Duration,
};

// Loads file and generate net from hvm-core syntax
fn load_from_core<P: AsRef<Path>>(file: P) -> (run::Book, run::Net) {
  let code = fs::read_to_string(file).unwrap();
  let (size, code) = extract_size(&code);

  let book = ast::do_parse_book(code);
  let rbook = ast::book_to_runtime(&book);

  let mut net = run::Net::new(size);
  net.boot(name_to_val("main"));
  (rbook, net)
}

// Loads file and generate net from hvm-lang syntax
fn load_from_lang<P: AsRef<Path>>(file: P) -> (run::Book, run::Net) {
  let prelude = fs::read_to_string(format!("{}/benches/prelude.hvm", env!("CARGO_MANIFEST_DIR"))).unwrap();
  let code = fs::read_to_string(file).unwrap();
  let (size, code) = extract_size(&code);

  let code = prelude + "\n" + code;
  let mut book = hvm_lang::term::parser::parse_definition_book(&code).unwrap();
  let (book, _) = hvm_lang::compile_book(&mut book).unwrap();
  let book = ast::book_to_runtime(&book);

  let mut net = run::Net::new(size);
  net.boot(name_to_val("main"));
  (book, net)
}

fn extract_size(code: &str) -> (usize, &str) {
  code
    .strip_prefix("// size = ")
    .and_then(|code| code.split_once('\n'))
    .and_then(|(size, rest)| {
      match size.split_ascii_whitespace().collect::<Vec<_>>().as_slice() {
        [a, "<<", b] => a.parse::<usize>().ok().zip(b.parse::<usize>().ok()).map(|(a, b)| a << b),
        [a] => a.parse().ok(),
        _ => None,
      }
      .map(|size| (size, rest))
    })
    .expect("failed to extract bench size")
}

fn run_programs_dir(c: &mut Criterion) {
  let root = PathBuf::from(format!("{}/benches/programs", env!("CARGO_MANIFEST_DIR")));
  run_dir(&root, None, c);
}

fn run_dir(dir: &PathBuf, group: Option<&str>, c: &mut Criterion) {
  let files = std::fs::read_dir(dir).unwrap();

  for file in files.flatten() {
    let file_path = &file.path();
    let file_name = file_path.file_stem().unwrap().to_string_lossy();

    let Some(ext) = file_path.extension() else {
      if file_path.is_dir() {
        match group {
          Some(group) => run_dir(file_path, Some(&format!("{group}/{file_name}")), c),
          None => run_dir(file_path, Some(file_name.as_ref()), c),
        };
      }
      continue;
    };

    let (book, net) = match ext.to_str() {
      Some("hvmc") => load_from_core(file_path),
      Some("hvm") => load_from_lang(file_path),
      _ => panic!("invalid file found: {}", file_path.to_string_lossy()),
    };

    match group {
      Some(group) => benchmark_group(&file_name, group, book, net, c),
      None => benchmark(&file_name, book, net, c),
    }
  }
}

fn benchmark(file_name: &str, book: run::Book, net: run::Net, c: &mut Criterion) {
  c.bench_function(file_name, |b| {
    b.iter_batched(
      || net.clone(),
      |net| black_box(black_box(net).normal(black_box(&book))),
      criterion::BatchSize::SmallInput,
    );
  });
}

fn benchmark_group(file_name: &str, group: &str, book: run::Book, net: run::Net, c: &mut Criterion) {
  c.benchmark_group(group).bench_function(file_name, |b| {
    b.iter_batched(
      || net.clone(),
      |net| black_box(black_box(net).normal(black_box(&book))),
      criterion::BatchSize::SmallInput,
    );
  });
}

fn interact_benchmark(c: &mut Criterion) {
  use ast::Tree::*;
  let mut group = c.benchmark_group("interact");
  group.sample_size(1000);

  let cases = [
    ("era-era", (Era, Era)),
    ("era-con", (Era, Ctr { lab: 0, lft: Era.into(), rgt: Era.into() })),
    ("con-con", ((Ctr { lab: 0, lft: Era.into(), rgt: Era.into() }), Ctr { lab: 0, lft: Era.into(), rgt: Era.into() })),
    ("con-dup", ((Ctr { lab: 0, lft: Era.into(), rgt: Era.into() }), Ctr { lab: 2, lft: Era.into(), rgt: Era.into() })),
  ];

  for (name, redex) in cases {
    let mut net = run::Net::new(10);
    let book = run::Book::new();
    ast::net_to_runtime(&mut net, &ast::Net { root: Era, rdex: vec![redex] });
    let (rdx_a, rdx_b) = net.rdex[0];
    group.bench_function(name, |b| {
      b.iter_batched(
        || net.clone(),
        |net| black_box(black_box(net).interact(black_box(&book), black_box(rdx_a), black_box(rdx_b))),
        criterion::BatchSize::SmallInput,
      );
    });
  }
}

criterion_group! {
  name = benches;
  config = Criterion::default()
    .measurement_time(Duration::from_millis(1000))
    .warm_up_time(Duration::from_millis(500));
  targets =
    run_programs_dir,
    interact_benchmark,
}
criterion_main!(benches);
