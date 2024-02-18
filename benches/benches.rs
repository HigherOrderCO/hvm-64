use criterion::{black_box, criterion_group, criterion_main, Criterion};
use hvmc::{ast, *};
use hvml::CompileOpts;
use std::{
  ffi::OsStr,
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

  let net = run::Net::new(size, false);
  
  (rbook, net)
}

// Loads file and generate net from hvm-lang syntax
fn load_from_lang<P: AsRef<Path>>(file: P) -> (run::Book, run::Net) {
  let code = fs::read_to_string(file).unwrap();
  let (size, code) = extract_size(&code);

  let mut book = hvml::term::parser::parse_book(code, hvml::term::Book::default, false).unwrap();
  let book = hvml::compile_book(&mut book, CompileOpts::heavy(), None).unwrap().core_book;
  let book = ast::book_to_runtime(&book);

  let net = run::Net::new(size, false);
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

fn run_dir(path: &PathBuf, group: Option<String>, c: &mut Criterion) {
  let dir_entries = std::fs::read_dir(path).unwrap().flatten();

  for entry in dir_entries {
    let entry = &entry.path();

    if entry.is_dir() {
      let dir_name = entry.file_stem().unwrap().to_string_lossy();

      let group = match group {
        Some(ref group) => format!("{group}/{dir_name}"),
        None => dir_name.to_string(),
      };

      run_dir(entry, Some(group), c)
    } else {
      run_file(entry, group.clone(), c);
    }
  }
}

fn run_file(path: &PathBuf, group: Option<String>, c: &mut Criterion) {
  match group {
    Some(group) => benchmark_group(path, group, c),
    None => benchmark(path, c),
  }
}

fn benchmark(path: &PathBuf, c: &mut Criterion) {
  let file_name = path.file_stem().unwrap().to_string_lossy();
  c.bench_function(&file_name, |b| {
    b.iter_batched(
      || {
        match path.extension().and_then(OsStr::to_str) {
          Some("hvmc") => load_from_core(path),
          Some("hvm") => load_from_lang(path),
          _ => panic!("invalid file found: {}", path.to_string_lossy()),
        }
      },
      |(book, net)| black_box(black_box(net).normal(black_box(&book))),
      criterion::BatchSize::PerIteration,
    );
  });
}

fn benchmark_group(path: &PathBuf, group: String, c: &mut Criterion) {
  let file_name = path.file_stem().unwrap().to_string_lossy();
  c.benchmark_group(group).bench_function(file_name, |b| {
    b.iter_batched(
      || {
        match path.extension().and_then(OsStr::to_str) {
          Some("hvmc") => load_from_core(path),
          Some("hvm") => load_from_lang(path),
          _ => panic!("invalid file found: {}", path.to_string_lossy()),
        }
      },
      |(book, net)| black_box(black_box(net).normal(black_box(&book))),
      criterion::BatchSize::PerIteration,
    );
  });
}

fn interact_benchmark(c: &mut Criterion) {
  use ast::Tree::*;
  let mut group = c.benchmark_group("interact");
  group.sample_size(1000);

  let cases = [
    ("era-era", (Era, Era)),
    ("era-con", (Era, Con { lft: Era.into(), rgt: Era.into() })),
    ("con-con", ((Con { lft: Era.into(), rgt: Era.into() }), Con { lft: Era.into(), rgt: Era.into() })),
    ("con-dup", ((Con { lft: Era.into(), rgt: Era.into() }), Dup { lab: 2, lft: Era.into(), rgt: Era.into() })),
  ];

  for (name, redex) in cases {
    group.bench_function(name, |b| {
      b.iter_batched(
        || {
          let book = run::Book::new();
          let net = ast::Net { root: Era, rdex: vec![redex.clone()] };

          let mut rnet = run::Net::new(1 << 4, false);

          rnet.net_to_runtime(&net);

          let (rdx_a, rdx_b) = match rnet {
            run::Net::Lazy(ref a) => a.net.rdex[0],
            run::Net::Eager(ref a) => a.net.rdex[0],
          };

          (book, rnet, rdx_a, rdx_b)
        },
        |(book, rnet, rdx_a, rdx_b)| black_box(black_box(rnet).interact(black_box(&book), black_box(rdx_a), black_box(rdx_b))),
        criterion::BatchSize::PerIteration,
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
