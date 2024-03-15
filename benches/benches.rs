use criterion::{black_box, criterion_group, criterion_main, Criterion};
use hvmc::{
  ast::{Book, Net},
  run::{Heap, Net as RtNet, Strict},
  stdlib::create_host,
};
use std::{
  fs,
  path::{Path, PathBuf},
  time::Duration,
};

// Loads file and generate net from hvm-core syntax
fn load_from_core<P: AsRef<Path>>(file: P) -> Book {
  let code = fs::read_to_string(file).unwrap();

  code.parse().unwrap()
}
fn run_programs_dir(c: &mut Criterion) {
  let root = PathBuf::from(format!("{}/tests/programs", env!("CARGO_MANIFEST_DIR")));
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
      // Skip stress tests and sort.
      if entry.components().any(|x| matches!(x.as_os_str().to_str(), Some("stress_tests" | "sort"))) {
        continue;
      }
      if entry.extension().unwrap().to_str().unwrap() == "hvmc" {
        run_file(entry, group.clone(), c);
      }
    }
  }
}

fn run_file(path: &PathBuf, group: Option<String>, c: &mut Criterion) {
  let book = load_from_core(path);
  let file_name = path.file_stem().unwrap().to_string_lossy();

  match group {
    Some(group) => benchmark_group(&file_name, group, book, c),
    None => benchmark(&file_name, book, c),
  }
}

fn benchmark(file_name: &str, book: Book, c: &mut Criterion) {
  let area = Heap::new_words(1 << 29);
  let host = create_host(&book);
  c.bench_function(file_name, |b| {
    b.iter(|| {
      let mut net = RtNet::<Strict>::new(&area);
      net.boot(host.lock().unwrap().defs.get("main").unwrap());
      black_box(black_box(net).normal())
    });
  });
}

fn benchmark_group(file_name: &str, group: String, book: Book, c: &mut Criterion) {
  let area = Heap::new_words(1 << 29);
  let host = create_host(&book);

  c.benchmark_group(group).bench_function(file_name, |b| {
    b.iter(|| {
      let mut net = RtNet::<Strict>::new(&area);
      net.boot(host.lock().unwrap().defs.get("main").unwrap());
      black_box(black_box(net).normal())
    });
  });
}

fn interact_benchmark(c: &mut Criterion) {
  use hvmc::ast::Tree::*;
  let mut group = c.benchmark_group("interact");
  group.sample_size(1000);

  let cases = [
    ("era-era", (Era, Era)),
    ("era-con", (Era, Ctr { lab: 0, ports: [Era.into(), Era.into()].into() })),
    (
      "con-con",
      ((Ctr { lab: 0, ports: [Era.into(), Era.into()].into() }, Ctr {
        lab: 0,
        ports: [Era.into(), Era.into()].into(),
      })),
    ),
    (
      "con-dup",
      ((Ctr { lab: 0, ports: [Era.into(), Era.into()].into() }, Ctr {
        lab: 2,
        ports: [Era.into(), Era.into()].into(),
      })),
    ),
  ];

  for (name, redex) in cases {
    let mut book = Book::default();
    book.insert("main".to_string(), Net { root: Era, redexes: vec![redex] });
    let area = Heap::new_words(1 << 24);
    let host = create_host(&book);
    group.bench_function(name, |b| {
      b.iter(|| {
        let mut net = RtNet::<Strict>::new(&area);
        net.boot(host.lock().unwrap().defs.get("main").unwrap());
        black_box(black_box(net).normal())
      });
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
