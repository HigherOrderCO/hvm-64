use criterion::{black_box, criterion_group, criterion_main, Criterion};
use hvmc::{ast::*, *};
use std::{
  ffi::OsStr,
  fs,
  path::{Path, PathBuf},
  time::Duration,
};

struct NetWithData<'a> (
  pub run::Net<'a>,
  Box<[(run::APtr, run::APtr)]>,
);

impl NetWithData<'_> {
  fn new(size: usize) -> Self {
    let data = Box::leak(run::Heap::init(size));
    let boxed = unsafe { Box::from_raw(data) };
    let mut net = run::Net::new(data);
    net.boot(name_to_val("main"));
    NetWithData(net, boxed)
  }
}

// Loads file and generate net from hvm-core syntax
fn load_from_core<'a, P: AsRef<Path>>(file: P) -> (run::Book, NetWithData<'a>) {
  let code = fs::read_to_string(file).unwrap();
  let (size, code) = extract_size(&code);

  let book = ast::do_parse_book(code);
  let rbook = ast::book_to_runtime(&book);

  let mut net = NetWithData::new(size);
  net.0.boot(name_to_val("main"));
  (rbook, net)
}

// Loads file and generate net from hvm-lang syntax
fn load_from_lang<'a, P: AsRef<Path>>(file: P) -> (run::Book, NetWithData<'a>) {
  let code = fs::read_to_string(file).unwrap();
  let (size, code) = extract_size(&code);

  let mut book = hvml::term::parser::parse_definition_book(&code).unwrap();
  let book = hvml::compile_book(&mut book, hvml::OptimizationLevel::Heavy).unwrap().core_book;
  let book = ast::book_to_runtime(&book);

  let mut net = NetWithData::new(size);
  net.0.boot(name_to_val("main"));
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

fn run_file(path: &PathBuf, mut group: Option<String>, c: &mut Criterion) {
  if cfg!(feature = "cuda") {
    group = Some(match group {
      Some(group) => format!("cuda/{group}"),
      None => "cuda".to_string(),
    });
  };

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
      |(book, net)| black_box(black_box(net.0).normal(black_box(&book))),
      criterion::BatchSize::PerIteration,
    );
  });
}

#[allow(unused_variables)]
fn benchmark_group(path: &PathBuf, group: String, c: &mut Criterion) {
  let file_name = path.file_stem().unwrap().to_string_lossy();
  #[cfg(not(feature = "cuda"))]
  c.benchmark_group(group).bench_function(file_name, |b| {
    b.iter_batched(
      || {
        match path.extension().and_then(OsStr::to_str) {
          Some("hvmc") => load_from_core(path),
          Some("hvm") => load_from_lang(path),
          _ => panic!("invalid file found: {}", path.to_string_lossy()),
        }
      },
      |(book, net)| black_box(black_box(net.0).normal(black_box(&book))),
      criterion::BatchSize::PerIteration,
    );
  });

  #[cfg(feature = "cuda")]
  c.benchmark_group(group).bench_function(file_name, |b| {
    b.iter_batched(
      || cuda::host::setup_gpu(&book, "main").unwrap(),
      |(dev, global_expand_prepare, global_expand, global_rewrite, gpu_net, gpu_book)| {
        black_box(
          cuda::host::cuda_normalize_net(
            black_box(global_expand_prepare),
            black_box(global_expand),
            black_box(global_rewrite),
            black_box(&gpu_net.device_net),
            black_box(&gpu_book),
          )
          .unwrap(),
        );

        black_box(dev.synchronize().unwrap());
      },
      criterion::BatchSize::PerIteration,
    )
  });
}

fn interact_benchmark(c: &mut Criterion) {
  if cfg!(feature = "cuda") {
    return;
  }

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
          let mut net = NetWithData::new(1 << 4);
          let book = run::Book::new();
          ast::net_to_runtime(&mut net.0, &ast::Net { root: Era, rdex: vec![redex.clone()] });
          let (rdx_a, rdx_b) = net.0.rdex[0];
          (book, net, rdx_a, rdx_b)
        },
        |(book, net, rdx_a, rdx_b)| black_box(black_box(net.0).interact(black_box(&book), black_box(rdx_a), black_box(rdx_b))),
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
    // run_programs_dir,
    interact_benchmark,
}
criterion_main!(benches);
