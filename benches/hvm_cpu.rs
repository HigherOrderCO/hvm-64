use criterion::{black_box, criterion_group, criterion_main, Criterion};
use hvmc::ast::*;
use hvmc::*;
use std::fs;

// Load file and generate net
fn load(file: &str, size: usize, replace: Option<(&str, &str)>) -> (run::Book, run::Net) {
  let code = fs::read_to_string(file).unwrap();
  let code = if let Some((from, to)) = replace { code.replace(from, to) } else { code };
  let book = ast::book_to_runtime(&ast::do_parse_book(&code));
  let mut net = run::Net::new(size);
  net.boot(name_to_val("main"));
  return (book, net);
}

fn church_benchmark(c: &mut Criterion) {
  let mut group = c.benchmark_group("church");
  for n in [2, 8, 24] {
    group.throughput(criterion::Throughput::Elements(n));
    let (book, net) = load("./benches/programs/church_mul.hvmc", 1 << 12, Some(("{n}", &n.to_string())));
    group.bench_with_input(criterion::BenchmarkId::new("multiplication", n), &n, |b, &n| {
      b.iter(|| black_box(net.clone().normal(&book)));
    });
    // Church exponentiation
    // n=24 uses at most 1107 elements in heap(1 << 12 = 4096)
    let (book, net) = load("./benches/programs/church_exp.hvmc", 1 << 12, Some(("{n}", &n.to_string())));
    group.bench_with_input(criterion::BenchmarkId::new("exponentiation", n), &n, |b, &n| {
      b.iter(|| black_box(net.clone().normal(&book)));
    });
  }
}

fn tree_benchmark(c: &mut Criterion) {
  let mut group = c.benchmark_group("tree");
  // Allocates a big tree
  for n in [4, 8, 16, 20] {
    let (book, net) = load("./benches/programs/alloc_big_tree.hvmc", 16 << n, Some(("{n}", &n.to_string())));
    group.bench_with_input(criterion::BenchmarkId::new("allocation", n), &n, |b, &n| {
      b.iter(|| black_box(net.clone().normal(&book)));
    });
  }
}

fn binary_counter_benchmark(c: &mut Criterion) {
  let mut group = c.benchmark_group("binary-counter");
  // Decrements a BitString until it is zero
  for n in [4, 8, 16, 20] {
    let (book, net) = load("./benches/programs/dec_bits.hvmc", 16 << n, Some(("{n}", &n.to_string())));
    group.bench_with_input(criterion::BenchmarkId::new("single", n), &n, |b, &n| {
      b.iter(|| black_box(net.clone().normal(&book)));
    });
  }
  // Decrements 2^N BitStrings until they reach zero (ex3)
  for n in [4, 8, 10] {
    let (book, net) = load("./benches/programs/dec_bits_tree.hvmc", 128 << n, Some(("{n}", &n.to_string())));
    group.bench_with_input(criterion::BenchmarkId::new("many", n), &n, |b, &n| {
      b.iter(|| black_box(net.clone().normal(&book)));
    });
  }
}

criterion_group! {
  name = benches;
  config = Criterion::default();
  targets =
    church_benchmark,
    tree_benchmark,
    binary_counter_benchmark,
}
criterion_main!(benches);
