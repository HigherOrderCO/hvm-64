use criterion::{black_box, criterion_group, criterion_main, Criterion};
use hvm_core::{book, define, name_to_val, Net};

fn church_benchmark(c: &mut Criterion) {
  // Initializes the book
  let mut book = book::setup_book();

  let mut group = c.benchmark_group("church");
  for n in [2, 8, 24] {
    group.throughput(criterion::Throughput::Elements(n));
    group.bench_with_input(criterion::BenchmarkId::new("multiplication", n), &n, |b, &n| {
      define(&mut book, "test", format!("$ root & (0 @c{n} (0 @c{n} root)) ~ @mul").as_str());
      let mut net = Net::new(1 << 12);
      net.boot(name_to_val("test"));
      // *out = net;

      b.iter(|| black_box(net.clone().normal(&book)));
    });
    // Church exponentiation (ex0)
    // n=24 uses at most 1107 elements in heap
    group.bench_with_input(criterion::BenchmarkId::new("exponentiation", n), &n, |b, &n| {
      define(&mut book, "test", format!("$ root & @c{n} ~ (0 @k{n} root)").as_str());
      let mut net = Net::new(1 << 12);
      net.boot(name_to_val("test"));

      b.iter(|| black_box(net.clone().normal(&book)));
    });
  }
}

fn tree_benchmark(c: &mut Criterion) {
  // Initializes the book
  let mut book = book::setup_book();

  let mut group = c.benchmark_group("tree");
  // Allocates a big tree (ex1)
  for n in [4, 8, 16, 20] {
    group.bench_with_input(criterion::BenchmarkId::new("allocation", n), &n, |b, &n| {
      define(&mut book, "test", format!("$ root & @c{n} ~ (0 @g_s (0 @g_z root))").as_str());
      let mut net = Net::new(16 << n);
      net.boot(name_to_val("test"));

      b.iter(|| black_box(net.clone().normal(&book)));
    });
  }
}

fn binary_counter_benchmark(c: &mut Criterion) {
  // Initializes the book
  let mut book = book::setup_book();

  let mut group = c.benchmark_group("binary-counter");
  // Decrements a BitString until it is zero (ex2)
  for n in [4, 8, 16, 20] {
    group.bench_with_input(criterion::BenchmarkId::new("single", n), &n, |b, &n| {
      let code = format!(
        " $ root
          & @c{n} ~ (0 @I (0 @E nie))
          & @run ~ (0 nie root)"
      );
      define(&mut book, "test", code.as_str());
      let mut net = Net::new(16 << n);
      net.boot(name_to_val("test"));

      b.iter(|| black_box(net.clone().normal(&book)));
    });
  }
  // Decrements 2^N BitStrings until they reach zero (ex3)
  for n in [4, 8, 10] {
    group.bench_with_input(criterion::BenchmarkId::new("many", n), &n, |b, &n| {
      let code = format!(
        " $ root
          & @c{n} ~ (0 @S (0 @Z dep))
          & @brn ~ (0 dep root)"
      );
      define(&mut book, "test", code.as_str());
      let mut net = Net::new(128 << n);
      net.boot(name_to_val("test"));

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
