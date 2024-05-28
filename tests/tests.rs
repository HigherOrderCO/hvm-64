use dyntest::{dyntest, DynTester};
use hvm64_transform::pre_reduce::PreReduce;
use std::{fs, path::Path, str::FromStr};

use hvm64_ast::{self as ast, Book, Net};
use hvm64_host::Host;
use hvm64_runtime as run;

use insta::assert_snapshot;

dyntest!(test);

fn test(t: &mut DynTester) {
  for (name, path) in t.glob("{examples,tests/programs}/**/*.hvm") {
    t.test(name.clone(), move || {
      let mut settings = insta::Settings::new();
      settings.set_prepend_module_to_snapshot(false);
      settings.set_input_file(&path);
      settings.set_snapshot_suffix(name);
      settings.bind(|| {
        test_path(&path);
      })
    });
  }
}

fn execute_host(host: &Host) -> Option<(run::Rewrites, Net)> {
  let heap = run::Heap::new(None).unwrap();
  let mut net = run::Net::new(&heap);
  let entrypoint = host.defs.get("main").unwrap();
  net.boot(entrypoint);
  net.parallel_normal();
  Some((net.rwts, host.readback(&net)))
}

fn test_run(host: &Host) {
  let Some((rwts, net)) = execute_host(host) else { return };

  let output = format!("{}\n{}", net, &rwts);
  assert_snapshot!(output);
}

fn test_pre_reduce_run(mut book: Book) {
  let pre_stats = book.pre_reduce(&|x| x == "main", None, u64::MAX);

  let host = Host::new(&book);
  let Some((rwts, net)) = execute_host(&host) else {
    assert_snapshot!(&pre_stats.rewrites);
    return;
  };

  let output = format!("{}\npre-reduce:\n{}run:\n{}", net, &pre_stats.rewrites, &rwts);
  assert_snapshot!(output);
}

fn test_path(path: &Path) {
  let code = fs::read_to_string(path).unwrap();
  let book = ast::Book::from_str(&code).unwrap();
  let host = Host::new(&book);

  test_pre_reduce_run(book.clone());
  test_run(&host);
}
