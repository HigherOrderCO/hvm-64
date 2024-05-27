include!("../prelude.rs");

mod compile;

mod args;
mod full;

use std::{
  env::consts::{DLL_PREFIX, DLL_SUFFIX},
  ffi::OsStr,
  fs, io,
  ops::DerefMut,
  path::PathBuf,
  process::{self, Stdio},
  sync::Arc,
  time::{Duration, Instant},
};

use parking_lot::Mutex;

use self::full::{CliMode, FullCli};

use args::{RunArgs, RuntimeOpts, TransformArgs, TransformPass};
use clap::Parser;

use hvm64_ast::{Book, Net, Tree};
use hvm64_host::{
  stdlib::{create_host, insert_stdlib},
  DefRef, Host,
};
use hvm64_runtime::{dispatch_dyn_net, Def, DynNet, Heap, Mode, Port, Trg};
use hvm64_transform::Transform;

fn main() {
  if cfg!(feature = "trace") {
    hvm64_runtime::trace::set_hook();
  }

  let cli = FullCli::parse();

  match cli.mode {
    CliMode::Compile { file, transform_args, output } => {
      let output = if let Some(output) = output {
        output
      } else if let Some("hvm") = file.extension().and_then(OsStr::to_str) {
        file.with_extension("")
      } else {
        eprintln!("file missing `.hvm` extension; explicitly specify an output path with `--output`.");

        process::exit(1);
      };

      let host = create_host(&load_book(&[file], transform_args));
      compile::create_temp_hvm(host).unwrap();

      compile_temp_hvm().unwrap();

      fs::copy(format!(".hvm/target/release/{DLL_PREFIX}hvm64_gen{DLL_SUFFIX}"), output).unwrap();
    }
    CliMode::Run { run_opts, mut transform_args, file, args } => {
      // Don't pre-reduce or prune the entry point
      transform_args.transform_opts.pre_reduce_skip.push(args.entry_point.clone());
      transform_args.transform_opts.prune_entrypoints.push(args.entry_point.clone());

      let host = load_host(&[file], transform_args, &run_opts.include);
      run(host, run_opts, args);
    }
    CliMode::Reduce { run_opts, transform_args, files, exprs } => {
      let host = load_host(&files, transform_args, &run_opts.include);
      let exprs: Vec<_> = exprs.iter().map(|x| x.parse().unwrap()).collect();
      reduce_exprs(host, &exprs, &run_opts);
    }
    CliMode::Transform { transform_args, files } => {
      let book = load_book(&files, transform_args);
      println!("{}", book);
    }
  };

  if cfg!(feature = "trace") {
    hvm64_runtime::trace::_read_traces(usize::MAX);
  }
}

fn run(host: Arc<Mutex<Host>>, opts: RuntimeOpts, args: RunArgs) {
  let mut net = Net { root: Tree::Ref { nam: args.entry_point }, redexes: vec![] };
  for arg in args.args {
    let arg: Net = arg.parse().unwrap();
    net.redexes.extend(arg.redexes);
    net.apply_tree(arg.root);
  }

  reduce_exprs(host, &[net], &opts);
}

fn load_host(
  files: &[PathBuf],
  transform_args: TransformArgs,
  include: &[PathBuf],
) -> Arc<parking_lot::lock_api::Mutex<parking_lot::RawMutex, Host>> {
  let host: Arc<Mutex<Host>> = Default::default();
  load_dylibs(host.clone(), include);
  insert_stdlib(host.clone());
  host.lock().insert_book(&load_book(files, transform_args));
  host
}

fn load_book(files: &[PathBuf], transform_args: TransformArgs) -> Book {
  let mut book = files
    .iter()
    .map(|name| {
      let contents = fs::read_to_string(name).unwrap_or_else(|_| {
        eprintln!("Input file {:?} not found", name);
        process::exit(1);
      });
      contents.parse().unwrap_or_else(|e| {
        eprintln!("Parsing error {e}");
        process::exit(1);
      })
    })
    .fold(Book::default(), |mut acc, i: Book| {
      acc.nets.extend(i.nets);
      acc
    });

  let transform_passes = TransformPass::to_passes(&transform_args.transform_passes[..]);
  book
    .transform(transform_passes, &hvm64_transform::TransformOpts {
      pre_reduce_skip: transform_args.transform_opts.pre_reduce_skip,
      pre_reduce_memory: transform_args.transform_opts.pre_reduce_memory,
      pre_reduce_rewrites: transform_args.transform_opts.pre_reduce_rewrites,
      prune_entrypoints: transform_args.transform_opts.prune_entrypoints,
    })
    .unwrap();

  book
}

fn load_dylibs(host: Arc<Mutex<Host>>, include: &[PathBuf]) {
  let current_dir = std::env::current_dir().unwrap();

  for file in include {
    unsafe {
      let lib = if file.is_absolute() {
        libloading::Library::new(file)
      } else {
        libloading::Library::new(current_dir.join(file))
      }
      .expect("failed to load dylib");

      let rust_version =
        lib.get::<fn() -> &'static str>(b"hvm64_dylib_v0__rust_version").expect("failed to load rust version");
      let rust_version = rust_version();
      if rust_version != env!("RUSTC_VERSION") {
        eprintln!(
          "warning: dylib {file:?} was compiled with rust version {rust_version}, but is being run with rust version {}",
          env!("RUSTC_VERSION")
        );
      }

      let hvm64_version =
        lib.get::<fn() -> &'static str>(b"hvm64_dylib_v0__hvm64_version").expect("failed to load hvm64 version");
      let hvm64_version = hvm64_version();
      if hvm64_version != env!("CARGO_PKG_VERSION") {
        eprintln!(
          "warning: dylib {file:?} was compiled with hvm64 version {hvm64_version}, but is being run with hvm64 version {}",
          env!("CARGO_PKG_VERSION")
        );
      }

      let insert_into = lib
        .get::<fn(&mut dyn FnMut(&str, Box<dyn DerefMut<Target = Def> + Send + Sync>))>(b"hvm64_dylib_v0__insert_into")
        .expect("failed to load insert_into");
      let mut host = host.lock();
      insert_into(&mut |name, def| {
        host.insert_def(name, DefRef::Owned(def));
      });

      // Leak the lib to avoid unloading it, as code from it is still referenced.
      std::mem::forget(lib);
    }
  }
}

fn reduce_exprs(host: Arc<Mutex<Host>>, exprs: &[Net], opts: &RuntimeOpts) {
  let heap = Heap::new(opts.memory).expect("memory allocation failed");
  for expr in exprs {
    let mut net = DynNet::new(&heap, opts.lazy_mode);
    dispatch_dyn_net!(&mut net => {
      host.lock().encode_net(net, Trg::port(Port::new_var(net.root.addr())), expr);
      let start_time = Instant::now();
      if opts.single_core {
        net.normal();
      } else {
        net.parallel_normal();
      }
      let elapsed = start_time.elapsed();
      println!("{}", host.lock().readback(net));
      if opts.show_stats {
        print_stats(net, elapsed);
      }
    });
  }
}

fn print_stats<M: Mode>(net: &hvm64_runtime::Net<M>, elapsed: Duration) {
  eprintln!("RWTS   : {:>15}", pretty_num(net.rwts.total()));
  eprintln!("- ANNI : {:>15}", pretty_num(net.rwts.anni));
  eprintln!("- COMM : {:>15}", pretty_num(net.rwts.comm));
  eprintln!("- ERAS : {:>15}", pretty_num(net.rwts.eras));
  eprintln!("- DREF : {:>15}", pretty_num(net.rwts.dref));
  eprintln!("- OPER : {:>15}", pretty_num(net.rwts.oper));
  eprintln!("TIME   : {:.3?}", elapsed);
  eprintln!("RPS    : {:.3} M", (net.rwts.total() as f64) / (elapsed.as_millis() as f64) / 1000.0);
}

fn pretty_num(n: u64) -> String {
  n.to_string()
    .as_bytes()
    .rchunks(3)
    .rev()
    .map(|x| std::str::from_utf8(x).unwrap())
    .flat_map(|x| ["_", x])
    .skip(1)
    .collect()
}

/// Compiles the `.hvm` directory, appending the provided `args` to `cargo`.
fn compile_temp_hvm() -> Result<(), io::Error> {
  let output = process::Command::new("cargo")
    .current_dir(".hvm/gen/")
    .arg("build")
    .arg("--release")
    .stderr(Stdio::inherit())
    .output()?;

  if !output.status.success() {
    process::exit(1);
  }

  Ok(())
}
