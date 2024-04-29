#![cfg_attr(feature = "trace", feature(const_type_name))]

use clap::{Args, Parser, Subcommand};
use hvmc::{
  ast::{Book, Net, Tree},
  host::Host,
  run::{DynNet, Mode, Trg},
  stdlib::{create_host, insert_stdlib},
  transform::{TransformOpts, TransformPass, TransformPasses},
  *,
};

use parking_lot::Mutex;
use std::{
  ffi::OsStr,
  fmt::Write,
  fs::{self, File},
  io::{self, BufRead},
  path::{Path, PathBuf},
  process::{self, Stdio},
  str::FromStr,
  sync::Arc,
  time::{Duration, Instant},
};

fn main() {
  if cfg!(feature = "trace") {
    trace::set_hook();
  }
  if cfg!(feature = "_full_cli") {
    let cli = FullCli::parse();

    match cli.mode {
      CliMode::Compile { file, dylib, transform_args, output } => {
        let output = if let Some(output) = output {
          output
        } else if let Some("hvmc") = file.extension().and_then(OsStr::to_str) {
          file.with_extension("")
        } else {
          eprintln!("file missing `.hvmc` extension; explicitly specify an output path with `--output`.");

          process::exit(1);
        };

        let host = create_host(&load_book(&[file], &transform_args));
        create_temp_hvm(host).unwrap();

        if dylib {
          prepare_temp_hvm_dylib().unwrap();
          compile_temp_hvm(&["--lib"]).unwrap();

          // TODO: this can be a different extension on different platforms
          // see: https://doc.rust-lang.org/reference/linkage.html
          fs::copy(".hvm/target/release/libhvmc.so", output).unwrap();
        } else {
          compile_temp_hvm(&[]).unwrap();

          fs::copy(".hvm/target/release/hvmc", output).unwrap();
        }
      }
      CliMode::Run { run_opts, mut transform_args, file, args } => {
        // Don't pre-reduce or prune the entry point
        transform_args.transform_opts.pre_reduce_skip.push(args.entry_point.clone());
        transform_args.transform_opts.prune_entrypoints.push(args.entry_point.clone());

        let host: Arc<Mutex<Host>> = Default::default();
        load_dylibs(host.clone(), &args.include);
        insert_stdlib(host.clone());
        host.lock().insert_book(&load_book(&[file], &transform_args));

        run(host, run_opts, args);
      }
      CliMode::Reduce { run_opts, transform_args, files, exprs } => {
        let host = create_host(&load_book(&files, &transform_args));
        let exprs: Vec<_> = exprs.iter().map(|x| Net::from_str(x).unwrap()).collect();
        reduce_exprs(host, &exprs, &run_opts);
      }
      CliMode::Transform { transform_args, files } => {
        let book = load_book(&files, &transform_args);
        println!("{}", book);
      }
    }
  } else {
    let cli = BareCli::parse();
    let host = create_host(&Book::default());
    gen::insert_into_host(&mut host.lock());
    run(host, cli.opts, cli.args);
  }
  if cfg!(feature = "trace") {
    hvmc::trace::_read_traces(usize::MAX);
  }
}

#[derive(Parser, Debug)]
#[command(
  author,
  version,
  about = "A massively parallel Interaction Combinator evaluator",
  long_about = r##"
A massively parallel Interaction Combinator evaluator

Examples: 
$ hvmc run examples/church_encoding/church.hvm
$ hvmc run examples/addition.hvmc "#16" "#3"
$ hvmc compile examples/addition.hvmc
$ hvmc reduce examples/addition.hvmc -- "a & @mul ~ (#3 (#4 a))"
$ hvmc reduce -- "a & #3 ~ <* #4 a>""##
)]
struct FullCli {
  #[command(subcommand)]
  pub mode: CliMode,
}

#[derive(Parser, Debug)]
#[command(author, version)]
struct BareCli {
  #[command(flatten)]
  pub opts: RuntimeOpts,
  #[command(flatten)]
  pub args: RunArgs,
}

#[derive(Subcommand, Clone, Debug)]
#[command(author, version)]
enum CliMode {
  /// Compile a hvm-core program into a Rust crate.
  Compile {
    /// hvm-core file to compile.
    file: PathBuf,
    /// Compile this hvm-core file to a dynamic library.
    ///
    /// These can be included when running with the `--include` option.
    #[arg(short, long)]
    dylib: bool,
    /// Output path; defaults to the input file with `.hvmc` stripped.
    #[arg(short, long)]
    output: Option<PathBuf>,
    #[command(flatten)]
    transform_args: TransformArgs,
  },
  /// Run a program, optionally passing a list of arguments to it.
  Run {
    /// Name of the file to load.
    file: PathBuf,
    #[command(flatten)]
    args: RunArgs,
    #[command(flatten)]
    run_opts: RuntimeOpts,
    #[command(flatten)]
    transform_args: TransformArgs,
  },
  /// Reduce hvm-core expressions to their normal form.
  ///
  /// The expressions are passed as command-line arguments.
  /// It is also possible to load files before reducing the expression,
  /// which makes it possible to reference definitions from the file
  /// in the expression.
  Reduce {
    /// Files to load before reducing the expressions.
    ///
    /// Multiple files will act as if they're concatenated together.
    #[arg(required = false)]
    files: Vec<PathBuf>,
    /// Expressions to reduce.
    ///
    /// The normal form of each expression will be
    /// printed on a new line. This list must be separated from the file list
    /// with a double dash ('--').
    #[arg(required = false, last = true)]
    exprs: Vec<String>,
    #[command(flatten)]
    run_opts: RuntimeOpts,
    #[command(flatten)]
    transform_args: TransformArgs,
  },
  /// Transform a hvm-core program using one of the optimization passes.
  Transform {
    /// Files to load before reducing the expressions.
    ///
    /// Multiple files will act as if they're concatenated together.
    #[arg(required = true)]
    files: Vec<PathBuf>,
    #[command(flatten)]
    transform_args: TransformArgs,
  },
}

#[derive(Args, Clone, Debug)]
struct TransformArgs {
  /// Enables or disables transformation passes.
  #[arg(short = 'O', value_delimiter = ' ', action = clap::ArgAction::Append)]
  transform_passes: Vec<TransformPass>,
  #[command(flatten)]
  transform_opts: TransformOpts,
}

#[derive(Args, Clone, Debug)]
struct RuntimeOpts {
  /// Show performance statistics.
  #[arg(short, long = "stats")]
  show_stats: bool,
  /// Single-core mode (no parallelism).
  #[arg(short = '1', long = "single")]
  single_core: bool,
  /// Lazy mode.
  ///
  /// Lazy mode only expands references that are reachable
  /// by a walk from the root of the net. This leads to a dramatic slowdown,
  /// but allows running programs that would expand indefinitely otherwise.
  #[arg(short, long = "lazy")]
  lazy_mode: bool,
  /// How much memory to allocate on startup.
  ///
  /// Supports abbreviations such as '4G' or '400M'.
  #[arg(short, long, value_parser = util::parse_abbrev_number::<usize>)]
  memory: Option<usize>,
}

#[derive(Args, Clone, Debug)]
struct RunArgs {
  /// Name of the definition that will get reduced.
  #[arg(short, default_value = "main")]
  entry_point: String,
  /// Dynamic library hvm-core files to include.
  ///
  /// hvm-core files can be compiled as dylibs with the `--dylib` option.
  #[arg(short, long, value_delimiter = ' ', action = clap::ArgAction::Append)]
  include: Vec<PathBuf>,
  /// List of arguments to pass to the program.
  ///
  /// Arguments are passed using the lambda-calculus interpretation
  /// of interaction combinators. So, for example, if the arguments are
  /// "#1" "#2" "#3", then the expression that will get reduced is
  /// `r & @main ~ (#1 (#2 (#3 r)))`.
  args: Vec<String>,
}

fn run(host: Arc<Mutex<Host>>, opts: RuntimeOpts, args: RunArgs) {
  let mut net = Net { root: Tree::Ref { nam: args.entry_point }, redexes: vec![] };
  for arg in args.args {
    let arg: Net = Net::from_str(&arg).unwrap();
    net.redexes.extend(arg.redexes);
    net.apply_tree(arg.root);
  }

  reduce_exprs(host, &[net], &opts);
}

fn load_book(files: &[PathBuf], transform_args: &TransformArgs) -> Book {
  let mut book = files
    .iter()
    .map(|name| {
      let contents = fs::read_to_string(name).unwrap_or_else(|_| {
        eprintln!("Input file {:?} not found", name);
        process::exit(1);
      });
      contents.parse::<Book>().unwrap_or_else(|e| {
        eprintln!("Parsing error {e}");
        process::exit(1);
      })
    })
    .fold(Book::default(), |mut acc, i| {
      acc.nets.extend(i.nets);
      acc
    });

  let transform_passes = TransformPasses::from(&transform_args.transform_passes[..]);
  book.transform(transform_passes, &transform_args.transform_opts).unwrap();

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

      let insert_into_host = lib.get::<fn(&mut Host)>(b"hvmc_dylib_v0__insert_host").expect("failed to load symbol");
      insert_into_host(&mut host.lock());

      std::mem::forget(lib);
    }
  }
}

fn reduce_exprs(host: Arc<Mutex<Host>>, exprs: &[Net], opts: &RuntimeOpts) {
  let heap = run::Heap::new(opts.memory).expect("memory allocation failed");
  for expr in exprs {
    let mut net = DynNet::new(&heap, opts.lazy_mode);
    dispatch_dyn_net!(&mut net => {
      host.lock().encode_net(net, Trg::port(run::Port::new_var(net.root.addr())), expr);
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

fn print_stats<M: Mode>(net: &run::Net<M>, elapsed: Duration) {
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

/// Copies the `hvm-core` source to a temporary `.hvm` directory.
/// Only a subset of `Cargo.toml` is included.
fn create_temp_hvm(host: Arc<Mutex<host::Host>>) -> Result<(), io::Error> {
  let gen = compile::compile_host(&host.lock());
  let outdir = ".hvm";
  if Path::new(&outdir).exists() {
    fs::remove_dir_all(outdir)?;
  }
  let cargo_toml = include_str!("../Cargo.toml");
  let mut cargo_toml = cargo_toml.split_once("##--COMPILER-CUTOFF--##").unwrap().0.to_owned();
  cargo_toml.push_str("[features]\ndefault = ['cli']\ncli = ['std', 'dep:clap']\nstd = []");

  macro_rules! include_files {
    ($([$($prefix:ident)*])? $mod:ident {$($sub:tt)*} $($rest:tt)*) => {
      fs::create_dir_all(concat!(".hvm/src/", $($(stringify!($prefix), "/",)*)? stringify!($mod)))?;
      include_files!([$($($prefix)* $mod)?] $($sub)*);
      include_files!([$($($prefix)*)?] $mod $($rest)*);
    };
    ($([$($prefix:ident)*])? $file:ident $($rest:tt)*) => {
      fs::write(
        concat!(".hvm/src/", $($(stringify!($prefix), "/",)*)* stringify!($file), ".rs"),
        include_str!(concat!($($(stringify!($prefix), "/",)*)* stringify!($file), ".rs")),
      )?;
      include_files!([$($($prefix)*)?] $($rest)*);
    };
    ($([$($prefix:ident)*])?) => {};
  }

  fs::create_dir_all(".hvm/src")?;
  fs::write(".hvm/Cargo.toml", cargo_toml)?;
  fs::write(".hvm/src/gen.rs", gen)?;

  include_files! {
    ast
    compile
    fuzz
    host {
      calc_labels
      encode
      readback
    }
    lib
    main
    ops {
      num
      word
    }
    prelude
    run {
      addr
      allocator
      def
      dyn_net
      instruction
      interact
      linker
      net
      node
      parallel
      port
      wire
    }
    stdlib
    trace
    transform {
      coalesce_ctrs
      encode_adts
      eta_reduce
      inline
      pre_reduce
      prune
    }
    util {
      apply_tree
      array_vec
      bi_enum
      create_var
      deref
      maybe_grow
      parse_abbrev_number
      stats
    }
  }

  Ok(())
}

/// Appends a function to `lib.rs` that will be dynamically loaded
/// by hvm-core when the generated dylib is included.
fn prepare_temp_hvm_dylib() -> Result<(), io::Error> {
  insert_crate_type_cargo_toml()?;

  let mut lib = fs::read_to_string(".hvm/src/lib.rs")?;

  writeln!(lib).unwrap();
  writeln!(
    lib,
    r#"
#[no_mangle]
pub fn hvmc_dylib_v0__insert_host(host: &mut host::Host) {{
  gen::insert_into_host(host)
}}
  "#
  )
  .unwrap();

  fs::write(".hvm/src/lib.rs", lib)
}

/// Adds `crate_type = ["dylib"]` under the `[lib]` section of `Cargo.toml`.
fn insert_crate_type_cargo_toml() -> Result<(), io::Error> {
  let mut cargo_toml = String::new();

  let file = File::open(".hvm/Cargo.toml")?;
  for line in io::BufReader::new(file).lines() {
    let line = line?;
    writeln!(cargo_toml, "{line}").unwrap();

    if line == "[lib]" {
      writeln!(cargo_toml, r#"crate_type = ["dylib"]"#).unwrap();
    }
  }

  fs::write(".hvm/Cargo.toml", cargo_toml)
}

/// Compiles the `.hvm` directory, appending the provided `args` to `cargo`.
fn compile_temp_hvm(args: &[&'static str]) -> Result<(), io::Error> {
  let output = process::Command::new("cargo")
    .current_dir(".hvm")
    .arg("build")
    .arg("--release")
    .args(args)
    .stderr(Stdio::inherit())
    .output()?;

  if !output.status.success() {
    process::exit(1);
  }

  Ok(())
}
