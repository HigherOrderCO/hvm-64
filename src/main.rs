#![cfg_attr(feature = "trace", feature(const_type_name))]

use clap::{Args, Parser, Subcommand};
use hvmc::{
  ast::{Net, Tree},
  host::Host,
  run::{DynNet, Mode, Strict, Trg},
  *,
};

use std::{
  fs, io,
  path::Path,
  process::{self, Stdio},
  str::FromStr,
  sync::{Arc, Mutex},
  time::{Duration, Instant},
};

fn main() {
  if cfg!(feature = "trace") {
    trace::set_hook();
  }
  if cfg!(feature = "_full_cli") {
    let cli = FullCli::parse();
    match cli.mode {
      CliMode::Compile { file } => {
        let host = load_files(&[file.clone()]);
        compile_executable(&file, &host.lock().unwrap()).unwrap();
      }
      CliMode::Run { opts, file, args } => {
        let host = load_files(&[file]);
        run(&host.lock().unwrap(), opts, args);
      }
      CliMode::Reduce { run_opts, files, exprs } => {
        let host = load_files(&files);
        let exprs: Vec<_> = exprs.iter().map(|x| Net::from_str(x).unwrap()).collect();
        reduce_exprs(&host.lock().unwrap(), &exprs, &run_opts);
      }
    }
  } else {
    let cli = BareCli::parse();
    let host = hvmc::gen::host();
    run(&host, cli.opts, cli.args);
  }
  if cfg!(feature = "trace") {
    hvmc::trace::_read_traces(usize::MAX);
  }
}

#[derive(Parser, Debug)]
#[command(author, version, about = "A massively parallel Interaction Combinator evaluator",
  long_about = r##"
A massively parallel Interaction Combinator evaluator

Examples: 
$ hvmc run examples/church_encoding/church.hvm
$ hvmc run tests/programs/addition.hvmc "#16" "#3"
$ hvmc compile tests/programs/addition.hvmc
$ hvmc reduce tests/programs/addition.hvmc -- "a & @mul ~ (#3 (#4 a))"
$ hvmc reduce -- "a & #3 ~ <* #4 a>""##)]
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

#[derive(Args, Clone, Debug)]
struct RuntimeOpts {
  #[arg(short = 's', long = "stats")]
  /// Show performance statistics
  show_stats: bool,
  #[arg(short = '1', long = "single")]
  /// Single-core mode (no parallelism)
  single_core: bool,
  #[arg(short = 'l', long = "lazy")]
  /// Lazy mode
  ///
  /// Lazy mode only expands references that are reachable
  /// by a walk from the root of the net. This leads to a dramatic slowdown,
  /// but allows running programs that would expand indefinitely otherwise
  lazy_mode: bool,
  #[arg(short = 'm', long = "memory", default_value = "1G", value_parser = mem_parser)]
  /// How much memory to allocate on startup.
  ///
  /// Supports abbreviations such as '4G' or '400M'
  memory: u64,
}

#[derive(Args, Clone, Debug)]
struct RunArgs {
  #[arg(short = 'e', default_value = "main")]
  /// Name of the definition that will get reduced
  entry_point: String,
  /// List of arguments to pass to the program
  ///
  /// Arguments are passed using the lambda-calculus interpretation
  /// of interaction combinators. So, for example, if the arguments are
  /// "#1" "#2" "#3", then the expression that will get reduced is
  /// `r & @main ~ (#1 (#2 (#3 r)))`
  args: Vec<String>,
}

#[derive(Subcommand, Clone, Debug)]
#[command(author, version)]
enum CliMode {
  /// Compile a hvm-core program into a Rust crate
  Compile {
    /// hvm-core file to compile
    file: String,
  },
  /// Run a program, optionally passing a list of arguments to it.
  Run {
    #[command(flatten)]
    opts: RuntimeOpts,
    /// Name of the file to load
    file: String,
    #[command(flatten)]
    args: RunArgs,
  },
  /// Reduce hvm-core expressions to their normal form
  ///
  /// The expressions are passed as command-line arguments.
  /// It is also possible to load files before reducing the expression,
  /// which makes it possible to reference definitions from the file
  /// in the expression
  Reduce {
    #[command(flatten)]
    run_opts: RuntimeOpts,
    #[arg(required = false)]
    /// Files to load before reducing the expressions.
    ///
    /// Multiple files will act as if they're concatenated together.
    files: Vec<String>,
    #[arg(required = false, last = true)]
    /// Expressions to reduce.
    ///
    /// The normal form of each expression will be
    /// printed on a new line. This list must be separated from the file list
    /// with a double dash ('--')
    exprs: Vec<String>,
  },
}

fn run(host: &Host, opts: RuntimeOpts, args: RunArgs) {
  let mut net = Net { root: Tree::Ref { nam: args.entry_point }, rdex: vec![] };
  for arg in args.args {
    let arg: Net = Net::from_str(&arg).unwrap();
    net.rdex.extend(arg.rdex);
    net.apply_tree(arg.root);
  }

  reduce_exprs(host, &[net], &opts);
}
/// Turn a string representation of a number, such as '1G' or '400K', into a
/// number.
///
/// This return a [`u64`] instead of [`usize`] to ensure that parsing CLI args
/// doesn't fail on 32-bit systems. We want it to fail later on, when attempting
/// to run the program
fn mem_parser(arg: &str) -> Result<u64, String> {
  let (base, mult) = match arg.to_lowercase().chars().last() {
    None => return Err("Mem size argument is empty".to_string()),
    Some('k') => (&arg[0 .. arg.len() - 1], 1 << 10),
    Some('m') => (&arg[0 .. arg.len() - 1], 1 << 20),
    Some('g') => (&arg[0 .. arg.len() - 1], 1 << 30),
    Some(_) => (arg, 1),
  };
  let base = base.parse::<u64>().map_err(|e| e.to_string())?;
  Ok(base * mult)
}

fn load_files(files: &[String]) -> Arc<Mutex<Host>> {
  let files: Vec<_> = files
    .iter()
    .map(|name| {
      fs::read_to_string(name).unwrap_or_else(|_| {
        eprintln!("Input file {:?} not found", name);
        process::exit(1);
      })
    })
    .collect();
  let host = Arc::new(Mutex::new(host::Host::default()));
  host.lock().unwrap().insert_def(
    "HVM.log",
    host::DefRef::Owned(Box::new(stdlib::LogDef::new({
      let host = Arc::downgrade(&host);
      move |wire| {
        println!("{}", host.upgrade().unwrap().lock().unwrap().readback_tree(&wire));
      }
    }))),
  );
  for file_contents in files {
    host.lock().unwrap().insert_book(&ast::Book::from_str(&file_contents).unwrap());
  }
  host
}

fn reduce_exprs(host: &Host, exprs: &[Net], opts: &RuntimeOpts) {
  let heap = run::Net::<Strict>::init_heap(opts.memory as usize);
  for expr in exprs {
    let mut net = DynNet::new(&heap, opts.lazy_mode);
    dispatch_dyn_net!(&mut net => {
      host.encode_net(net, Trg::port(run::Port::new_var(net.root.addr())), expr);
      let start_time = Instant::now();
      if opts.single_core {
        net.normal();
      } else {
        net.parallel_normal();
      }
      let elapsed = start_time.elapsed();
      println!("{}", host.readback(net));
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

fn compile_executable(file_name: &str, host: &host::Host) -> Result<(), io::Error> {
  let gen = compile::compile_host(host);
  let outdir = ".hvm";
  if Path::new(&outdir).exists() {
    fs::remove_dir_all(outdir)?;
  }
  let cargo_toml = include_str!("../Cargo.toml");
  let cargo_toml = cargo_toml.split("##--COMPILER-CUTOFF--##").next().unwrap();
  fs::create_dir_all(format!("{}/src", outdir))?;
  fs::write(".hvm/Cargo.toml", cargo_toml)?;
  fs::write(".hvm/src/ast.rs", include_str!("../src/ast.rs"))?;
  fs::write(".hvm/src/compile.rs", include_str!("../src/compile.rs"))?;
  fs::write(".hvm/src/host/encode.rs", include_str!("../src/host/encode.rs"))?;
  fs::write(".hvm/src/fuzz.rs", include_str!("../src/fuzz.rs"))?;
  fs::write(".hvm/src/host.rs", include_str!("../src/host.rs"))?;
  fs::write(".hvm/src/lib.rs", include_str!("../src/lib.rs"))?;
  fs::write(".hvm/src/main.rs", include_str!("../src/main.rs"))?;
  fs::write(".hvm/src/ops.rs", include_str!("../src/ops.rs"))?;
  fs::write(".hvm/src/run.rs", include_str!("../src/run.rs"))?;
  fs::write(".hvm/src/stdlib.rs", include_str!("../src/stdlib.rs"))?;
  fs::write(".hvm/src/trace.rs", include_str!("../src/trace.rs"))?;
  fs::write(".hvm/src/util.rs", include_str!("../src/util.rs"))?;
  fs::write(".hvm/src/gen.rs", gen)?;

  let output = process::Command::new("cargo")
    .current_dir("./.hvm")
    .arg("build")
    .arg("--release")
    .stderr(Stdio::inherit())
    .output()?;
  if !output.status.success() {
    process::exit(1);
  }

  let target = format!("./{}", file_name.strip_suffix(".hvmc").unwrap_or(file_name));
  fs::copy("./.hvm/target/release/hvmc", target)?;

  Ok(())
}
