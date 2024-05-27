use std::path::PathBuf;

use clap::{Parser, Subcommand};

use crate::{args::TransformArgs, RunArgs, RuntimeOpts};

#[derive(Parser, Debug)]
#[command(
  author,
  version,
  about = "A massively parallel Interaction Combinator evaluator",
  long_about = r##"
A massively parallel Interaction Combinator evaluator

Examples: 
$ hvm64 run examples/church_encoding/church.hvm
$ hvm64 run examples/addition.hvm "#16" "#3"
$ hvm64 compile examples/addition.hvm
$ hvm64 reduce examples/addition.hvm -- "a & @mul ~ (#3 (#4 a))"
$ hvm64 reduce -- "a & #3 ~ <* #4 a>""##
)]
pub struct FullCli {
  #[command(subcommand)]
  pub mode: CliMode,
}

#[derive(Subcommand, Clone, Debug)]
#[command(author, version)]
pub enum CliMode {
  /// Compile a hvm-64 program into a Rust crate.
  Compile {
    /// hvm-64 file to compile.
    file: PathBuf,
    /// Output path; defaults to the input file with `.hvm` stripped.
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
  /// Reduce hvm-64 expressions to their normal form.
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
  /// Transform a hvm-64 program using one of the optimization passes.
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
