use clap::Args;
use hvm64_transform::TransformPasses;
use std::path::PathBuf;

#[derive(Args, Clone, Debug)]
pub struct RunArgs {
  /// Name of the definition that will get reduced.
  #[arg(short, default_value = "main")]
  pub entry_point: String,

  /// List of arguments to pass to the program.
  ///
  /// Arguments are passed using the lambda-calculus interpretation
  /// of interaction combinators. So, for example, if the arguments are
  /// "#1" "#2" "#3", then the expression that will get reduced is
  /// `r & @main ~ (#1 (#2 (#3 r)))`.
  pub args: Vec<String>,
}

#[derive(Args, Clone, Debug)]
pub struct RuntimeOpts {
  /// Show performance statistics.
  #[arg(short, long = "stats")]
  pub show_stats: bool,

  /// Single-core mode (no parallelism).
  #[arg(short = '1', long = "single")]
  pub single_core: bool,

  /// How much memory to allocate on startup.
  ///
  /// Supports abbreviations such as '4G' or '400M'.
  #[arg(short, long, value_parser = hvm64_util::parse_abbrev_number::<usize>)]
  pub memory: Option<usize>,

  /// Dynamic library hvm-64 files to include.
  ///
  /// hvm-64 files can be compiled as dylibs with the `--dylib` option.
  #[arg(short, long, value_delimiter = ' ', action = clap::ArgAction::Append)]
  pub include: Vec<PathBuf>,
}

#[derive(Clone, Debug, Args)]
#[non_exhaustive]
pub struct TransformOpts {
  /// Names of the definitions that should not get pre-reduced.
  ///
  /// For programs that don't take arguments and don't have side effects this is
  /// usually the entry point of the program (otherwise, the whole program will
  /// get reduced to normal form).
  #[arg(long = "pre-reduce-skip", value_delimiter = ' ', action = clap::ArgAction::Append)]
  pub pre_reduce_skip: Vec<String>,

  /// How much memory to allocate when pre-reducing.
  ///
  /// Supports abbreviations such as '4G' or '400M'.
  #[arg(long = "pre-reduce-memory", value_parser = hvm64_util::parse_abbrev_number::<usize>)]
  pub pre_reduce_memory: Option<usize>,

  /// Maximum amount of rewrites to do when pre-reducing.
  ///
  /// Supports abbreviations such as '4G' or '400M'.
  #[arg(long = "pre-reduce-rewrites", default_value = "100M", value_parser = hvm64_util::parse_abbrev_number::<u64>)]
  pub pre_reduce_rewrites: u64,

  /// Names of the definitions that should not get pruned.
  #[arg(long = "prune-entrypoints", default_value = "main")]
  pub prune_entrypoints: Vec<String>,
}

#[derive(Args, Clone, Debug)]
pub struct TransformArgs {
  /// Enables or disables transformation passes.
  #[arg(short = 'O', value_delimiter = ' ', action = clap::ArgAction::Append)]
  pub transform_passes: Vec<TransformPass>,

  #[command(flatten)]
  pub transform_opts: TransformOpts,
}

macro_rules! transform_passes {
  ($($pass:ident: $name:literal $(| $alias:literal)*),* $(,)?) => {
    #[derive(Debug, Clone, Copy)]
    #[allow(non_camel_case_types)]
    pub enum TransformPass {
      all(bool),
      $($pass(bool),)*
    }

    impl clap::ValueEnum for TransformPass {
      fn value_variants<'a>() -> &'a [Self] {
        &[
          Self::all(true), Self::all(false),
          $(Self::$pass(true), Self::$pass(false),)*
        ]
      }

      fn to_possible_value(&self) -> Option<clap::builder::PossibleValue> {
        use TransformPass::*;
        Some(match self {
          all(true) => clap::builder::PossibleValue::new("all"),
          all(false) => clap::builder::PossibleValue::new("no-all"),
          $(
            $pass(true) => clap::builder::PossibleValue::new($name)$(.alias($alias))*,
            $pass(false) => clap::builder::PossibleValue::new(concat!("no-", $name))$(.alias(concat!("no-", $alias)))*,
          )*
        })
      }
    }

    impl TransformPass {
      pub fn to_passes(args: &[TransformPass]) -> TransformPasses {
        use TransformPass::*;
        let mut opts = TransformPasses::NONE;
        for arg in args {
          match arg {
            all(true) => opts = TransformPasses::ALL,
            all(false) => opts = TransformPasses::NONE,
            $(&$pass(b) => opts.$pass = b,)*
          }
        }
        opts
      }
    }
  };
}

transform_passes! {
  pre_reduce: "pre-reduce" | "pre",
  eta_reduce: "eta-reduce" | "eta",
  inline: "inline",
  prune: "prune",
}
