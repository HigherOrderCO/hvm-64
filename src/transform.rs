use crate::prelude::*;

use crate::ast::Book;

pub mod coalesce_ctrs;
pub mod encode_adts;
pub mod eta_reduce;
pub mod inline;
pub mod pre_reduce;
pub mod prune;

#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
#[cfg_attr(feature = "std", derive(Error))]
pub enum TransformError {
  #[cfg_attr(feature = "std", error("infinite reference cycle in `@{0}`"))]
  InfiniteRefCycle(String),
}

impl Book {
  pub fn transform(&mut self, passes: TransformPasses, opts: &TransformOpts) -> Result<(), TransformError> {
    if passes.prune {
      self.prune(&opts.prune_entrypoints);
    }
    if passes.pre_reduce {
      if passes.eta_reduce {
        for (_, def) in &mut self.nets {
          def.eta_reduce();
        }
      }
      self.pre_reduce(
        &|x| opts.pre_reduce_skip.iter().any(|y| x == y),
        opts.pre_reduce_memory,
        opts.pre_reduce_rewrites,
      );
    }
    for (_, def) in &mut self.nets {
      if passes.eta_reduce {
        def.eta_reduce();
      }
      for tree in def.trees_mut() {
        if passes.coalesce_ctrs {
          tree.coalesce_constructors();
        }
        if passes.encode_adts {
          tree.encode_scott_adts();
        }
      }
    }
    if passes.inline {
      loop {
        let inline_changed = self.inline()?;
        if inline_changed.is_empty() {
          break;
        }
        if !(passes.eta_reduce || passes.encode_adts) {
          break;
        }
        for name in inline_changed {
          let def = self.get_mut(&name).unwrap();
          if passes.eta_reduce {
            def.eta_reduce();
          }
          if passes.encode_adts {
            for tree in def.trees_mut() {
              tree.encode_scott_adts();
            }
          }
        }
      }
    }
    if passes.prune {
      self.prune(&opts.prune_entrypoints);
    }
    Ok(())
  }
}

#[derive(Clone, Debug)]
#[cfg_attr(feature = "cli", derive(clap::Args))]
#[non_exhaustive]
pub struct TransformOpts {
  /// Names of the definitions that should not get pre-reduced.
  ///
  /// For programs that don't take arguments and don't have side effects this is
  /// usually the entry point of the program (otherwise, the whole program will
  /// get reduced to normal form).
  #[cfg_attr(feature = "cli", arg(long = "pre-reduce-skip", value_delimiter = ' ', action = clap::ArgAction::Append))]
  pub pre_reduce_skip: Vec<String>,

  /// How much memory to allocate when pre-reducing.
  ///
  /// Supports abbreviations such as '4G' or '400M'.
  #[cfg_attr(feature = "cli", arg(long = "pre-reduce-memory", value_parser = crate::util::parse_abbrev_number::<usize>))]
  pub pre_reduce_memory: Option<usize>,

  /// Maximum amount of rewrites to do when pre-reducing.
  ///
  /// Supports abbreviations such as '4G' or '400M'.
  #[cfg_attr(feature = "cli", arg(long = "pre-reduce-rewrites", default_value = "100M", value_parser = crate::util::parse_abbrev_number::<u64>))]
  pub pre_reduce_rewrites: u64,

  /// Names of the definitions that should not get pruned.
  #[cfg_attr(feature = "cli", arg(long = "prune-entrypoints", default_value = "main"))]
  pub prune_entrypoints: Vec<String>,
}

impl TransformOpts {
  pub fn add_entrypoint(&mut self, entrypoint: &str) {
    self.pre_reduce_skip.push(entrypoint.to_owned());
    self.prune_entrypoints.push(entrypoint.to_owned());
  }
}

macro_rules! transform_passes {
  ($($pass:ident: $name:literal $(| $alias:literal)*),* $(,)?) => {
    #[derive(Debug, Default, Clone, Copy)]
    #[non_exhaustive]
    pub struct TransformPasses {
      $(pub $pass: bool),*
    }

    impl TransformPasses {
      pub const NONE: Self = Self { $($pass: false),* };
      pub const ALL: Self = Self { $($pass: true),* };
    }

    #[derive(Debug, Clone, Copy)]
    #[allow(non_camel_case_types)]
    pub enum TransformPass {
      all(bool),
      $($pass(bool),)*
    }

    #[cfg(feature = "cli")]
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

    impl From<&[TransformPass]> for TransformPasses {
      fn from(args: &[TransformPass]) -> TransformPasses {
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
  coalesce_ctrs: "coalesce-ctrs" | "coalesce",
  encode_adts: "encode-adts" | "adts",
  eta_reduce: "eta-reduce" | "eta",
  inline: "inline",
  prune: "prune",
}
