#![cfg_attr(not(feature = "std"), no_std)]

include!("../../prelude.rs");

use crate::prelude::*;

use hvm64_ast::Book;

pub mod eta_reduce;
pub mod inline;
pub mod pre_reduce;
pub mod prune;

use eta_reduce::EtaReduce;
use inline::Inline;
use pre_reduce::PreReduce;
use prune::Prune;

#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum TransformError {
  InfiniteRefCycle(String),
}

impl fmt::Display for TransformError {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    match self {
      TransformError::InfiniteRefCycle(name) => write!(f, "infinite reference cycle in `@{name}`"),
    }
  }
}

pub trait Transform {
  fn transform(&mut self, passes: TransformPasses, opts: &TransformOpts) -> Result<(), TransformError>;
}

impl Transform for Book {
  fn transform(&mut self, passes: TransformPasses, opts: &TransformOpts) -> Result<(), TransformError> {
    if passes.prune {
      self.prune(&opts.prune_entrypoints);
    }
    if passes.pre_reduce {
      if passes.eta_reduce {
        for def in self.nets.values_mut() {
          def.eta_reduce();
        }
      }
      self.pre_reduce(
        &|x| opts.pre_reduce_skip.iter().any(|y| x == y),
        opts.pre_reduce_memory,
        opts.pre_reduce_rewrites,
      );
    }
    for def in &mut self.nets.values_mut() {
      if passes.eta_reduce {
        def.eta_reduce();
      }
    }
    if passes.inline {
      loop {
        let inline_changed = self.inline()?;
        if inline_changed.is_empty() {
          break;
        }
        if !passes.eta_reduce {
          break;
        }
        for name in inline_changed {
          let def = self.get_mut(&name).unwrap();
          def.eta_reduce();
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
pub struct TransformOpts {
  pub pre_reduce_skip: Vec<String>,
  pub pre_reduce_memory: Option<usize>,
  pub pre_reduce_rewrites: u64,
  pub prune_entrypoints: Vec<String>,
}

impl TransformOpts {
  pub fn add_entrypoint(&mut self, entrypoint: &str) {
    self.pre_reduce_skip.push(entrypoint.to_owned());
    self.prune_entrypoints.push(entrypoint.to_owned());
  }
}

macro_rules! transform_passes {
  ($($pass:ident),* $(,)?) => {
    #[derive(Debug, Default, Clone, Copy)]
    #[non_exhaustive]
    pub struct TransformPasses {
      $(pub $pass: bool),*
    }

    impl TransformPasses {
      pub const NONE: Self = Self { $($pass: false),* };
      pub const ALL: Self = Self { $($pass: true),* };
    }
  }
}

transform_passes! {
  pre_reduce,
  eta_reduce,
  inline,
  prune,
}
