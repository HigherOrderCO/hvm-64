#![cfg_attr(not(feature = "std"), no_std)]

include!("../../prelude.rs");

use hvm64_ast::Book;

pub mod coalesce_ctrs;
pub mod encode_adts;
pub mod eta_reduce;
pub mod inline;
pub mod pre_reduce;
pub mod prune;

use coalesce_ctrs::CoalesceCtrs;
use encode_adts::EncodeAdts;
use eta_reduce::EtaReduce;
use inline::Inline;
use pre_reduce::PreReduce;
use prune::Prune;

#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
#[cfg_attr(feature = "std", derive(thiserror::Error))]
pub enum TransformError {
  #[cfg_attr(feature = "std", error("infinite reference cycle in `@{0}`"))]
  InfiniteRefCycle(String),
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
  coalesce_ctrs,
  encode_adts,
  eta_reduce,
  inline,
  prune,
}
