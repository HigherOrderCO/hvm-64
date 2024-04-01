use thiserror::Error;

pub mod coalesce_ctrs;
pub mod encode_adts;
pub mod eta_reduce;
pub mod inline;
pub mod pre_reduce;
pub mod prune;

#[derive(Debug, Error, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum TransformError {
  #[error("infinite reference cycle")]
  InfiniteRefCycle,
}
