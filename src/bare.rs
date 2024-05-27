use crate::{RunArgs, RuntimeOpts};

use clap::Parser;

#[derive(Parser, Debug)]
#[command(author, version)]
pub struct BareCli {
  #[command(flatten)]
  pub opts: RuntimeOpts,
  #[command(flatten)]
  pub args: RunArgs,
}
