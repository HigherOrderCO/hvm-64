#![cfg_attr(not(feature = "std"), no_std)]

pub mod prelude;

pub mod ops;

mod bi_enum;
mod deref;
mod multi_iterator;

mod create_var;
mod maybe_grow;
mod new_uninit_slice;
mod parse_abbrev_number;
mod pretty_num;

pub use create_var::*;
pub use maybe_grow::*;
pub use new_uninit_slice::*;
pub use parse_abbrev_number::*;
pub use pretty_num::*;
