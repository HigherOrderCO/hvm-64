#![cfg_attr(feature = "trace", feature(const_type_name))]
#![allow(non_snake_case)]

pub mod ast;
pub mod compile;
pub mod host;
pub mod ops;
pub mod run;

mod util;

#[doc(hidden)] // not public api
pub mod fuzz;
#[doc(hidden)] // not public api
pub mod trace;

#[doc(hidden)]
pub mod gen;
