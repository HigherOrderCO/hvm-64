#![feature(const_type_id, extern_types, inline_const, generic_const_exprs, new_uninit)]
#![cfg_attr(feature = "trace", feature(const_type_name))]
#![cfg_attr(not(feature = "std"), no_std)]
#![allow(
  non_snake_case,
  incomplete_features,
  clippy::field_reassign_with_default,
  clippy::missing_safety_doc,
  clippy::new_ret_no_self
)]
#![warn(
  clippy::alloc_instead_of_core,
  clippy::std_instead_of_core,
  clippy::std_instead_of_alloc,
  clippy::absolute_paths
)]

extern crate alloc;

mod prelude;

pub mod ast;
pub mod compile;
pub mod host;
pub mod ops;
pub mod run;
pub mod stdlib;
pub mod transform;
pub mod util;

#[doc(hidden)] // not public api
pub mod fuzz;
#[doc(hidden)] // not public api
pub mod trace;

#[doc(hidden)] // shim for compilation
pub mod gen;
