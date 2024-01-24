#![cfg_attr(feature = "trace", feature(const_type_name))]
#![feature(extern_types)]
#![allow(non_snake_case)]

pub mod ast;
pub mod jit;
pub mod ops;
pub mod run;

#[doc(hidden)] // not public api
pub mod fuzz;
#[doc(hidden)] // not public api
pub mod trace;

#[doc(hidden)] // not public api
pub mod gen;
