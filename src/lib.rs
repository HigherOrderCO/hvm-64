#![cfg_attr(feature = "trace", feature(const_type_name))]
#![allow(non_snake_case)]

pub mod ast;
pub mod fuzz;
pub mod jit;
pub mod ops;
pub mod run;
pub mod trace;

pub mod gen;
