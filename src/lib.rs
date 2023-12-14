#![cfg_attr(feature = "trace", feature(const_type_name))]
#![allow(dead_code)]
#![allow(non_snake_case)]
#![allow(non_upper_case_globals)]

pub mod ast;
pub mod jit;
pub mod ops;
pub mod run;
pub mod trace;

pub mod gen;
