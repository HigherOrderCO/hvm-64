use alloc::alloc::alloc;
use core::{alloc::Layout, mem::MaybeUninit, slice};

use crate::prelude::*;

// TODO: use `Box::new_uninit_slice` once stabilized
// https://github.com/rust-lang/rust/issues/63291
pub fn new_uninit_slice<T>(len: usize) -> Box<[MaybeUninit<T>]> {
  unsafe { Box::from_raw(slice::from_raw_parts_mut(alloc(Layout::array::<T>(len).unwrap()).cast(), len)) }
}
