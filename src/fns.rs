use crate::run::*;

impl Net {
  #[inline]
  pub fn call_native(&mut self, book: &Book, ptr: Ptr, x: Ptr) -> bool {
    (self.call_native)(self, book, ptr, x)
  }
}