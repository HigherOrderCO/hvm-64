use crate::run::{*};

impl<'a, const LAZY: bool> NetFields<'a, LAZY> where [(); LAZY as usize]: {
  pub fn call_native(&mut self, book: &Book, ptr: Ptr, x: Ptr) -> bool {
    match ptr.loc() {
      _ => { return false; }
    }
  }

}
