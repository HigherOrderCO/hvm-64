use crate::run::{*};

impl<'a> Net<'a> {

  pub fn call_native(&mut self, book: &Book, ptr: Ptr, x: Ptr) -> bool {
    match ptr.val() {
      _ => { return false; }
    }
  }

}
