extern crate alloc;

#[allow(unused)]
mod prelude {
  pub use alloc::{
    borrow::ToOwned,
    boxed::Box,
    format,
    string::{String, ToString},
    vec,
    vec::Vec,
  };

  pub use core::{fmt, hint, iter, mem, ptr};

  #[cfg(feature = "std")]
  pub use std::collections::{hash_map::Entry, HashMap as Map, HashSet as Set};

  #[cfg(not(feature = "std"))]
  pub use alloc::collections::{btree_map::Entry, BTreeMap as Map, BTreeSet as Set};
}
