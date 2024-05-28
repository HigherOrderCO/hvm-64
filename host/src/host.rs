//! The runtime's host, which acts as a translation layer between the AST and
//! the runtime.
#![cfg_attr(not(feature = "std"), no_std)]

include!("../../prelude.rs");

use crate::prelude::*;
use hvm64_ast::{Book, Tree};
use hvm64_runtime::{Addr, Def, DynDef, InterpretedDef, LabSet, Port, Tag, Wire};

mod calc_labels;
mod encode;
mod readback;

use calc_labels::calculate_label_sets;

/// Stores a bidirectional mapping between names and runtime defs.
#[derive(Default)]
pub struct Host {
  /// the forward mapping, from a name to the runtime def
  pub defs: Map<String, Box<DynDef>>,
  /// the backward mapping, from the address of a runtime def to the name
  pub back: Map<Addr, String>,
}

impl Host {
  pub fn new(book: &Book) -> Host {
    let mut host = Host::default();
    host.insert_book(book);
    host
  }

  /// Converts all of the nets from the book into runtime defs, and inserts them
  /// into the host. The book must not have refs that are not in the book or the
  /// host.
  pub fn insert_book(&mut self, book: &Book) {
    self.insert_book_with_default(book, &mut |x| panic!("Found reference {x:?}, which is not in the book!"))
  }

  /// Like `insert_book`, but allows specifying a function (`default_def`) that
  /// will be run when the name of a definition is not found in the book.
  /// The return value of the function will be inserted into the host.
  pub fn insert_book_with_default(&mut self, book: &Book, default_def: &mut dyn FnMut(&str) -> Box<DynDef>) {
    #[cfg(feature = "std")]
    {
      self.defs.reserve(book.len());
      self.back.reserve(book.len());
    }

    // Because there may be circular dependencies, inserting the definitions
    // must be done in two phases:

    // First, we insert empty defs into the host. Even though their instructions
    // are not yet set, the address of the def will not change, meaning that
    // `net_to_runtime_def` can safely use `Port::new_def` on them.
    for (name, labs) in calculate_label_sets(book, |nam| match self.defs.get(nam) {
      Some(x) => x.labs.clone(),
      None => {
        self.insert_def(nam, default_def(nam));
        self.defs[nam].labs.clone()
      }
    })
    .into_iter()
    {
      let def = Box::new(Def::new(labs, InterpretedDef::default()));
      self.insert_def(name, def);
    }

    // Now that `defs` is fully populated, we can fill in the instructions of
    // each of the new defs.
    for (nam, net) in book.iter() {
      let data = self.encode_def(net);
      self.get_mut::<InterpretedDef>(nam).data = data;
    }
  }

  /// Inserts a singular def into the mapping.
  pub fn insert_def(&mut self, name: &str, def: Box<DynDef>) {
    self.back.insert(Port::new_ref(&def).addr(), name.to_owned());
    self.defs.insert(name.to_owned(), def);
  }

  /// Returns a mutable [`Def`] named `name`.
  pub fn get_mut<T: Send + Sync + 'static>(&mut self, name: &str) -> &mut Def<T> {
    Def::downcast_mut(self.defs.get_mut(name).unwrap()).unwrap()
  }
}
