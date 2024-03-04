//! The runtime's host, which acts as a translation layer between the AST and
//! the runtime.

use crate::{
  ast::{Book, Net, Tree},
  run::{self, Addr, Def, Instruction, InterpretedDef, LabSet, Mode, Port, Tag, TrgId, Wire},
  util::create_var,
};
use std::{
  collections::{hash_map::Entry, HashMap},
  ops::{Deref, DerefMut, RangeFrom},
};

mod calc_labels;
mod encode_def;
mod encode_net;
mod readback;

use calc_labels::calculate_label_sets;
use encode_def::ast_net_to_instructions;

/// Stores a bidirectional mapping between names and runtime defs.
#[derive(Default)]
pub struct Host {
  /// the forward mapping, from a name to the runtime def
  pub defs: HashMap<String, DefRef>,
  /// the backward mapping, from the address of a runtime def to the name
  pub back: HashMap<Addr, String>,
}

/// A potentially-owned reference to a [`Def`]. Vitally, the address of the
/// `Def` is stable, even if the `DefRef` moves -- this is why
/// [`std::Borrow::Cow`] cannot be used here.
pub enum DefRef {
  Owned(Box<dyn DerefMut<Target = Def> + Send + Sync>),
  Static(&'static Def),
}

impl Deref for DefRef {
  type Target = Def;
  fn deref(&self) -> &Def {
    match self {
      DefRef::Owned(x) => x,
      DefRef::Static(x) => x,
    }
  }
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
  pub fn insert_book_with_default(&mut self, book: &Book, default_def: &mut dyn FnMut(&str) -> DefRef) {
    self.defs.reserve(book.len());
    self.back.reserve(book.len());
    // Because there may be circular dependencies, inserting the definitions
    // must be done in two phases:

    // First, we insert empty defs into the host. Even though their instructions
    // are not yet set, the address of the def will not change, meaning that
    // `net_to_runtime_def` can safely use `Port::new_def` on them.
    for (name, labs) in calculate_label_sets(book, |nam| match self.defs.get(nam) {
      Some(x) => x.labs.clone(),
      None => {
        self.insert_def(&nam, default_def(nam));
        self.defs[nam].labs.clone()
      }
    })
    .into_iter()
    {
      let def = DefRef::Owned(Box::new(Def::new(labs, InterpretedDef::default())));
      self.insert_def(name, def);
    }

    // Now that `defs` is fully populated, we can fill in the instructions of
    // each of the new defs.
    for (nam, net) in book.iter() {
      let instr = ast_net_to_instructions(net, |nam| {
        Port::new_ref(&self.defs[nam] /* calculate_label_sets already ensures all ref names are in self.defs */)
      });
      match self.defs.get_mut(nam).unwrap() {
        DefRef::Owned(def) => def.downcast_mut::<InterpretedDef>().unwrap().data.instr = instr,
        DefRef::Static(_) => unreachable!(),
      }
    }
  }

  /// Inserts a singular def into the mapping.
  pub fn insert_def(&mut self, name: &str, def: DefRef) {
    self.back.insert(Port::new_ref(&def).addr(), name.to_owned());
    self.defs.insert(name.to_owned(), def);
  }
}
