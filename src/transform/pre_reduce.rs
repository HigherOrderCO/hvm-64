// Reduce the compiled networks, solving any annihilations and commutations.

use std::sync::Mutex;

use crate::{
  ast::Book,
  host::{DefRef, Host},
  run::{self, Def, LabSet},
};

/// A Def that pushes all interactions to its inner Vec.
#[derive(Default)]
struct InertDef(Mutex<Vec<(run::Port, run::Port)>>);

impl run::AsDef for InertDef {
  unsafe fn call<M: run::Mode>(def: *const run::Def<Self>, _: &mut run::Net<M>, port: run::Port) {
    let def = unsafe { &*def };
    def.data.0.lock().unwrap().push((run::Port::new_ref(def), port));
  }
}

impl Book {
  /// Reduces the definitions in the book individually, except for the skipped
  /// ones.
  ///
  /// Defs that are not in the book are treated as inert defs.
  pub fn pre_reduce(&mut self, skip: &dyn Fn(&str) -> bool, max_memory: usize, max_rwts: u64) -> Result<(), String> {
    let mut host = Host::default();
    // When a ref is not found in the `Host`, then
    // put an inert def in its place
    host.insert_book_with_default(self, &mut |_| DefRef::Owned(Box::new(Def::new(LabSet::ALL, InertDef::default()))));
    let area = run::Heap::new_bytes(max_memory);

    for (nam, net) in self.nets.iter_mut() {
      // Skip unnecessary work
      if net.redexes.is_empty() || skip(nam) {
        continue;
      }

      let mut rt = run::Net::<run::Strict>::new(&area);
      rt.boot(host.defs.get(nam).expect("No function."));
      rt.reduce(max_rwts as usize);

      // Move interactions with inert defs back into the net redexes array
      for def in host.defs.values() {
        if let Some(def) = def.downcast_ref::<InertDef>() {
          let mut stored_redexes = def.data.0.lock().unwrap();
          rt.redexes.extend(core::mem::take(&mut *stored_redexes));
        }
      }
      // Place the reduced net back into the def map
      *net = host.readback(&mut rt);
    }
    Ok(())
  }
}
