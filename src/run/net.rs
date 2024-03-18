use std::mem::MaybeUninit;

use super::*;

/// An interaction combinator net.
pub struct Net<'a, M: Mode> {
  linker: Linker<'a, M>,
  pub tid: usize,  // thread id
  pub tids: usize, // thread count
  pub trgs: Box<[MaybeUninit<Trg>]>,
  pub root: Wire,
}

deref!({<'a, M: Mode>} Net<'a, M> => self.linker: Linker<'a, M>);

impl<'h, M: Mode> Net<'h, M> {
  /// Creates an empty net with a given heap.
  pub fn new(heap: &'h Heap) -> Self {
    let mut net = Net::new_with_root(heap, Wire(0));
    net.root = Wire::new(Align2, net.alloc(Align2));
    net
  }

  pub(super) fn new_with_root(heap: &'h Heap, root: Wire) -> Self {
    Net { linker: Linker::new(heap), tid: 0, tids: 1, trgs: Box::new_uninit_slice(1 << 16), root }
  }

  /// Boots a net from a Def.
  pub fn boot(&mut self, def: &Def) {
    self.call(Port::new_ref(def), self.root.as_var());
  }
}

impl<'a, M: Mode> Net<'a, M> {
  /// Reduces at most `limit` redexes.
  #[inline(always)]
  pub fn reduce(&mut self, limit: usize) -> usize {
    assert!(!M::LAZY);
    let mut count = 0;
    while let Some((a, b)) = self.redexes.pop() {
      self.interact(a, b);
      count += 1;
      if count >= limit {
        break;
      }
    }
    count
  }

  // Lazy mode weak head normalizer
  #[inline(always)]
  fn weak_normal(&mut self, mut prev: Port, root: Wire) -> Port {
    assert!(M::LAZY);

    let mut path: Vec<Port> = vec![];

    loop {
      trace!(self, prev);
      // Load ptrs
      let next = self.get_target_full(prev.clone());
      trace!(self, next);

      // If next is root, stop.
      if next == root.as_var() || next == self.root.as_var() {
        break;
      }

      // If next is a main port...
      if next.is_principal() {
        // If prev is a main port, reduce the active pair.
        if prev.is_principal() {
          self.interact(next, prev.clone());
          prev = path.pop().unwrap();
          continue;
        // Otherwise, if it is a ref, expand it.
        } else if next.is(Tag::Ref) && next != Port::ERA {
          self.call(next, prev.clone());
          continue;
        // Otherwise, we're done.
        } else {
          break;
        }
      }

      // If next is an aux port, pass through.
      let main = self.get_header(next.addr().floor(next.alloc_align()));
      path.push(prev);
      prev = main.this.clone();
    }

    return self.get_target_full(prev);
  }

  pub fn normal_from(&mut self, root: Wire) {
    assert!(M::LAZY);
    let mut visit = vec![root.as_var()];
    while let Some(prev) = visit.pop() {
      trace!(self, "visit", prev);
      //println!("normal {} | {}", prev.view(), self.rewrites());
      let next = self.weak_normal(prev, root.clone());
      trace!(self, "got", next);
      if next.is_full_node() {
        todo!();
        // visit.push(Port::new_var( next.addr()));
        // visit.push(Port::new_var( next.addr().other_half()));
      }
    }
  }

  /// Reduces a net to normal form.
  pub fn normal(&mut self) {
    if M::LAZY {
      self.normal_from(self.root.clone());
    } else {
      self.expand();
      while !self.redexes.is_empty() {
        self.reduce(usize::MAX);
      }
    }
  }
}

impl<'h, M: Mode> Net<'h, M> {
  /// Expands [`Tag::Ref`] nodes in the tree connected to `root`.
  pub fn expand(&mut self) {
    assert!(!M::LAZY);
    let (new_root, out_port) = self.create_wire();
    let old_root = std::mem::replace(&mut self.root, new_root);
    self.link_wire_port(old_root, ExpandDef::new(out_port));
  }
}

struct ExpandDef {
  out: Port,
}

impl ExpandDef {
  fn new(out: Port) -> Port {
    Port::new_ref(Box::leak(Box::new(Def::new(LabSet::ALL, ExpandDef { out }))))
  }
}

impl AsDef for ExpandDef {
  unsafe fn call<M: Mode>(def: *const Def<Self>, net: &mut Net<M>, port: Port) {
    if port.tag() == Tag::Ref && port != Port::ERA {
      let other: *const Def = port.addr().def() as *const _;
      if let Some(other) = Def::downcast_ptr::<Self>(other) {
        let def = *Box::from_raw(def as *mut Def<Self>);
        let other = *Box::from_raw(other as *mut Def<Self>);
        return net.link_port_port(def.data.out, other.data.out);
      } else {
        return net.call(port, Port::new_ref(Def::upcast(unsafe { &*def })));
      }
    }
    let def = *Box::from_raw(def as *mut Def<Self>);
    match port.tag() {
      Tag::Red => {
        unreachable!()
      }
      Tag::Ref | Tag::Num | Tag::Var => net.link_port_port(def.data.out, port),
      tag @ (Tag::Op | Tag::Mat | Tag::Ctr) => {
        let old = port.consume_node();
        let new = net.create_node(tag, old.lab);
        net.link_port_port(def.data.out, new.p0);
        net.link_wire_port(old.p1, ExpandDef::new(new.p1));
        net.link_wire_port(old.p2, ExpandDef::new(new.p2));
      }
    }
  }
}
