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
    let mut net = Net::new_with_root(heap, Wire(std::ptr::null()));
    net.root = Wire::new(net.alloc());
    net
  }

  pub(super) fn new_with_root(heap: &'h Heap, root: Wire) -> Self {
    Net { linker: Linker::new(heap), tid: 0, tids: 1, trgs: Box::new_uninit_slice(1 << 16), root }
  }

  /// Boots a net from a Ref.
  pub fn boot(&mut self, def: &Def) {
    let def = Port::new_ref(def);
    trace!(self, def);
    self.root.set_target(def);
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

  /// Expands [`Ref`] nodes in the tree connected to `root`.
  #[inline(always)]
  pub fn expand(&mut self) {
    assert!(!M::LAZY);
    fn go<M: Mode>(net: &mut Net<M>, wire: Wire, len: usize, key: usize) {
      trace!(net.tracer, wire);
      let port = wire.load_target();
      trace!(net.tracer, port);
      if port == Port::LOCK {
        return;
      }
      if port.tag() == Ctr {
        let node = port.traverse_node();
        if len >= net.tids || key % 2 == 0 {
          go(net, node.p1, len.saturating_mul(2), key / 2);
        }
        if len >= net.tids || key % 2 == 1 {
          go(net, node.p2, len.saturating_mul(2), key / 2);
        }
      } else if port.tag() == Ref && port != Port::ERA {
        let got = wire.swap_target(Port::LOCK);
        if got != Port::LOCK {
          trace!(net.tracer, port, wire);
          net.call(port, Port::new_var(wire.addr()));
        }
      }
    }
    go(self, self.root.clone(), 1, self.tid);
  }

  // Lazy mode weak head normalizer
  #[inline(always)]
  fn weak_normal(&mut self, mut prev: Port, root: Wire) -> Port {
    assert!(M::LAZY);

    let mut path: Vec<Port> = vec![];

    loop {
      trace!(self.tracer, prev);
      // Load ptrs
      let next = self.get_target_full(prev.clone());
      trace!(self.tracer, next);

      // If next is root, stop.
      if next == Port::new_var(root.addr()) || next == Port::new_var(self.root.addr()) {
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
        } else if next.tag() == Ref && next != Port::ERA {
          self.call(next, prev.clone());
          continue;
        // Otherwise, we're done.
        } else {
          break;
        }
      }

      // If next is an aux port, pass through.
      let main = self.get_header(next.addr().left_half());
      path.push(prev);
      prev = main.this.clone();
    }

    return self.get_target_full(prev);
  }

  pub fn normal_from(&mut self, root: Wire) {
    assert!(M::LAZY);
    let mut visit = vec![Port::new_var(root.addr())];
    while let Some(prev) = visit.pop() {
      trace!(self.tracer, "visit", prev);
      //println!("normal {} | {}", prev.view(), self.rewrites());
      let next = self.weak_normal(prev, root.clone());
      trace!(self.tracer, "got", next);
      if next.is_full_node() {
        visit.push(Port::new_var(next.addr()));
        visit.push(Port::new_var(next.addr().other_half()));
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
        self.expand();
      }
    }
  }
}
