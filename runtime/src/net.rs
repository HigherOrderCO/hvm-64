use super::*;

use hvm64_util::new_uninit_slice;
use mem::MaybeUninit;

/// An interaction combinator net.
pub struct Net<'a> {
  pub(super) linker: Linker<'a>,
  pub tid: usize,  // thread id
  pub tids: usize, // thread count
  pub trgs: Box<[MaybeUninit<Trg>]>,
  pub root: Wire,
}

deref_to!({<'a, >} Net<'a> => self.linker: Linker<'a>);

impl<'h> Net<'h> {
  /// Creates an empty net with a given heap.
  pub fn new(heap: &'h Heap) -> Self {
    let mut net = Net::new_with_root(heap, Wire(ptr::null()));
    net.root = Wire::new(net.alloc());
    net
  }

  pub(super) fn new_with_root(heap: &'h Heap, root: Wire) -> Self {
    Net { linker: Linker::new(heap), tid: 0, tids: 1, trgs: new_uninit_slice(1 << 16), root }
  }

  /// Boots a net from a Def.
  pub fn boot(&mut self, def: &Def) {
    self.call(Port::new_ref(def), self.root.as_var());
  }
}

impl<'a> Net<'a> {
  /// Reduces at most `limit` redexes.
  ///
  /// If normalized, returns `Some(num_redexes)`.
  /// If stopped because the limit was reached, returns `None`.
  #[inline(always)]
  pub fn reduce(&mut self, limit: usize) -> Option<usize> {
    let mut count = 0;

    while let Some((a, b)) = self.redexes.pop() {
      self.interact(a, b);
      count += 1;
      if count >= limit {
        return None;
      }
    }
    Some(count)
  }

  /// Reduces a net to normal form.
  pub fn normal(&mut self) {
    self.expand();
    while !self.redexes.is_empty() {
      self.reduce(usize::MAX);
    }
  }
}

impl<'h> Net<'h> {
  /// Expands [`Tag::Ref`] nodes in the tree connected to `root`.
  pub fn expand(&mut self) {
    let (new_root, out_port) = self.create_wire();
    let old_root = mem::replace(&mut self.root, new_root);
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
  unsafe fn call(def: *const Def<Self>, net: &mut Net, port: Port) {
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
      tag @ (Tag::Op | Tag::Switch | Tag::Ctr) => {
        let old = port.consume_node();
        let new = net.create_node(tag, old.lab);
        net.link_port_port(def.data.out, new.p0);
        net.link_wire_port(old.p1, ExpandDef::new(new.p1));
        net.link_wire_port(old.p2, ExpandDef::new(new.p2));
      }
    }
  }
}
