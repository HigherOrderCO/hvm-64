use hvm64_util::prelude::*;

use super::{Addr, Host, Port, Tag, Wire};

use core::ops::RangeFrom;

use hvm64_ast::{Net, Tree};
use hvm64_num::{Num, NumTag};
use hvm64_util::{create_var, maybe_grow};

impl Host {
  /// Creates an ast tree from a wire in a runtime net.
  pub fn readback_tree(&self, wire: &Wire) -> Tree {
    ReadbackState { host: self, vars: Default::default(), var_id: 0 .. }.read_wire(wire.clone())
  }

  /// Creates an ast net from a runtime net.
  ///
  /// Note that vicious circles and disconnected subnets will not be in the
  /// resulting ast net, as it is impossible to read these back from the runtime
  /// net representation. In the case of vicious circles, this may result in
  /// unbound variables.
  pub fn readback(&self, rt_net: &hvm64_runtime::Net) -> Net {
    let mut state = ReadbackState { host: self, vars: Default::default(), var_id: 0 .. };
    let mut net = Net::default();

    net.root = state.read_wire(rt_net.root.clone());
    for (a, b) in rt_net.redexes.iter() {
      net.redexes.push((state.read_port(a.clone(), None), state.read_port(b.clone(), None)))
    }

    net
  }
}

/// See [`Host::readback`].
struct ReadbackState<'a> {
  host: &'a Host,
  vars: Map<Addr, usize>,
  var_id: RangeFrom<usize>,
}

impl<'a> ReadbackState<'a> {
  /// Reads a tree out from a given `wire`.
  fn read_wire(&mut self, wire: Wire) -> Tree {
    let port = wire.load_target();
    self.read_port(port, Some(wire))
  }

  /// Reads a tree out from a given `port`. If this is a var port, the
  /// `wire` this port was reached from must be supplied to key into the
  /// `vars` map.
  fn read_port(&mut self, port: Port, wire: Option<Wire>) -> Tree {
    maybe_grow(move || match port.tag() {
      Tag::Var | Tag::Red => {
        // todo: resolve redirects
        let key = wire.unwrap().addr().min(port.addr());
        Tree::Var(create_var(match self.vars.entry(key) {
          Entry::Occupied(e) => e.remove(),
          Entry::Vacant(e) => *e.insert(self.var_id.next().unwrap()),
        }))
      }
      Tag::Ref if port == Port::ERA => Tree::Era,
      Tag::Ref => Tree::Ref(self.host.back[&port.addr()].clone()),
      Tag::Num => Tree::Num(port.num()),
      Tag::Op => {
        let op = port.op();
        let node = port.traverse_node();
        let node = Tree::Op { rhs: Box::new(self.read_wire(node.p1)), out: Box::new(self.read_wire(node.p2)) };
        if op == NumTag::Sym {
          node
        } else {
          Tree::Op { rhs: Box::new(Tree::Num(Num::new_sym(op))), out: Box::new(node) }
        }
      }
      Tag::Ctr => {
        let node = port.traverse_node();
        Tree::Ctr { lab: node.lab, p1: Box::new(self.read_wire(node.p1)), p2: Box::new(self.read_wire(node.p2)) }
      }
      Tag::Switch => {
        let node = port.traverse_node();
        let arms = self.read_wire(node.p1);
        let out = self.read_wire(node.p2);
        Tree::Switch { arms: Box::new(arms), out: Box::new(out) }
      }
    })
  }
}
