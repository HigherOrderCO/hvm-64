use std::sync::Arc;

use crate::run::{AsDef, Def, LabSet, Net, Port, Tag, Trg, Wire};

fn call_identity(net: &mut Net, port: Port) {
  let (a, b) = net.do_ctr(0, Trg::port(port));
  net.link_trg(a, b);
}

pub const IDENTITY: &Def = const { &Def::new(LabSet::from_bits(&[1]), call_identity) }.upcast();

pub struct LogDef<F: Fn(Wire) + 'static>(pub F);

impl<F: Fn(Wire) + Clone + Send + Sync + 'static> AsDef for LogDef<F> {
  unsafe fn call(slf: *const Def<Self>, net: &mut Net, port: Port) {
    let slf = unsafe { &*slf };
    let (arg, seq) = net.do_ctr(0, Trg::port(port));
    let (wire, port) = net.create_wire();
    net.link_trg_port(
      arg,
      Port::new_ref(Box::leak(Box::new(Def::new(LabSet::ALL, ActiveLogDef {
        logger: Arc::new(Logger { f: slf.data.0.clone(), root: wire, seq }),
        out: port,
      })))),
    );
  }
}

struct Logger<F> {
  f: F,
  root: Wire,
  seq: Trg,
}

impl<F: Fn(Wire)> Logger<F> {
  fn maybe_log(self: Arc<Self>, net: &mut Net) {
    let Some(slf) = Arc::into_inner(self) else { return };
    (slf.f)(slf.root.clone());
    net.link_wire_port(slf.root, Port::ERA);
    net.link_trg_port(slf.seq, Port::new_ref(IDENTITY));
  }
}

struct ActiveLogDef<F> {
  logger: Arc<Logger<F>>,
  out: Port,
}

impl<F: Fn(Wire) + Send + Sync + 'static> AsDef for ActiveLogDef<F> {
  unsafe fn call(slf: *const Def<Self>, net: &mut Net, port: Port) {
    let slf = *Box::from_raw(slf as *mut Def<Self>);
    match port.tag() {
      Tag::Red | Tag::Var => unreachable!(),
      Tag::Ref if port != Port::ERA => {
        let def: *const Def = port.addr().def() as *const _;
        if let Some(other) = Def::downcast_ptr::<Self>(def) {
          let other = *Box::from_raw(other as *mut Def<Self>);
          net.link_port_port(slf.data.out, other.data.out);
          other.data.logger.maybe_log(net);
        } else {
          net.link_port_port(slf.data.out, port);
        }
      }
      Tag::Ref | Tag::Num => net.link_port_port(slf.data.out, port),
      tag @ (Tag::Op2 | Tag::Op1 | Tag::Mat | Tag::Ctr) => {
        let old = port.consume_node();
        let new = net.create_node(tag, old.lab);
        net.link_port_port(slf.data.out, new.p0);
        net.link_wire_port(
          old.p1,
          Port::new_ref(Box::leak(Box::new(Def::new(LabSet::ALL, ActiveLogDef {
            logger: slf.data.logger.clone(),
            out: new.p1,
          })))),
        );
        net.link_wire_port(
          old.p2,
          Port::new_ref(Box::leak(Box::new(Def::new(LabSet::ALL, ActiveLogDef {
            logger: slf.data.logger.clone(),
            out: new.p2,
          })))),
        );
      }
    }
    slf.data.logger.maybe_log(net);
  }
}
