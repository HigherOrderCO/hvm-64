use std::sync::{Arc, Mutex};

use crate::{
  ast::Book,
  host::{DefRef, Host},
  run::{AsDef, Def, LabSet, Mode, Net, Port, Tag, Trg, Wire},
};

/// `@IDENTITY = (x x)`
pub const IDENTITY: *const Def = const { &Def::new(LabSet::from_bits(&[1]), (call_identity, call_identity)) }.upcast();

fn call_identity<M: Mode>(net: &mut Net<M>, port: Port) {
  let (a, b) = net.do_ctr(0, Trg::port(port));
  net.link_trg(a, b);
}

/// The definition of `HVM.log`, parameterized by the readback function.
///
/// `@HVM.log ~ (arg out)` waits for `arg` to be normalized, prints it using the
/// supplied readback function, and then reduces to `arg ~ * & out ~ @IDENTITY`.
///
/// The output can therefore either be erased, or used to sequence other
/// operations after the log.
pub struct LogDef<F>(F);

impl<F: Fn(Wire) + Clone + Send + Sync + 'static> LogDef<F> {
  pub fn new(f: F) -> Def<Self> {
    Def::new(LabSet::ALL, LogDef(f))
  }
}

impl<F: Fn(Wire) + Clone + Send + Sync + 'static> AsDef for LogDef<F> {
  unsafe fn call<M: Mode>(def: *const Def<Self>, net: &mut Net<M>, port: Port) {
    let def = unsafe { &*def };
    let (arg, seq) = net.do_ctr(0, Trg::port(port));
    let (wire, port) = net.create_wire();
    if M::LAZY {
      net.link_trg_port(arg, port);
      net.link_trg_port(seq, Port::new_ref(unsafe { &*IDENTITY }));
      net.normal_from(wire.clone());
      def.data.0(wire.clone());
      net.link_wire_port(wire, Port::ERA);
    } else {
      let logger = Arc::new(Logger { f: def.data.0.clone(), root: wire, seq });
      net.link_trg_port(arg, ActiveLogDef::new(logger, port));
    }
  }
}

struct Logger<F> {
  f: F,
  root: Wire,
  seq: Trg,
}

impl<F: Fn(Wire)> Logger<F> {
  fn maybe_log<M: Mode>(self: Arc<Self>, net: &mut Net<M>) {
    let Some(slf) = Arc::into_inner(self) else { return };
    (slf.f)(slf.root.clone());
    net.link_wire_port(slf.root, Port::ERA);
    net.link_trg_port(slf.seq, Port::new_ref(unsafe { &*IDENTITY }));
  }
}

struct ActiveLogDef<F> {
  logger: Arc<Logger<F>>,
  out: Port,
}

impl<F: Fn(Wire) + Send + Sync + 'static> ActiveLogDef<F> {
  fn new(logger: Arc<Logger<F>>, out: Port) -> Port {
    Port::new_ref(Box::leak(Box::new(Def::new(LabSet::ALL, ActiveLogDef { logger, out }))))
  }
}

impl<F: Fn(Wire) + Send + Sync + 'static> AsDef for ActiveLogDef<F> {
  unsafe fn call<M: Mode>(def: *const Def<Self>, net: &mut Net<M>, port: Port) {
    let def = *Box::from_raw(def as *mut Def<Self>);
    match port.tag() {
      Tag::Red => {
        unreachable!()
      }
      Tag::Ref if port != Port::ERA => {
        let other: *const Def = port.addr().def() as *const _;
        if let Some(other) = Def::downcast_ptr::<Self>(other) {
          let other = *Box::from_raw(other as *mut Def<Self>);
          net.link_port_port(def.data.out, other.data.out);
          other.data.logger.maybe_log(net);
        } else {
          net.link_port_port(def.data.out, port);
        }
      }
      Tag::Ref | Tag::Num | Tag::Var => net.link_port_port(def.data.out, port),
      tag @ (Tag::Op2 | Tag::Op1 | Tag::Mat | Tag::Ctr) => {
        let old = port.consume_node();
        let new = net.create_node(tag, old.lab);
        net.link_port_port(def.data.out, new.p0);
        net.link_wire_port(old.p1, ActiveLogDef::new(def.data.logger.clone(), new.p1));
        net.link_wire_port(old.p2, ActiveLogDef::new(def.data.logger.clone(), new.p2));
      }
    }
    def.data.logger.maybe_log(net);
  }
}

/// Create a `Host` from a `Book`, including `hvm-core`'s built-in definitions
pub fn create_host(book: &Book) -> Arc<Mutex<Host>> {
  let host = Arc::new(Mutex::new(Host::default()));
  host.lock().unwrap().insert_def(
    "HVM.log",
    DefRef::Owned(Box::new(crate::stdlib::LogDef::new({
      let host = Arc::downgrade(&host);
      move |wire| {
        println!("{}", host.upgrade().unwrap().lock().unwrap().readback_tree(&wire));
      }
    }))),
  );
  host.lock().unwrap().insert_book(&book);
  host
}
