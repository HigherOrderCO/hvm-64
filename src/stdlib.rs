use std::{
  marker::PhantomData,
  sync::{Arc, Mutex},
};

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

impl<F: Fn(Wire) + Clone + Send + Sync + 'static> LogDef<F> {
  /// # SAFETY
  /// The caller must ensure that the returned value lives at least as long as
  /// the port where it is used.
  pub unsafe fn new(f: F) -> DefRef {
    HostedDef::new_hosted(LabSet::ALL, LogDef(f))
  }
}

pub struct LogDef<F>(F);

impl<F: Fn(Wire) + Clone + Send + Sync + 'static> AsHostedDef for LogDef<F> {
  fn call<M: Mode>(def: &Def<Self>, net: &mut Net<M>, port: Port) {
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
    Port::new_ref(Box::leak(BoxDef::new_boxed(LabSet::ALL, ActiveLogDef { logger, out })))
  }
}

impl<F: Fn(Wire) + Send + Sync + 'static> AsBoxDef for ActiveLogDef<F> {
  fn call<M: Mode>(def: Box<Def<Self>>, net: &mut Net<M>, port: Port) {
    match port.tag() {
      Tag::Red => {
        unreachable!()
      }
      Tag::Ref if port != Port::ERA => {
        if let Some(other) = unsafe { BoxDef::<Self>::try_downcast_box(port.clone()) } {
          net.link_port_port(def.data.out, other.data.0.out);
          other.data.0.logger.maybe_log(net);
        } else {
          net.link_port_port(def.data.out, port);
        }
      }
      Tag::Ref | Tag::Num | Tag::Var => net.link_port_port(def.data.out, port),
      tag @ (Tag::Op | Tag::Mat | Tag::Ctr) => {
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
  host.lock().unwrap().insert_def("HVM.log", unsafe {
    crate::stdlib::LogDef::new({
      let host = Arc::downgrade(&host);
      move |wire| {
        println!("{}", host.upgrade().unwrap().lock().unwrap().readback_tree(&wire));
      }
    })
  });
  host.lock().unwrap().insert_def("HVM.black_box", DefRef::Static(unsafe { &*IDENTITY }));
  host.lock().unwrap().insert_book(&book);
  host
}

#[repr(transparent)]
pub struct BoxDef<T: AsBoxDef>(pub T, PhantomData<()>);

impl<T: AsBoxDef> BoxDef<T> {
  pub fn new_boxed(labs: LabSet, data: T) -> Box<Def<Self>> {
    Box::new(Def::new(labs, BoxDef(data, PhantomData)))
  }
  /// SAFETY: if port is a ref, it must be a valid pointer.
  pub unsafe fn try_downcast_box(port: Port) -> Option<Box<Def<Self>>> {
    if port.is(Tag::Ref) {
      if let Some(port) = unsafe { Def::downcast_ptr::<Self>(port.addr().0 as *const _) } {
        Some(unsafe { Box::from_raw(port as *mut Def<Self>) })
      } else {
        None
      }
    } else {
      None
    }
  }
}

pub trait AsBoxDef: Send + Sync + 'static {
  fn call<M: Mode>(slf: Box<Def<Self>>, net: &mut Net<M>, port: Port);
}

impl<T: AsBoxDef> AsDef for BoxDef<T> {
  unsafe fn call<M: Mode>(slf: *const Def<Self>, net: &mut Net<M>, port: Port) {
    T::call(Box::from_raw(slf as *mut _), net, port)
  }
}

#[repr(transparent)]
pub struct ArcDef<T: AsArcDef>(pub T, PhantomData<()>);

impl<T: AsArcDef> ArcDef<T> {
  pub fn new_arc(labs: LabSet, data: T) -> Arc<Def<Self>> {
    Arc::new(Def::new(labs, ArcDef(data, PhantomData)))
  }
  pub fn to_port(slf: Arc<Def<T>>) -> Port {
    unsafe { Port::new_ref(Arc::into_raw(slf).as_ref().unwrap()) }
  }
  /// SAFETY: if port is a ref, it must be a valid pointer.
  pub unsafe fn try_downcast_arc(port: Port) -> Option<Arc<Def<Self>>> {
    if port.is(Tag::Ref) && port != Port::ERA {
      if let Some(port) = unsafe { Def::downcast_ptr::<Self>(port.addr().0 as *const _) } {
        Some(unsafe { Arc::from_raw(port as *mut Def<Self>) })
      } else {
        None
      }
    } else {
      None
    }
  }
}

pub trait AsArcDef: Send + Sync + 'static {
  fn call<M: Mode>(slf: Arc<Def<Self>>, net: &mut Net<M>, port: Port);
}

impl<T: AsArcDef> AsDef for ArcDef<T> {
  unsafe fn call<M: Mode>(slf: *const Def<Self>, net: &mut Net<M>, port: Port) {
    T::call(Arc::from_raw(slf as *mut _), net, port);
  }
}

#[repr(transparent)]
pub struct HostedDef<T: AsHostedDef>(pub T, PhantomData<()>);

impl<T: AsHostedDef> HostedDef<T> {
  pub unsafe fn new_hosted(labs: LabSet, data: T) -> DefRef {
    DefRef::Owned(Box::new(Def::new(labs, HostedDef(data, PhantomData))))
  }
}

pub trait AsHostedDef: Send + Sync + 'static {
  fn call<M: Mode>(slf: &Def<Self>, net: &mut Net<M>, port: Port);
}

impl<T: AsHostedDef> AsDef for HostedDef<T> {
  unsafe fn call<M: Mode>(slf: *const Def<Self>, net: &mut Net<M>, port: Port) {
    T::call((slf as *const Def<T>).as_ref().unwrap(), net, port)
  }
}

use crate::ast::Tree;

pub struct UniqueTreePtr(*mut Tree);
unsafe impl Send for UniqueTreePtr {}
unsafe impl Sync for UniqueTreePtr {}

pub struct ReadbackDef<F: FnOnce() + Send + Sync + 'static> {
  root: Arc<F>,
  host: Arc<Mutex<Host>>,
  tree: UniqueTreePtr,
  out: Port,
}

impl<F: FnOnce() + Send + Sync + 'static> ReadbackDef<F> {
  fn maybe_finish(root: Arc<F>) {
    let Some(root) = Arc::into_inner(root) else { return };
    (root)()
  }
  fn with(&self, tree: *mut Tree, out: Port) -> Port {
    Port::new_ref(&*BoxDef::new_boxed(LabSet::ALL, Self {
      tree: UniqueTreePtr(tree),
      out,
      root: self.root.clone(),
      host: self.host.clone(),
    }))
  }
}

impl<F: FnOnce() + Send + Sync + 'static> AsBoxDef for ReadbackDef<F> {
  fn call<M: Mode>(def: Box<Def<Self>>, net: &mut Net<M>, port: Port) {
    match port.tag() {
      Tag::Red => {
        unreachable!()
      }
      Tag::Ref if port != Port::ERA => {
        if let Some(other) = unsafe { BoxDef::<Self>::try_downcast_box(port.clone()) } {
          net.link_port_port(def.data.out, other.data.0.out);
          Self::maybe_finish(other.data.0.root);
        } else {
          net.link_port_port(def.data.out, port);
        }
      }
      Tag::Ref => {
        unsafe { *(def.data.tree.0) = Tree::Era };
        net.link_port_port(def.data.out, port);
      }
      Tag::Num => {
        unsafe { *(def.data.tree.0) = Tree::Num { val: port.num() } };
        net.link_port_port(def.data.out, port)
      }
      Tag::Var => net.link_port_port(def.data.out, port),
      tag @ (Tag::Op | Tag::Mat | Tag::Ctr) => {
        let old = port.clone().consume_node();
        let new = net.create_node(tag, old.lab);
        let (lhs, rhs): (*mut Tree, *mut Tree) = match tag {
          Tag::Op => {
            unsafe {
              *(def.data.tree.0) = Tree::Op { op: port.op(), rhs: Box::new(Tree::Era), out: Box::new(Tree::Era) }
            };
            let Tree::Op { rhs, out, .. } = (unsafe { &mut *(def.data.tree.0) }) else { unreachable!() };
            (out.as_mut(), rhs.as_mut())
          }
          Tag::Mat => {
            unsafe { *(def.data.tree.0) = Tree::Mat { sel: Box::new(Tree::Era), ret: Box::new(Tree::Era) } };
            let Tree::Mat { sel, ret } = (unsafe { &mut *(def.data.tree.0) }) else { unreachable!() };
            (sel.as_mut(), ret.as_mut())
          }
          Tag::Ctr => {
            unsafe {
              *(def.data.tree.0) = Tree::Ctr { lab: port.lab(), lft: Box::new(Tree::Era), rgt: Box::new(Tree::Era) }
            };
            let Tree::Ctr { lft, rgt, .. } = (unsafe { &mut *(def.data.tree.0) }) else { unreachable!() };
            (lft.as_mut(), rgt.as_mut())
          }
          _ => unreachable!(),
        };
        net.link_port_port(def.data.out.clone(), new.p0);
        net.link_wire_port(old.p1, def.data.with(rhs, new.p1));
        net.link_wire_port(old.p2, def.data.with(lhs, new.p2));
      }
    }
    Self::maybe_finish(def.data.root);
  }
}

// public api:
// readback: Wire -> FnOnce(Tree) -> ()
// readback_and_wait: Net -> Wire -> Tree
