use crate::prelude::*;

use alloc::sync::Arc;
use core::{
  marker::PhantomData,
  sync::atomic::{AtomicUsize, Ordering},
};
use parking_lot::Mutex;

use crate::{
  dispatch_dyn_net,
  host::{DefRef, Host},
  run::{AsDef, Def, DynNetMut, LabSet, Mode, Net, Port, Tag, Trg},
  util::create_var,
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

impl<F: Fn(Tree) + Clone + Send + Sync + 'static> LogDef<F> {
  /// # SAFETY
  /// The caller must ensure that the returned value lives at least as long as
  /// the port where it is used.
  pub unsafe fn new(host: Arc<Mutex<Host>>, f: F) -> DefRef {
    HostedDef::new_hosted(LabSet::ALL, LogDef(host, f))
  }
}

pub struct LogDef<F>(Arc<Mutex<Host>>, F);

impl<F: Fn(Tree) + Clone + Send + Sync + 'static> AsHostedDef for LogDef<F> {
  fn call<M: Mode>(def: &Def<Self>, net: &mut Net<M>, port: Port) {
    let (arg, seq) = net.do_ctr(0, Trg::port(port));
    let seq = net.wire_to_trg(seq);
    // SAFETY: the function inside `readback` won't live longer than
    // the net, and thus won't live longer than the host, where the
    // `&Def<Self>` points to
    let def: &'static Def<Self> = unsafe { mem::transmute(def) };
    readback(net, def.data.0.clone(), arg, |net, tree| {
      (def.data.1)(tree);
      dispatch_dyn_net!(net => {
        net.link_wire_port(seq, Port::new_ref(unsafe { &*IDENTITY }));
      });
    });
  }
}

/// Create a `Host` from a `Book`, including `hvm-core`'s built-in definitions
#[cfg(feature = "std")]
#[allow(clippy::absolute_paths)]
pub fn create_host(book: &crate::ast::Book) -> Arc<Mutex<Host>> {
  let host: Arc<Mutex<Host>> = Default::default();

  insert_stdlib(host.clone());
  host.lock().insert_book(book);

  host
}

/// Insert `hvm-core`'s built-in definitions into a host.
#[cfg(feature = "std")]
#[allow(clippy::absolute_paths)]
pub fn insert_stdlib(host: Arc<Mutex<Host>>) {
  host.lock().insert_def("HVM.log", unsafe {
    crate::stdlib::LogDef::new(host.clone(), {
      move |tree| {
        println!("{}", tree);
      }
    })
  });
  host.lock().insert_def("HVM.black_box", DefRef::Static(unsafe { &*IDENTITY }));
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
      unsafe { Def::downcast_ptr::<Self>(port.addr().0 as *const _) }
        .map(|port| unsafe { Box::from_raw(port as *mut Def<Self>) })
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
  pub fn new_arc_port(labs: LabSet, data: T) -> Port {
    unsafe { Port::new_ref(Arc::into_raw(Arc::new(Def::new(labs, ArcDef(data, PhantomData)))).as_ref().unwrap()) }
  }
  pub fn to_port(slf: Arc<Def<T>>) -> Port {
    unsafe { Port::new_ref(Arc::into_raw(slf).as_ref().unwrap()) }
  }
  /// SAFETY: if port is a ref, it must be a valid pointer.
  pub unsafe fn try_downcast_arc(port: Port) -> Option<Arc<Def<Self>>> {
    if port.is(Tag::Ref) && port != Port::ERA {
      unsafe { Def::downcast_ptr::<Self>(port.addr().0 as *const _) }
        .map(|port| unsafe { Arc::from_raw(port as *mut Def<Self>) })
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
  pub unsafe fn new(labs: LabSet) -> DefRef
  where
    T: Default,
  {
    Self::new_hosted(labs, T::default())
  }

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

#[derive(Copy, Clone)]
pub struct UniqueTreePtr(*mut Tree);
unsafe impl Send for UniqueTreePtr {}
unsafe impl Sync for UniqueTreePtr {}

impl UniqueTreePtr {
  unsafe fn to_box(self) -> Box<Tree> {
    Box::from_raw(self.0)
  }
}

pub struct ReadbackDef<F: FnOnce(DynNetMut) + Send + Sync + 'static> {
  root: Arc<F>,
  host: Arc<Mutex<Host>>,
  var_idx: Arc<AtomicUsize>,
  tree: UniqueTreePtr,
}

impl<F: FnOnce(DynNetMut) + Send + Sync + 'static> ReadbackDef<F> {
  fn maybe_finish(net: DynNetMut<'_, '_>, root: Arc<F>) {
    let Some(root) = Arc::into_inner(root) else { return };
    (root)(net)
  }
  fn with(&self, tree: *mut Tree) -> Port {
    Port::new_ref(Box::leak(BoxDef::new_boxed(LabSet::ALL, Self {
      tree: UniqueTreePtr(tree),
      root: self.root.clone(),
      var_idx: self.var_idx.clone(),
      host: self.host.clone(),
    })))
  }
}

impl<F: FnOnce(DynNetMut) + Send + Sync + 'static> AsBoxDef for ReadbackDef<F> {
  fn call<M: Mode>(def: Box<Def<Self>>, net: &mut Net<M>, port: Port) {
    match port.tag() {
      Tag::Var | Tag::Red => {
        unreachable!()
      }
      Tag::Ref if port != Port::ERA => {
        if let Some(other) = unsafe { BoxDef::<Self>::try_downcast_box(port.clone()) } {
          let var = def.data.var_idx.fetch_add(1, Ordering::AcqRel);
          let var = Tree::Var { nam: create_var(var) };
          unsafe {
            (*def.data.tree.0) = var.clone();
            (*other.data.0.tree.0) = var;
          }
          Self::maybe_finish(DynNetMut::from(&mut *net), other.data.0.root);
        } else if let Some(back) = def.data.host.lock().back.get(&port.addr()) {
          unsafe { *(def.data.tree.0) = Tree::Ref { nam: back.clone() } };
        } else {
          unsafe { *(def.data.tree.0) = Tree::Era };
        }
      }
      Tag::Ref => {
        unsafe { *(def.data.tree.0) = Tree::Era };
      }
      Tag::Int => {
        unsafe { *(def.data.tree.0) = Tree::Int { val: port.int() } };
      }
      Tag::F32 => {
        unsafe { *(def.data.tree.0) = Tree::F32 { val: port.float().into() } };
      }
      Tag::Mat => {
        unsafe {
          *(def.data.tree.0) =
            Tree::Mat { zero: Box::new(Tree::Era), succ: Box::new(Tree::Era), out: Box::new(Tree::Era) }
        };
        let Tree::Mat { zero, succ, out } = (unsafe { &mut *(def.data.tree.0) }) else { unreachable!() };
        let old = port.clone().consume_node();
        let old_sel = old.p1.load_target().consume_node();
        net.link_wire_port(old.p2, def.data.with(out.as_mut()));
        net.link_wire_port(old_sel.p1, def.data.with(zero.as_mut()));
        net.link_wire_port(old_sel.p2, def.data.with(succ.as_mut()));
      }
      tag @ (Tag::Op | Tag::Ctr) => {
        let old = port.clone().consume_node();
        let (lhs, rhs): (*mut Tree, *mut Tree) = match tag {
          Tag::Op => {
            unsafe {
              *(def.data.tree.0) = Tree::Op { op: port.op(), rhs: Box::new(Tree::Era), out: Box::new(Tree::Era) }
            };
            let Tree::Op { rhs, out, .. } = (unsafe { &mut *(def.data.tree.0) }) else { unreachable!() };
            (rhs.as_mut(), out.as_mut())
          }
          Tag::Ctr => {
            unsafe { *(def.data.tree.0) = Tree::Ctr { lab: port.lab(), ports: vec![Tree::Era, Tree::Era] } };
            let Tree::Ctr { ports, .. } = (unsafe { &mut *(def.data.tree.0) }) else { unreachable!() };
            (&mut ports[0], &mut ports[1])
          }
          _ => unreachable!(),
        };
        net.link_wire_port(old.p1, def.data.with(lhs));
        net.link_wire_port(old.p2, def.data.with(rhs));
      }
    }
    Self::maybe_finish(DynNetMut::from(net), def.data.root);
  }
}

pub fn readback<M: Mode>(
  net: &mut Net<M>,
  host: Arc<Mutex<Host>>,
  from: Trg,
  f: impl FnOnce(DynNetMut, Tree) + Send + Sync + 'static,
) {
  let root = UniqueTreePtr(Box::leak(Box::default()));

  if M::LAZY {
    let from = net.wire_to_trg(from);
    net.normal_from(from.clone());
    let tree = host.lock().readback_tree(&from);
    net.link_wire_port(from, Port::ERA);
    f(DynNetMut::from(net), tree);
  } else {
    let closure: Box<dyn FnOnce(DynNetMut) + Send + Sync + 'static> = Box::new(move |net| {
      let root = unsafe { root.to_box() };
      f(net, *root);
    });

    net.link_trg_port(
      from,
      Port::new_ref(Box::leak(BoxDef::new_boxed(LabSet::ALL, ReadbackDef {
        root: Arc::new(closure),
        host,
        tree: root,
        var_idx: Arc::new(AtomicUsize::from(0)),
      }))),
    );
  }
}
