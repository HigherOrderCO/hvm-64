use super::{Heap, Lazy, Mode, Net, Strict};

/// A [`Net`] whose mode is determined dynamically, at runtime.
///
/// Use [`dispatch_dyn_net!`] to wrap operations on the inner net.
pub type DynNet<'a> = DynNetInner<Net<'a, Lazy>, Net<'a, Strict>>;

/// A mutable reference to a [`Net`] whose mode is determined dynamically, at
/// runtime.
///
/// Use [`dispatch_dyn_net!`] to wrap operations on the inner net.
pub type DynNetMut<'r, 'h> = DynNetInner<&'r mut Net<'h, Lazy>, &'r mut Net<'h, Strict>>;

pub enum DynNetInner<L, S> {
  Lazy(L),
  Strict(S),
}

impl<'h> DynNet<'h> {
  pub fn new(heap: &'h Heap, lazy: bool) -> Self {
    if lazy { DynNet::Lazy(Net::new(heap)) } else { DynNet::Strict(Net::new(heap)) }
  }
}

impl<'r, 'h, M: Mode> From<&'r mut Net<'h, M>> for DynNetMut<'r, 'h> {
  fn from(value: &'r mut Net<'h, M>) -> Self {
    match value.match_laziness_mut() {
      Ok(net) => DynNetMut::Lazy(net),
      Err(net) => DynNetMut::Strict(net),
    }
  }
}
#[macro_export]
macro_rules! dispatch_dyn_net {
  ($pat:pat = $expr:expr => $body:expr) => {
    match $expr {
      $crate::run::DynNetInner::Lazy($pat) => $body,
      $crate::run::DynNetInner::Strict($pat) => $body,
    }
  };
  ($net:ident => $body:expr) => {
    dispatch_dyn_net! { $net = $net => $body }
  };
  (mut $net:ident => $body:expr) => {
    dispatch_dyn_net! { mut $net = $net => $body }
  };
  (&$net:ident => $body:expr) => {
    dispatch_dyn_net! { $net = &$net => $body }
  };
  (&mut $net:ident => $body:expr) => {
    dispatch_dyn_net! { $net = &mut $net => $body }
  };
}
