use super::*;

/// A wire in the interaction net.
///
/// More accurately, this is a *directed view* of a wire. If ports `a` and `b`
/// are connected, then the wire leaving `a` and the wire leaving `b` are the
/// same wire, but viewed from opposite directions.
///
/// This is represented by a pointer to an `AtomicU64` storing the *target* of
/// the wire -- the port on the other side. (The target of the wire leaving `a`
/// is `b`.)
///
/// Changes to the target are handled by the linker.
#[derive(Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[must_use]
pub struct Wire(pub *const AtomicU64);

impl fmt::Debug for Wire {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    write!(f, "{:012x?}", self.0 as usize)
  }
}

unsafe impl Send for Wire {}
unsafe impl Sync for Wire {}

impl Wire {
  #[inline(always)]
  pub fn addr(&self) -> Addr {
    Addr(self.0 as _)
  }

  #[inline(always)]
  pub fn new(addr: Addr) -> Wire {
    Wire(addr.0 as _)
  }

  #[inline(always)]
  fn target<'a>(&self) -> &'a AtomicU64 {
    unsafe { &*self.0 }
  }

  #[inline(always)]
  pub fn load_target(&self) -> Port {
    let port = Port(self.target().load(Relaxed));
    port
  }

  #[inline(always)]
  pub fn set_target(&self, port: Port) {
    self.target().store(port.0, Relaxed);
  }

  #[inline(always)]
  pub fn cas_target(&self, expected: Port, value: Port) -> Result<Port, Port> {
    self.target().compare_exchange(expected.0, value.0, Relaxed, Relaxed).map(Port).map_err(Port)
  }

  #[inline(always)]
  pub fn swap_target(&self, value: Port) -> Port {
    let port = Port(self.target().swap(value.0, Relaxed));
    port
  }

  // Takes a pointer's target.
  #[inline(always)]
  pub fn lock_target(&self) -> Port {
    loop {
      let got = self.swap_target(Port::LOCK);
      if got != Port::LOCK {
        return got;
      }
      spin_loop();
    }
  }

  #[inline(always)]
  pub(super) fn as_var(&self) -> Port {
    Port::new_var(self.addr())
  }
}
