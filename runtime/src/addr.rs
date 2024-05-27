use super::*;

/// A memory address to be used in a [`Port`] or a [`Wire`].
///
/// The bottom three bits must be zero; i.e. this address must be at least
/// 8-byte-aligned.
///
/// Additionally, all bits other than the lowest 48 must be zero. On a 32-bit
/// system, this has no effect, but on a 64-bit system, this means that the top
/// 16 bits much be zero.
#[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[must_use]
pub struct Addr(pub usize);

impl fmt::Debug for Addr {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    write!(f, "{:012x?}", self.0)
  }
}

impl Addr {
  pub const NULL: Addr = Addr(0);

  /// Casts this address into an `&AtomicU64`, which may or may not be valid.
  #[inline(always)]
  pub fn val<'a>(&self) -> &'a AtomicU64 {
    unsafe { &*(self.0 as *const _) }
  }

  /// Casts this address into an `&Def`, which may or may not be valid.
  #[inline(always)]
  pub fn def<'a>(&self) -> &'a Def {
    unsafe { &*(self.0 as *const _) }
  }

  const HALF_MASK: usize = 0b1000;

  /// Given an address to one word of a two-word allocation, returns the address
  /// of the first word of that allocation.
  #[inline(always)]
  pub(super) fn left_half(&self) -> Self {
    Addr(self.0 & !Addr::HALF_MASK)
  }

  /// Given an address to one word of a two-word allocation, returns the address
  /// of the other word of that allocation.
  #[inline(always)]
  pub fn other_half(&self) -> Self {
    Addr(self.0 ^ Addr::HALF_MASK)
  }
}
