use super::*;

/// A port in the interaction net.
///
/// The type of a port is determined by its *tag*, which is stored in the bottom
/// three bits.
///
/// All tags other than [`Num`] divide the bits of the port as follows:
/// - the top 16 bits are the *label*, accessible with [`Port::lab`]
/// - the middle 45 bits are the non-alignment bits of the *address*, an
///   8-byte-aligned pointer accessible with [`Port::addr`]
/// - the bottom 3 bits are the tag, as always
///
/// The semantics of these fields depend upon the tag; see the documentation for
/// each [`Tag`] variant.
#[derive(Clone, Eq, PartialEq, PartialOrd, Hash, Default)]
#[repr(transparent)]
#[must_use]
pub struct Port(pub u64);

bi_enum! {
  #[repr(u8)]
  #[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
  pub enum Tag {
    /// `Red` ports represent redirects, which are an implementation detail of
    /// the atomic linking algorithm, and don't have a precise analogue in
    /// interaction nets.
    ///
    /// These ports are never directly held, but rather replace the backlinks of
    /// some var ports. They are used to resolve inter-thread conflicts, and
    /// thus will never appear when single-threaded.
    ///
    /// See the documentation for the linking algorithm for more.
    Red = 0,
    /// A `Var` port represents an auxiliary port in the net.
    ///
    /// The address of this port represents the wire leaving this port,
    /// accessible with `Port::wire`.
    ///
    /// The label of this port is currently unused and always 0.
    Var = 1,
    /// A `Ref` port represents the principal port of a nilary reference node.
    ///
    /// The address of this port is a pointer to the corresponding [`Def`].
    ///
    /// The label of this port is always equivalent to `def.labs.min_safe`, and
    /// is used as an optimization for the ref commutation interaction.
    ///
    /// Eraser nodes are represented by a null-pointer `Ref`, available as the
    /// constant [`Port::ERA`].
    Ref = 2,
    /// A `Num` port represents the principal port of a U60 node.
    ///
    /// The top 60 bits of the port are the value of this node, and are
    /// accessible with [`Port::num`].
    ///
    /// The 4th bit from the bottom is currently unused in this port.
    Num = 3,
    /// An `Op` port represents the principal port of an Op node.
    ///
    /// The label of this port is the corresponding operation, which can be
    /// accessed with [`Port::op`].
    ///
    /// The address of this port is the address of a two-word allocation,
    /// storing the targets of the wires connected to the two auxiliary ports of
    /// this node.
    Op = 4,
    /// A `Mat` port represents the principal port of a Mat node.
    ///
    /// The address of this port is the address of a two-word allocation,
    /// storing the targets of the wires connected to the two auxiliary ports of
    /// the node.
    ///
    /// The label of this port is currently unused and always 0.
    Mat = 6,
    /// A `Ctr` port represents the principal port of an binary interaction
    /// combinator node.
    ///
    /// The label of this port is the label of the combinator; two combinators
    /// annihilate if they have the same label, or commute otherwise.
    ///
    /// The address of this port is the address of a two-word allocation,
    /// storing the targets of the wires connected to the two auxiliary ports of
    /// the node.
    Ctr = 7,
  }
}

impl fmt::Debug for Port {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    write!(f, "{:016x?} ", self.0)?;
    match *self {
      Port::ERA => write!(f, "[ERA]"),
      Port::FREE => write!(f, "[FREE]"),
      Port::GONE => write!(f, "[GONE]"),
      Port::LOCK => write!(f, "[LOCK]"),
      _ => match self.tag() {
        Num => write!(f, "[Num {}]", self.num()),
        Var | Red | Mat => write!(f, "[{:?} {:?}]", self.tag(), self.addr()),
        Op | Ctr | Ref => write!(f, "[{:?} {:?} {:?}]", self.tag(), self.lab(), self.addr()),
      },
    }
  }
}

impl Port {
  /// The principal port of an eraser node.
  pub const ERA: Port = Port(Ref as _);
  /// A sentinel value used to indicate free memory; see the allocator for more
  /// details.
  pub const FREE: Port = Port(0x8000_0000_0000_0000);
  /// A sentinel value used to lock a wire; see the linking algorithm for more
  /// details.
  pub const LOCK: Port = Port(0xFFFF_FFFF_FFFF_FFF0);
  /// A sentinel value used in the atomic linking algorithm; see it for more
  /// details.
  pub const GONE: Port = Port(0xFFFF_FFFF_FFFF_FFFF);

  /// Creates a new port with a given tag, label, and addr.
  #[inline(always)]
  pub fn new(tag: Tag, lab: Lab, addr: Addr) -> Self {
    Port(((lab as u64) << 48) | (addr.0 as u64) | (tag as u64))
  }

  /// Creates a new [`Var`] port with a given addr.
  #[inline(always)]
  pub fn new_var(addr: Addr) -> Self {
    Port::new(Var, 0, addr)
  }

  /// Creates a new [`Num`] port with a given 60-bit numeric value.
  #[inline(always)]
  pub const fn new_num(val: i64) -> Self {
    Port((val << 4) as u64 | (Num as u64))
  }

  /// Creates a new [`Ref`] port corresponding to a given definition.
  #[inline(always)]
  pub fn new_ref(def: &Def) -> Port {
    Port::new(Ref, def.labs.min_safe, Addr(def as *const _ as _))
  }

  /// Accesses the tag of this port; this is valid for all ports.
  #[inline(always)]
  pub fn tag(&self) -> Tag {
    unsafe { Tag::from_unchecked((self.0 & 0x7) as u8) }
  }

  #[inline(always)]
  pub fn is(&self, tag: Tag) -> bool {
    self.tag() == tag
  }

  /// Accesses the label of this port; this is valid for all non-`Num` ports.
  #[inline(always)]
  pub const fn lab(&self) -> Lab {
    (self.0 >> 48) as Lab
  }

  /// Accesses the addr of this port; this is valid for all non-`Num` ports.
  #[inline(always)]
  pub const fn addr(&self) -> Addr {
    Addr((self.0 & 0x0000_FFFF_FFFF_FFF8) as usize as _)
  }

  /// Accesses the operation of this port; this is valid for [`Op1`] and [`Op2`]
  /// ports.
  #[inline(always)]
  pub fn op(&self) -> Op {
    self.lab().into()
  }

  /// Accesses the numeric value of this port; this is valid for [`Num`] ports.
  #[inline(always)]
  pub const fn num(&self) -> i64 {
    (self.0 >> 4) as i64
  }

  /// Accesses the wire leaving this port; this is valid for [`Var`] ports and
  /// non-sentinel [`Red`] ports.
  #[inline(always)]
  pub fn wire(&self) -> Wire {
    Wire::new(self.addr())
  }

  #[inline(always)]
  pub fn is_principal(&self) -> bool {
    self.tag() >= Ref
  }

  /// Given a principal port, returns whether this principal port may be part of
  /// a skippable active pair -- an active pair like `ERA-ERA` that does not
  /// need to be added to the redex list.
  #[inline(always)]
  pub fn is_skippable(&self) -> bool {
    self.tag() == Num || self.tag() == Ref && self.lab() != u16::MAX
  }

  /// Converts a [`Var`] port into a [`Red`] port with the same address.
  #[inline(always)]
  pub(super) fn redirect(&self) -> Port {
    Port::new(Red, 0, self.addr())
  }

  /// Converts a [`Red`] port into a [`Var`] port with the same address.
  #[inline(always)]
  pub(super) fn unredirect(&self) -> Port {
    Port::new(Var, 0, self.addr())
  }

  pub(super) fn is_full_node(&self) -> bool {
    self.tag() > Num
  }
}
