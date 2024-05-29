use hvm64_util::new_uninit_slice;

use super::*;

/// A bitset representing the set of labels used in a def.
#[derive(Default, Debug, Clone, PartialEq, Eq)]
pub struct LabSet {
  /// The least label greater than every label in the set.
  ///
  /// `lab >= set.min_safe` implies `!set.has(lab)`.
  pub(crate) min_safe: Lab,
  pub(crate) bits: Cow<'static, [u64]>,
}

impl LabSet {
  pub const NONE: LabSet = LabSet { min_safe: 0, bits: Cow::Borrowed(&[]) };
  pub const ALL: LabSet = LabSet { min_safe: Lab::MAX, bits: Cow::Borrowed(&[u64::MAX; 1024]) };

  pub fn add(&mut self, lab: Lab) {
    self.min_safe = self.min_safe.max(lab + 1);
    let index = (lab >> 6) as usize;
    let bit = lab & 63;
    let bits = self.bits.to_mut();
    if index >= bits.len() {
      bits.resize(index + 1, 0);
    }
    bits[index] |= 1 << bit;
  }

  pub fn has(&self, lab: Lab) -> bool {
    if lab >= self.min_safe {
      return false;
    }
    let index = (lab >> 6) as usize;
    let bit = lab & 63;
    unsafe { self.bits.get_unchecked(index) & 1 << bit != 0 }
  }

  /// Adds all of the labels in `other` to this set.
  pub fn union(&mut self, other: &LabSet) {
    self.min_safe = self.min_safe.max(other.min_safe);
    let bits = self.bits.to_mut();
    for (a, b) in bits.iter_mut().zip(other.bits.iter()) {
      *a |= b;
    }
    if other.bits.len() > bits.len() {
      bits.extend_from_slice(&other.bits[bits.len() ..])
    }
  }

  pub fn bits(&self) -> &[u64] {
    &self.bits
  }

  pub const fn from_bits(bits: &'static [u64]) -> Self {
    if bits.is_empty() {
      return LabSet::NONE;
    }
    let min_safe = (bits.len() << 6) as u16 - bits[bits.len() - 1].leading_zeros() as u16;
    LabSet { min_safe, bits: Cow::Borrowed(bits) }
  }
}

impl FromIterator<Lab> for LabSet {
  fn from_iter<T: IntoIterator<Item = Lab>>(iter: T) -> Self {
    let mut set = LabSet::default();
    for lab in iter {
      set.add(lab);
    }
    set
  }
}

/// A custom nilary interaction net agent.
///
/// This is *roughly* equivalent to the following definition:
/// ```rust,ignore
/// struct Def<T: ?Sized + AsDef = dyn AsDef> {
///   pub labs: LabSet,
///   pub data: T,
/// }
/// ```
///
/// Except that, with this definition, `&Def` is a thin pointer, as the vtable
/// data is effectively stored inline in `Def`.
#[repr(C)] // ensure that the fields will have a consistent alignment, regardless of `T`
#[repr(align(32))] // `Ref` ports are `Align4`
pub struct Def<T: ?Sized + Send + Sync = Dynamic> {
  /// The set of labels used by this agent; the agent commutes with any
  /// interaction combinator whose label is not in this set.
  pub labs: LabSet,
  ty: TypeId,
  call: unsafe fn(*const Def<T>, &mut Net, port: Port),
  pub data: T,
}

pub type DynDef = dyn DerefMut<Target = Def> + Send + Sync;

/// An internal type used to mark dynamic `Def`s.
///
/// This should be unsized, but there is no stable way to do this at the moment.
pub struct Dynamic(());

unsafe impl Send for Dynamic {}
unsafe impl Sync for Dynamic {}

pub trait AsDef: Any + Send + Sync {
  unsafe fn call(slf: *const Def<Self>, net: &mut Net, port: Port);
}

impl<T: Send + Sync> Def<T> {
  pub fn new(labs: LabSet, data: T) -> Self
  where
    T: AsDef,
  {
    Def { labs, ty: TypeId::of::<T>(), call: T::call, data }
  }

  #[inline(always)]
  pub const fn upcast(&self) -> &Def {
    unsafe { &*(self as *const _ as *const _) }
  }

  #[inline(always)]
  pub fn upcast_mut(&mut self) -> &mut Def {
    unsafe { &mut *(self as *mut _ as *mut _) }
  }
}

impl Def {
  #[inline(always)]
  pub unsafe fn downcast_ptr<T: Send + Sync + 'static>(slf: *const Def) -> Option<*const Def<T>> {
    if (*slf).ty == TypeId::of::<T>() { Some(slf as *const Def<T>) } else { None }
  }
  #[inline(always)]
  pub unsafe fn downcast_mut_ptr<T: Send + Sync + 'static>(slf: *mut Def) -> Option<*mut Def<T>> {
    if (*slf).ty == TypeId::of::<T>() { Some(slf as *mut Def<T>) } else { None }
  }
  #[inline(always)]
  pub fn downcast_ref<T: Send + Sync + 'static>(&self) -> Option<&Def<T>> {
    unsafe { Def::downcast_ptr(self).map(|x| &*x) }
  }
  #[inline(always)]
  pub fn downcast_mut<T: Send + Sync + 'static>(&mut self) -> Option<&mut Def<T>> {
    unsafe { Def::downcast_mut_ptr(self).map(|x| &mut *x) }
  }
  #[inline(always)]
  pub unsafe fn call(slf: *const Def, net: &mut Net, port: Port) {
    ((*slf).call)(slf as *const _, net, port)
  }
}

impl<T: Send + Sync> Deref for Def<T> {
  type Target = Def;
  #[inline(always)]
  fn deref(&self) -> &Self::Target {
    Def::upcast(self)
  }
}

impl<T: Send + Sync> DerefMut for Def<T> {
  #[inline(always)]
  fn deref_mut(&mut self) -> &mut Self::Target {
    Def::upcast_mut(self)
  }
}

impl<F: Fn(&mut Net, Port) + Send + Sync + 'static> AsDef for F {
  unsafe fn call(slf: *const Def<Self>, net: &mut Net, port: Port) {
    ((*slf).data)(net, port)
  }
}

impl<'a> Net<'a> {
  /// Expands a [`Ref`] node connected to `trg`.
  #[inline(never)]
  pub fn call(&mut self, port: Port, trg: Port) {
    trace!(self, port, trg);

    let def = port.addr().def();

    if trg.tag() == Ctr && !def.labs.has(trg.lab()) {
      return self.comm02(port, trg);
    }

    self.rwts.dref += 1;

    unsafe { Def::call(port.addr().0 as *const _, self, trg) }
  }
}

/// [`Def`]s, when not pre-compiled, are represented as lists of instructions.
#[derive(Debug, Default, Clone)]
pub struct InterpretedDef {
  pub instr: Vec<Instruction>,
  /// The number of targets used in the def; must be greater than all of the
  /// `TrgId` indices in `instr`.
  pub trgs: usize,
}

impl InterpretedDef {
  #[inline(always)]
  pub fn instructions(&self) -> &[Instruction] {
    &self.instr
  }
}

impl InterpretedDef {
  pub fn new_trg_id(&mut self) -> TrgId {
    let index = self.trgs;
    self.trgs += 1;
    TrgId::new(index)
  }
}

impl AsDef for InterpretedDef {
  unsafe fn call(def: *const Def<InterpretedDef>, net: &mut Net, trg: Port) {
    let def = unsafe { &*def };
    let def = &def.data;
    let instructions = &def.instr;

    if def.trgs >= net.trgs.len() {
      net.trgs = new_uninit_slice(def.trgs);
    }

    let mut trgs = Trgs(&mut net.trgs[..] as *mut _ as *mut _);

    /// Points to an array of `Trg`s of length at least `def.trgs`. The `Trg`s
    /// may not all be initialized.
    ///
    /// Only `TrgId`s with index less than `def.trgs` may be passed to `get_trg`
    /// and `set_trg`, and `get_trg` can only be called after `set_trg` was
    /// called with the same `TrgId`.
    struct Trgs(*mut Trg);

    impl Trgs {
      #[inline(always)]
      fn get_trg(&self, i: TrgId) -> Trg {
        unsafe { (*self.0.byte_offset(i.byte_offset as _)).clone() }
      }

      #[inline(always)]
      fn set_trg(&mut self, i: TrgId, trg: Trg) {
        unsafe { *self.0.byte_offset(i.byte_offset as _) = trg }
      }
    }

    trgs.set_trg(TrgId::new(0), Trg::port(trg));
    for i in instructions {
      unsafe {
        match *i {
          Instruction::Const { trg, ref port } => trgs.set_trg(trg, Trg::port(port.clone())),
          Instruction::Link { a, b } => net.link_trg(trgs.get_trg(a), trgs.get_trg(b)),
          Instruction::LinkConst { trg, ref port } => {
            if !port.is_principal() {
              unreachable_unchecked()
            }
            net.link_trg_port(trgs.get_trg(trg), port.clone())
          }
          Instruction::Ctr { lab, trg, p1, p2 } => {
            let (t1, t2) = net.do_ctr(lab, trgs.get_trg(trg));
            trgs.set_trg(p1, t1);
            trgs.set_trg(p2, t2);
          }
          Instruction::Op { op, trg, rhs, out } => {
            let (r, o) = net.do_op(op, trgs.get_trg(trg));
            trgs.set_trg(rhs, r);
            trgs.set_trg(out, o);
          }
          Instruction::OpNum { op, trg, ref rhs, out } => {
            let o = net.do_op_num(op, trgs.get_trg(trg), rhs.clone());
            trgs.set_trg(out, o);
          }
          Instruction::Mat { trg, arms, out } => {
            let (a, o) = net.do_mat(trgs.get_trg(trg));
            trgs.set_trg(arms, a);
            trgs.set_trg(out, o);
          }
          Instruction::Wires { av, aw, bv, bw } => {
            let (avt, awt, bvt, bwt) = net.do_wires();
            trgs.set_trg(av, avt);
            trgs.set_trg(aw, awt);
            trgs.set_trg(bv, bvt);
            trgs.set_trg(bw, bwt);
          }
        }
      }
    }
  }
}
