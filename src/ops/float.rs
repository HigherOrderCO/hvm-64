use crate::util::bi_enum;

use ordered_float::OrderedFloat;

pub trait Float: Eq + Ord + Sized {
  const ZERO: Self;
  const ONE: Self;

  fn add(a: Self, b: Self) -> Self;
  fn sub(a: Self, b: Self) -> Self;
  fn mul(a: Self, b: Self) -> Self;
  fn div(a: Self, b: Self) -> Self;
  fn rem(a: Self, b: Self) -> Self;

  fn from_bool(b: bool) -> Self {
    if b { Self::ONE } else { Self::ZERO }
  }
}

impl Float for OrderedFloat<f32> {
  const ZERO: Self = OrderedFloat(0f32);
  const ONE: Self = OrderedFloat(1f32);

  fn add(a: Self, b: Self) -> Self {
    a + b
  }
  fn sub(a: Self, b: Self) -> Self {
    a - b
  }
  fn mul(a: Self, b: Self) -> Self {
    a * b
  }
  fn div(a: Self, b: Self) -> Self {
    a / b
  }
  fn rem(a: Self, b: Self) -> Self {
    a % b
  }
}

bi_enum! {
  #[repr(u16)]
  /// Native operations on floats.
  ///
  /// Each operation has a swapped counterpart (accessible with `.swap()`),
  /// where the order of the operands is swapped.
  ///
  /// Operations without an already-named counterpart (e.g. `Add <-> Add` and
  /// `Lt <-> Gt`) are suffixed with `$`/`S`: `(-$ 1 2) = (- 2 1) = 1`.
  #[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
  pub enum Op {
    "+":   Add  = 0,
    "-":   Sub  = 1,
    "-$":  SubS = 2,
    "*":   Mul  = 3,
    "/":   Div  = 4,
    "/$":  DivS = 5,
    "%":   Rem  = 6,
    "%$":  RemS = 7,
    "==":  Eq   = 8,
    "!=":  Ne   = 9,
    "<":   Lt   = 10,
    ">":   Gt   = 11,
    "<=":  Le   = 12,
    ">=":  Ge   = 13,
  }
}

impl Op {
  /// Returns this operation's swapped counterpart.
  ///
  /// For all `op, a, b`, `op.swap().op(a, b) == op.op(b, a)`.
  #[inline]
  pub fn swap(self) -> Self {
    match self {
      Self::Add => Self::Add,
      Self::Sub => Self::SubS,
      Self::SubS => Self::Sub,
      Self::Mul => Self::Mul,
      Self::Div => Self::DivS,
      Self::DivS => Self::Div,
      Self::Rem => Self::RemS,
      Self::RemS => Self::Rem,
      Self::Eq => Self::Eq,
      Self::Ne => Self::Ne,
      Self::Lt => Self::Gt,
      Self::Gt => Self::Lt,
      Self::Le => Self::Ge,
      Self::Ge => Self::Le,
    }
  }

  pub fn op<T: Float>(self, a: T, b: T) -> T {
    match self {
      Self::Add => T::add(a, b),
      Self::Sub => T::sub(a, b),
      Self::SubS => T::sub(b, a),
      Self::Mul => T::mul(a, b),
      Self::Div => T::div(a, b),
      Self::DivS => T::div(b, a),
      Self::Rem => T::rem(a, b),
      Self::RemS => T::rem(b, a),
      Self::Eq => T::from_bool(a == b),
      Self::Ne => T::from_bool(a != b),
      Self::Lt => T::from_bool(a < b),
      Self::Le => T::from_bool(a <= b),
      Self::Gt => T::from_bool(a > b),
      Self::Ge => T::from_bool(a >= b),
    }
  }
}
