use ordered_float::OrderedFloat;

use crate::util::bi_enum;
use std::{
  cmp::{Eq, Ord},
  fmt::Display,
  str::FromStr,
};

trait Num: Eq + Ord + Sized {
  const ZERO: Self;
  const ONE: Self;

  fn add(a: Self, b: Self) -> Self;
  fn sub(a: Self, b: Self) -> Self;
  fn mul(a: Self, b: Self) -> Self;
  fn div(a: Self, b: Self) -> Self;
  fn rem(a: Self, b: Self) -> Self;
  fn and(a: Self, b: Self) -> Self;
  fn or(a: Self, b: Self) -> Self;
  fn xor(a: Self, b: Self) -> Self;
  fn shl(a: Self, b: Self) -> Self;
  fn shr(a: Self, b: Self) -> Self;

  fn from_bool(b: bool) -> Self {
    if b { Self::ONE } else { Self::ZERO }
  }
}

macro_rules! impl_num {
  ( $($ty:ty),+ ) => {
    $(
    impl Num for $ty {
      const ZERO: Self = 0;
      const ONE: Self = 1;

      fn add(a: Self, b: Self) -> Self { a.wrapping_add(b) }
      fn sub(a: Self, b: Self) -> Self { a.wrapping_sub(b) }
      fn mul(a: Self, b: Self) -> Self { a.wrapping_mul(b) }
      fn div(a: Self, b: Self) -> Self { a.checked_div(b).unwrap_or(0) }
      fn rem(a: Self, b: Self) -> Self { a.checked_rem(b).unwrap_or(0) }
      fn and(a: Self, b: Self) -> Self { a & b }
      fn or(a: Self, b: Self) -> Self { a | b }
      fn xor(a: Self, b: Self) -> Self { a ^ b }
      fn shl(a: Self, b: Self) -> Self { a.wrapping_shl(b as u32) }
      fn shr(a: Self, b: Self) -> Self { a.wrapping_shr(b as u32) }
    }
    )*
  }
}

impl_num! { u8, u16, u32, u64, i8, i16, i32 }

impl Num for OrderedFloat<f32> {
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
  fn and(_: Self, _: Self) -> Self {
    unimplemented!("f32 & f32 unsupported")
  }
  fn or(_: Self, _: Self) -> Self {
    unimplemented!("f32 | f32 unsupported")
  }
  fn xor(_: Self, _: Self) -> Self {
    unimplemented!("f32 ^ f32 unsupported")
  }
  fn shl(_: Self, _: Self) -> Self {
    unimplemented!("f32 << f32 unsupported")
  }
  fn shr(_: Self, _: Self) -> Self {
    unimplemented!("f32 >> f32 unsupported")
  }
}

bi_enum! {
  #[repr(u8)]
  /// Native operations on mixed-width integers (u8, u16, u32, u60, i8, i16, i32).
  ///
  /// Each operation has a swapped counterpart (accessible with `.swap()`),
  /// where the order of the operands is swapped.
  ///
  /// Operations without an already-named counterpart (e.g. `Add <-> Add` and
  /// `Lt <-> Gt`) are suffixed with `$`/`S`: `(-$ 1 2) = (- 2 1) = 1`.
  #[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
  pub enum BinOp {
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
    "&":   And  = 14,
    "|":   Or   = 15,
    "^":   Xor  = 16,
    "<<":  Shl  = 17,
    "<<$": ShlS = 18,
    ">>":  Shr  = 19,
    ">>$": ShrS = 20,
  }
}

impl BinOp {
  /// Returns this operation's swapped counterpart.
  ///
  /// For all `op, a, b`, `op.swap().op(a, b) == op.op(b, a)`.
  #[inline]
  fn swap(self) -> Self {
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
      Self::And => Self::And,
      Self::Or => Self::Or,
      Self::Xor => Self::Xor,
      Self::Shl => Self::ShlS,
      Self::ShlS => Self::Shl,
      Self::Shr => Self::ShrS,
      Self::ShrS => Self::Shr,
    }
  }

  fn op<T: Num>(self, a: T, b: T) -> T {
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
      Self::And => T::and(a, b),
      Self::Or => T::or(a, b),
      Self::Xor => T::xor(a, b),
      Self::Shl => T::shl(a, b),
      Self::ShlS => T::shl(b, a),
      Self::Shr => T::shr(a, b),
      Self::ShrS => T::shr(b, a),
    }
  }
}

bi_enum! {
  #[repr(u8)]
  #[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
  pub enum Ty {
    "u8":  U8  = 0,
    "u16": U16 = 1,
    "u32": U32 = 2,
    "u60": U60 = 3,
    "i8":  I8  = 4,
    "i16": I16 = 5,
    "i32": I32 = 6,
  }
}

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Op {
  pub ty: Ty,
  pub op: BinOp,
}

impl Display for Op {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    match self.ty {
      Ty::U60 => write!(f, "{}", self.op),
      _ => write!(f, "{}.{}", self.ty, self.op),
    }
  }
}

impl TryFrom<u16> for Op {
  type Error = ();

  fn try_from(value: u16) -> Result<Self, Self::Error> {
    let [ty, op] = value.to_be_bytes();

    Ok(Self { ty: Ty::try_from(ty)?, op: BinOp::try_from(op)? })
  }
}

impl From<Op> for u16 {
  fn from(op: Op) -> Self {
    (op.ty as u16) << 8 | op.op as u16
  }
}

impl FromStr for Op {
  type Err = ();

  fn from_str(s: &str) -> Result<Self, Self::Err> {
    match s.split('.').collect::<Vec<_>>().as_slice() {
      [ty, op] => Ok(Self { ty: Ty::from_str(ty)?, op: BinOp::from_str(op)? }),
      [op] => Ok(Self { ty: Ty::U60, op: BinOp::from_str(op)? }),

      _ => Err(()),
    }
  }
}

impl Op {
  pub fn swap(self) -> Self {
    Self { op: self.op.swap(), ty: self.ty }
  }

  #[inline]
  pub fn op_int(self, a: i64, b: i64) -> i64 {
    const U60: i64 = 0xFFF_FFFF_FFFF_FFFF;

    match self.ty {
      Ty::U8 => self.op.op(a as u8, b as u8) as i64,
      Ty::U16 => self.op.op(a as u16, b as u16) as i64,
      Ty::U32 => self.op.op(a as u32, b as u32) as i64,
      Ty::U60 => self.op.op(a as u64, b as u64) as i64 & U60,
      Ty::I8 => self.op.op(a as i8, b as i8) as i64,
      Ty::I16 => self.op.op(a as i16, b as i16) as i64,
      Ty::I32 => self.op.op(a as i32, b as i32) as i64,
    }
  }

  #[inline]
  pub fn op_float(self, a: f32, b: f32) -> f32 {
    self.op.op(OrderedFloat(a), OrderedFloat(b)).into()
  }
}
