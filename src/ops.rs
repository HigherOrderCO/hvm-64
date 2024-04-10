pub mod word;

use ordered_float::OrderedFloat;

use crate::util::bi_enum;

use self::word::{FromWord, ToWord};
use std::{
  cmp::{Eq, Ord},
  fmt::Display,
  str::FromStr,
};

#[derive(Clone, Copy, Debug)]
pub enum Num {
  Int(i64),
  Float(f32),
}

impl ToWord for Num {
  fn to_word(self) -> u64 {
    match self {
      Self::Int(int) => int as u64,
      Self::Float(float) => unsafe { std::mem::transmute::<_, u32>(float) as u64 },
    }
  }
}

trait Numeric: Eq + Ord + Sized {
  const ZERO: Self;
  const ONE: Self;

  fn add(_: Self, _: Self) -> Self {
    return Self::ZERO;
  }
  fn sub(_: Self, _: Self) -> Self {
    return Self::ZERO;
  }
  fn mul(_: Self, _: Self) -> Self {
    return Self::ZERO;
  }
  fn div(_: Self, _: Self) -> Self {
    return Self::ZERO;
  }
  fn rem(_: Self, _: Self) -> Self {
    return Self::ZERO;
  }
  fn and(_: Self, _: Self) -> Self {
    return Self::ZERO;
  }
  fn or(_: Self, _: Self) -> Self {
    return Self::ZERO;
  }
  fn xor(_: Self, _: Self) -> Self {
    return Self::ZERO;
  }
  fn shl(_: Self, _: Self) -> Self {
    return Self::ZERO;
  }
  fn shr(_: Self, _: Self) -> Self {
    return Self::ZERO;
  }
  fn from_bool(b: bool) -> Self {
    if b { Self::ONE } else { Self::ZERO }
  }
}

macro_rules! impl_numeric {
  ( $($ty:ty),+ ) => {
    $(
    impl Numeric for $ty {
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

impl_numeric! { u8, u16, u32, u64, i8, i16, i32 }

impl Numeric for OrderedFloat<f32> {
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
  #[repr(u8)]
  /// The type of a numeric operation.
  ///
  /// This dictates how the bits of the operands will be interpreted,
  /// and the return type of the operation.
  #[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
  pub enum Ty {
    "u8":  U8  = 0,
    "u16": U16 = 1,
    "u32": U32 = 2,
    "u60": U60 = 3,
    "i8":  I8  = 4,
    "i16": I16 = 5,
    "i32": I32 = 6,
    "f32": F32 = 7,
  }
}

bi_enum! {
  #[repr(u8)]
  /// Native operations on numerics (u8, u16, u32, u60, i8, i16, i32, f32).
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
    "&":   And  = 14,
    "|":   Or   = 15,
    "^":   Xor  = 16,
    "<<":  Shl  = 17,
    "<<$": ShlS = 18,
    ">>":  Shr  = 19,
    ">>$": ShrS = 20,
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
      Self::And => Self::And,
      Self::Or => Self::Or,
      Self::Xor => Self::Xor,
      Self::Shl => Self::ShlS,
      Self::ShlS => Self::Shl,
      Self::Shr => Self::ShrS,
      Self::ShrS => Self::Shr,
    }
  }

  fn op<T: Numeric>(self, a: T, b: T) -> T {
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

/// A numeric operator.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct TypedOp {
  /// The type of the operands.
  pub ty: Ty,
  /// The operation. An opaque type whose interpretation depends on `ty`.
  pub op: Op,
}

impl TypedOp {
  pub unsafe fn from_unchecked(val: u16) -> Self {
    std::mem::transmute(val)
  }

  pub fn is_int(&self) -> bool {
    self.ty < Ty::F32
  }

  pub fn swap(self) -> Self {
    Self { op: self.op.swap(), ty: self.ty }
  }

  #[inline]
  pub fn op(self, a: u64, b: u64) -> u64 {
    match self.ty {
      Ty::I8 => self.op.op(i8::from_word(a), i8::from_word(b)).to_word(),
      Ty::I16 => self.op.op(i16::from_word(a), i16::from_word(b)).to_word(),
      Ty::I32 => self.op.op(i32::from_word(a), i32::from_word(b)).to_word(),

      Ty::U8 => self.op.op(u8::from_word(a), u8::from_word(b)).to_word(),
      Ty::U16 => self.op.op(u16::from_word(a), u16::from_word(b)).to_word(),
      Ty::U32 => self.op.op(u32::from_word(a), u32::from_word(b)).to_word(),
      Ty::U60 => self.op.op(u64::from_word(a), u64::from_word(b)).to_word(),

      Ty::F32 => self.op.op(OrderedFloat::<f32>::from_word(a), OrderedFloat::<f32>::from_word(b)).to_word(),
    }
  }
}

impl Display for TypedOp {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    match self.ty {
      Ty::U60 => write!(f, "{}", self.op),
      _ => write!(f, "{}.{}", self.ty, self.op),
    }
  }
}

#[derive(thiserror::Error, Debug)]
pub enum Error {
  #[error("invalid type: {0}")]
  Type(String),
  #[error("invalid operator: {0}")]
  Op(String),
}

impl TryFrom<u16> for TypedOp {
  type Error = ();

  fn try_from(value: u16) -> Result<Self, Self::Error> {
    let [ty, op] = value.to_be_bytes();

    Ok(Self { ty: Ty::try_from(ty)?, op: Op::try_from(op)? })
  }
}

impl From<TypedOp> for u16 {
  fn from(TypedOp { ty, op }: TypedOp) -> Self {
    ((ty as u16) << 8) | (op as u16)
  }
}

impl FromStr for TypedOp {
  type Err = Error;

  fn from_str(s: &str) -> Result<Self, Self::Err> {
    match s.split('.').collect::<Vec<_>>().as_slice() {
      [ty, op] => Ok(Self {
        ty: Ty::from_str(ty).map_err(|_| Error::Type(ty.to_string()))?,
        op: Op::from_str(op).map_err(|_| Error::Op(op.to_string()))?,
      }),
      [op] => Ok(Self { ty: Ty::U60, op: Op::from_str(op).map_err(|_| Error::Op(op.to_string()))? }),

      _ => Err(Error::Op(s.to_string())),
    }
  }
}
