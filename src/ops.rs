mod num;
mod word;

use crate::util::bi_enum;

use self::{
  num::Numeric,
  word::{FromWord, ToWord},
};
use std::{
  cmp::{Eq, Ord},
  fmt::Display,
  str::FromStr,
};

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

  fn op<T: Numeric + PartialEq + PartialOrd>(self, a: T, b: T) -> T {
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

  fn op_word<T: Numeric + PartialOrd + PartialEq + FromWord + ToWord>(self, a: u64, b: u64) -> u64 {
    self.op(T::from_word(a), T::from_word(b)).to_word()
  }
}

/// A numeric operator.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(C, align(2))]
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
    const U60: u64 = 0xFFF_FFFF_FFFF_FFFF;

    match self.ty {
      Ty::I8 => self.op.op_word::<i8>(a, b),
      Ty::I16 => self.op.op_word::<i16>(a, b),
      Ty::I32 => self.op.op_word::<i32>(a, b),

      Ty::U8 => self.op.op_word::<u8>(a, b),
      Ty::U16 => self.op.op_word::<u16>(a, b),
      Ty::U32 => self.op.op_word::<u32>(a, b),
      Ty::U60 => self.op.op_word::<u64>(a, b) & U60,

      Ty::F32 => self.op.op_word::<f32>(a, b),
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
    let [ty, op] = value.to_ne_bytes();

    Ok(Self { ty: Ty::try_from(ty)?, op: Op::try_from(op)? })
  }
}

impl From<TypedOp> for u16 {
  fn from(TypedOp { ty, op }: TypedOp) -> Self {
    u16::from_ne_bytes([ty as u8, op as u8])
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
