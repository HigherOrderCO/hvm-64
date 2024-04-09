mod float;
mod int;

use ordered_float::OrderedFloat;

use self::{
  float::Op as FloatOp,
  int::{Op as IntOp, Ty as IntTy, TypedOp as IntTypedOp},
};
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

impl From<Num> for u64 {
  fn from(num: Num) -> u64 {
    unsafe {
      match num {
        Num::Int(int) => std::mem::transmute(int),
        Num::Float(float) => std::mem::transmute::<_, u32>(float) as u64,
      }
    }
  }
}

/// A numeric operator.
///
/// Represented as a `u16` with
/// [1 bit for int | float][15 bits variant-dependent]
///   - int:   [7 bits type][8 bits operator]
///   - float: [15 bits operator]
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Op {
  Int(IntTypedOp),
  Float(FloatOp),
}

impl TryFrom<u16> for Op {
  type Error = ();

  fn try_from(value: u16) -> Result<Self, Self::Error> {
    const U15: u16 = 0b1000_0000_0000_0000;

    if U15 & value == 0 {
      Ok(Self::Int(IntTypedOp::try_from(value)?))
    } else {
      Ok(Self::Float(FloatOp::try_from(U15 ^ value)?))
    }
  }
}

impl From<Op> for u16 {
  fn from(op: Op) -> Self {
    const U15: u16 = 0b1000_0000_0000_0000;

    match op {
      Op::Int(op) => u16::from(op),
      Op::Float(op) => U15 | u16::from(op),
    }
  }
}

impl FromStr for Op {
  type Err = ();

  fn from_str(s: &str) -> Result<Self, Self::Err> {
    match s.split('.').collect::<Vec<_>>().as_slice() {
      ["f32", op] => Ok(Self::Float(FloatOp::from_str(op)?)),
      [ty, op] => Ok(Self::Int(IntTypedOp::new(IntTy::from_str(ty)?, IntOp::from_str(op)?))),
      [op] => Ok(Self::Int(IntTypedOp::new(IntTy::U60, IntOp::from_str(op)?))),

      _ => Err(()),
    }
  }
}

impl Display for Op {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    match self {
      Self::Int(op) => write!(f, "{op}"),
      Self::Float(op) => write!(f, "f32.{op}"),
    }
  }
}

impl Op {
  pub fn swap(self) -> Self {
    match self {
      Self::Int(op) => Self::Int(op.swap()),
      Self::Float(op) => Self::Float(op.swap()),
    }
  }

  #[inline]
  pub fn op(self, a: u64, b: u64) -> u64 {
    // unsafe is only for `std::mem::transmute`
    unsafe {
      match self {
        Self::Int(op) => {
          let a: i64 = std::mem::transmute(a);
          let b: i64 = std::mem::transmute(b);

          std::mem::transmute(op.op(a, b))
        }
        Self::Float(op) => {
          let a: OrderedFloat<f32> = std::mem::transmute(a as u32);
          let b: OrderedFloat<f32> = std::mem::transmute(b as u32);

          std::mem::transmute::<_, u32>(op.op(a, b)) as u64
        }
      }
    }
  }
}
