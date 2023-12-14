// Implements u48: 48-bit unsigned integers using u64 and u128

use std::{fmt, str::FromStr};

use crate::run::Lab;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(u16)]
pub enum Op {
  Add = 0,
  Sub = 1,
  Mul = 2,
  Div = 3,
  Mod = 4,
  Eq = 5,
  Ne = 6,
  Lt = 7,
  Gt = 8,
  Lte = 9,
  Gte = 10,
  And = 11,
  Or = 12,
  Xor = 13,
  Lsh = 14,
  Rsh = 15,
  Not = 16,
}

use Op::*;

impl TryFrom<Lab> for Op {
  type Error = ();

  #[inline(always)]
  fn try_from(value: Lab) -> Result<Self, Self::Error> {
    Ok(match value {
      0 => Add,
      1 => Sub,
      2 => Mul,
      3 => Div,
      4 => Mod,
      5 => Eq,
      6 => Ne,
      7 => Lt,
      8 => Gt,
      9 => Lte,
      10 => Gte,
      11 => And,
      12 => Or,
      13 => Xor,
      14 => Lsh,
      15 => Rsh,
      16 => Not,
      _ => Err(())?,
    })
  }
}

const U60: u64 = 0xFFF_FFFF_FFFF_FFFF;

impl Op {
  #[inline]
  pub fn op(self, a: u64, b: u64) -> u64 {
    match self {
      Add => a.wrapping_add(b) & U60,
      Sub => a.wrapping_sub(b) & U60,
      Mul => a.wrapping_mul(b) & U60,
      Div => a / b,
      Mod => a % b,
      Eq => (a == b) as u64,
      Ne => (a != b) as u64,
      Lt => (a < b) as u64,
      Gt => (a > b) as u64,
      Lte => (a <= b) as u64,
      Gte => (a >= b) as u64,
      And => a & b,
      Or => a | b,
      Xor => a ^ b,
      Lsh => (a << b) & U60,
      Rsh => a >> b,
      Not => !a,
    }
  }
}

impl fmt::Display for Op {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    f.write_str(match self {
      Add => "+",
      Sub => "-",
      Mul => "*",
      Div => "/",
      Mod => "%",
      Eq => "==",
      Ne => "!=",
      Lt => "<",
      Gt => ">",
      Lte => "<=",
      Gte => ">=",
      And => "&&",
      Or => "||",
      Xor => "^",
      Lsh => "<<",
      Rsh => ">>",
      Not => "!",
    })
  }
}

impl FromStr for Op {
  type Err = ();

  fn from_str(s: &str) -> Result<Self, Self::Err> {
    Ok(match s {
      "+" => Add,
      "-" => Sub,
      "*" => Mul,
      "/" => Div,
      "%" => Mod,
      "==" => Eq,
      "!=" => Ne,
      "<" => Lt,
      ">" => Gt,
      "<=" => Lte,
      ">=" => Gte,
      "&&" => And,
      "||" => Or,
      "^" => Xor,
      "<<" => Lsh,
      ">>" => Rsh,
      "!" => Not,
      _ => Err(())?,
    })
  }
}
