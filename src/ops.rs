use crate::util::bi_enum;

bi_enum! {
  #[repr(u16)]
  /// A native operation on 60-bit integers.
  ///
  /// Each operation has a swapped counterpart (accessible with `.swap()`),
  /// where the order of the operands is swapped.
  ///
  /// Operations without an already-named counterpart (e.g. `Add <-> Add` and
  /// `Lt <-> Gt`) are suffixed with `$`/`S`: `(-$ 1 2) = (- 2 1) = 1`.
  #[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
  pub enum Op {
    "+":   Add  = 0,
    "-":   Sub  = 1,
    "-$":  SubS = 2,
    "*":   Mul  = 3,
    "/":   Div  = 4,
    "/$":  DivS = 5,
    "%":   Mod  = 6,
    "%$":  ModS = 7,
    "==":  Eq   = 8,
    "!=":  Ne   = 9,
    "<":   Lt   = 10,
    ">":   Gt   = 11,
    "<=":  Lte  = 12,
    ">=":  Gte  = 13,
    "&":   And  = 14,
    "|":   Or   = 15,
    "^":   Xor  = 16,
    "<<":  Shl  = 17,
    "<<$": ShlS = 18,
    ">>":  Shr  = 19,
    ">>$": ShrS = 20,
  }
}

use Op::*;

impl Op {
  /// Returns this operation's swapped counterpart.
  ///
  /// For all `op, a, b`, `op.swap().op(a, b) == op.op(b, a)`.
  #[inline]
  pub fn swap(self) -> Self {
    match self {
      Add => Add,
      Sub => SubS,
      SubS => Sub,
      Mul => Mul,
      Div => DivS,
      DivS => Div,
      Mod => ModS,
      ModS => Mod,
      Eq => Eq,
      Ne => Ne,
      Lt => Gt,
      Gt => Lt,
      Lte => Gte,
      Gte => Lte,
      And => And,
      Or => Or,
      Xor => Xor,
      Shl => ShlS,
      ShlS => Shl,
      Shr => ShrS,
      ShrS => Shr,
    }
  }
  #[inline]
  pub fn op(self, a: u64, b: u64) -> u64 {
    const U60: u64 = 0xFFF_FFFF_FFFF_FFFF;
    match self {
      Add => a.wrapping_add(b) & U60,
      Sub => a.wrapping_sub(b) & U60,
      SubS => b.wrapping_sub(a) & U60,
      Mul => a.wrapping_mul(b) & U60,
      Div => a.checked_div(b).unwrap_or(0),
      DivS => b.checked_div(a).unwrap_or(0),
      Mod => a.checked_rem(b).unwrap_or(0),
      ModS => b.checked_rem(a).unwrap_or(0),
      Eq => (a == b) as u64,
      Ne => (a != b) as u64,
      Lt => (a < b) as u64,
      Gt => (a > b) as u64,
      Lte => (a <= b) as u64,
      Gte => (a >= b) as u64,
      And => a & b,
      Or => a | b,
      Xor => a ^ b,
      Shl => (a << b) & U60,
      ShlS => (b << a) & U60,
      Shr => a >> b,
      ShrS => b >> a,
    }
  }
}
