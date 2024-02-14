use crate::util::bi_enum;

bi_enum! {
  #[repr(u16)]
  /// A native operation on 60-bit integers.
  #[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
  pub enum Op {
    "+":  Add = 0,
    "-":  Sub = 1,
    "*":  Mul = 2,
    "/":  Div = 3,
    "%":  Mod = 4,
    "==": Eq  = 5,
    "!=": Ne  = 6,
    "<":  Lt  = 7,
    ">":  Gt  = 8,
    "<=": Lte = 9,
    ">=": Gte = 10,
    "&":  And = 11,
    "|":  Or  = 12,
    "^":  Xor = 13,
    "<<": Lsh = 14,
    ">>": Rsh = 15,
    "!":  Not = 16,
  }
}

impl Op {
  #[inline]
  pub fn op(self, a: u64, b: u64) -> u64 {
    use Op::*;
    const U60: u64 = 0xFFF_FFFF_FFFF_FFFF;
    match self {
      Add => a.wrapping_add(b) & U60,
      Sub => a.wrapping_sub(b) & U60,
      Mul => a.wrapping_mul(b) & U60,
      Div => a.checked_div(b).unwrap_or(0),
      Mod => a.checked_rem(b).unwrap_or(0),
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
