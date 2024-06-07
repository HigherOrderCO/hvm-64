#![no_std]

use hvm64_util::prelude::*;

use core::hint::unreachable_unchecked;

use hvm64_util::bi_enum;

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Num {
  raw: u32,
}

bi_enum! {
  #[repr(u8)]
  #[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
  pub enum NumTag {
    "u24": U24  = 0x00,
    "i24": I24  = 0x01,
    "f24": F24  = 0x02,
    " ":   Sym  = 0x03,
    "+":   Add  = 0x04,
    "-":   Sub  = 0x05,
    ":-":  SubS = 0x06,
    "*":   Mul  = 0x07,
    "/":   Div  = 0x08,
    ":/":  DivS = 0x09,
    "%":   Rem  = 0x0a,
    ":%":  RemS = 0x0b,
    "&":   And  = 0x0c,
    "|":   Or   = 0x0d,
    "^":   Xor  = 0x0e,
    "<<":  Shl  = 0x0f,
    ":<<": ShlS = 0x10,
    ">>":  Shr  = 0x11,
    ":>>": ShrS = 0x12,
    "=":   Eq   = 0x13,
    "!":   Ne   = 0x14,
    "<":   Lt   = 0x15,
    ">":   Gt   = 0x16,
  }
}

impl NumTag {
  #[inline(always)]
  pub fn is_op(self) -> bool {
    self > NumTag::Sym
  }

  #[inline(always)]
  pub fn is_ty(self) -> bool {
    self < NumTag::Sym
  }
}

impl Num {
  pub const ID: Self = Num::new_sym(NumTag::Sym);
  pub const INVALID: Self = Num::new_u24(0);
  pub const TAG: u8 = 3;

  pub const unsafe fn new(tag: NumTag, payload: u32) -> Self {
    Self { raw: (payload << 8) | ((tag as u32) << 3) | (Self::TAG as u32) }
  }

  #[inline(always)]
  pub const unsafe fn from_raw(raw: u32) -> Self {
    Num { raw }
  }

  #[inline(always)]
  pub const fn new_u24(val: u32) -> Self {
    unsafe { Self::new(NumTag::U24, val) }
  }

  #[inline(always)]
  pub const fn new_i24(val: i32) -> Self {
    unsafe { Self::new(NumTag::I24, val as u32) }
  }

  #[inline]
  pub fn new_f24(val: f32) -> Self {
    let bits = val.to_bits();
    let mut shifted_bits = bits >> 8;
    let lost_bits = bits & 0xFF;
    // round ties to even
    shifted_bits += u32::from(!val.is_nan()) & ((lost_bits - ((lost_bits >> 7) & !shifted_bits)) >> 7);
    // ensure NaNs don't become infinities
    shifted_bits |= u32::from(val.is_nan());
    unsafe { Self::new(NumTag::F24, shifted_bits) }
  }

  #[inline(always)]
  pub const fn new_sym(val: NumTag) -> Self {
    unsafe { Self::new(NumTag::Sym, val as u32) }
  }

  #[inline(always)]
  pub fn raw(self) -> u32 {
    self.raw
  }

  #[inline(always)]
  pub fn tag(self) -> NumTag {
    unsafe { NumTag::from_unchecked((self.raw >> 3 & 0x1f) as u8) }
  }

  #[inline(always)]
  pub fn payload(self) -> u32 {
    self.raw >> 8
  }

  #[inline(always)]
  pub fn get_u24(self) -> u32 {
    self.payload()
  }

  #[inline(always)]
  pub fn get_i24(self) -> i32 {
    (self.raw as i32) >> 8
  }

  #[inline(always)]
  pub fn get_f24(self) -> f32 {
    f32::from_bits(self.raw & !0xff)
  }

  #[inline(always)]
  pub unsafe fn get_sym(self) -> NumTag {
    unsafe { NumTag::from_unchecked(self.get_u24() as u8) }
  }

  #[inline]
  pub fn operate_unary(op: NumTag, num: Self) -> Self {
    const U24_MAX: u32 = (1 << 24) - 1;
    const U24_MIN: u32 = 0;
    const I24_MAX: i32 = (1 << 23) - 1;
    const I24_MIN: i32 = (-1) << 23;

    match op {
      NumTag::Sym => num,
      NumTag::U24 => match num.tag() {
        NumTag::U24 => num,
        NumTag::I24 => Num::new_u24(num.get_u24()),
        NumTag::F24 => Num::new_u24((num.get_f24() as u32).clamp(U24_MIN, U24_MAX)),
        _ => Self::INVALID,
      },
      NumTag::I24 => match num.tag() {
        NumTag::U24 => Num::new_i24(num.get_i24()),
        NumTag::I24 => num,
        NumTag::F24 => Num::new_i24((num.get_f24() as i32).clamp(I24_MIN, I24_MAX)),
        _ => Self::INVALID,
      },
      NumTag::F24 => match num.tag() {
        NumTag::U24 => Num::new_f24(num.get_u24() as f32),
        NumTag::I24 => Num::new_f24(num.get_i24() as f32),
        NumTag::F24 => num,
        _ => Self::INVALID,
      },
      _ => unsafe { Self::new(op, num.payload()) },
    }
  }

  #[inline]
  pub fn operate_sym(a: Self, b: Self) -> Self {
    let at = a.tag();
    let bt = b.tag();
    let (op, ty, a, b) = match ((at, at.is_op(), a), (bt, bt.is_op(), b)) {
      ((NumTag::Sym, ..), (NumTag::Sym, ..)) => return Self::INVALID,
      sym!((NumTag::Sym, _, a), (.., b)) => return Self::operate_unary(unsafe { a.get_sym() }, b),
      ((_, false, _), (_, false, _)) | ((_, true, _), (_, true, _)) => return Self::INVALID,
      sym!((op, true, a), (ty, false, b)) => (op, ty, a, b),
    };
    match ty {
      NumTag::U24 => {
        let a = a.get_u24();
        let b = b.get_u24();
        match op {
          NumTag::U24 | NumTag::I24 | NumTag::F24 | NumTag::Sym => unsafe { unreachable_unchecked() },
          NumTag::Add => Num::new_u24(a.wrapping_add(b)),
          NumTag::Sub => Num::new_u24(a.wrapping_sub(b)),
          NumTag::SubS => Num::new_u24(b.wrapping_sub(a)),
          NumTag::Mul => Num::new_u24(a.wrapping_mul(b)),
          NumTag::Div => Num::new_u24(a.wrapping_div(b)),
          NumTag::DivS => Num::new_u24(b.wrapping_div(a)),
          NumTag::Rem => Num::new_u24(a.wrapping_rem(b)),
          NumTag::RemS => Num::new_u24(b.wrapping_rem(a)),
          NumTag::Eq => Num::new_u24((a == b) as u32),
          NumTag::Ne => Num::new_u24((a != b) as u32),
          NumTag::Lt => Num::new_u24((a < b) as u32),
          NumTag::Gt => Num::new_u24((a > b) as u32),
          NumTag::And => Num::new_u24(a & b),
          NumTag::Or => Num::new_u24(a | b),
          NumTag::Xor => Num::new_u24(a ^ b),
          NumTag::Shl => Num::new_u24(a.wrapping_shl(b)),
          NumTag::ShlS => Num::new_u24(b.wrapping_shl(a)),
          NumTag::Shr => Num::new_u24(a.wrapping_shr(b)),
          NumTag::ShrS => Num::new_u24(b.wrapping_shr(a)),
        }
      }
      NumTag::I24 => {
        let a = a.get_i24();
        let b = b.get_i24();
        match op {
          NumTag::U24 | NumTag::I24 | NumTag::F24 | NumTag::Sym => unsafe { unreachable_unchecked() },
          NumTag::Add => Num::new_i24(a.wrapping_add(b)),
          NumTag::Sub => Num::new_i24(a.wrapping_sub(b)),
          NumTag::SubS => Num::new_i24(b.wrapping_sub(a)),
          NumTag::Mul => Num::new_i24(a.wrapping_mul(b)),
          NumTag::Div => Num::new_i24(a.wrapping_div(b)),
          NumTag::DivS => Num::new_i24(b.wrapping_div(a)),
          NumTag::Rem => Num::new_i24(a.wrapping_rem(b)),
          NumTag::RemS => Num::new_i24(b.wrapping_rem(a)),
          NumTag::Eq => Num::new_u24((a == b) as u32),
          NumTag::Ne => Num::new_u24((a != b) as u32),
          NumTag::Lt => Num::new_u24((a < b) as u32),
          NumTag::Gt => Num::new_u24((a > b) as u32),
          NumTag::And => Num::new_i24(a & b),
          NumTag::Or => Num::new_i24(a | b),
          NumTag::Xor => Num::new_i24(a ^ b),
          NumTag::Shl => Num::new_i24(a.wrapping_shl(b as u32)),
          NumTag::ShlS => Num::new_i24(b.wrapping_shl(a as u32)),
          NumTag::Shr => Num::new_i24(a.wrapping_shr(b as u32)),
          NumTag::ShrS => Num::new_i24(b.wrapping_shr(a as u32)),
        }
      }
      NumTag::F24 => {
        let a = a.get_f24();
        let b = b.get_f24();
        match op {
          NumTag::U24 | NumTag::I24 | NumTag::F24 | NumTag::Sym => unsafe { unreachable_unchecked() },
          NumTag::Add => Num::new_f24(a + b),
          NumTag::Sub => Num::new_f24(a - b),
          NumTag::SubS => Num::new_f24(b - a),
          NumTag::Mul => Num::new_f24(a * b),
          NumTag::Div => Num::new_f24(a / b),
          NumTag::DivS => Num::new_f24(b / a),
          NumTag::Rem => Num::new_f24(a % b),
          NumTag::RemS => Num::new_f24(b % a),
          NumTag::Eq => Num::new_u24((a == b) as u32),
          NumTag::Ne => Num::new_u24((a != b) as u32),
          NumTag::Lt => Num::new_u24((a < b) as u32),
          NumTag::Gt => Num::new_u24((a > b) as u32),
          #[cfg(feature = "std")]
          NumTag::And => Num::new_f24(a.atan2(b)),
          #[cfg(feature = "std")]
          NumTag::Or => Num::new_f24(b.log(a)),
          #[cfg(feature = "std")]
          NumTag::Xor => Num::new_f24(a.powf(b)),
          _ => Num::INVALID,
        }
      }
      _ => unsafe { unreachable_unchecked() },
    }
  }

  #[inline]
  pub fn operate_binary(a: Self, op: NumTag, b: Self) -> Self {
    if op == NumTag::Sym {
      Self::operate_sym(a, b)
    } else {
      Self::operate_sym(Self::operate_sym(a, Num::new_sym(op)), b)
    }
  }
}

macro_rules! sym {
  ($a:pat, $b:pat) => {
    ($a, $b) | ($b, $a)
  };
}

use sym;

impl fmt::Debug for Num {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    write!(f, "{:?}", self.tag())?;
    match self.tag() {
      NumTag::Sym => write!(f, "({:?})", unsafe { self.get_sym() }),
      NumTag::U24 => write!(f, "({:?})", self.get_u24()),
      NumTag::I24 => write!(f, "({:?})", self.get_i24()),
      NumTag::F24 => write!(f, "({:?})", self.get_f24()),
      _ => write!(f, "(0x{:06x})", self.payload()),
    }
  }
}

impl fmt::Display for Num {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    match self.tag() {
      NumTag::Sym => write!(f, "[{}]", unsafe { self.get_sym() }),
      NumTag::U24 => write!(f, "{}", self.get_u24()),
      NumTag::I24 => write!(f, "{:+}", self.get_i24()),
      NumTag::F24 => {
        let val = self.get_f24();
        if val.is_infinite() {
          if val.is_sign_positive() { write!(f, "+inf") } else { write!(f, "-inf") }
        } else if val.is_nan() {
          write!(f, "+NaN")
        } else {
          write!(f, "{val:?}")
        }
      }
      _ => write!(f, "[{}{}]", self.tag(), self.payload()),
    }
  }
}
