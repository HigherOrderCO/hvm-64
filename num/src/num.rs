#![no_std]

use hvm64_util::prelude::*;

use core::hint::unreachable_unchecked;

use hvm64_util::bi_enum;

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Num(u32);

bi_enum! {
  #[repr(u8)]
  #[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
  pub enum NumTag {
    "u24": U24  = 0x00,
    "i24": I24  = 0x01,
    "f24": F24  = 0x02,
    "[]":  Sym  = 0x03,
    "+":   Add  = 0x04,
    "-":   Sub  = 0x05,
    ":-":  SubS = 0x06,
    "*":   Mul  = 0x07,
    "/":   Div  = 0x08,
    ":/":  DivS = 0x09,
    "%":   Rem  = 0x0a,
    ":%":  RemS = 0x0b,
    "=":   Eq   = 0x0c,
    "!":   Ne   = 0x0d,
    "<":   Lt   = 0x0e,
    ">":   Gt   = 0x0f,
    "&":   And  = 0x10,
    "|":   Or   = 0x11,
    "^":   Xor  = 0x12,
    "<<":  Shl  = 0x13,
    ":<<": ShlS = 0x14,
    ">>":  Shr  = 0x15,
    ":>>": ShrS = 0x16,
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
  pub const INVALID: Self = Num::new(NumTag::U24, 0);
  pub const TAG: u8 = 3;

  #[inline(always)]
  pub const fn new(tag: NumTag, payload: u32) -> Self {
    Self((payload << 8) | ((tag as u32) << 3) | (Self::TAG as u32))
  }

  #[inline(always)]
  pub const unsafe fn from_raw(raw: u32) -> Self {
    Num(raw)
  }

  #[inline(always)]
  pub const fn new_u24(val: u32) -> Self {
    Self::new(NumTag::U24, val << 8)
  }

  #[inline(always)]
  pub const fn new_i24(val: i32) -> Self {
    Self::new(NumTag::I24, (val << 8) as u32)
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
    Self::new(NumTag::F24, shifted_bits << 8)
  }

  #[inline(always)]
  pub const fn new_sym(val: NumTag) -> Self {
    Self::new(NumTag::Sym, (val as u32) << 8)
  }

  #[inline(always)]
  pub fn raw(self) -> u32 {
    self.0
  }

  #[inline(always)]
  pub fn tag(self) -> NumTag {
    unsafe { NumTag::from_unchecked((self.0 >> 3 & 0x1f) as u8) }
  }

  #[inline(always)]
  pub fn payload(self) -> u32 {
    self.0 >> 8
  }

  #[inline(always)]
  pub fn get_u24(self) -> u32 {
    self.payload()
  }

  #[inline(always)]
  pub fn get_i24(self) -> i32 {
    (self.0 as i32) >> 8
  }

  #[inline(always)]
  pub fn get_f24(self) -> f32 {
    f32::from_bits(self.0 & !0xff)
  }

  #[inline(always)]
  pub unsafe fn get_sym(self) -> NumTag {
    unsafe { NumTag::from_unchecked(self.get_u24() as u8) }
  }

  #[inline]
  pub fn operate(a: Self, b: Self) -> Self {
    let at = a.tag();
    let bt = b.tag();
    let (op, ty, a, b) = match ((at, at.is_op(), a), (bt, bt.is_op(), b)) {
      ((NumTag::Sym, ..), (NumTag::Sym, ..)) | ((_, false, _), (_, false, _)) | ((_, true, _), (_, true, _)) => {
        return Self::INVALID;
      }
      sym!((NumTag::Sym, _, a), (.., b)) => return Self::new(unsafe { a.get_sym() }, b.payload()),
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
          NumTag::Mul => Num::new_u24(b.wrapping_sub(a)),
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
          NumTag::Mul => Num::new_i24(b.wrapping_sub(a)),
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
          NumTag::And => Num::new_f24(a.atan2(b)),
          NumTag::Or => Num::new_f24(b.log(a)),
          NumTag::Xor => Num::new_f24(a.powf(b)),
          NumTag::Shl => Num::INVALID,
          NumTag::ShlS => Num::INVALID,
          NumTag::Shr => Num::INVALID,
          NumTag::ShrS => Num::INVALID,
        }
      }
      _ => unsafe { unreachable_unchecked() },
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
      _ => write!(f, "["),
    }
  }
}
