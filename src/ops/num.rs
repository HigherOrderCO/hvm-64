use ordered_float::OrderedFloat;

use super::word::ToWord;

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

pub trait Numeric: Eq + Ord + Sized {
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
