pub trait FromWord {
  fn from_word(bits: u64) -> Self;
}

pub trait ToWord {
  fn to_word(self) -> u64;
}

macro_rules! impl_signed {
  ( $($ty:ty),+ ) => {
    $(
    impl FromWord for $ty {
      #[inline(always)]
      fn from_word(bits: u64) -> Self {
        unsafe { std::mem::transmute::<_, i64>(bits) as Self }
      }
    }

    impl ToWord for $ty {
      #[inline(always)]
      fn to_word(self) -> u64 {
        unsafe { std::mem::transmute(self as i64) }
      }
    }
    )*
  };
}

macro_rules! impl_unsigned {
  ( $($ty:ty),+ ) => {
    $(
    impl FromWord for $ty {
      #[inline(always)]
      fn from_word(bits: u64) -> Self {
        bits as $ty
      }
    }

    impl ToWord for $ty {
      #[inline(always)]
      fn to_word(self) -> u64 {
        self as u64
      }
    }
    )*
  };
}

impl_signed! { i8, i16, i32 }
impl_unsigned! { u8, u16, u32, u64 }

impl FromWord for f32 {
  #[inline(always)]
  fn from_word(bits: u64) -> Self {
    f32::from_bits(bits as u32)
  }
}

impl ToWord for f32 {
  #[inline(always)]
  fn to_word(self) -> u64 {
    self.to_bits() as u64
  }
}
