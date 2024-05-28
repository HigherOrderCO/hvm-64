#[macro_export]
macro_rules! deref_to {
  ($({$($gen:tt)*})? $ty:ty => self.$field:ident: $trg:ty) => {
    impl $($($gen)*)? core::ops::Deref for $ty {
      type Target = $trg;
      fn deref(&self) -> &Self::Target {
        &self.$field
      }
    }
    impl $($($gen)*)? core::ops::DerefMut for $ty {
      fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.$field
      }
    }
  };
}
