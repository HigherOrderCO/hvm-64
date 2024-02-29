macro_rules! deref {
  ($({$($gen:tt)*})? $ty:ty => self.$field:ident: $trg:ty) => {
    impl $($($gen)*)? std::ops::Deref for $ty {
      type Target = $trg;
      fn deref(&self) -> &Self::Target {
        &self.$field
      }
    }
    impl $($($gen)*)? std::ops::DerefMut for $ty {
      fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.$field
      }
    }
  };
}

pub(crate) use deref;
