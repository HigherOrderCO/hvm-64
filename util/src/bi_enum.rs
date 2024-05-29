/// Defines bi-directional mappings for a numeric enum.
#[macro_export]
macro_rules! bi_enum {
  (
    #[repr($uN:ident)]
    $(#$attr:tt)*
    $vis:vis enum $Ty:ident {
      $($(#$var_addr:tt)* $Variant:ident = $value:literal),* $(,)?
    }
  ) => {
    #[repr($uN)] $(#$attr)* $vis enum $Ty { $($(#$var_addr)* $Variant = $value,)* }

    impl TryFrom<$uN> for $Ty {
      type Error = ();
      fn try_from(value: $uN) -> Result<Self, Self::Error> {
        Ok(match value { $($value => $Ty::$Variant,)* _ => Err(())?, })
      }
    }

    impl $Ty {
      #[allow(unused)]
      pub unsafe fn from_unchecked(value: $uN) -> $Ty {
        Self::try_from(value).unwrap_unchecked()
      }
    }

    impl From<$Ty> for $uN {
      fn from(value: $Ty) -> Self { value as Self }
    }
  };
  (
    #[repr($uN:ident)]
    $(#$attr:tt)*
    $vis:vis enum $Ty:ident {
      $($(#$var_addr:tt)* $str:literal: $Variant:ident = $value:literal),* $(,)?
    }
  ) => {
    bi_enum! { #[repr($uN)] $(#$attr)* $vis enum $Ty { $($(#$var_addr)* $Variant = $value,)* } }

    impl core::fmt::Display for $Ty {
      fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str(match self { $($Ty::$Variant => $str,)* })
      }
    }

    impl core::str::FromStr for $Ty {
      type Err = ();
      fn from_str(s: &str) -> Result<Self, Self::Err> {
        Ok(match s { $($str => $Ty::$Variant,)* _ => Err(())?, })
      }
    }
  };
}

#[test]
fn test_bi_enum() {
  use crate::prelude::*;
  use alloc::string::ToString;
  use core::str::FromStr;

  bi_enum! {
    #[repr(u8)]
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    enum Trit {
      Nil = 0,
      One = 1,
      Two = 2,
    }
  }
  assert_eq!(u8::from(Trit::Nil), 0);
  assert_eq!(Trit::try_from(1), Ok(Trit::One));
  assert_eq!(Trit::try_from(100), Err(()));

  bi_enum! {
    #[repr(u8)]
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    enum Op {
      "+": Add = 0,
      "-": Sub = 1,
      "*": Mul = 2,
      "/": Div = 3,
    }
  }
  assert_eq!(Op::Add.to_string(), "+");
  assert_eq!(Op::from_str("-"), Ok(Op::Sub));
  assert_eq!(Op::from_str("#"), Err(()));
}
