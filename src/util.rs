pub fn num_to_str(mut num: usize) -> String {
  let mut txt = Vec::new();
  num += 1;
  while num > 0 {
    num -= 1;
    txt.push((num % 26) as u8 + b'a');
    num /= 26;
  }
  txt.reverse();
  String::from_utf8(txt).unwrap()
}

#[macro_export]
macro_rules! bi_enum {
  (
    #[repr($uN:ident)]
    $(#$attr:tt)*
    $vis:vis enum $Ty:ident {
      $($Variant:ident = $value:literal),* $(,)?
    }
  ) => {
    #[repr($uN)] $(#$attr)* $vis enum $Ty { $($Variant = $value,)* }

    impl TryFrom<$uN> for $Ty {
      type Error = ();
      fn try_from(value: $uN) -> Result<Self, Self::Error> {
        Ok(match value { $($value => $Ty::$Variant,)* _ => Err(())?, })
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
      $($str:literal: $Variant:ident = $value:literal),* $(,)?
    }
  ) => {
    bi_enum! { #[repr($uN)] $(#$attr)* $vis enum $Ty { $($Variant = $value,)* } }

    impl std::fmt::Display for $Ty {
      fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self { $($Ty::$Variant => $str,)* })
      }
    }

    impl std::str::FromStr for $Ty {
      type Err = ();
      fn from_str(s: &str) -> Result<Self, Self::Err> {
        Ok(match s { $($str => $Ty::$Variant,)* _ => Err(())?, })
      }
    }
  };
}
