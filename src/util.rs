/// Creates a variable uniquely identified by `id`.
pub fn create_var(mut id: usize) -> String {
  let mut txt = Vec::new();
  id += 1;
  while id > 0 {
    id -= 1;
    txt.push((id % 26) as u8 + b'a');
    id /= 26;
  }
  txt.reverse();
  String::from_utf8(txt).unwrap()
}

/// Inverse function of [`create_var`].
/// Returns None when the provided string is not an output of
/// `create_var`.
pub fn var_to_num(s: &str) -> Option<usize> {
  let mut n = 0usize;
  for i in s.chars() {
    let i = (i as u32).checked_sub('a' as u32)? as usize;
    if i > 'z' as usize {
      return None;
    }
    n *= 26;
    n += i;
    n += 1;
  }
  n.checked_sub(1) // if it's none, then it means the initial string was ''
}

#[test]
fn test_create_var() {
  assert_eq!(create_var(0), "a");
  assert_eq!(create_var(1), "b");
  assert_eq!(create_var(25), "z");
  assert_eq!(create_var(26), "aa");
  assert_eq!(create_var(27), "ab");
  assert_eq!(create_var(51), "az");
  assert_eq!(create_var(52), "ba");
  assert_eq!(create_var(676), "za");
  assert_eq!(create_var(701), "zz");
  assert_eq!(create_var(702), "aaa");
  assert_eq!(create_var(703), "aab");
  assert_eq!(create_var(728), "aba");
  assert_eq!(create_var(1351), "ayz");
  assert_eq!(create_var(1352), "aza");
  assert_eq!(create_var(1378), "baa");
}
#[test]
fn test_var_to_num() {
  for i in [0, 1, 2, 3, 10, 26, 27, 30, 50, 70] {
    assert_eq!(Some(i), var_to_num(&create_var(i)));
  }
}

/// Defines bi-directional mappings for a numeric enum.
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

pub(crate) use bi_enum;

#[test]
fn test_bi_enum() {
  use std::str::FromStr;
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

impl crate::ast::Net {
  /// Transforms the net `x & ...` into `y & x ~ (arg y) & ...`
  pub fn with_argument(&mut self, arg: crate::ast::Tree) {
    use crate::ast::Tree;
    let mut fresh = 0usize;
    fn ensure_no_conflicts(tree: &Tree, fresh: &mut usize) {
      match tree {
        Tree::Ctr { lft, rgt, .. } | Tree::Op2 { lft, rgt, .. } | Tree::Mat { sel: lft, ret: rgt } => {
          ensure_no_conflicts(lft, fresh);
          ensure_no_conflicts(rgt, fresh);
        }
        Tree::Op1 { rgt, .. } => {
          ensure_no_conflicts(rgt, fresh);
        }
        Tree::Var { nam } => {
          if let Some(var_num) = var_to_num(nam) {
            *fresh = (*fresh).max(var_num);
          }
        }
        _ => (),
      }
    }
    ensure_no_conflicts(&self.root, &mut fresh);
    for (l, r) in &self.rdex {
      ensure_no_conflicts(l, &mut fresh);
      ensure_no_conflicts(r, &mut fresh);
    }
    let fresh_str = create_var(fresh + 1);

    let fun = core::mem::take(&mut self.root);
    let oth = Tree::Ctr { lab: 0, lft: Box::new(arg), rgt: Box::new(Tree::Var { nam: fresh_str.clone() }) };
    self.root = Tree::Var { nam: fresh_str };
    self.rdex.push((fun, oth));
  }
}

pub(crate) use deref;
