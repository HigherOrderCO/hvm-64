use crate::{
  ast::{Net, Tree},
  run::Rewrites,
};
use std::time::Duration;

/// Creates a variable uniquely identified by `id`.
pub(crate) fn create_var(mut id: usize) -> String {
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
///
/// Returns None when the provided string is not an output of
/// `create_var`.
pub(crate) fn var_to_num(s: &str) -> Option<usize> {
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
#[allow(non_local_definitions)]
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

pub(crate) use deref;

impl Tree {
  /// Increases `fresh` until `create_var(*fresh)` does not conflict
  /// with a [`Tree::Var`]  in `tree`
  ///
  /// This function can be called multiple times with many trees to
  /// ensure that `fresh` does not conflict with any of them.
  pub(crate) fn ensure_no_conflicts(&self, fresh: &mut usize) {
    match self {
      Tree::Var { nam } => {
        if let Some(var_num) = var_to_num(nam) {
          *fresh = (*fresh).max(var_num);
        }
      }
      // Recurse on children
      Tree::Ctr { lft, rgt, .. } | Tree::Op2 { lft, rgt, .. } | Tree::Mat { sel: lft, ret: rgt } => {
        lft.ensure_no_conflicts(fresh);
        rgt.ensure_no_conflicts(fresh);
      }
      Tree::Op1 { rgt, .. } => {
        rgt.ensure_no_conflicts(fresh);
      }
      Tree::Era | Tree::Num { .. } | Tree::Ref { .. } => {}
    }
  }
}

impl Net {
  pub(crate) fn ensure_no_conflicts(&self, fresh: &mut usize) {
    self.root.ensure_no_conflicts(fresh);
    for (a, b) in &self.rdex {
      a.ensure_no_conflicts(fresh);
      b.ensure_no_conflicts(fresh);
    }
  }
  /// Transforms the net `x & ...` into `y & x ~ (arg y) & ...`
  ///
  /// The result is equivalent a λ-calculus application. Thus,
  /// if the net is a λ-calculus term, then this function will
  /// apply an argument to it.
  pub fn apply_tree(&mut self, arg: Tree) {
    let mut fresh = 0usize;
    self.ensure_no_conflicts(&mut fresh);
    arg.ensure_no_conflicts(&mut fresh);

    let fresh_str = create_var(fresh + 1);

    let fun = core::mem::take(&mut self.root);
    let app = Tree::Ctr { lab: 0, lft: Box::new(arg), rgt: Box::new(Tree::Var { nam: fresh_str.clone() }) };
    self.root = Tree::Var { nam: fresh_str };
    self.rdex.push((fun, app));
  }
}

pub fn show_rewrites(rwts: &Rewrites) -> String {
  format!(
    "{}{}{}{}{}{}",
    format_args!("RWTS   : {:>15}\n", pretty_num(rwts.total())),
    format_args!("- ANNI : {:>15}\n", pretty_num(rwts.anni)),
    format_args!("- COMM : {:>15}\n", pretty_num(rwts.comm)),
    format_args!("- ERAS : {:>15}\n", pretty_num(rwts.eras)),
    format_args!("- DREF : {:>15}\n", pretty_num(rwts.dref)),
    format_args!("- OPER : {:>15}\n", pretty_num(rwts.oper)),
  )
}

pub fn show_stats(rwts: &Rewrites, elapsed: Duration) -> String {
  format!(
    "{}{}{}",
    show_rewrites(rwts),
    format_args!("TIME   : {:.3?}\n", elapsed),
    format_args!("RPS    : {:.3} M\n", (rwts.total() as f64) / (elapsed.as_millis() as f64) / 1000.0),
  )
}

fn pretty_num(n: u64) -> String {
  n.to_string()
    .as_bytes()
    .rchunks(3)
    .rev()
    .map(|x| std::str::from_utf8(x).unwrap())
    .flat_map(|x| ["_", x])
    .skip(1)
    .collect()
}
