#[macro_export]
macro_rules! specialize {
  (
    $(# $branch_attr:tt)*
    $branch_vis:vis fn $branch:ident$params:tt $(-> $ret:ty)?;
    $impl:ident$args:tt;
    match $value:tt {
      $($pat:pat => $(# $inner_attr:tt)* fn $inner:ident,)*
      $($expr_pat:pat => $expr:expr,)*
    }
  ) => {
    $(# $branch_attr)*
    $branch_vis fn $branch $params $(-> $ret)? {
      match $value {
        $($pat => $inner $args,)*
        $($expr_pat => $expr,)*
      }
    }
    $(
      $(# $inner_attr)*
      fn $inner $params $(-> $ret)? {
        match $value {
          $pat => {},
          _ => unsafe { std::hint::unreachable_unchecked() },
        };
        $impl $args
      }
    )*
  };
}
