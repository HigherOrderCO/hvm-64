use crate::prelude::*;

/// Turn a string representation of a number, such as '1G' or '400K', into a
/// number.
pub fn parse_abbrev_number<T: TryFrom<u64>>(arg: &str) -> Result<T, String>
where
  <T as TryFrom<u64>>::Error: fmt::Debug,
{
  let (base, scale) = match arg.to_lowercase().chars().last() {
    None => return Err("Mem size argument is empty".to_string()),
    Some('k') => (&arg[0 .. arg.len() - 1], 1u64 << 10),
    Some('m') => (&arg[0 .. arg.len() - 1], 1u64 << 20),
    Some('g') => (&arg[0 .. arg.len() - 1], 1u64 << 30),
    Some('t') => (&arg[0 .. arg.len() - 1], 1u64 << 40),
    Some(_) => (arg, 1),
  };
  let base = base.parse::<u64>().map_err(|e| e.to_string())?;
  Ok((base * scale).try_into().map_err(|e| format!("{:?}", e))?)
}
