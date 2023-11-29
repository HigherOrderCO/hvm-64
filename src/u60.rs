// Implements u48: 48-bit unsigned integers using u64 and u128

type U60 = u64;

#[inline(always)]
pub fn new(a: u64) -> U60 {
  return a & 0xFFF_FFFF_FFFF_FFFF;
}

#[inline(always)]
pub fn val(a: u64) -> U60 {
  return a;
}

#[inline(always)]
pub fn add(a: U60, b: U60) -> U60 {
  return new(a + b);
}

#[inline(always)]
pub fn sub(a: U60, b: U60) -> U60 {
  return if a >= b { a - b } else { 0x1000000000000000 - (b - a) };
}

#[inline(always)]
pub fn mul(a: U60, b: U60) -> U60 {
  return new((a as u128 * b as u128) as u64);
}

#[inline(always)]
pub fn div(a: U60, b: U60) -> U60 {
  return a / b;
}

#[inline(always)]
pub fn rem(a: U60, b: U60) -> U60 {
  return a % b;
}

#[inline(always)]
pub fn and(a: U60, b: U60) -> U60 {
  return a & b;
}

#[inline(always)]
pub fn or(a: U60, b: U60) -> U60 {
  return a | b;
}

#[inline(always)]
pub fn xor(a: U60, b: U60) -> U60 {
  return a ^ b;
}

#[inline(always)]
pub fn lsh(a: U60, b: U60) -> U60 {
  return new(a << b);
}

#[inline(always)]
pub fn rsh(a: U60, b: U60) -> U60 {
  return a >> b;
}

#[inline(always)]
pub fn lt(a: U60, b: U60) -> U60 {
  return if a < b { 1 } else { 0 };
}

#[inline(always)]
pub fn gt(a: U60, b: U60) -> U60 {
  return if a > b { 1 } else { 0 };
}

#[inline(always)]
pub fn lte(a: U60, b: U60) -> U60 {
  return if a <= b { 1 } else { 0 };
}

#[inline(always)]
pub fn gte(a: U60, b: U60) -> U60 {
  return if a >= b { 1 } else { 0 };
}

#[inline(always)]
pub fn eq(a: U60, b: U60) -> U60 {
  return if a == b { 1 } else { 0 };
}

#[inline(always)]
pub fn ne(a: U60, b: U60) -> U60 {
  return if a != b { 1 } else { 0 };
}

#[inline(always)]
pub fn min(a: U60, b: U60) -> U60 {
  return if a < b { a } else { b };
}

#[inline(always)]
pub fn max(a: U60, b: U60) -> U60 {
  return if a > b { a } else { b };
}

#[inline(always)]
pub fn not(a: U60) -> U60 {
  return !a & 0xFFF_FFFF_FFFF_FFFF;
}

#[inline(always)]
pub fn show(a: U60) -> String {
  return format!("{}", a);
}
