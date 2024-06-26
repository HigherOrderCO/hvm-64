/// Guard against stack overflows in recursive functions.
#[cfg(feature = "std")]
pub fn maybe_grow<R>(f: impl FnOnce() -> R) -> R {
  stacker::maybe_grow(1024 * 32, 1024 * 1024, f)
}

#[cfg(not(feature = "std"))]
pub fn maybe_grow<R>(f: impl FnOnce() -> R) -> R {
  f()
}
