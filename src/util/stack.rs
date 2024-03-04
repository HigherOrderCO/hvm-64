/// Function that adds a guard against stack overflows in recursive functions.
pub fn maybe_grow<R>(f: impl FnOnce() -> R) -> R {
  stacker::maybe_grow(1024 * 32, 1024 * 1024, f)
}
