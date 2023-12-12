use std::{
  fmt::{self, Formatter, Write},
  io::stdout,
  sync::{
    atomic::{AtomicU64, Ordering},
    Mutex,
  },
};

use crate::run::Ptr;

#[cfg(not(feature = "trace"))]
#[macro_export]
macro_rules! trace {
  ($tracer:expr, $str:lit $(, $x:expr)* $(,)?) => {
    if false { let _: (&mut $crate::trace::Tracer, [$crate::run::Ptr]) = (&mut $tracer, &[$($x as Ptr),*]); }
  };
  ($tracer:expr $(, $x:expr)* $(,)?) => {
    if false { let _: (&mut $crate::trace::Tracer, [$crate::run::Ptr]) = (&mut $tracer, &[$($x as Ptr),*]); }
  };
}

#[cfg(not(feature = "trace"))]
pub struct Tracer;

#[cfg(not(feature = "trace"))]
impl Tracer {
  pub fn new() -> Self {
    Tracer
  }
}

#[cfg(feature = "trace")]
#[macro_export]
macro_rules! trace {
  ($tracer:expr, $str:literal $(, $x:expr)* $(,)?) => {{
    struct __;
    static POINT: $crate::trace::TracePoint = $crate::trace::TracePoint {
      func: &std::any::type_name::<__>(),
      file: file!(),
      line: line!(),
      str: $str,
      args: &[$(stringify!($x)),*],
    };
    $tracer.trace(&POINT, &[$($x),*]);
  }};
  ($tracer:expr $(, $x:expr)* $(,)?) => {
    trace!($tracer, "", $($x),*)
  };
}

#[cfg(feature = "trace")]
pub struct Tracer(OwnedTracer);

#[cfg(feature = "trace")]
impl Tracer {
  #[inline(always)]
  pub fn new() -> Tracer {
    Tracer(OwnedTracer::new())
  }
  #[inline(always)]
  pub fn trace(&mut self, point: &'static TracePoint, args: &[Ptr]) {
    self.0.trace(point, args)
  }
}

#[doc(hidden)]
pub struct TracePoint {
  pub func: &'static str,
  pub file: &'static str,
  pub line: u32,
  pub str: &'static str,
  pub args: &'static [&'static str],
}

const TRACE_SIZE: usize = 1 << 16;

static TRACE_NONCE: AtomicU64 = AtomicU64::new(0);

struct TracerInner {
  data: Box<[u64; TRACE_SIZE]>,
  cursor: usize,
  nonce: u64,
}

struct GlobalTracerRef(*const TracerInner);
unsafe impl Send for GlobalTracerRef {}

static ACTIVE_TRACERS: Mutex<Vec<Option<GlobalTracerRef>>> = Mutex::new(Vec::new());

struct OwnedTracer {
  inner: Box<TracerInner>,
  id: usize,
}

impl OwnedTracer {
  fn new() -> Self {
    let inner = Box::new(TracerInner { data: Box::new([0; TRACE_SIZE]), cursor: 0, nonce: 0 });
    let mut active_tracers = ACTIVE_TRACERS.lock().unwrap();
    let id = active_tracers.len();
    active_tracers.push(Some(GlobalTracerRef((&*inner) as *const _)));
    OwnedTracer { inner, id }
  }
  fn trace(&mut self, point: &'static TracePoint, args: &[Ptr]) {
    let nonce = TRACE_NONCE.fetch_add(1, Ordering::Relaxed);
    // let nonce = self.inner.nonce;
    self.inner.nonce += 1;
    for arg in args {
      self.write_word(arg.0);
    }
    self.write_word(point as *const _ as u64);
    self.write_word(nonce);
  }
  fn write_word(&mut self, word: u64) {
    self.inner.data[self.inner.cursor] = word;
    self.inner.cursor = (self.inner.cursor + 1) % TRACE_SIZE;
  }
}

impl Drop for OwnedTracer {
  fn drop(&mut self) {
    let mut active_tracers = ACTIVE_TRACERS.lock().unwrap();
    active_tracers[self.id] = None;
  }
}

struct TracerReader<'a> {
  tracer: &'a TracerInner,
  cursor: usize,
  id: usize,
}

impl<'a> TracerReader<'a> {
  fn new(tracer: &'a TracerInner, id: usize) -> Self {
    TracerReader { tracer, cursor: (TRACE_SIZE + tracer.cursor - 1) % TRACE_SIZE, id }
  }
  fn read_entry(&mut self, f: &mut impl Write) -> Option<fmt::Result> {
    let _nonce = self.read_word()?;
    let point = unsafe { &*(self.read_word()? as *const TracePoint) };
    if self.remaining() < point.args.len() {
      return None;
    }
    Some((|| {
      writeln!(
        f,
        "{:04x} {}{}{} [{}:{}]",
        self.id,
        point.func.strip_suffix("::__").unwrap_or(point.func).rsplit("::").next().unwrap(),
        if point.str.is_empty() { "" } else { " " },
        point.str,
        point.file,
        point.line
      )?;
      let max_len = point.args.iter().map(|x| x.len()).max().unwrap_or(0);
      for arg in point.args {
        let ptr = Ptr(self.read_word().unwrap());
        for _ in 0 .. (6 + max_len - arg.len()) {
          f.write_char(' ')?;
        }
        writeln!(f, "{}: {:?}", arg, ptr)?;
      }
      Ok(())
    })())
  }
  fn peek_word(&self) -> Option<u64> {
    if self.cursor == self.tracer.cursor { None } else { Some(self.tracer.data[self.cursor]) }
  }
  fn read_word(&mut self) -> Option<u64> {
    let word = self.peek_word()?;
    self.cursor = (self.cursor + TRACE_SIZE - 1) % TRACE_SIZE;
    Some(word)
  }
  fn remaining(&self) -> usize {
    (self.cursor + TRACE_SIZE - self.tracer.cursor) % TRACE_SIZE
  }
}

#[no_mangle]
pub fn _read_traces(limit: usize) {
  let active_tracers = &mut *ACTIVE_TRACERS.lock().unwrap();
  let mut readers = active_tracers
    .iter_mut()
    .enumerate()
    .filter_map(|(i, t)| Some(TracerReader::new(unsafe { &*t.as_ref()?.0 }, i)))
    .collect::<Vec<_>>();
  let mut out = String::new();
  for _ in 0 .. limit {
    let Some((_, r)) = readers.iter_mut().map(|x| (x.peek_word(), x)).max_by_key(|x| x.0) else { break };
    r.read_entry(&mut out);
  }
  print!("{}", out);
}

pub fn _reset_traces() {
  ACTIVE_TRACERS.lock().unwrap().clear()
}
