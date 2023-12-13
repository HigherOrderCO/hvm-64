use std::{
  fmt::{self, Formatter, Write},
  io::stdout,
  sync::{
    atomic::{AtomicU64, AtomicUsize, Ordering},
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
  #[inline(always)]
  pub fn new() -> Self {
    Tracer
  }
  #[inline(always)]
  pub fn sync_nonce(&mut self) {}
  #[inline(always)]
  pub fn set_tid(&mut self, tid: usize) {}
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
  pub fn sync_nonce(&mut self) {
    self.0.sync_nonce()
  }
  #[inline(always)]
  pub fn trace(&mut self, point: &'static TracePoint, args: &[Ptr]) {
    self.0.trace(point, args)
  }
  #[inline(always)]
  pub fn set_tid(&mut self, tid: usize) {
    self.0.set_tid(tid)
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

const TRACE_SIZE: usize = 1 << 20;

static TRACE_NONCE: AtomicU64 = AtomicU64::new(1);

struct TracerInner {
  data: Box<[u64; TRACE_SIZE]>,
  cursor: usize,
  safe: AtomicUsize,
  nonce: u64,
  tid: usize,
}

#[derive(Debug)]
struct GlobalTracerRef(*const TracerInner);
unsafe impl Send for GlobalTracerRef {}

static ACTIVE_TRACERS: Mutex<Vec<Option<GlobalTracerRef>>> = Mutex::new(Vec::new());

struct OwnedTracer {
  inner: Box<TracerInner>,
  id: usize,
}

impl OwnedTracer {
  fn new() -> Self {
    let inner =
      Box::new(TracerInner { data: Box::new([0; TRACE_SIZE]), cursor: 0, safe: AtomicUsize::new(0), nonce: 0, tid: 0 });

    let mut active_tracers = ACTIVE_TRACERS.lock().unwrap();
    let id = active_tracers.len();
    let ptr = (&*inner) as *const _;
    active_tracers.push(Some(GlobalTracerRef(ptr)));
    let mut tracer = OwnedTracer { inner, id };
    tracer.sync_nonce();
    tracer
  }
  fn sync_nonce(&mut self) {
    self.inner.nonce = TRACE_NONCE.fetch_add(1, Ordering::Relaxed);
  }
  fn trace(&mut self, point: &'static TracePoint, args: &[Ptr]) {
    let nonce = self.inner.nonce;
    assert!(args.len() == point.args.len());
    for arg in args.iter().rev() {
      self.write_word(arg.0);
    }
    self.write_word(point as *const _ as u64);
    self.write_word(nonce);
    self.inner.safe.store(self.inner.cursor, Ordering::Release);
  }
  fn write_word(&mut self, word: u64) {
    self.inner.data[self.inner.cursor] = word;
    unsafe {
      std::ptr::write_volatile(&mut self.inner.cursor as *mut _, (self.inner.cursor + 1) % TRACE_SIZE);
    }
  }
  fn set_tid(&mut self, tid: usize) {
    self.inner.tid = tid;
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
    let safe = tracer.safe.load(Ordering::Acquire);
    dbg!(tracer.cursor, safe, tracer.data.last());
    TracerReader { tracer, cursor: (TRACE_SIZE + safe - 1) % TRACE_SIZE, id }
  }
  fn read_entry(&mut self, f: &mut impl Write) -> Option<fmt::Result> {
    let nonce = self.read_word()?;
    let point = self.read_word()?;
    dbg!(self.id, self.cursor, self.tracer.cursor, point as *const u8);
    if point == 0 {
      self.cursor = self.tracer.cursor;
      return None;
    }
    let point = unsafe { &*(point as *const TracePoint) };
    if self.remaining() < point.args.len() {
      self.cursor = self.tracer.cursor;
      return None;
    }
    Some((|| {
      writeln!(
        f,
        "{:02x}t{:02x} {}{}{} [{}:{}] #{}",
        self.id,
        self.tracer.tid,
        point.func.strip_suffix("::__").unwrap_or(point.func).rsplit("::").next().unwrap(),
        if point.str.is_empty() { "" } else { " " },
        point.str,
        point.file,
        point.line,
        nonce,
      )?;
      let max_len = point.args.iter().map(|x| x.len()).max().unwrap_or(0);
      for arg in point.args {
        let ptr = Ptr(self.read_word().unwrap());
        for _ in 0 .. (8 + max_len - arg.len()) {
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
pub unsafe fn _read_traces(limit: usize) {
  let active_tracers = &*ACTIVE_TRACERS.lock().unwrap();
  let mut readers = active_tracers
    .iter()
    .enumerate()
    .filter_map(|(i, t)| Some(TracerReader::new(unsafe { &*t.as_ref()?.0 }, i)))
    .collect::<Vec<_>>();
  let mut out = String::new();
  for _ in 0 .. limit {
    let Some((1 .., r)) = readers.iter_mut().filter_map(|x| Some((x.peek_word()?, x))).max_by_key(|x| x.0) else {
      break;
    };
    r.read_entry(&mut out);
    eprintln!("{}", out);
    out.clear();
  }
  eprintln!("{}", out.len());
}

pub fn _reset_traces() {
  ACTIVE_TRACERS.lock().unwrap().clear();
  TRACE_NONCE.store(1, Ordering::Relaxed);
}
