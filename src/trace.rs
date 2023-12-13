use std::{
  cell::UnsafeCell,
  fmt::{self, Formatter, Write},
  io::stdout,
  sync::{
    atomic::{AtomicBool, AtomicU64, AtomicU8, AtomicUsize, Ordering},
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
pub struct Tracer(TraceWriter);

#[cfg(feature = "trace")]
impl Tracer {
  #[inline(always)]
  pub fn new() -> Tracer {
    Tracer(TraceWriter::new())
  }
  #[inline(always)]
  pub fn sync_nonce(&mut self) {
    self.0.sync()
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

const TRACE_SIZE: usize = 1 << 22;

static TRACE_NONCE: AtomicU64 = AtomicU64::new(1);

struct TraceLock {
  locked: AtomicBool,
  data: UnsafeCell<TraceData>,
}

struct TraceData {
  tid: usize,
  cursor: usize,
  data: Box<[u64; TRACE_SIZE]>,
}

impl TraceData {
  fn write_word(&mut self, word: u64) {
    self.data[self.cursor] = word;
    self.cursor = (self.cursor + 1) % TRACE_SIZE;
  }
}

#[derive(Debug)]
struct GlobalTracerRef(*const TraceData);
unsafe impl Send for GlobalTracerRef {}

static ACTIVE_TRACERS: Mutex<Vec<Box<TraceLock>>> = Mutex::new(Vec::new());

struct TraceWriter {
  lock: &'static TraceLock,
  nonce: u64,
}

unsafe impl Send for TraceWriter {}

impl TraceWriter {
  fn new() -> Self {
    let boxed = Box::new(TraceLock {
      locked: AtomicBool::new(false),
      data: UnsafeCell::new(TraceData { tid: 0, cursor: 0, data: Box::new([0; TRACE_SIZE]) }),
    });
    let lock = unsafe { &*(&*boxed as *const _) };
    let mut active_tracers = ACTIVE_TRACERS.lock().unwrap();
    active_tracers.push(boxed);
    TraceWriter { lock, nonce: TRACE_NONCE.fetch_add(1, Ordering::Relaxed) }
  }
  fn sync(&mut self) {
    self.nonce = TRACE_NONCE.fetch_add(1, Ordering::Relaxed);
  }
  fn acquire(&self, cb: impl FnOnce(&mut TraceData)) {
    while self.lock.locked.compare_exchange_weak(false, true, Ordering::Relaxed, Ordering::Relaxed).is_err() {
      std::hint::spin_loop();
    }
    cb(unsafe { &mut *self.lock.data.get() });
    self.lock.locked.store(false, Ordering::Release);
  }
  fn trace(&self, point: &'static TracePoint, args: &[Ptr]) {
    self.acquire(|data| {
      let nonce = self.nonce;
      assert!(args.len() == point.args.len());
      for arg in args.iter().rev() {
        data.write_word(arg.0);
      }
      data.write_word(point as *const _ as u64);
      data.write_word(nonce);
    })
  }
  fn set_tid(&self, tid: usize) {
    self.acquire(|data| data.tid = tid);
  }
}

struct TraceReader<'a> {
  data: &'a TraceData,
  cursor: usize,
  id: usize,
}

impl<'a> TraceReader<'a> {
  fn new(data: &'a TraceData, id: usize) -> Self {
    TraceReader { data, cursor: (TRACE_SIZE + data.cursor - 1) % TRACE_SIZE, id }
  }
  fn read_entry(&mut self, f: &mut impl Write) -> Option<fmt::Result> {
    let nonce = self.read_word()?;
    let point = self.read_word()?;
    if point == 0 {
      self.cursor = self.data.cursor;
      return None;
    }
    let point = unsafe { &*(point as *const TracePoint) };
    if self.remaining() < point.args.len() {
      self.cursor = self.data.cursor;
      return None;
    }
    Some((|| {
      writeln!(
        f,
        "{:02x}t{:02x} {}{}{} [{}:{}] #{}",
        self.id,
        self.data.tid,
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
    if self.cursor == self.data.cursor { None } else { Some(self.data.data[self.cursor]) }
  }
  fn read_word(&mut self) -> Option<u64> {
    let word = self.peek_word()?;
    self.cursor = (self.cursor + TRACE_SIZE - 1) % TRACE_SIZE;
    Some(word)
  }
  fn remaining(&self) -> usize {
    (self.cursor + TRACE_SIZE - self.data.cursor) % TRACE_SIZE
  }
}

#[no_mangle]
pub fn _read_traces(limit: usize) {
  let active_tracers = &*ACTIVE_TRACERS.lock().unwrap();
  let mut readers = active_tracers
    .iter()
    .enumerate()
    .map(|(i, t)| {
      while t.locked.compare_exchange_weak(false, true, Ordering::Acquire, Ordering::Relaxed).is_err() {
        std::hint::spin_loop();
      }
      TraceReader::new(unsafe { &*t.data.get() }, i)
    })
    .collect::<Vec<_>>();
  let mut out = String::new();
  for _ in 0 .. limit {
    let Some((1 .., r)) = readers.iter_mut().filter_map(|x| Some((x.peek_word()?, x))).max_by_key(|x| x.0) else {
      break;
    };
    r.read_entry(&mut out);
  }
  eprintln!("{}", out);
  active_tracers.iter().for_each(|t| t.locked.store(false, Ordering::Relaxed));
}

pub unsafe fn _reset_traces() {
  ACTIVE_TRACERS.lock().unwrap().clear();
  TRACE_NONCE.store(1, Ordering::Relaxed);
}
