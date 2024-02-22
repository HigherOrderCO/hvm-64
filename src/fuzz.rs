//! An 'atomic fuzzer' to exhaustively test an atomic algorithm for correctness.
//!
//! This fuzzer can test an algorithm with every possible ordering of parallel
//! instructions / update propagation.
//!
//! This module implements a subset of the [`std::sync::atomic`] api. To test an
//! algorithm with the fuzzer, it must first be compiled to use this module's
//! atomics, rather than std's. Then, it can be called from within
//! [`Fuzzer::fuzz`].
//!
//! For example, here's a test of the ['nomicon][nomicon-example]'s example of
//! atomic ordering:
//!
//! [nomicon-example]:
//!     https://doc.rust-lang.org/nomicon/atomics.html#hardware-reordering
//!
//! ```
//! # use std::collections::HashSet;
//! use hvmc::fuzz::{Fuzzer, AtomicU64, Ordering};
//!
//! let mut results = HashSet::new();
//! Fuzzer::default().fuzz(|f| {
//!   let x = AtomicU64::new(0);
//!   let y = AtomicU64::new(1);
//!   f.scope(|s| { // use `f.scope` instead of `std::thread::scope`
//!     s.spawn(|| {
//!       y.store(3, Ordering::Relaxed);
//!       x.store(1, Ordering::Relaxed);
//!     });
//!     s.spawn(|| {
//!       if x.load(Ordering::Relaxed) == 1 {
//!         y.store(y.load(Ordering::Relaxed) * 2, Ordering::Relaxed);
//!       }
//!     });
//!   });
//!   results.insert(y.read());
//! });
//!
//! assert_eq!(results, [6, 3, 2].into_iter().collect());
//! ```
//!
//! Note that the atomic types exposed by this module will panic if used outside
//! of a thread spawned by [`Fuzzer::fuzz`].
//!
//! Internally, the fuzzer works by iterating through *decision paths*,
//! sequences of integers indicating which branch is taken at each
//! non-deterministic instruction. This might, for example, correspond to which
//! thread is switched to at a given yield point.
//!
//! The shape of the decision tree is not known ahead of time, and is instead
//! progressively discovered through execution. Thus, when a path is first
//! executed, it may not be the full path -- there may be branches that are not
//! specified. In the extreme case, at the very beginning, the path starts as
//! `[]`, with no branches specified.
//!
//! When a branching point is reached that is not specified in the path, the
//! *last* branch is automatically chosen, and this decision is appended to the
//! path. For example, executing path `[]` might disambiguate it to `[1, 2, 1]`.
//! The last path is chosen because this alleviates the need to store the number
//! of branches at any given point -- instead, branches are selected in
//! decreasing order, so when the index reaches `0`, we know that there are no
//! more branches to explore.
//!
//! By default, fuzzing starts at path `[]`, which will test the full decision
//! tree. Alternatively, one can specify a different path to start at with
//! [`Fuzzer::with_path`] -- this is useful, for example, to debug a failing
//! path late in the tree. (It's important to note, though, that the semantics
//! of the path are very dependent on the specifics of the algorithm, so if the
//! algorithm is changed (particularly, the atomic instructions it executes),
//! old paths may no longer be valid.)

use std::{
  any::Any,
  cell::{OnceCell, RefCell},
  fmt::Debug,
  marker::PhantomData,
  ops::Add,
  sync::{atomic, Arc, Condvar, Mutex},
  thread::{self, Scope, ThreadId},
};

use nohash_hasher::IntMap;

#[repr(transparent)]
pub struct Atomic<T: HasAtomic> {
  value: T::Atomic,
}

impl<T: HasAtomic + Default> Default for Atomic<T> {
  fn default() -> Self {
    Atomic::new(T::default())
  }
}

impl<T: HasAtomic> Atomic<T> {
  pub fn new(value: T) -> Self {
    Atomic { value: T::new_atomic(value) }
  }
  /// Reads the final value of this atomic -- should only be called in a
  /// non-parallel context; i.e., at the end of a test.
  pub fn read(&self) -> T {
    T::load(&self.value)
  }
  fn with<R>(&self, f: impl FnOnce(&Fuzzer, &mut Vec<T>, &mut usize) -> (bool, R)) -> R {
    ThreadContext::with(|ctx| {
      if !ctx.just_started {
        ctx.fuzzer.yield_point();
      }
      ctx.just_started = false;
      let key = &self.value as *const _ as usize;
      let view = ctx.views.entry(key).or_insert_with(|| AtomicView {
        history: ctx
          .fuzzer
          .atomics
          .lock()
          .unwrap()
          .entry(key)
          .or_insert_with(|| Arc::new(Mutex::new(vec![T::load(&self.value)])))
          .clone(),
        index: 0,
      });
      let history: &AtomicHistory<T> = view.history.to_any().downcast_ref().unwrap();
      let mut history = history.lock().unwrap();
      let (changed, r) = f(&ctx.fuzzer, &mut history, &mut view.index);
      if changed {
        ctx.fuzzer.unblock_threads();
      }
      r
    })
  }
  pub fn load(&self, _: Ordering) -> T {
    self.with(|fuzzer, history, index| {
      let delta = fuzzer.decide(history.len() - *index);
      *index += delta;
      let value = history[*index];
      (false, value)
    })
  }
  pub fn store(&self, value: T, _: Ordering) {
    self.with(|_, history, index| {
      *index = history.len();
      history.push(value);
      T::store(&self.value, value);
      (true, ())
    })
  }
  pub fn swap(&self, new: T, _: Ordering) -> T {
    self.with(|_, history, index| {
      *index = history.len();
      let old = *history.last().unwrap();
      history.push(new);
      T::store(&self.value, new);
      (true, old)
    })
  }
  pub fn compare_exchange(&self, expected: T, new: T, _: Ordering, _: Ordering) -> Result<T, T> {
    self.with(|_, history, index| {
      let old = *history.last().unwrap();
      if old == expected {
        *index = history.len();
        history.push(new);
        T::store(&self.value, new);
        (true, Ok(old))
      } else {
        *index = history.len() - 1;
        (false, Err(old))
      }
    })
  }
  pub fn compare_exchange_weak(&self, expected: T, new: T, _: Ordering, _: Ordering) -> Result<T, T> {
    self.with(|fuzzer, history, index| {
      let old = *history.last().unwrap();
      if old == expected && fuzzer.decide(2) == 1 {
        *index = history.len();
        history.push(new);
        T::store(&self.value, new);
        (true, Ok(old))
      } else {
        *index = history.len() - 1;
        (false, Err(old))
      }
    })
  }
  pub fn fetch_add(&self, delta: T, _: Ordering) -> T {
    self.with(|_, history, index| {
      *index = history.len();
      let old = *history.last().unwrap();
      let new = old + delta;
      history.push(new);
      T::store(&self.value, new);
      (true, old)
    })
  }
}

pub fn spin_loop() {
  ThreadContext::with(|ctx| {
    let mut unsynced = ctx
      .views
      .values_mut()
      .map(|x| {
        let l = x.history.len();
        (x, l)
      })
      .filter(|x| x.0.index + 1 < x.1)
      .collect::<Vec<_>>();
    if !unsynced.is_empty() {
      ctx.fuzzer.yield_point();
      let idx = ctx.fuzzer.decide(unsynced.len());
      let unsynced = &mut unsynced[idx];
      let amount = 1 + ctx.fuzzer.decide(unsynced.1 - unsynced.0.index - 1);
      unsynced.0.index += amount;
    } else {
      ctx.fuzzer.block_thread();
      ctx.fuzzer.yield_point();
      ctx.just_started = true;
    }
  })
}

/// One thread's view of an atomic variable -- a reference to the history, and
/// an index into it that denotes the currently propagated value.
struct AtomicView<H: ?Sized> {
  history: Arc<H>,
  index: usize,
}

/// The history of an atomic variable (a vector of the vales it has held).
type AtomicHistory<T> = Mutex<Vec<T>>;

/// Currently, only `Ordering::Relaxed` is supported; other orderings could
/// theoretically be supported, but this has not been implemented, as HVM
/// currently only uses `Relaxed` operations.
pub enum Ordering {
  Relaxed,
}

struct ThreadContext {
  fuzzer: Arc<Fuzzer>,
  views: IntMap<usize, AtomicView<dyn AnyAtomicHistory>>,
  just_started: bool,
}

impl ThreadContext {
  fn init(fuzzer: Arc<Fuzzer>) {
    CONTEXT.with(|ctx| {
      assert!(ctx.get().is_none(), "thread context already initialized");
      ctx.get_or_init(|| RefCell::new(ThreadContext { fuzzer, views: Default::default(), just_started: true }));
    });
  }
  fn with<T>(f: impl FnOnce(&mut ThreadContext) -> T) -> T {
    CONTEXT.with(|ctx| f(&mut ctx.get().expect("cannot use fuzz atomics outside of Fuzzer::fuzz").borrow_mut()))
  }
}

impl<T: ?Sized> Clone for AtomicView<T> {
  fn clone(&self) -> Self {
    AtomicView { history: self.history.clone(), index: self.index }
  }
}

trait AnyAtomicHistory: Any + Send + Sync {
  fn len(&self) -> usize;
  fn to_any(&self) -> &dyn Any;
}

impl<T: 'static + Send> AnyAtomicHistory for AtomicHistory<T> {
  fn len(&self) -> usize {
    self.lock().unwrap().len()
  }
  fn to_any(&self) -> &dyn Any {
    self
  }
}

thread_local! {
  static CONTEXT: OnceCell<RefCell<ThreadContext>> = const { OnceCell::new() };
}

#[derive(Default)]
struct DecisionPath {
  path: Vec<usize>,
  index: usize,
}

impl DecisionPath {
  fn decide(&mut self, options: usize) -> usize {
    if options == 1 {
      return 0;
    }
    if options == 0 {
      panic!("you left me no choice");
    }
    if self.index == self.path.len() {
      self.path.push(options - 1);
    }
    let choice = self.path[self.index];
    self.index += 1;
    choice
  }
  fn next_path(&mut self) -> bool {
    self.index = 0;
    while self.path.last() == Some(&0) || self.path.len() > 100 {
      self.path.pop();
    }
    if let Some(branch) = self.path.last_mut() {
      *branch -= 1;
      true
    } else {
      false
    }
  }
}

#[derive(Default)]
pub struct Fuzzer {
  path: Mutex<DecisionPath>,
  atomics: Mutex<IntMap<usize, Arc<dyn AnyAtomicHistory + Send + Sync>>>,
  current_thread: Mutex<Option<ThreadId>>,
  active_threads: Mutex<Vec<ThreadId>>,
  blocked_threads: Mutex<Vec<ThreadId>>,
  condvar: Condvar,
  main: Option<ThreadId>,
}

impl Fuzzer {
  pub fn with_path(path: Vec<usize>) -> Self {
    Fuzzer { path: Mutex::new(DecisionPath { path, index: 0 }), ..Default::default() }
  }

  pub fn fuzz(mut self, mut f: impl FnMut(&Arc<Fuzzer>) + Send) {
    thread::scope(move |s| {
      s.spawn(move || {
        self.main = Some(thread::current().id());
        let fuzzer = Arc::new(self);
        ThreadContext::init(fuzzer.clone());
        let mut i = 0;
        loop {
          println!("{:6} {:?}", i, &fuzzer.path.lock().unwrap().path);
          fuzzer.atomics.lock().unwrap().clear();
          ThreadContext::with(|ctx| ctx.views.clear());
          f(&fuzzer);
          i += 1;
          if !fuzzer.path.lock().unwrap().next_path() {
            break;
          }
        }
        println!("checked all {} paths", i);
      });
    });
  }

  /// Makes a non-deterministic decision, returning an integer within
  /// `0..options`.
  pub fn decide(&self, options: usize) -> usize {
    self.path.lock().unwrap().decide(options)
  }

  pub fn maybe_swap<T>(&self, a: T, b: T) -> (T, T) {
    if self.decide(2) == 1 { (b, a) } else { (a, b) }
  }

  /// Yields to the "scheduler", potentially switching to another thread.
  pub fn yield_point(&self) {
    if self.switch_thread() {
      self.pause_thread();
    }
  }

  fn unblock_threads(&self) {
    let mut active_threads = self.active_threads.lock().unwrap();
    let mut blocked_threads = self.blocked_threads.lock().unwrap();
    active_threads.extend(blocked_threads.drain(..));
  }

  fn block_thread(&self) {
    let thread_id = thread::current().id();
    let mut active_threads = self.active_threads.lock().unwrap();
    let mut blocked_threads = self.blocked_threads.lock().unwrap();
    let idx = active_threads.iter().position(|x| x == &thread_id).unwrap();
    active_threads.swap_remove(idx);
    blocked_threads.push(thread_id);
  }

  fn switch_thread(&self) -> bool {
    let thread_id = thread::current().id();
    let active_threads = self.active_threads.lock().unwrap();
    if active_threads.is_empty() {
      if self.main != Some(thread_id) {
        panic!("deadlock");
      }
      return false;
    }
    let new_idx = self.decide(active_threads.len());
    let new_thread = active_threads[new_idx];
    if new_thread == thread_id {
      return false;
    }
    let mut current_thread = self.current_thread.lock().unwrap();
    *current_thread = Some(new_thread);
    self.condvar.notify_all();
    true
  }

  fn pause_thread(&self) {
    let thread_id = thread::current().id();
    let mut current_thread = self.current_thread.lock().unwrap();
    while *current_thread != Some(thread_id) {
      current_thread = self.condvar.wait(current_thread).unwrap();
    }
  }

  pub fn scope<'e>(self: &Arc<Self>, f: impl for<'s, 'p> FnOnce(&FuzzScope<'s, 'p, 'e>)) {
    thread::scope(|scope| {
      let scope = FuzzScope { fuzzer: self.clone(), scope, __: PhantomData };
      f(&scope);
      assert_eq!(*self.current_thread.lock().unwrap(), None);
      self.switch_thread();
    });
  }
}

pub struct FuzzScope<'s, 'p: 's, 'e: 'p> {
  fuzzer: Arc<Fuzzer>,
  scope: &'s Scope<'s, 'p>,
  __: PhantomData<&'e mut &'e ()>,
}

impl<'s, 'p: 's, 'e: 's> FuzzScope<'s, 'p, 'e> {
  pub fn spawn<F: FnOnce() + Send + 's>(&self, f: F) {
    let fuzzer = self.fuzzer.clone();
    let ready = Arc::new(atomic::AtomicBool::new(false));
    let views = ThreadContext::with(|ctx| ctx.views.clone());
    self.scope.spawn({
      let ready = ready.clone();
      move || {
        ThreadContext::init(fuzzer.clone());
        ThreadContext::with(|ctx| ctx.views = views);
        let thread_id = thread::current().id();
        fuzzer.active_threads.lock().unwrap().push(thread_id);
        ready.store(true, atomic::Ordering::Relaxed);
        fuzzer.pause_thread();
        f();
        let mut active_threads = fuzzer.active_threads.lock().unwrap();
        let i = active_threads.iter().position(|&t| t == thread_id).unwrap();
        active_threads.swap_remove(i);
        if !active_threads.is_empty() {
          drop(active_threads);
          fuzzer.switch_thread();
        } else {
          *fuzzer.current_thread.lock().unwrap() = None;
        }
      }
    });
    while !ready.load(atomic::Ordering::Relaxed) {
      std::hint::spin_loop()
    }
  }
}

pub trait HasAtomic: 'static + Copy + Eq + Send + Add<Output = Self> + Debug {
  type Atomic;
  fn new_atomic(value: Self) -> Self::Atomic;
  fn load(atomic: &Self::Atomic) -> Self;
  fn store(atomic: &Self::Atomic, value: Self);
}

macro_rules! decl_atomic {
  ($($T:ident: $A:ident),* $(,)?) => {$(
    pub type $A = Atomic<$T>;

    impl HasAtomic for $T {
      type Atomic = atomic::$A;
      fn new_atomic(value: Self) -> Self::Atomic {
        atomic::$A::new(value)
      }
      fn load(atomic: &Self::Atomic) -> Self {
        atomic.load(atomic::Ordering::SeqCst)
      }
      fn store(atomic: &Self::Atomic, value: Self) {
        atomic.store(value, atomic::Ordering::SeqCst)
      }
    }
  )*};
}

decl_atomic! {
  u8: AtomicU8,
  u16: AtomicU16,
  u32: AtomicU32,
  u64: AtomicU64,
  usize: AtomicUsize,
}
