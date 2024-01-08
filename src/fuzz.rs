use std::{
  any::Any,
  cell::{OnceCell, RefCell},
  marker::PhantomData,
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
  pub fn read(self) -> T {
    T::load(&self.value)
  }
  fn with<R>(&self, yield_point: bool, f: impl FnOnce(&Fuzzer, &mut Vec<T>, &mut usize) -> R) -> R {
    ThreadContext::with(|ctx| {
      if yield_point {
        ctx.fuzzer.yield_point()
      }
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
      let history: &AtomicHistory<T> = view.history.downcast_ref().unwrap();
      let mut history = history.lock().unwrap();
      let r = f(&ctx.fuzzer, &mut history, &mut view.index);
      r
    })
  }
  pub fn load(&self, _: Ordering) -> T {
    self.with(true, |fuzzer, history, index| {
      *index += fuzzer.decide(history.len() - *index);
      history[*index]
    })
  }
  pub fn store(&self, value: T, _: Ordering) {
    self.with(true, |_, history, index| {
      *index = history.len();
      history.push(value);
      T::store(&self.value, value);
    })
  }
  pub fn swap(&self, value: T, _: Ordering) -> T {
    self.with(true, |_, history, index| {
      *index = history.len();
      let old = *history.last().unwrap();
      history.push(value);
      T::store(&self.value, value);
      old
    })
  }
  pub fn compare_exchange(&self, expected: T, value: T, _: Ordering, _: Ordering) -> Result<T, T> {
    self.with(true, |_, history, index| {
      let old = *history.last().unwrap();
      if old == expected {
        *index = history.len();
        history.push(value);
        T::store(&self.value, value);
        Ok(old)
      } else {
        *index = history.len() - 1;
        Err(old)
      }
    })
  }
  pub fn compare_exchange_weak(&self, expected: T, value: T, _: Ordering, _: Ordering) -> Result<T, T> {
    self.with(true, |fuzzer, history, index| {
      let old = *history.last().unwrap();
      if old == expected && fuzzer.decide(2) == 1 {
        *index = history.len();
        history.push(value);
        T::store(&self.value, value);
        Ok(old)
      } else {
        *index = history.len() - 1;
        Err(old)
      }
    })
  }
}

struct AtomicView<H: ?Sized> {
  history: Arc<H>,
  index: usize,
}

type AtomicHistory<T> = Mutex<Vec<T>>;

pub enum Ordering {
  Relaxed,
}

struct ThreadContext {
  fuzzer: Arc<Fuzzer>,
  views: IntMap<usize, AtomicView<dyn Any>>,
}

impl ThreadContext {
  fn init(fuzzer: Arc<Fuzzer>) {
    CONTEXT.with(|ctx| {
      assert!(ctx.get().is_none(), "thread context already initialized");
      ctx.get_or_init(|| RefCell::new(ThreadContext { fuzzer, views: Default::default() }));
    });
  }
  fn with<T>(f: impl FnOnce(&mut ThreadContext) -> T) -> T {
    CONTEXT.with(|ctx| f(&mut ctx.get().expect("cannot use fuzz atomics outside of fuzz::fuzz").borrow_mut()))
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
  fn decide(&mut self, choices: usize) -> usize {
    if choices == 1 {
      return 0;
    }
    if self.index == self.path.len() {
      self.path.push(choices - 1);
    }
    let choice = self.path[self.index];
    self.index += 1;
    choice
  }
  fn next_path(&mut self) -> bool {
    self.index = 0;
    while self.path.last() == Some(&0) {
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
  atomics: Mutex<IntMap<usize, Arc<dyn Any + Send + Sync>>>,
  current_thread: Mutex<Option<ThreadId>>,
  active_threads: Mutex<Vec<ThreadId>>,
  condvar: Condvar,
}

impl Fuzzer {
  pub fn fuzz(mut f: impl FnMut(&Arc<Fuzzer>) + Send) {
    thread::scope(move |s| {
      s.spawn(move || {
        let fuzzer = Arc::new(Fuzzer::default());
        ThreadContext::init(fuzzer.clone());
        loop {
          fuzzer.atomics.lock().unwrap().clear();
          f(&fuzzer);
          if !fuzzer.path.lock().unwrap().next_path() {
            break;
          }
        }
      });
    });
  }

  pub fn decide(&self, options: usize) -> usize {
    self.path.lock().unwrap().decide(options)
  }
  pub fn yield_point(&self) {
    self.switch_thread();
    self.block_thread();
  }

  fn switch_thread(&self) {
    let thread_id = thread::current().id();
    let active_threads = self.active_threads.lock().unwrap();
    let new_idx = self.decide(active_threads.len());
    let new_thread = active_threads[new_idx];
    if new_thread == thread_id {
      return;
    }
    let mut current_thread = self.current_thread.lock().unwrap();
    *current_thread = Some(new_thread);
    self.condvar.notify_all();
  }

  fn block_thread(&self) {
    let thread_id = thread::current().id();
    let mut current_thread = self.current_thread.lock().unwrap();
    while *current_thread != Some(thread_id) {
      current_thread = self.condvar.wait(current_thread).unwrap();
    }
  }

  pub fn scope<'e>(self: &Arc<Self>, f: impl for<'s, 'p> FnOnce(&FuzzScope<'s, 'p, 'e>)) {
    let pending = &atomic::AtomicUsize::new(0);
    thread::scope(|scope| {
      let scope = FuzzScope { fuzzer: self.clone(), scope, pending, __: PhantomData };
      f(&scope);
      while scope.pending.load(std::sync::atomic::Ordering::Relaxed) != 0 {}
      assert_eq!(*self.current_thread.lock().unwrap(), None);
      self.switch_thread();
    });
  }
}

pub struct FuzzScope<'s, 'p: 's, 'e: 'p> {
  fuzzer: Arc<Fuzzer>,
  scope: &'s Scope<'s, 'p>,
  pending: &'p std::sync::atomic::AtomicUsize,
  __: PhantomData<&'e mut &'e ()>,
}

impl<'s, 'p: 's, 'e: 's> FuzzScope<'s, 'p, 'e> {
  pub fn spawn<F: FnOnce() + Send + 's>(&self, f: F) {
    self.pending.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let pending = self.pending;
    let fuzzer = self.fuzzer.clone();
    let ready = Arc::new(std::sync::atomic::AtomicBool::new(false));
    self.scope.spawn(move || {
      ThreadContext::init(fuzzer.clone());
      let thread_id = thread::current().id();
      fuzzer.active_threads.lock().unwrap().push(thread_id);
      ready.store(true, std::sync::atomic::Ordering::Relaxed);
      pending.fetch_sub(1, std::sync::atomic::Ordering::Relaxed);
      fuzzer.block_thread();
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
    });
  }
}

pub trait HasAtomic: 'static + Copy + Eq + Send {
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
        atomic.load(atomic::Ordering::Relaxed)
      }
      fn store(atomic: &Self::Atomic, value: Self) {
        atomic.store(value, atomic::Ordering::Relaxed)
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
