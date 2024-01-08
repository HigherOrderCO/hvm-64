use std::{
  cell::{OnceCell, RefCell},
  collections::HashMap,
  sync::{Arc, Condvar, Mutex, OnceLock},
  thread::{self, Scope, ThreadId},
};

static ATOMIC_ID: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

#[repr(transparent)]
pub struct AtomicU64 {
  id: u64,
}

impl AtomicU64 {
  pub fn new(value: u64) -> Self {
    let id = ATOMIC_ID.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    BACKING.get_or_init(Default::default).lock().unwrap().insert(id, Arc::new(Mutex::new(vec![value])));
    AtomicU64 { id }
  }
  pub fn final_value(self) -> u64 {
    *BACKING.get().unwrap().lock().unwrap().get(&self.id).unwrap().lock().unwrap().last().unwrap()
  }
  fn with<T>(&self, f: impl FnOnce(&mut AtomicView) -> T) -> T {
    VIEWS.with(|views| {
      let mut views = views.borrow_mut();
      f(&mut views.entry(self.id).or_insert_with(|| AtomicView {
        backing: BACKING.get().unwrap().lock().unwrap().get(&self.id).unwrap().clone(),
        current: 0,
      }))
    })
  }
  pub fn load(&self, _: Ordering) -> u64 {
    yield_point();
    self.with(|view| {
      let backing = view.backing.lock().unwrap();
      view.current += decide(backing.len() - view.current);
      backing[view.current]
    })
  }
  pub fn store(&self, value: u64, _: Ordering) {
    yield_point();
    self.with(|view| {
      let mut backing = view.backing.lock().unwrap();
      view.current = backing.len();
      backing.push(value);
    })
  }
  pub fn swap(&self, value: u64, _: Ordering) -> u64 {
    yield_point();
    self.with(|view| {
      let mut backing = view.backing.lock().unwrap();
      view.current = backing.len();
      let old = *backing.last().unwrap();
      backing.push(value);
      old
    })
  }
  pub fn compare_exchange(&self, expected: u64, value: u64, _: Ordering, _: Ordering) -> Result<u64, u64> {
    yield_point();
    self.with(|view| {
      let mut backing = view.backing.lock().unwrap();
      let old = *backing.last().unwrap();
      if old == expected {
        view.current = backing.len();
        backing.push(value);
        Ok(old)
      } else {
        view.current = backing.len() - 1;
        Err(old)
      }
    })
  }
  pub fn compare_exchange_weak(&self, expected: u64, value: u64, _: Ordering, _: Ordering) -> Result<u64, u64> {
    yield_point();
    self.with(|view| {
      let mut backing = view.backing.lock().unwrap();
      let old = *backing.last().unwrap();
      if old == expected && decide(2) == 1 {
        view.current = backing.len();
        backing.push(value);
        Ok(old)
      } else {
        view.current = backing.len() - 1;
        Err(old)
      }
    })
  }
}

struct AtomicView {
  backing: AtomicBacking,
  current: usize,
}

type AtomicBacking = Arc<Mutex<Vec<u64>>>;

pub enum Ordering {
  Relaxed,
}

thread_local! {
  static COORDINATOR: OnceCell<Arc<Coordinator>> = const { OnceCell::new() };
}

static BACKING: OnceLock<Mutex<HashMap<u64, AtomicBacking>>> = OnceLock::new();

thread_local! {
  static VIEWS: RefCell<HashMap<u64, AtomicView>> = RefCell::new(HashMap::new());
}

#[derive(Default)]
pub struct Coordinator {
  path: Mutex<(Vec<usize>, usize)>,
  current_thread: Mutex<Option<ThreadId>>,
  active_threads: Mutex<Vec<ThreadId>>,
  condvar: Condvar,
  pending: std::sync::atomic::AtomicUsize,
}

impl Coordinator {
  fn decide(&self, options: usize) -> usize {
    let mut path = self.path.lock().unwrap();
    if path.1 == path.0.len() {
      path.0.push(options - 1);
    }
    let choice = options - 1 - path.0[path.1];
    path.1 += 1;
    choice
  }
  fn yield_point(&self) {
    self.switch();
    self.wait();
  }
  fn switch(&self) {
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
  fn wait(&self) {
    let thread_id = thread::current().id();
    let mut current_thread = self.current_thread.lock().unwrap();
    while *current_thread != Some(thread_id) {
      current_thread = self.condvar.wait(current_thread).unwrap();
    }
  }
  pub fn spawn<'s, 'e: 's, F: FnOnce() + Send + 's>(self: &Arc<Self>, scope: &'s Scope<'s, 'e>, f: F) {
    self.pending.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let coord = self.clone();
    let ready = Arc::new(std::sync::atomic::AtomicBool::new(false));
    scope.spawn(move || {
      COORDINATOR.with(|c| {
        c.get_or_init(|| coord.clone());
      });
      let thread_id = thread::current().id();
      coord.active_threads.lock().unwrap().push(thread_id);
      ready.store(true, std::sync::atomic::Ordering::Relaxed);
      coord.pending.fetch_sub(1, std::sync::atomic::Ordering::Relaxed);
      coord.wait();
      f();
      let mut active_threads = coord.active_threads.lock().unwrap();
      let i = active_threads.iter().position(|&t| t == thread_id).unwrap();
      active_threads.swap_remove(i);
      if !active_threads.is_empty() {
        drop(active_threads);
        coord.switch();
      } else {
        *coord.current_thread.lock().unwrap() = None;
      }
    });
  }
  pub fn start(&self) {
    while self.pending.load(std::sync::atomic::Ordering::Relaxed) != 0 {}
    assert_eq!(*self.current_thread.lock().unwrap(), None);
    self.switch();
  }
}

pub fn fuzz(f: impl Fn(&Arc<Coordinator>)) {
  let coord = Arc::new(Coordinator::default());
  loop {
    f(&coord);
    let mut path = coord.path.lock().unwrap();
    path.1 = 0;
    while path.0.last() == Some(&0) {
      path.0.pop();
    }
    let Some(last) = path.0.last_mut() else { break };
    *last -= 1;
  }
}

fn decide(options: usize) -> usize {
  if options == 1 {
    return 0;
  }
  COORDINATOR.with(|c| c.get().unwrap().decide(options))
}

fn yield_point() {
  COORDINATOR.with(|c| c.get().unwrap().yield_point())
}
