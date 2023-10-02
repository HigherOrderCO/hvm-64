use crate::core::{*};

use std::sync::atomic::{AtomicU32, Ordering};

pub struct AtomicPtr(AtomicU32);

pub struct Heap {
  data: Vec<AtomicPtr>,
  next: usize,
  used: usize,
}

impl Heap {
  pub fn new(size: usize) -> Heap {
    let mut data = vec![];
    for _ in 0 .. size * 2 {
      data.push(AtomicPtr(AtomicU32::new(NULL.0)));
    }
    return Heap {
      data,
      next: 1,
      used: 0,
    };
  }

  #[inline(always)]
  pub fn alloc(&mut self, size: usize) -> Val {
    if size == 0 {
      return 0;
    } else {
      let mut space = 0;
      loop {
        if self.next >= self.data.len() {
          space = 0;
          self.next = 1;
        }
        if self.get(self.next as Val, P1).is_nil() {
          space += 1;
        } else {
          space = 0;
        }
        self.next += 1;
        if space == size {
          self.used += size;
          return (self.next - space) as Val;
        }
      }
    }
  }

  #[inline(always)]
  pub fn free(&mut self, index: Val) {
    self.used -= 1;
    self.set(index, P1, NULL);
    self.set(index, P2, NULL);
  }

  #[inline(always)]
  pub fn lock(&self, index: Val) {
    return;
  }

  #[inline(always)]
  pub fn unlock(&self, index: Val) {
    return;
  }

  #[inline(always)]
  pub fn get(&self, index: Val, port: Port) -> Ptr {
    unsafe {
      return Ptr(self.data.get_unchecked((index * 2 + port) as usize).0.load(Ordering::Relaxed));
    }
  }

  #[inline(always)]
  pub fn set(&mut self, index: Val, port: Port, value: Ptr) {
    unsafe {
      self.data.get_unchecked_mut((index * 2 + port) as usize).0.store(value.0, Ordering::Relaxed);
    }
  }

  #[inline(always)]
  pub fn swap(&self, index: Val, port: Port, value: Ptr) -> Ptr {
    unsafe {
      return Ptr(self.data.get_unchecked((index * 2 + port) as usize).0.swap(value.0, Ordering::Relaxed));
    }
  }

  #[inline(always)]
  pub fn cas(&self, index: Val, port: Port, from: Ptr, to: Ptr) -> Ptr {
    unsafe {
      let ptr_ref = self.data.get_unchecked((index * 2 + port) as usize);
      match ptr_ref.0.compare_exchange_weak(from.0, to.0, Ordering::Relaxed, Ordering::Relaxed) {
        Ok(old)  => { Ptr(old) }
        Err(old) => { Ptr(old) }
      }
    }
  }

  #[inline(always)]
  pub fn get_root(&self) -> Ptr {
    return self.get(0, P2);
  }

  #[inline(always)]
  pub fn set_root(&mut self, value: Ptr) {
    self.set(0, P2, value);
  }

  #[inline(always)]
  pub fn compact(&mut self) -> (Ptr, Vec<(Ptr,Ptr)>) {
    let root = self.get(0, P2);
    let mut node = vec![];
    for i in 0 .. self.used {
      node.push((self.get((i+1) as Val, P1), self.get((i+1) as Val, P2)));
    }
    return (root, node);
  }
}
