use crate::core::{*};

#[derive(Clone, Debug, Eq, PartialEq, Hash)]
pub struct Heap {
  data: Vec<Ptr>,
  next: usize,
  used: usize,
}

impl Heap {
  pub fn new(size: usize) -> Heap {
    return Heap {
      data: vec![NULL; size * 2],
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
      return *self.data.get_unchecked((index * 2 + port) as usize);
    }
  }

  #[inline(always)]
  pub fn set(&mut self, index: Val, port: Port, value: Ptr) {
    unsafe {
      *self.data.get_unchecked_mut((index * 2 + port) as usize) = value;
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
    let root = self.data[1];
    let mut node = vec![];
    for i in 0 .. self.used {
      node.push((self.data[(i+1)*2+0], self.data[(i+1)*2+1]));
    }
    return (root, node);
  }
}
