use super::*;

/// The memory behind a two-word allocation.
///
/// This must be aligned to 16 bytes so that the left word's address always ends
/// with `0b0000` and the right word's address always ends with `0b1000`.
#[repr(C)]
#[repr(align(16))]
#[derive(Default)]
pub(super) struct Node(pub AtomicU64, pub AtomicU64);

#[repr(align(16))]
pub struct Heap(pub(super) [Node]);

impl Heap {
  #[inline]
  /// Allocate a new heap with a given size in words.
  pub fn new_words(words: usize) -> Box<Self> {
    let nodes = words / 2;
    unsafe {
      Box::from_raw(core::ptr::slice_from_raw_parts_mut(
        alloc::alloc(Layout::array::<Node>(nodes).unwrap()) as *mut _,
        nodes,
      ) as *mut _)
    }
  }
  #[inline(always)]
  /// Allocate a new heap with a given size in bytes.
  pub fn new_bytes(bytes: usize) -> Box<Self> {
    Heap::new_words(bytes / 8)
  }
}

pub struct Allocator<'h> {
  pub(super) tracer: Tracer,
  pub(super) heap: &'h Heap,
  pub(super) next: usize,
  pub(super) head: Addr,
}

deref!({<'h>} Allocator<'h> => self.tracer: Tracer);

impl<'h> Allocator<'h> {
  pub fn new(heap: &'h Heap) -> Self {
    Allocator { tracer: Tracer::default(), heap, next: 0, head: Addr::NULL }
  }

  /// Frees one word of a two-word allocation.
  #[inline(always)]
  pub fn half_free(&mut self, addr: Addr) {
    trace!(self.tracer, addr);
    const FREE: u64 = Port::FREE.0;
    if cfg!(feature = "_fuzz") {
      if cfg!(not(feature = "_fuzz_no_free")) {
        assert_ne!(addr.val().swap(FREE, Relaxed), FREE, "double free");
      }
    } else {
      addr.val().store(FREE, Relaxed);
      if addr.other_half().val().load(Relaxed) == FREE {
        trace!(self.tracer, "other free");
        let addr = addr.left_half();
        if addr.val().compare_exchange(FREE, self.head.0 as u64, Relaxed, Relaxed).is_ok() {
          let old_head = &self.head;
          let new_head = addr;
          trace!(self.tracer, "appended", old_head, new_head);
          self.head = new_head;
        } else {
          trace!(self.tracer, "too slow");
        };
      }
    }
  }

  /// Allocates a two-word node.
  #[inline(never)]
  pub fn alloc(&mut self) -> Addr {
    trace!(self.tracer, self.head);
    let addr = if self.head != Addr::NULL {
      let addr = self.head.clone();
      let next = Addr(self.head.val().load(Relaxed) as usize);
      trace!(self.tracer, next);
      self.head = next;
      addr
    } else {
      let index = self.next;
      self.next += 1;
      Addr(&self.heap.0.get(index).expect("OOM").0 as *const _ as _)
    };
    trace!(self.tracer, addr, self.head);
    addr.val().store(Port::LOCK.0, Relaxed);
    addr.other_half().val().store(Port::LOCK.0, Relaxed);
    addr
  }

  #[inline(always)]
  pub(crate) fn free_wire(&mut self, wire: Wire) {
    self.half_free(wire.addr());
  }

  /// If `trg` is a wire, frees the backing memory.
  #[inline(always)]
  pub(crate) fn free_trg(&mut self, trg: Trg) {
    if trg.is_wire() {
      self.free_wire(trg.as_wire());
    }
  }
}
