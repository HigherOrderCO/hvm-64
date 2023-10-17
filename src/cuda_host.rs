use crate::cuda_device::*;
use crate::run;
use cudarc::driver::{DeviceRepr, ValidAsZeroBits, sys::CUdeviceptr};
use cudarc::driver::{CudaDevice, CudaSlice, DriverError, DevicePtr};
use std::{collections::{BTreeMap, HashMap}, slice, sync::Arc};

// TODO: Factor out duplication with `gen_cuda_book` in main.rs

// Generate data to pass to CUDA runtime.
// Returns (book_data, jump_data, function_ids)
pub fn gen_cuda_book_data(book: &run::Book) -> (Vec<u32>, Vec<u32>, HashMap<String, u32>) {
  // Sort the book.defs by key
  let mut defs = BTreeMap::new();
  for i in 0 .. book.defs.len() {
    if book.defs[i].node.len() > 0 {
      defs.insert(i as u32, book.defs[i].clone());
    }
  }

  // Generate map from function name to id
  let function_ids = defs.keys().enumerate().map(|(i, &id)|  {
    (crate::ast::val_to_name(id), id)
  }).collect::<HashMap<_, _>>();

  // Generate book data
  let mut book_data = vec![];
  for (i, (id, net)) in defs.iter().enumerate() {
    // Collect all pointers from root, nodes and rdex into a single buffer
    book_data.push(net.node.len() as u32);
    book_data.push(net.rdex.len() as u32);
    // .node
    for (i, node) in net.node.iter().enumerate() {
      book_data.push(node.0.data());
      book_data.push(node.1.data());
    }
    // .rdex
    for (i, (a, b)) in net.rdex.iter().enumerate() {
      book_data.push(a.data());
      book_data.push(b.data());
    }
  }

  let mut jump_data = vec![];
  let mut index = 0;
  for (i, id) in defs.keys().enumerate() {
    jump_data.push(*id);
    jump_data.push(index);
    index += 2 + 2 * defs[id].node.len() as u32 + 2 * defs[id].rdex.len() as u32;
  }

  (book_data, jump_data, function_ids)
}


// High-level representation of `CudaNet` used on the CPU
pub struct HostNet {
  bags: Box<[Wire]>,
  heap: Box<[Node]>,
  head: Box<[Wire]>,
  jump: Box<[u32]>,
  rwts: u64,
}

// Gets the target ref of a var or redirection pointer
#[inline(always)]
fn target(net: &mut HostNet, ptr: Ptr) -> &mut Ptr {
  /* let heap = unsafe {
    std::slice::from_raw_parts_mut(net.heap, HEAD_SIZE as usize)
  }; */
  let heap = &mut net.heap;
  &mut heap[val(ptr) as usize].ports[(ptr & 1) as usize]
}

pub fn mknet(root_fn: u32, jump_data: &[u32]) -> HostNet {
  let mut net = HostNet {
    bags: vec![Default::default(); BAGS_SIZE as usize].into_boxed_slice(),
    heap: vec![Default::default(); HEAP_SIZE as usize].into_boxed_slice(),
    head: vec![Default::default(); HEAD_SIZE as usize].into_boxed_slice(),
    jump: vec![Default::default(); JUMP_SIZE as usize].into_boxed_slice(),
    rwts: 0,
  };

  *target(&mut net, ROOT) = mkptr(REF, root_fn);
  for i in 0 .. jump_data.len() / 2 {
    net.jump[jump_data[i*2+0] as usize] = jump_data[i*2+1];
  }

  net
}

pub fn net_to_gpu(dev: &Arc<CudaDevice>, host_net: &HostNet) -> Result<CudaSlice<CudaNet>, DriverError> {
	// TODO: Async copy? (Requires passing owned Vec)
	let device_bags = dev.htod_sync_copy(&host_net.bags)?;
  let device_heap = dev.htod_sync_copy(&host_net.heap)?;
  let device_head = dev.htod_sync_copy(&host_net.head)?;
  let device_jump = dev.htod_sync_copy(&host_net.jump)?;

	let temp_net = CudaNet {
    bags: *(&device_bags).device_ptr() as _,
    heap: *(&device_heap).device_ptr() as _,
    head: *(&device_head).device_ptr() as _,
    jump: *(&device_jump).device_ptr() as _,
    rwts: 0,
	};

	let device_net = dev.htod_sync_copy(slice::from_ref(&temp_net))?;

  // TODO: Keep these alive in the returned value
  std::mem::forget(device_bags);
  std::mem::forget(device_heap);
  std::mem::forget(device_head);
  std::mem::forget(device_jump);

	Ok(device_net)
}