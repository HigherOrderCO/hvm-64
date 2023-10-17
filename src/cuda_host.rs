use crate::cuda_device::*;
use crate::run;
use cudarc::driver::{DeviceRepr, ValidAsZeroBits, sys::CUdeviceptr};
use cudarc::driver::{CudaDevice, CudaSlice, DriverError, DevicePtr};
use cudarc::driver::{LaunchAsync, LaunchConfig};
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

pub fn book_to_gpu(dev: &Arc<CudaDevice>, book_data: &[u32]) -> Result<CudaSlice<u32>, DriverError> {
  let device_book_data = dev.htod_sync_copy(book_data)?;
  Ok(device_book_data)
}

pub fn net_to_cpu(dev: &Arc<CudaDevice>, device_net: CudaSlice<CudaNet>) -> Result<HostNet, DriverError> {
	use cudarc::driver::{result, sys::CUdeviceptr};

	fn dtoh_sync_copy_into<T: DeviceRepr>(
		dev: &Arc<CudaDevice>,
		src: CUdeviceptr,
		src_len: usize,
		dst: &mut [T],
	) -> Result<(), DriverError> {
		assert_eq!(src_len, dst.len());
		dev.bind_to_thread()?;
		unsafe { result::memcpy_dtoh_sync(dst, src) }?;
		dev.synchronize()
	}

	fn dtoh_sync_copy<T: DeviceRepr>(
		dev: &Arc<CudaDevice>,
		src: CUdeviceptr,
		src_len: usize,
	) -> Result<Vec<T>, DriverError> {
		let mut dst = Vec::with_capacity(src_len);
		unsafe { dst.set_len(src_len) };
		dtoh_sync_copy_into(dev, src, src_len, &mut dst)?;
		Ok(dst)
	}

	// let mut net_vec: Vec<CudaNet> = dev.sync_reclaim(device_net)?;
	let mut net_vec = dev.dtoh_sync_copy(&device_net)?;
	let net = net_vec.remove(0);

	let bags = dtoh_sync_copy(dev, net.bags as CUdeviceptr, BAGS_SIZE as usize)?;
  let heap = dtoh_sync_copy(dev, net.heap as CUdeviceptr, HEAP_SIZE as usize)?;
  let head = dtoh_sync_copy(dev, net.head as CUdeviceptr, HEAD_SIZE as usize)?;
  let jump = dtoh_sync_copy(dev, net.jump as CUdeviceptr, JUMP_SIZE as usize)?;

	let net = HostNet {
    bags: bags.into_boxed_slice(),
    heap: heap.into_boxed_slice(),
    head: head.into_boxed_slice(),
    jump: jump.into_boxed_slice(),
    rwts: net.rwts,
	};
	Ok(net)
}

// Prints a net in hexadecimal, limited to a given size
fn print_net(net: &HostNet) {
  println!("Bags:");
  for i in 0..BAGS_SIZE {
    if i % RBAG_SIZE == 0 && net.bags[i as usize] > 0 {
      println!("- [{:07X}] LEN={}", i, net.bags[i as usize]);
    } else if i % RBAG_SIZE >= 1 {
      //let a = wire_lft(net.bags[i as usize]);
      //let b = wire_rgt(net.bags[i as usize]);
      //if a != 0 || b != 0 {
        //println!("- [{:07X}] {} {}", i, show_ptr(a,0), show_ptr(b,1));
      //}
    }
  }
  //println!("Heap:");
  //for i in 0..HEAP_SIZE {
    //let a = net.heap[i as usize].ports[P1];
    //let b = net.heap[i as usize].ports[P2];
    //if a != 0 || b != 0 {
      //println!("- [{:07X}] {} {}", i, show_ptr(a,0), show_ptr(b,1));
    //}
  //}
  println!("Rwts: {}", net.rwts);
}

pub fn run_on_gpu(book: &run::Book, entry_point_function: &str) -> Result<HostNet, Box<dyn std::error::Error>> {
  let (book_data, jump_data, function_ids) = gen_cuda_book_data(book);

  // println!("book_data: {{{}}}", book_data.iter().map(|x| format!("0x{:08X}", x)).collect::<Vec<_>>().join(", "));
  // println!("jump_data: {{{}}}", jump_data.iter().map(|x| format!("0x{:08X}", x)).collect::<Vec<_>>().join(", "));
  // println!("function_ids: {{{}}}", function_ids.iter().map(|(k, v)| format!("{}: 0x{:08X}", k, v)).collect::<Vec<_>>().join(", "));

  let root_fn_id = *function_ids.get(entry_point_function).unwrap_or_else(|| {
    // TODO: Proper error handling
    panic!("Entry point function not found: {}", entry_point_function);
  });

  // Allocates net on CPU
  let cpu_net = mknet(root_fn_id, &jump_data);

  // Prints the input net
  println!("\nINPUT\n=====\n");
  print_net(&cpu_net);

  let gpu_index = 0; // TODO: Receive GPU index as argument to let user choose which GPU to use
  let dev = CudaDevice::new(gpu_index)?;

  // Uploads net and book to GPU
  let gpu_net = net_to_gpu(&dev, &cpu_net)?;

  // Equivalent to: Book* gpu_book = init_book_on_gpu(BOOK_DATA, BOOK_DATA_SIZE);
  let gpu_book = book_to_gpu(&dev, &book_data)?;

  // Load CUDA runtime
  let ptx = cudarc::nvrtc::compile_ptx(include_str!("cuda/runtime.cu"))?;
  const MODULE_NAME: &str = "runtime";
  dev.load_ptx(ptx, MODULE_NAME, &["do_global_expand", "do_global_rewrite"])?;
  let do_global_expand = dev.get_func(MODULE_NAME, "do_global_expand").expect("Function `do_global_expand` not found");
  let do_global_rewrite = dev.get_func(MODULE_NAME, "do_global_rewrite").expect("Function `do_global_rewrite` not found");

  let time_before = std::time::Instant::now();

  // TODO: Uncomment and adjust
  /* // Normalizes
  // do_global_expand(gpu_net, gpu_book);
  unsafe { do_global_expand.clone().launch(LaunchConfig::for_num_elems(1), (&gpu_net,)) }?;
  for tick in 0 .. 128 {
    // do_global_rewrite(gpu_net, gpu_book, 16, tick, (tick / BAGS_WIDTH_L2) % 2);
    unsafe { do_global_rewrite.clone().launch(LaunchConfig::for_num_elems(1), (&gpu_net, &gpu_book, 16, tick, (tick / BAGS_WIDTH_L2) % 2)) }?;
  }
  // do_global_expand(gpu_net, gpu_book);
  unsafe { do_global_expand.launch(LaunchConfig::for_num_elems(1), (&gpu_net,)) }?;
  // do_global_rewrite(gpu_net, gpu_book, 200000, 0, 0);
  unsafe { do_global_rewrite.launch(LaunchConfig::for_num_elems(1), (&gpu_net, &gpu_book, 200000, 0, 0)) }?;
  dev.synchronize()?; */

  let time_elapsed_secs = time_before.elapsed().as_secs_f64();

  // Reads result back to cpu
  let norm = net_to_cpu(&dev, gpu_net)?;

  // Prints the output
  println!("\nNORMAL ~ rewrites={}\n======\n", norm.rwts);
  print_net(&norm);
  println!("Time: {:.3} s", time_elapsed_secs);
  println!("RPS : {:.3} million", norm.rwts as f64 / time_elapsed_secs);

  // TODO: Uncomment and adjust
  /* // Clears CPU memory
  net_free_on_gpu(gpu_net);
  book_free_on_gpu(gpu_book);

  // Clears GPU memory
  net_free_on_cpu(cpu_net);
  net_free_on_cpu(norm); */
  Ok(norm)
}