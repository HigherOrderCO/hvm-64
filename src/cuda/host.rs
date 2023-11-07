use crate::cuda::device::*;
use crate::run;
use cudarc::driver::CudaFunction;
use cudarc::driver::{DeviceRepr, ValidAsZeroBits, sys::CUdeviceptr};
use cudarc::driver::{CudaDevice, CudaSlice, DriverError, DevicePtr};
use cudarc::driver::{LaunchAsync, LaunchConfig};
use cudarc::nvrtc::CompileError;
use std::{collections::{BTreeMap, HashMap}, slice, sync::Arc};

// Generate data to pass to CUDA runtime.
// Based on `gen_cuda_book`.
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

impl HostNet {
  pub fn to_runtime_net(self) -> run::Net {
    fn wire_to_rdex(wire: Wire) -> (run::Ptr, run::Ptr) {
      (run::Ptr((wire & 0xFFFFFFFF) as Val), run::Ptr((wire >> 32) as Val))
    }

    fn node_to_pair(node: Node) -> (run::Ptr, run::Ptr) {
      let Node { ports: [a, b] } = node;
      (run::Ptr(a), run::Ptr(b))
    }

    let mut rdex = Vec::new();

    for i in 0..BAGS_SIZE {
      let wire: u64 = self.bags[i as usize];
      if i % RBAG_SIZE == 0 && wire > 0 {
        rdex.push(wire_to_rdex(wire));
      } else if i % RBAG_SIZE >= 1 {
        let (a, b) = wire_to_rdex(wire);
        if a.0 != 0 || b.0 != 0 {
          rdex.push((a, b));
        }
      }
    }

    let data = self.heap.into_iter().map(|node| node_to_pair(*node)).collect();

    // We don't have detailed rewrite statistics,
    // so we put the total rewrite count on an arbitrary rewrite field on the net 
    let oper = self.rwts.try_into().unwrap();

    run::Net {
      rdex,
      heap: run::Heap::from_data(data),
      anni: 0,
      comm: 0,
      eras: 0,
      dref: 0,
      oper,
    }
  }
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

pub struct CudaNetHandle {
  device_net: CudaSlice<CudaNet>,
  device_bags: CudaSlice<Wire>,
  device_heap: CudaSlice<Node>,
  device_head: CudaSlice<Wire>,
  device_jump: CudaSlice<u32>,
}

pub fn net_to_gpu(dev: &Arc<CudaDevice>, host_net: &HostNet) -> Result<CudaNetHandle, DriverError> {
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

  Ok(CudaNetHandle {
    device_net,
    device_bags,
    device_heap,
    device_head,
    device_jump,
  })
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

fn show_ptr(ptr: Ptr) -> String {
  match ptr {
    0x00000000 => "           ".to_string(),
    0xFFFFFFFF => "[LOCK.....]".to_string(),
    _ => {
      let tag_str = match (ptr & 0xF) as Tag {
        run::VR1 => "VR1",
        run::VR2 => "VR2",
        run::RD1 => "RD1",
        run::RD2 => "RD2",
        run::REF => "REF",
        run::ERA => "ERA",
        run::NUM => "NUM",
        run::OP2 => "OP2",
        run::OP1 => "OP1",
        run::MAT => "MAT",
        run::CT0 => "CT0",
        run::CT1 => "CT1",
        run::CT2 => "CT2",
        run::CT3 => "CT3",
        run::CT4 => "CT4",
        run::CT5 => "CT5",
        _ => "???"
      };
  
      let val = (ptr >> 4) as Val;
      format!("{}:{:07X}", tag_str, val)
    }
  }
}

// Prints a net in hexadecimal, limited to a given size
fn print_net(net: &HostNet) {
  println!("Bags:");
  for i in 0..BAGS_SIZE {
    let wire = net.bags[i as usize];
    if i % RBAG_SIZE == 0 && wire > 0 {
      println!("- [{:07X}] LEN={}", i, wire);
    } else if i % RBAG_SIZE >= 1 {
      // let a = (wire & 0xFFFFFFFF) as Val;
      // let b = (wire >> 32) as Val;
      // if a != 0 || b != 0 {
      //   println!("- [{:07X}] {} {}", i, show_ptr(a), show_ptr(b));
      // }
    }
  }
  // println!("Heap:");
  // for i in 0..HEAP_SIZE {
  //   let Node { ports: [a, b] } = net.heap[i as usize];
  //   if a != 0 || b != 0 {
  //     println!("- [{:07X}] {} {}", i, show_ptr(a), show_ptr(b));
  //   }
  // }
  println!("Rwts: {}", net.rwts);
}

pub fn run_on_gpu(book: &run::Book, entry_point_function: &str) -> Result<HostNet, Box<dyn std::error::Error>> {
  let (book_data, jump_data, function_ids) = gen_cuda_book_data(book);

  let root_fn_id = *function_ids.get(entry_point_function).unwrap_or_else(|| {
    // TODO: Proper error handling
    panic!("Entry point function not found: {}", entry_point_function);
  });

  let gpu_index = 0; // TODO: Receive GPU index as argument to let user choose which GPU to use
  let dev = CudaDevice::new(gpu_index)?;

  // Load CUDA runtime
  let ptx = match cudarc::nvrtc::compile_ptx(include_str!("../../cuda/runtime.cu")) {
    Ok(ptx) => ptx,
    Err(CompileError::CompileError { nvrtc, options, log }) => {
      let log_str = log.to_str().unwrap();
      println!("\n=== Error compiling CUDA runtime ===\n");
      for diagnostic in log_str.split("default_program").filter(|s| {
        !s.is_empty() && s.find(": ").map_or(false, |i| s[i + 2 ..].starts_with("error: "))
      }) {
        print!("{}", diagnostic);
      }
      return Err(CompileError::CompileError { nvrtc, options, log }.into());
    }
    Err(e) => return Err(e.into()),
  };
  const MODULE_NAME: &str = "runtime";
  println!("Loading module `{}`...", MODULE_NAME);
  dev.load_ptx(ptx, MODULE_NAME, &["global_expand_prepare", "global_expand", "global_rewrite"])?;
  println!("Module `{}` loaded.", MODULE_NAME);
  let global_expand_prepare = dev.get_func(MODULE_NAME, "global_expand_prepare").expect("Function `global_expand_prepare` not found");
  let global_expand = dev.get_func(MODULE_NAME, "global_expand").expect("Function `global_expand` not found");
  let global_rewrite = dev.get_func(MODULE_NAME, "global_rewrite").expect("Function `global_rewrite` not found");


  // Allocates net on CPU
  let cpu_net = mknet(root_fn_id, &jump_data);

  // Prints the input net
  println!("\nINPUT\n=====\n");
  print_net(&cpu_net);

  // Uploads net and book to GPU
  let gpu_net = net_to_gpu(&dev, &cpu_net)?;

  // Equivalent to: Book* gpu_book = init_book_on_gpu(BOOK_DATA, BOOK_DATA_SIZE);
  let gpu_book = book_to_gpu(&dev, &book_data)?;


  let time_before = std::time::Instant::now();

  // Normalizes
  cuda_normalize_net(global_expand_prepare, global_expand, global_rewrite, &gpu_net.device_net, &gpu_book)?;
  dev.synchronize()?;

  let time_elapsed_secs = time_before.elapsed().as_secs_f64();

  // Reads result back to cpu
  let norm = net_to_cpu(&dev, gpu_net.device_net)?;

  // Prints the output
  println!("\nNORMAL ~ rewrites={}\n======\n", norm.rwts);
  print_net(&norm);
  println!("Time: {:.3} s", time_elapsed_secs);
  println!("RPS : {:.3} million", norm.rwts as f64 / time_elapsed_secs / 1_000_000.);

  Ok(norm)
}

#[inline(always)]
fn cuda_normalize_net(
  global_expand_prepare: CudaFunction,
  global_expand: CudaFunction,
  global_rewrite: CudaFunction,
  gpu_net: &CudaSlice<CudaNet>,
  gpu_book: &CudaSlice<u32>,
) -> Result<(), DriverError> {
  // Normalizes
  do_global_expand(global_expand_prepare.clone(), global_expand.clone(), &gpu_net, &gpu_book)?;
  for tick in 0 .. 128 {
    do_global_rewrite(global_rewrite.clone(), &gpu_net, &gpu_book, 16, tick, (tick / BAGS_WIDTH_L2) % 2 != 0)?;
  }
  do_global_expand(global_expand_prepare, global_expand, &gpu_net, &gpu_book)?;
  do_global_rewrite(global_rewrite, &gpu_net, &gpu_book, 200000, 0, false)?;
  Ok(())
}

// Performs a global head expansion (1 deref per bag)
fn do_global_expand(
  global_expand_prepare: CudaFunction,
  global_expand: CudaFunction,
  gpu_net: &CudaSlice<CudaNet>,
  gpu_book: &CudaSlice<u32>,
) -> Result<(), DriverError> {
  // global_expand_prepare<<<BAGS_HEIGHT, GROUP_SIZE>>>(net);
  unsafe {
    global_expand_prepare.launch(
      LaunchConfig {
        grid_dim: (BAGS_HEIGHT, 1, 1),
        block_dim: (GROUP_SIZE, 1, 1),
        shared_mem_bytes: 0,
      },
      (gpu_net,)
    )
  }?;

  // global_expand<<<BAGS_HEIGHT, BLOCK_SIZE>>>(net, book);
  unsafe {
    global_expand.launch(
      LaunchConfig {
        grid_dim: (BAGS_HEIGHT, 1, 1),
        block_dim: (BLOCK_SIZE, 1, 1),
        shared_mem_bytes: 0,
      },
      (gpu_net, gpu_book)
    )
  }?;
  Ok(())
}

fn do_global_rewrite(
  global_rewrite: CudaFunction,
  net: &CudaSlice<CudaNet>,
  book: &CudaSlice<u32>,
  repeat: u32,
  tick: u32,
  flip: bool,
) -> Result<(), DriverError> {
  // global_rewrite<<<BAGS_HEIGHT, BLOCK_SIZE>>>(net, book, repeat, tick, flip);
  unsafe {
    global_rewrite.launch(
      LaunchConfig {
        grid_dim: (BAGS_HEIGHT, 1, 1),
        block_dim: (BLOCK_SIZE, 1, 1),
        shared_mem_bytes: 0,
      },
      (net, book, repeat, tick, flip)
    )
  }?;
  Ok(())
}
