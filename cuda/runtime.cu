#include "runtime.cuh"

// Allocates a new node in memory
__device__ inline u32 alloc(Worker *worker, Net *net) {
  u32 K = 0;
  while (true) {
    //dbug(&K, "alloc");
    u32  idx = (worker->unit * ALLOC_PAD + worker->aloc * 4 + worker->frac) % NODE_SIZE;
    a64* ref = &((a64*)net->node)[idx];
    u64  got = atomicCAS(ref, 0, ((u64)NEO << 32) | ((u64)NEO)); // Wire{NEO,NEO}
    if (got == 0) {
      //printf("[%d] alloc at %d\n", worker->gid, idx);
      return idx;
    } else {
      worker->aloc = (worker->aloc + 1) % NODE_SIZE;
    }
  }
}

// Creates a new active pair
__device__ inline void put_redex(Worker* worker, Ptr a_ptr, Ptr b_ptr) {
  worker->bag[worker->frac] = (Wire){a_ptr, b_ptr};
}

// Gets the value of a ref; waits if busy
__device__ Ptr take(Ptr* ref) {
  Ptr got = atomicExch((u32*)ref, BSY);
  u32 K = 0;
  while (got == BSY) {
    //dbug(&K, "take");
    got = atomicExch((u32*)ref, BSY);
  }
  return got;
}

// Attempts to replace 'exp' by 'neo', until it succeeds
__device__ void replace(u32 id, Ptr* ref, Ptr exp, Ptr neo) {
  Ptr got = atomicCAS((u32*)ref, exp, neo);
  u32 K = 0;
  while (got != exp) {
    //dbug(&K, "replace");
    got = atomicCAS((u32*)ref, exp, neo);
  }
}

// Links the node in 'nod_ref' towards the destination of 'dir_ptr'
// - If the dest is a redirection => clear it and aim forwards
// - If the dest is a variable    => pass the node into it and halt
// - If the dest is a node        => create an active pair and halt
__device__ void link(Worker* worker, Net* net, Ptr* nod_ref, Ptr dir_ptr) {
  //printf("[%d] linking node=%8X towards %8X\n", worker->gid, *nod_ref, dir_ptr);

  u32 K = 0;
  while (true) {
    //dbug(&K, "link");
    //printf("[%d] step\n", worker->gid);

    // We must be careful to not cross boundaries. When 'trg_ptr' is a VAR, it
    // isn't owned by us. As such, we can't 'take()' it, and must peek instead.
    Ptr* trg_ref = target(net, dir_ptr);
    Ptr  trg_ptr = atomicAdd(trg_ref, 0);

    // If trg_ptr is a redirection, clear it
    if (tag(trg_ptr) >= RDR && tag(trg_ptr) <= RD2) {
      //printf("[%d] redir\n", worker->gid);
      u32 cleared = atomicCAS((u32*)trg_ref, trg_ptr, 0);
      if (cleared == trg_ptr) {
        dir_ptr = trg_ptr;
      }
      continue;
    }

    // If trg_ptr is a var, try replacing it by the node
    else if (tag(trg_ptr) >= VRR && tag(trg_ptr) <= VR2) {
      //printf("[%d] var\n", worker->gid);
      // Peeks our own node
      Ptr nod_ptr = atomicAdd((u32*)nod_ref, 0);
      // We don't own the var, so we must try replacing with a CAS
      u32 replaced = atomicCAS((u32*)trg_ref, trg_ptr, nod_ptr);
      // If it worked, we successfully moved our node to another region
      if (replaced == trg_ptr) {
        // Collects the backwards path, which is now orphan
        trg_ref = target(net, trg_ptr);
        trg_ptr = atomicAdd(trg_ref, 0);
        u32 K2 = 0;
        while (tag(trg_ptr) >= RDR && tag(trg_ptr) <= RD2) {
          //dbug(&K2, "inner-link");
          u32 cleared = atomicCAS((u32*)trg_ref, trg_ptr, 0);
          if (cleared == trg_ptr) {
            trg_ref = target(net, trg_ptr);
            trg_ptr = atomicAdd(trg_ref, 0);
          }
        }
        // Clear our node
        atomicCAS((u32*)nod_ref, nod_ptr, 0);
        return;
      // Otherwise, things probably changed, so we step back and try again
      } else {
        continue;
      }
    }

    // If it is a node, two threads will reach this branch
    // The first to arrive makes a redex, the second exits
    else if (tag(trg_ptr) >= CON && tag(trg_ptr) <= QUI || trg_ptr == GON) {
      //printf("[%d] con (from %8X to %8X)\n", worker->gid, *nod_ref, trg_ptr);
      Ptr *fst_ref = nod_ref < trg_ref ? nod_ref : trg_ref;
      Ptr *snd_ref = nod_ref < trg_ref ? trg_ref : nod_ref;
      Ptr  fst_ptr = atomicExch((u32*)fst_ref, GON);
      if (fst_ptr == GON) {
        atomicCAS((u32*)fst_ref, GON, 0);
        atomicCAS((u32*)snd_ref, GON, 0);
        return;
      } else {
        Ptr snd_ptr = atomicExch((u32*)snd_ref, GON);
        put_redex(worker, fst_ptr, snd_ptr);
        return;
      }
    }

    // If it is busy, we wait
    else if (trg_ptr == BSY) {
      //printf("[%d] waits\n", worker->gid);
      continue;
    }

    else {
      //printf("[%d] WTF?? ~ %8X\n", worker->gid, trg_ptr);
      return;
    }
  }
}

// Kernels
// -------

// Performs a local scan sum
__device__ int scansum(u32* arr) {
  int tid = threadIdx.x;
  int bid = blockIdx.x;
  int gid = bid * blockDim.x + tid;

  // upsweep
  for (int d = 0; d < BLOCK_LOG2; ++d) {
    if (tid % (1 << (d + 1)) == 0) {
      arr[tid + (1 << (d + 1)) - 1] += arr[tid + (1 << d) - 1];
    }
    __syncthreads();
  }

  // gets sum
  int sum = arr[BLOCK_SIZE - 1];
  __syncthreads();

  // clears last
  if (tid == 0) {
    arr[BLOCK_SIZE - 1] = 0;
  }
  __syncthreads();

  // downsweep
  for (int d = BLOCK_LOG2 - 1; d >= 0; --d) {
    if (tid % (1 << (d + 1)) == 0) {
      u32 tmp = arr[tid + (1 << d) - 1];
      arr[tid + (1 << (d + 0)) - 1] = arr[tid + (1 << (d + 1)) - 1];
      arr[tid + (1 << (d + 1)) - 1] += tmp;
    }
    __syncthreads();
  }

  return sum;
}

// Local scatter
__device__ void local_scatter(Net* net) {
  u32   tid = threadIdx.x;
  u32   bid = blockIdx.x;
  u32   gid = bid * blockDim.x + tid;
  u32*  loc = net->gidx + bid * BLOCK_SIZE;
  Wire* bag = net->bags + bid * BLOCK_SIZE;

  // Initializes the index object
  loc[tid] = bag[tid].lft > 0 ? 1 : 0;
  __syncthreads();

  // Computes our bag len
  u32 bag_len = scansum(loc);
  __syncthreads();

  // Takes our wire
  Wire wire = bag[tid];
  bag[tid] = Wire{0,0};
  __syncthreads();

  // Moves our wire to target location
  if (wire.lft != 0) {
    u32 chunks  = BLOCK_SIZE / UNIT_SIZE;
    u32 filled  = (bag_len + chunks - 1) / chunks;
    u32 index   = loc[tid];
    u32 target  = (index / filled) * UNIT_SIZE + (index % filled);
    bag[target] = wire;
  }
}

// Prepares a global scatter, step 0
__global__ void global_scatter_prepare_0(Net* net) {
  u32  tid = threadIdx.x;
  u32  bid = blockIdx.x;
  u32  gid = bid * blockDim.x + tid;
  u32* loc = net->gidx + bid * BLOCK_SIZE;

  // Initializes the index object
  loc[tid] = net->bags[gid].lft > 0 ? 1 : 0;
  __syncthreads();

  // Computes our bag len
  u32 bag_len = scansum(loc);
  __syncthreads();

  // Broadcasts our bag len 
  net->gtmp[bid] = bag_len;
}

// Prepares a global scatter, step 1
__global__ void global_scatter_prepare_1(Net* net) {
  net->blen = scansum(net->gtmp);
}

// Global scatter
__global__ void global_scatter(Net* net) {
  u32 tid = threadIdx.x;
  u32 bid = blockIdx.x;
  u32 gid = bid * blockDim.x + tid;

  // Takes our wire
  u64 wire = atomicExch((a64*)&net->bags[gid], 0);

  // Moves wire to target location
  if (wire != 0) {
    u32 chunks = BAGS_SIZE / BLOCK_SIZE;
    u32 filled = (net->blen + chunks - 1) / chunks;
    u32 index  = net->gidx[gid] + net->gtmp[bid];
    u32 target = (index / filled) * BLOCK_SIZE + (index % filled);
    //printf("[%d on %d] moves %d to %d | put={%8X,%8X} | global=%d bags_size=%d bag_len=%d chunk_size=%d per_chunk=%d index=%d target=%d \n", lft, wire.rgt, global, BAGS_SIZE, bag_len, chunk_size, per_chunk, index, target);
    while (atomicCAS((a64*)&net->bags[target], 0, wire) != 0) {};
  }
}

// Performs a global scatter
void do_global_scatter(Net* net) {
  global_scatter_prepare_0<<<TOTAL_BLOCKS, BLOCK_SIZE>>>(net);
  global_scatter_prepare_1<<<1, BLOCK_SIZE>>>(net);
  global_scatter<<<TOTAL_BLOCKS, BLOCK_SIZE>>>(net);
}

// An active wire is reduced by 4 parallel threads, each one performing "1/4" of
// the work. Each thread will be pointing to a node of the active pair, and an
// aux port of that node. So, when nodes A-B interact, we have 4 thread types:
// - Thread A1: points to node A and its aux1
// - Thread A2: points to node A and its aux2
// - Thread B1: points to node B and its aux1
// - Thread B2: points to node B and its aux2
// This is organized so that local threads can perform the same instructions
// whenever possible. So, for example, in a commutation rule, all the 4 clones
// would be allocated at the same time.
__global__ void global_rewrite(Net* net) {
  __shared__ u32 XLOC[BLOCK_SIZE]; // aux arr for clone locs

  // Initializes local vars
  Worker worker;
  worker.tid  = threadIdx.x;
  worker.bid  = blockIdx.x;
  worker.gid  = worker.bid * blockDim.x + worker.tid;
  worker.aloc = 0;
  worker.rwts = 0;
  worker.frac = worker.tid % 4;
  worker.port = worker.tid % 2;
  worker.bag  = (net->bags + worker.gid / UNIT_SIZE * UNIT_SIZE);

  // Scatters redexes
  local_scatter(net);

  // Gets the active wire
  Wire wire = Wire{0,0};
  u32  widx = 0;
  u32  wlen = 0;
  for (u32 r = 0; r < 4; ++r) {
    if (worker.bag[r].lft > 0) {
      wire = worker.bag[r];
      widx = r;
      wlen += 1;
    }
  }
  if (wlen == 1) {
    worker.bag[widx] = (Wire){0,0};
  }
  __syncwarp();

  // Reads redex ptrs
  worker.a_ptr = worker.frac <= A2 ? wire.lft : wire.rgt;
  worker.b_ptr = worker.frac <= A2 ? wire.rgt : wire.lft;

  // Checks if we got redex, and what type
  bool rdex = wlen == 1;
  bool anni = rdex && tag(worker.a_ptr) == tag(worker.b_ptr);
  bool comm = rdex && tag(worker.a_ptr) != tag(worker.b_ptr);

  // Prints message
  if (rdex && worker.frac == A1) {
    //printf("[%04X] rewrites: %8X ~ %8X | %d\n", worker.gid, worker.a_ptr, worker.b_ptr, comm ? 1 : 0);
    worker.rwts += 1;
  }

  // Local variables
  Ptr *ak_ref; // ref to our aux port
  Ptr *bk_ref; // ref to other aux port
  Ptr  ak_ptr; // val of our aux port
  u32  xk_loc; // loc of ptr to send to other side
  Ptr  xk_ptr; // val of ptr to send to other side
  u32  y0_idx; // idx of other clone idx

  // Gets relevant ptrs and refs
  if (rdex) {
    ak_ref = at(net, val(worker.a_ptr), worker.port);
    bk_ref = at(net, val(worker.b_ptr), worker.port);
    ak_ptr = take(ak_ref);
  }

  // If anni, send a redirection
  if (anni) {
    xk_ptr = redir(ak_ptr); // redirection ptr to send
  }

  // If comm, send a clone
  if (comm) {
    xk_loc = alloc(&worker, net); // alloc a clone
    xk_ptr = mkptr(tag(worker.a_ptr),xk_loc); // cloned node ptr to send
    XLOC[worker.tid] = xk_loc; // send cloned index to other threads
  }

  // Receive cloned indices from local threads
  __syncwarp();

  // If comm, create inner wires between clones
  if (comm) {
    const u32 ADD[4] = {2, 1, -2, -3}; // deltas to get the other clone index
    const u32 VRK    = worker.port == P1 ? VR1 : VR2; // type of inner wire var
    replace(10, at(net, xk_loc, P1), NEO, mkptr(VRK, XLOC[worker.tid + ADD[worker.frac] + 0]));
    replace(20, at(net, xk_loc, P2), NEO, mkptr(VRK, XLOC[worker.tid + ADD[worker.frac] + 1]));
  }
  __syncwarp();

  // Send ptr to other side
  if (rdex) {
    replace(30, bk_ref, BSY, xk_ptr);
  }

  // If anni and we sent a NOD, link the node there, towards our port
  // If comm and we have a VAR, link the clone here, towards that var
  if (anni && !var(ak_ptr) || comm && var(ak_ptr)) {
    u32  RDK  = worker.port == P1 ? RD1 : RD2;
    Ptr *self = comm ? ak_ref        : bk_ref;
    Ptr  targ = comm ? redir(ak_ptr) : mkptr(RDK, val(worker.a_ptr)); 
    link(&worker, net, self, targ);
  }

  // If comm and we have a NOD, form an active pair with the clone we got
  if (comm && !var(ak_ptr)) {
    put_redex(&worker, ak_ptr, take(ak_ref));
    atomicCAS((u32*)ak_ref, BSY, 0);
  }

  // When the work ends, sum stats
  if (worker.frac == A1) {
    atomicAdd(&net->rwts, worker.rwts);
  }
}

void do_global_rewrite(Net* net) {
  global_rewrite<<<TOTAL_BLOCKS, BLOCK_SIZE>>>(net);
}
