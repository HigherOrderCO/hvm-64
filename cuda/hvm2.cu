#include <stdarg.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

typedef uint8_t u8;
typedef uint32_t u32;
typedef uint64_t u64;
typedef unsigned long long int a64;

// Configuration
// -------------

// This code is initially optimized for nVidia RTX 4090
const u32 BLOCK_LOG2    = 8;                         // log2 of block size
const u32 BLOCK_SIZE    = 1 << BLOCK_LOG2;           // threads per block
const u32 TOTAL_BLOCKS  = BLOCK_SIZE;                // must be = BLOCK_SIZE
const u32 TOTAL_THREADS = TOTAL_BLOCKS * BLOCK_SIZE; // total threads
const u32 UNIT_SIZE     = 4;                         // threads per rewrite unit
const u32 TOTAL_UNITS   = TOTAL_THREADS / UNIT_SIZE; // total rewrite units
const u32 NODE_SIZE     = 1 << 28;                   // total nodes (2GB addressable)
const u32 BAGS_SIZE     = TOTAL_BLOCKS * BLOCK_SIZE; // total parallel redexes
const u32 GTMP_SIZE     = BLOCK_SIZE;                // TODO: rename
const u32 GIDX_SIZE     = BAGS_SIZE;                 // TODO: rename
const u32 ALLOC_PAD     = NODE_SIZE / TOTAL_UNITS;   // space between unit alloc areas

// Types
// -----

// Pointer value (28-bit)
typedef u32 Val;

// Pointer tags (4-bit)
typedef u8 Tag;
const Tag NIL = 0x0; // empty node
const Tag REF = 0x1; // reference to a definition (closed net)
const Tag NUM = 0x2; // unboxed number
const Tag ERA = 0x3; // unboxed eraser
const Tag VRR = 0x4; // variable pointing to root
const Tag VR1 = 0x5; // variable pointing to aux1 port of node
const Tag VR2 = 0x6; // variable pointing to aux2 port of node
const Tag RDR = 0x7; // redirection to root
const Tag RD1 = 0x8; // redirection to aux1 port of node
const Tag RD2 = 0x9; // redirection to aux2 port of node
const Tag CON = 0xA; // points to main port of con node
const Tag DUP = 0xB; // points to main port of dup node
const Tag TRI = 0xC; // points to main port of tri node
const Tag QUA = 0xD; // points to main port of qua node
const Tag QUI = 0xE; // points to main port of qui node
const Tag SEX = 0xF; // points to main port of sex node
const u32 NEO = 0xFFFFFFFD; // recently allocated value
const u32 GON = 0xFFFFFFFE; // node has been moved to redex bag
const u32 BSY = 0xFFFFFFFF; // value taken by another thread, will be replaced soon

// Rewrite fractions
const u32 A1 = 0;
const u32 A2 = 1;
const u32 B1 = 2;
const u32 B2 = 3;

// Ports (P1 or P2)
typedef u8 Port;
const u32 P1 = 0;
const u32 P2 = 1;

// Pointers = 4-bit tag + 28-bit val
typedef u32 Ptr;

// Nodes are pairs of pointers
typedef struct alignas(8) {
  Ptr ports[2];
} Node;

// Wires are pairs of pointers
typedef struct alignas(8) {
  Ptr lft;
  Ptr rgt;
} Wire;

// An interaction net 
typedef struct {
  Ptr   root; // root wire
  u32   blen; // total bag length (redex count)
  Wire* bags; // redex bags (active pairs)
  Node* node; // memory buffer with all nodes
  u32*  gidx; // ............
  u32*  gtmp; // aux obj for communication
  u32   done; // number of completed threads
  u32   rwts; // number of rewrites performed
} Net;

// A worker local data
typedef struct {
  u32   tid;   // thread id
  u32   bid;   // block id 
  u32   gid;   // global id
  u32   unit;  // unit id (index on redex array)
  u32   frac;  // worker frac (A1|A2|B1|B2)
  u32   port;  // worker port (P1|P2)
  Ptr   a_ptr; // left pointer of active wire
  Ptr   b_ptr; // right pointer of active wire
  u32   aloc;  // where to alloc next node
  u32   rwts;  // total rewrites this performed
  Wire* bag;   // local redex bag
} Worker;

// Debug
// -----

__device__ __host__ void stop(const char* tag) {
  //printf(tag);
  //printf("\n");
}

__device__ __host__ bool dbug(u32* K, const char* tag) {
  *K += 1;
  if (*K > 1000000) {
    stop(tag);
    return false;
  }
  return true;
}

// Runtime
// -------

// Creates a new pointer
__host__ __device__ inline Ptr mkptr(Tag tag, Val val) {
  return ((u32)tag << 28) | (val & 0x0FFFFFFF);
}

// Gets the tag of a pointer
__host__ __device__ inline Tag tag(Ptr ptr) {
  return (Tag)(ptr >> 28);
}

// Gets the value of a pointer
__host__ __device__ inline Val val(Ptr ptr) {
  return ptr & 0x0FFFFFFF;
}

// Is this pointer a variable?
__host__ __device__ inline bool var(Ptr ptr) {
  return tag(ptr) >= VRR && tag(ptr) <= VR2;
}

// Gets the target ref of a var or redirection pointer
__host__ __device__ inline Ptr* target(Net* net, Ptr ptr) {
  if (tag(ptr) == VRR || tag(ptr) == RDR) {
    return &net->root;
  } else if (tag(ptr) == VR1 || tag(ptr) == RD1) {
    return &net->node[val(ptr)].ports[P1];
  } else if (tag(ptr) == VR2 || tag(ptr) == RD2) {
    return &net->node[val(ptr)].ports[P2];
  } else {
    return NULL;
  }
}

// Traverses to the other side of a wire
__host__ __device__ Ptr enter(Net* net, Ptr ptr) {
  ptr = *target(net, ptr);
  u32 K = 0;
  while (tag(ptr) >= RDR && tag(ptr) <= RD2) {
    //dbug(&K, "enter");
    ptr = *target(net, ptr);
  }
  return ptr;
}

// Transforms a variable into a redirection
__host__ __device__ inline Ptr redir(Ptr ptr) {
  return mkptr(tag(ptr) + (var(ptr) ? 3 : 0), val(ptr));
}

// Creates a new node
__host__ __device__ inline Node mknode(Ptr p1, Ptr p2) {
  Node node;
  node.ports[P1] = p1;
  node.ports[P2] = p2;
  return node;
}

// Creates a nil node
__host__ __device__ inline Node Node_nil() {
  return mknode(mkptr(NIL, 0), mkptr(NIL, 0));
}

// Gets a reference to the index/port Ptr on the net
__device__ inline Ptr* at(Net* net, Val idx, Port port) {
  return &net->node[idx].ports[port];
}

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
    u32 target  = (BLOCK_SIZE / bag_len) * loc[tid];
    bag[target] = wire;
  }
  __syncthreads();
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
    u32 target = (BAGS_SIZE / net->blen) * (net->gidx[gid] + net->gtmp[bid]);
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
    //printf("[%04X] redex: %8X ~ %8X | %d\n", worker.gid, worker.a_ptr, worker.b_ptr, comm ? 1 : 0);
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

  //local_scatter(net);

  // When the work ends, sum stats
  if (rdex && worker.frac == A1) {
    atomicAdd(&net->rwts, worker.rwts);
  }
}

void do_global_rewrite(Net* net) {
  global_rewrite<<<TOTAL_BLOCKS, BLOCK_SIZE>>>(net);
}

// Host<->Device
// -------------

__host__ Net* mknet() {
  Net* net  = (Net*)malloc(sizeof(Net));
  net->root = mkptr(NIL, 0);
  net->rwts = 0;
  net->done = 0;
  net->blen = 0;
  net->bags = (Wire*)malloc(BAGS_SIZE * sizeof(Wire));
  net->gidx = (u32*) malloc(GIDX_SIZE * sizeof(u32));
  net->gtmp = (u32*) malloc(GTMP_SIZE * sizeof(u32));
  net->node = (Node*)malloc(NODE_SIZE * sizeof(Node));
  memset(net->bags, 0, BAGS_SIZE * sizeof(Wire));
  memset(net->gidx, 0, GIDX_SIZE * sizeof(u32));
  memset(net->gtmp, 0, GTMP_SIZE * sizeof(u32));
  memset(net->node, 0, NODE_SIZE * sizeof(Node));
  return net;
}

__host__ Net* net_to_device(Net* host_net) {
  // Allocate memory on the device for the Net object, and its data
  Net*  device_net;
  Wire* device_bags;
  u32*  device_gidx;
  u32*  device_gtmp;
  Node* device_node;

  cudaMalloc((void**)&device_net, sizeof(Net));
  cudaMalloc((void**)&device_bags, BAGS_SIZE * sizeof(Wire));
  cudaMalloc((void**)&device_gidx, GIDX_SIZE * sizeof(u32));
  cudaMalloc((void**)&device_gtmp, GTMP_SIZE * sizeof(u32));
  cudaMalloc((void**)&device_node, NODE_SIZE * sizeof(Node));

  // Copy the host data to the device memory
  cudaMemcpy(device_bags, host_net->bags, BAGS_SIZE * sizeof(Wire), cudaMemcpyHostToDevice);
  cudaMemcpy(device_gidx, host_net->gidx, GIDX_SIZE * sizeof(u32),  cudaMemcpyHostToDevice);
  cudaMemcpy(device_gtmp, host_net->gtmp, GTMP_SIZE * sizeof(u32),  cudaMemcpyHostToDevice);
  cudaMemcpy(device_node, host_net->node, NODE_SIZE * sizeof(Node), cudaMemcpyHostToDevice);

  // Create a temporary host Net object with device pointers
  Net temp_net  = *host_net;
  temp_net.bags = device_bags;
  temp_net.gidx = device_gidx;
  temp_net.gtmp = device_gtmp;
  temp_net.node = device_node;

  // Copy the temporary host Net object to the device memory
  cudaMemcpy(device_net, &temp_net, sizeof(Net), cudaMemcpyHostToDevice);

  // Return the device pointer to the created Net object
  return device_net;
}

__host__ Net* net_to_host(Net* device_net) {
  // Create a new host Net object
  Net* host_net = (Net*)malloc(sizeof(Net));

  // Copy the device Net object to the host memory
  cudaMemcpy(host_net, device_net, sizeof(Net), cudaMemcpyDeviceToHost);

  // Allocate host memory for data
  host_net->bags = (Wire*)malloc(BAGS_SIZE * sizeof(Wire));
  host_net->gidx = (u32*) malloc(GIDX_SIZE * sizeof(u32));
  host_net->gtmp = (u32*) malloc(GTMP_SIZE * sizeof(u32));
  host_net->node = (Node*)malloc(NODE_SIZE * sizeof(Node));

  // Retrieve the device pointers for data
  Wire* device_bags;
  u32*  device_gidx;
  u32*  device_gtmp;
  Node* device_node;
  cudaMemcpy(&device_bags, &(device_net->bags), sizeof(Wire*), cudaMemcpyDeviceToHost);
  cudaMemcpy(&device_gidx, &(device_net->gidx), sizeof(u32*),  cudaMemcpyDeviceToHost);
  cudaMemcpy(&device_gtmp, &(device_net->gtmp), sizeof(u32*),  cudaMemcpyDeviceToHost);
  cudaMemcpy(&device_node, &(device_net->node), sizeof(Node*), cudaMemcpyDeviceToHost);

  // Copy the device data to the host memory
  cudaMemcpy(host_net->bags, device_bags, BAGS_SIZE * sizeof(Wire), cudaMemcpyDeviceToHost);
  cudaMemcpy(host_net->gidx, device_gidx, GIDX_SIZE * sizeof(u32),  cudaMemcpyDeviceToHost);
  cudaMemcpy(host_net->gtmp, device_gtmp, GTMP_SIZE * sizeof(u32),  cudaMemcpyDeviceToHost);
  cudaMemcpy(host_net->node, device_node, NODE_SIZE * sizeof(Node), cudaMemcpyDeviceToHost);

  return host_net;
}

__host__ void net_free_on_device(Net* device_net) {
  // Retrieve the device pointers for data
  Wire* device_bags;
  u32*  device_gidx;
  u32*  device_gtmp;
  Node* device_node;
  cudaMemcpy(&device_bags, &(device_net->bags), sizeof(Wire*), cudaMemcpyDeviceToHost);
  cudaMemcpy(&device_gidx, &(device_net->gidx), sizeof(u32*),  cudaMemcpyDeviceToHost);
  cudaMemcpy(&device_gtmp, &(device_net->gtmp), sizeof(u32*),  cudaMemcpyDeviceToHost);
  cudaMemcpy(&device_node, &(device_net->node), sizeof(Node*), cudaMemcpyDeviceToHost);

  // Free the device memory
  cudaFree(device_bags);
  cudaFree(device_gidx);
  cudaFree(device_gtmp);
  cudaFree(device_node);
  cudaFree(device_net);
}

__host__ void net_free_on_host(Net* host_net) {
  free(host_net->bags);
  free(host_net->gidx);
  free(host_net->gtmp);
  free(host_net->node);
  free(host_net);
}

// Debugging
// ---------

__host__ const char* Ptr_show(Ptr ptr, u32 slot) {
  static char buffer[8][12];
  if (ptr == 0) {
    strcpy(buffer[slot], "           ");
    return buffer[slot];
  } else if (ptr == BSY) {
    strcpy(buffer[slot], "[.........]");
    return buffer[slot];
  } else {
    const char* tag_str = NULL;
    switch (tag(ptr)) {
      case VR1: tag_str = "VR1"; break;
      case VR2: tag_str = "VR2"; break;
      case NIL: tag_str = "NIL"; break;
      case REF: tag_str = "REF"; break;
      case NUM: tag_str = "NUM"; break;
      case ERA: tag_str = "ERA"; break;
      case VRR: tag_str = "VRR"; break;
      case RDR: tag_str = "RDR"; break;
      case RD1: tag_str = "RD1"; break;
      case RD2: tag_str = "RD2"; break;
      case CON: tag_str = "CON"; break;
      case DUP: tag_str = "DUP"; break;
      case TRI: tag_str = "TRI"; break;
      case QUA: tag_str = "QUA"; break;
      case QUI: tag_str = "QUI"; break;
      case SEX: tag_str = "SEX"; break;
    }
    snprintf(buffer[slot], sizeof(buffer[slot]), "%s:%07X", tag_str, val(ptr));
    return buffer[slot];
  }
}

// Prints a net in hexadecimal, limited to a given size
void print_net(Net* net) {
  printf("Root:\n");
  printf("- %s\n", Ptr_show(net->root,0));
  printf("Bags:\n");
  for (u32 i = 0; i < BAGS_SIZE; ++i) {
    Ptr a = net->bags[i].lft;
    Ptr b = net->bags[i].rgt;
    if (a != 0 || b != 0) {
      printf("- [%07X] %s %s\n", i, Ptr_show(a,0), Ptr_show(b,1));
    }
  }
  printf("Node:\n");
  for (u32 i = 0; i < NODE_SIZE; ++i) {
    Ptr a = net->node[i].ports[P1];
    Ptr b = net->node[i].ports[P2];
    if (a != 0 || b != 0) {
      printf("- [%07X] %s %s\n", i, Ptr_show(a,0), Ptr_show(b,1));
    }
  }
  printf("BLen: %u\n", net->blen);
  printf("Rwts: %u\n", net->rwts);
  //printf("GTMP: ");
  //for (u32 i = 0; i < GTMP_SIZE; ++i) {
    //printf("%d ", net->gtmp[i]);
  //}
  printf("\n");

}

// Struct to represent a Map of entries using a simple array of (key,id) pairs
typedef struct {
  u32 keys[65536];
  u32 vals[65536];
  u32 size;
} Map;

// Function to insert a new entry into the map
__host__ void map_insert(Map* map, u32 key, u32 val) {
  map->keys[map->size] = key;
  map->vals[map->size] = val;
  map->size++;
}

// Function to lookup an id in the map by key
__host__ u32 map_lookup(Map* map, u32 key) {
  for (u32 i = 0; i < map->size; ++i) {
    if (map->keys[i] == key) {
      return map->vals[i];
    }
  }
  return map->size;
}

// Recursive function to print a term as a tree with unique variable IDs
__host__ void print_tree_go(Net* net, Ptr ptr, Map* var_ids) {
  if (var(ptr)) {
    u32 got = map_lookup(var_ids, ptr);
    if (got == var_ids->size) {
      u32 name = var_ids->size;
      Ptr targ = enter(net, ptr);
      map_insert(var_ids, targ, name);
      printf("x%d", name);
    } else {
      printf("x%d", got);
    }
  } else if (tag(ptr) == ERA) {
    printf("*");
  } else {
    switch (tag(ptr)) {
      case RDR: case RD1: case RD2:
        print_tree_go(net, *target(net, ptr), var_ids);
        break;
      default:
        printf("(%d ", tag(ptr) - CON);
        print_tree_go(net, net->node[val(ptr)].ports[P1], var_ids);
        printf(" ");
        print_tree_go(net, net->node[val(ptr)].ports[P2], var_ids);
        printf(")");
    }
  }
}

__host__ void print_tree(Net* net, Ptr ptr) {
  Map var_ids = { .size = 0 };
  print_tree_go(net, ptr, &var_ids);
  printf("\n");
}

// ~
// ~
// ~

// Tests
// -----

// For each test case below, create a function 'make_test_X', which injects the
// values of the respective test on the net. Start with 'term_a', then do all.

// term_a = (λx(x) λy(y))
__host__ void inject_term_a(Net* net) {
  net->root     = 0x60000001;
  net->bags[ 0] = (Wire) {0xa0000000,0xa0000001};
  net->node[ 0] = (Node) {0x60000000,0x50000000};
  net->node[ 1] = (Node) {0xa0000002,0x40000000};
  net->node[ 2] = (Node) {0x60000002,0x50000002};
}

// term_b = (λfλx(f x) λy(y))
__host__ void inject_term_b(Net* net) {
  net->root     = 0x60000003;
  net->bags[ 0] = (Wire) {0xa0000000,0xa0000003};
  net->node[ 0] = (Node) {0xa0000001,0xa0000002};
  net->node[ 1] = (Node) {0x50000002,0x60000002};
  net->node[ 2] = (Node) {0x50000001,0x60000001};
  net->node[ 3] = (Node) {0xa0000004,0x40000000};
  net->node[ 4] = (Node) {0x60000004,0x50000004};
}

// term_c = (λfλx(f (f x)) λx(x))
__host__ void inject_term_c(Net* net) {
  net->root     = 0x60000005;
  net->bags[ 0] = (Wire) {0xa0000000,0xa0000005};
  net->node[ 0] = (Node) {0xb0000001,0xa0000004};
  net->node[ 1] = (Node) {0xa0000002,0xa0000003};
  net->node[ 2] = (Node) {0x50000004,0x50000003};
  net->node[ 3] = (Node) {0x60000002,0x60000004};
  net->node[ 4] = (Node) {0x50000002,0x60000003};
  net->node[ 5] = (Node) {0xa0000006,0x40000000};
  net->node[ 6] = (Node) {0x60000006,0x50000006};
}

// term_d = (λfλx(f (f x)) λgλy(g (g y)))
__host__ void inject_term_d(Net* net) {
  net->root     = 0x60000005;
  net->bags[ 0] = (Wire) {0xa0000000,0xa0000005};
  net->node[ 0] = (Node) {0xb0000001,0xa0000004};
  net->node[ 1] = (Node) {0xa0000002,0xa0000003};
  net->node[ 2] = (Node) {0x50000004,0x50000003};
  net->node[ 3] = (Node) {0x60000002,0x60000004};
  net->node[ 4] = (Node) {0x50000002,0x60000003};
  net->node[ 5] = (Node) {0xa0000006,0x40000000};
  net->node[ 6] = (Node) {0xc0000007,0xa000000a};
  net->node[ 7] = (Node) {0xa0000008,0xa0000009};
  net->node[ 8] = (Node) {0x5000000a,0x50000009};
  net->node[ 9] = (Node) {0x60000008,0x6000000a};
  net->node[10] = (Node) {0x50000008,0x60000009};
}

// term_e = (c2 g_s g_z)
__host__ void inject_term_e(Net* net) {
  net->root     = 0x6000000b;
  net->bags[ 0] = (Wire) {0xa0000000,0xa0000005};
  net->node[ 0] = (Node) {0xb0000001,0xa0000004};
  net->node[ 1] = (Node) {0xa0000002,0xa0000003};
  net->node[ 2] = (Node) {0x50000004,0x50000003};
  net->node[ 3] = (Node) {0x60000002,0x60000004};
  net->node[ 4] = (Node) {0x50000002,0x60000003};
  net->node[ 5] = (Node) {0xa0000006,0xa000000b};
  net->node[ 6] = (Node) {0xc0000007,0xa0000008};
  net->node[ 7] = (Node) {0x50000009,0x5000000a};
  net->node[ 8] = (Node) {0xa0000009,0x6000000a};
  net->node[ 9] = (Node) {0x50000007,0xa000000a};
  net->node[10] = (Node) {0x60000007,0x60000008};
  net->node[11] = (Node) {0xa000000c,0x40000000};
  net->node[12] = (Node) {0x6000000c,0x5000000c};
}

// term_f = (c3 g_s g_z)
__host__ void inject_term_f(Net* net) {
  net->root     = 0x6000000d;
  net->bags[ 0] = (Wire) {0xa0000000,0xa0000007};
  net->node[ 0] = (Node) {0xb0000001,0xa0000006};
  net->node[ 1] = (Node) {0xb0000002,0xa0000005};
  net->node[ 2] = (Node) {0xa0000003,0xa0000004};
  net->node[ 3] = (Node) {0x50000006,0x50000004};
  net->node[ 4] = (Node) {0x60000003,0x50000005};
  net->node[ 5] = (Node) {0x60000004,0x60000006};
  net->node[ 6] = (Node) {0x50000003,0x60000005};
  net->node[ 7] = (Node) {0xa0000008,0xa000000d};
  net->node[ 8] = (Node) {0xc0000009,0xa000000a};
  net->node[ 9] = (Node) {0x5000000b,0x5000000c};
  net->node[10] = (Node) {0xa000000b,0x6000000c};
  net->node[11] = (Node) {0x50000009,0xa000000c};
  net->node[12] = (Node) {0x60000009,0x6000000a};
  net->node[13] = (Node) {0xa000000e,0x40000000};
  net->node[14] = (Node) {0x6000000e,0x5000000e};
}

// term_g = (c8 g_s g_z)
__host__ void inject_term_g(Net* net) {
  net->root     = 0x60000017;
  net->bags[ 0] = (Wire) {0xa0000000,0xa0000011};
  net->node[ 0] = (Node) {0xb0000001,0xa0000010};
  net->node[ 1] = (Node) {0xb0000002,0xa000000f};
  net->node[ 2] = (Node) {0xb0000003,0xa000000e};
  net->node[ 3] = (Node) {0xb0000004,0xa000000d};
  net->node[ 4] = (Node) {0xb0000005,0xa000000c};
  net->node[ 5] = (Node) {0xb0000006,0xa000000b};
  net->node[ 6] = (Node) {0xb0000007,0xa000000a};
  net->node[ 7] = (Node) {0xa0000008,0xa0000009};
  net->node[ 8] = (Node) {0x50000010,0x50000009};
  net->node[ 9] = (Node) {0x60000008,0x5000000a};
  net->node[10] = (Node) {0x60000009,0x5000000b};
  net->node[11] = (Node) {0x6000000a,0x5000000c};
  net->node[12] = (Node) {0x6000000b,0x5000000d};
  net->node[13] = (Node) {0x6000000c,0x5000000e};
  net->node[14] = (Node) {0x6000000d,0x5000000f};
  net->node[15] = (Node) {0x6000000e,0x60000010};
  net->node[16] = (Node) {0x50000008,0x6000000f};
  net->node[17] = (Node) {0xa0000012,0xa0000017};
  net->node[18] = (Node) {0xc0000013,0xa0000014};
  net->node[19] = (Node) {0x50000015,0x50000016};
  net->node[20] = (Node) {0xa0000015,0x60000016};
  net->node[21] = (Node) {0x50000013,0xa0000016};
  net->node[22] = (Node) {0x60000013,0x60000014};
  net->node[23] = (Node) {0xa0000018,0x40000000};
  net->node[24] = (Node) {0x60000018,0x50000018};
}

// term_h = (c12 g_s g_z)
__host__ void inject_term_h(Net* net) {
  net->root     = 0x6000001f;
  net->bags[ 0] = (Wire) {0xa0000000,0xa0000019};
  net->node[ 0] = (Node) {0xb0000001,0xa0000018};
  net->node[ 1] = (Node) {0xb0000002,0xa0000017};
  net->node[ 2] = (Node) {0xb0000003,0xa0000016};
  net->node[ 3] = (Node) {0xb0000004,0xa0000015};
  net->node[ 4] = (Node) {0xb0000005,0xa0000014};
  net->node[ 5] = (Node) {0xb0000006,0xa0000013};
  net->node[ 6] = (Node) {0xb0000007,0xa0000012};
  net->node[ 7] = (Node) {0xb0000008,0xa0000011};
  net->node[ 8] = (Node) {0xb0000009,0xa0000010};
  net->node[ 9] = (Node) {0xb000000a,0xa000000f};
  net->node[10] = (Node) {0xb000000b,0xa000000e};
  net->node[11] = (Node) {0xa000000c,0xa000000d};
  net->node[12] = (Node) {0x50000018,0x5000000d};
  net->node[13] = (Node) {0x6000000c,0x5000000e};
  net->node[14] = (Node) {0x6000000d,0x5000000f};
  net->node[15] = (Node) {0x6000000e,0x50000010};
  net->node[16] = (Node) {0x6000000f,0x50000011};
  net->node[17] = (Node) {0x60000010,0x50000012};
  net->node[18] = (Node) {0x60000011,0x50000013};
  net->node[19] = (Node) {0x60000012,0x50000014};
  net->node[20] = (Node) {0x60000013,0x50000015};
  net->node[21] = (Node) {0x60000014,0x50000016};
  net->node[22] = (Node) {0x60000015,0x50000017};
  net->node[23] = (Node) {0x60000016,0x60000018};
  net->node[24] = (Node) {0x5000000c,0x60000017};
  net->node[25] = (Node) {0xa000001a,0xa000001f};
  net->node[26] = (Node) {0xc000001b,0xa000001c};
  net->node[27] = (Node) {0x5000001d,0x5000001e};
  net->node[28] = (Node) {0xa000001d,0x6000001e};
  net->node[29] = (Node) {0x5000001b,0xa000001e};
  net->node[30] = (Node) {0x6000001b,0x6000001c};
  net->node[31] = (Node) {0xa0000020,0x40000000};
  net->node[32] = (Node) {0x60000020,0x50000020};
}

// term_i = (c14 g_s g_z)
__host__ void inject_term_i(Net* net) {
  net->root     = 0x60000023;
  net->bags[ 0] = (Wire) {0xa0000000,0xa000001d};
  net->node[ 0] = (Node) {0xb0000001,0xa000001c};
  net->node[ 1] = (Node) {0xb0000002,0xa000001b};
  net->node[ 2] = (Node) {0xb0000003,0xa000001a};
  net->node[ 3] = (Node) {0xb0000004,0xa0000019};
  net->node[ 4] = (Node) {0xb0000005,0xa0000018};
  net->node[ 5] = (Node) {0xb0000006,0xa0000017};
  net->node[ 6] = (Node) {0xb0000007,0xa0000016};
  net->node[ 7] = (Node) {0xb0000008,0xa0000015};
  net->node[ 8] = (Node) {0xb0000009,0xa0000014};
  net->node[ 9] = (Node) {0xb000000a,0xa0000013};
  net->node[10] = (Node) {0xb000000b,0xa0000012};
  net->node[11] = (Node) {0xb000000c,0xa0000011};
  net->node[12] = (Node) {0xb000000d,0xa0000010};
  net->node[13] = (Node) {0xa000000e,0xa000000f};
  net->node[14] = (Node) {0x5000001c,0x5000000f};
  net->node[15] = (Node) {0x6000000e,0x50000010};
  net->node[16] = (Node) {0x6000000f,0x50000011};
  net->node[17] = (Node) {0x60000010,0x50000012};
  net->node[18] = (Node) {0x60000011,0x50000013};
  net->node[19] = (Node) {0x60000012,0x50000014};
  net->node[20] = (Node) {0x60000013,0x50000015};
  net->node[21] = (Node) {0x60000014,0x50000016};
  net->node[22] = (Node) {0x60000015,0x50000017};
  net->node[23] = (Node) {0x60000016,0x50000018};
  net->node[24] = (Node) {0x60000017,0x50000019};
  net->node[25] = (Node) {0x60000018,0x5000001a};
  net->node[26] = (Node) {0x60000019,0x5000001b};
  net->node[27] = (Node) {0x6000001a,0x6000001c};
  net->node[28] = (Node) {0x5000000e,0x6000001b};
  net->node[29] = (Node) {0xa000001e,0xa0000023};
  net->node[30] = (Node) {0xc000001f,0xa0000020};
  net->node[31] = (Node) {0x50000021,0x50000022};
  net->node[32] = (Node) {0xa0000021,0x60000022};
  net->node[33] = (Node) {0x5000001f,0xa0000022};
  net->node[34] = (Node) {0x6000001f,0x60000020};
  net->node[35] = (Node) {0xa0000024,0x40000000};
  net->node[36] = (Node) {0x60000024,0x50000024};
}

// term_j = (c18 g_s g_z)
__host__ void inject_term_j(Net* net) {
  net->root    = 0x6000002b;
  net->bags[ 0] = (Wire) {0xa0000000,0xa0000025};
  net->node[ 0] = (Node) {0xb0000001,0xa0000024};
  net->node[ 1] = (Node) {0xb0000002,0xa0000023};
  net->node[ 2] = (Node) {0xb0000003,0xa0000022};
  net->node[ 3] = (Node) {0xb0000004,0xa0000021};
  net->node[ 4] = (Node) {0xb0000005,0xa0000020};
  net->node[ 5] = (Node) {0xb0000006,0xa000001f};
  net->node[ 6] = (Node) {0xb0000007,0xa000001e};
  net->node[ 7] = (Node) {0xb0000008,0xa000001d};
  net->node[ 8] = (Node) {0xb0000009,0xa000001c};
  net->node[ 9] = (Node) {0xb000000a,0xa000001b};
  net->node[10] = (Node) {0xb000000b,0xa000001a};
  net->node[11] = (Node) {0xb000000c,0xa0000019};
  net->node[12] = (Node) {0xb000000d,0xa0000018};
  net->node[13] = (Node) {0xb000000e,0xa0000017};
  net->node[14] = (Node) {0xb000000f,0xa0000016};
  net->node[15] = (Node) {0xb0000010,0xa0000015};
  net->node[16] = (Node) {0xb0000011,0xa0000014};
  net->node[17] = (Node) {0xa0000012,0xa0000013};
  net->node[18] = (Node) {0x50000024,0x50000013};
  net->node[19] = (Node) {0x60000012,0x50000014};
  net->node[20] = (Node) {0x60000013,0x50000015};
  net->node[21] = (Node) {0x60000014,0x50000016};
  net->node[22] = (Node) {0x60000015,0x50000017};
  net->node[23] = (Node) {0x60000016,0x50000018};
  net->node[24] = (Node) {0x60000017,0x50000019};
  net->node[25] = (Node) {0x60000018,0x5000001a};
  net->node[26] = (Node) {0x60000019,0x5000001b};
  net->node[27] = (Node) {0x6000001a,0x5000001c};
  net->node[28] = (Node) {0x6000001b,0x5000001d};
  net->node[29] = (Node) {0x6000001c,0x5000001e};
  net->node[30] = (Node) {0x6000001d,0x5000001f};
  net->node[31] = (Node) {0x6000001e,0x50000020};
  net->node[32] = (Node) {0x6000001f,0x50000021};
  net->node[33] = (Node) {0x60000020,0x50000022};
  net->node[34] = (Node) {0x60000021,0x50000023};
  net->node[35] = (Node) {0x60000022,0x60000024};
  net->node[36] = (Node) {0x50000012,0x60000023};
  net->node[37] = (Node) {0xa0000026,0xa000002b};
  net->node[38] = (Node) {0xc0000027,0xa0000028};
  net->node[39] = (Node) {0x50000029,0x5000002a};
  net->node[40] = (Node) {0xa0000029,0x6000002a};
  net->node[41] = (Node) {0x50000027,0xa000002a};
  net->node[42] = (Node) {0x60000027,0x60000028};
  net->node[43] = (Node) {0xa000002c,0x40000000};
  net->node[44] = (Node) {0x6000002c,0x5000002c};
}

// Main
// ----

int main() {
  // Prints device info
  int device;
  cudaDeviceProp prop;
  cudaGetDevice(&device);
  cudaGetDeviceProperties(&prop, device);
  printf("CUDA Device: %s, Compute Capability: %d.%d\n\n", prop.name, prop.major, prop.minor);
  // print all info about the GPU, including SMs, shared memory size, etc
  printf("Total Global Memory: %zu\n", prop.totalGlobalMem);
  printf("Shared Memory per Block: %zu\n", prop.sharedMemPerBlock);
  printf("Registers per Block: %d\n", prop.regsPerBlock);
  printf("Warp Size: %d\n", prop.warpSize);
  printf("Memory Pitch: %zu\n", prop.memPitch);
  printf("Max Threads per Block: %d\n", prop.maxThreadsPerBlock);
  printf("Max Threads Dimension: %d x %d x %d\n", prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
  printf("Max Grid Size: %d x %d x %d\n", prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
  printf("Clock Rate: %d\n", prop.clockRate);
  printf("Total Constant Memory: %zu\n", prop.totalConstMem);
  printf("Compute Capability Major: %d\n", prop.major);
  printf("Compute Capability Minor: %d\n", prop.minor);
  printf("Texture Alignment: %zu\n", prop.textureAlignment);
  printf("Device Overlap: %s\n", prop.deviceOverlap ? "Yes" : "No");
  printf("Multiprocessor Count: %d\n", prop.multiProcessorCount);

  // Allocates the initial net on device
  Net* h_net = mknet();
  inject_term_i(h_net);

  // Prints the initial net
  printf("\n");
  printf("INPUT\n");
  printf("=====\n\n");
  print_net(h_net);

  // Sends the net from host to device
  Net* d_net = net_to_device(h_net);

  // Performs parallel reductions
  printf("\n");
  for (u64 i = 0; i < 64; ++i) {
    do_global_rewrite(d_net);
    do_global_scatter(d_net);
  }
  cudaDeviceSynchronize();

  // Reads the normalized net from device to host
  Net* norm = net_to_host(d_net);

  // Prints the normal form (raw data)
  printf("\n");
  printf("NORMAL (%d | %d)\n", norm->rwts, norm->blen);
  printf("======\n\n");
  //print_tree(norm, norm->root);
  //print_net(norm);
  printf("\n");

  // ----
  
  // Free device memory
  net_free_on_device(d_net);

  // Free host memory
  net_free_on_host(h_net);
  net_free_on_host(norm);

  return 0;
}
