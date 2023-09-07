#include <stdarg.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>

typedef uint8_t u8;
typedef uint16_t u16;
typedef unsigned long long int u64;

// Configuration
// -------------

// This code is initially optimized for nVidia RTX 4090
const u64 BLOCK_LOG2    = 8;                                     // log2 of block size
const u64 BLOCK_SIZE    = 1 << BLOCK_LOG2;                       // threads per block
const u64 UNIT_SIZE     = 4;                                     // threads per rewrite unit
const u64 NODE_SIZE     = 1 << 28;                               // max total nodes (2GB addressable)
const u64 HEAD_SIZE     = BLOCK_SIZE * BLOCK_SIZE * BLOCK_SIZE;  // size of unormalized heads
const u64 BAGS_SIZE     = BLOCK_SIZE * BLOCK_SIZE * BLOCK_SIZE;  // size of global redex bag
const u64 GROUP_SIZE    = BLOCK_SIZE * BLOCK_SIZE;               // size os a group of bags
const u64 GIDX_SIZE     = BAGS_SIZE + GROUP_SIZE + BLOCK_SIZE;   // aux object to hold scatter indices
const u64 GMOV_SIZE     = BAGS_SIZE;                             // aux object to hold scatter indices
const u64 REPEAT_RATE   = 64;                                    // local rewrites per global rewrite
const u64 MAX_TERM_SIZE = 16;                                    // max number of nodes in a term

// Types
// -----

typedef u16 Tag; // pointer tag: 16-bit
typedef u64 Val; // pointer val: 48-bit

// Core terms
const Tag NIL = 0x0000; // empty node
const Tag REF = 0x0001; // reference to a definition (closed net)
const Tag ERA = 0x0002; // unboxed eraser
const Tag VRR = 0x0003; // variable pointing to root
const Tag VR1 = 0x0004; // variable pointing to aux1 port of node
const Tag VR2 = 0x0005; // variable pointing to aux2 port of node
const Tag RDR = 0x0006; // redirection to root
const Tag RD1 = 0x0007; // redirection to aux1 port of node
const Tag RD2 = 0x0008; // redirection to aux2 port of node
const Tag CON = 0x1000; // points to main port of con node
const Tag DUP = 0x1001; // points to main port of dup node
const Tag CTR = 0x2000; // last constructor
const Tag NUM = 0x0100; // unboxed number

// Special values
const u64 NEO = 0xFFFFFFFFFFFFFFFD; // recently allocated value
const u64 GON = 0xFFFFFFFFFFFFFFFE; // node has been moved to redex bag
const u64 BSY = 0xFFFFFFFFFFFFFFFF; // value taken by another thread, will be replaced soon

// Worker types
const u64 A1 = 0; // focuses on the A node, P1 port
const u64 A2 = 1; // focuses on the A node, P2 port
const u64 B1 = 2; // focuses on the B node, P1 port
const u64 B2 = 3; // focuses on the B node, P2 port

// Ports (P1 or P2)
typedef u8 Port;
const u64 P1 = 0;
const u64 P2 = 1;

// Pointers = 4-bit tag + 28-bit val
typedef u64 Ptr;

// Nodes are pairs of pointers
typedef struct alignas(8) {
  Ptr ports[2];
} Node;

// Wires are pairs of pointers
typedef struct alignas(8) {
  Ptr lft;
  Ptr rgt;
} Wire;

// Maximum number of defs in a book
const u64 MAX_DEFS = 1 << 24; // FIXME: make a proper HashMap

typedef struct {
  Ptr   root;
  u64   alen;
  Wire* acts;
  u64   nlen;
  Node* node;
} Term;

// A book
typedef struct {
  Term** defs;
} Book;

// An interaction net 
typedef struct {
  Ptr   root; // root wire
  u64   blen; // total bag length (redex count)
  Wire* bags; // redex bags (active pairs)
  Node* node; // memory buffer with all nodes
  u64   hlen; // total head length
  Ptr*  head; // unormalized heads
  u64*  gidx; // aux buffer used on scatter fns
  Wire* gmov; // aux buffer used on scatter fns
  u64   done; // number of completed threads
  u64   rwts; // number of rewrites performed
} Net;

// A worker local data
typedef struct {
  u64   tid;   // thread id
  u64   bid;   // block id 
  u64   gid;   // global id
  u64   type;  // worker type (A1|A2|B1|B2)
  u64   port;  // worker port (P1|P2)
  u64   aloc;  // where to alloc next node
  u64   rwts;  // local rewrites performed
  Wire* bag;   // local redex bag
  u64*  locs;  // local alloc locs
} Worker;

// Debug
// -----

__device__ __host__ void stop(const char* tag) {
  printf(tag);
  printf("\n");
}

__device__ __host__ bool dbug(u64* K, const char* tag) {
  *K += 1;
  if (*K > 5000000) {
    stop(tag);
    return false;
  }
  return true;
}

// Runtime
// -------

// Integer ceil division
__host__ __device__ u64 div(u64 a, u64 b) {
  return (a + b - 1) / b;
}

// Pseudorandom Number Generator
__host__ __device__ u64 rng(u64 a) {
  return a * 214013 + 2531011;
}

// Creates a new pointer
__host__ __device__ inline Ptr mkptr(Tag tag, Val val) {
  return ((u64)tag << 48) | (val & 0xFFFFFFFFFFFF);
}

// Gets the tag of a pointer
__host__ __device__ inline Tag tag(Ptr ptr) {
  return (Tag)(ptr >> 48);
}

// Gets the value of a pointer
__host__ __device__ inline Val val(Ptr ptr) {
  return ptr & 0xFFFFFFFFFFFF;
}

// Is this pointer a variable?
__host__ __device__ inline bool is_var(Ptr ptr) {
  return tag(ptr) >= VRR && tag(ptr) <= VR2;
}

// Is this pointer a redirection?
__host__ __device__ inline bool is_red(Ptr ptr) {
  return tag(ptr) >= RDR && tag(ptr) <= RD2;
}

// Is this pointer a constructor?
__host__ __device__ inline bool is_ctr(Ptr ptr) {
  return tag(ptr) >= CON && tag(ptr) < CTR;
}

// Is this pointer an eraser?
__host__ __device__ inline bool is_era(Ptr ptr) {
  return tag(ptr) == ERA;
}

// Is this pointer a number?
__host__ __device__ inline bool is_num(Ptr ptr) {
  return tag(ptr) == NUM;
}

// Is this pointer a reference?
__host__ __device__ inline bool is_ref(Ptr ptr) {
  return tag(ptr) == REF;
}

// Is this pointer a main port?
__host__ __device__ inline bool is_pri(Ptr ptr) {
  return is_era(ptr)
      || is_ctr(ptr)
      || is_ref(ptr)
      || is_num(ptr);
}

// Is this pointer carrying a location (that needs adjustment)?
__host__ __device__ inline bool has_loc(Ptr ptr) {
  return is_ctr(ptr)
      || is_var(ptr) && tag(ptr) != VRR
      || is_red(ptr) && tag(ptr) != RDR;
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
__host__ __device__ Ptr* enter(Net* net, Ptr ptr) {
  Ptr* ref = target(net, ptr);
  while (tag(*ref) >= RDR && tag(*ref) <= RD2) {
    ref = target(net, *ref);
  }
  return ref;
}

// Transforms a variable into a redirection
__host__ __device__ inline Ptr redir(Ptr ptr) {
  return mkptr(tag(ptr) + (is_var(ptr) ? 3 : 0), val(ptr));
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
__device__ inline u64 alloc(Worker *worker, Net *net) {
  u64 K = 0;
  while (true) {
    u64  idx = (worker->aloc * 4 + worker->type) % NODE_SIZE;
    u64* ref = (u64*)&net->node[idx].ports[P1];
    u64  got = atomicCAS(ref, 0, NEO);
    if (got == 0) {
      if (atomicCAS(ref + 1, 0, NEO) == 0) {
        return idx;
      }
      *ref = 0;
      //atomicExch(ref, 0);
    }
    worker->aloc = (worker->aloc + 1) % NODE_SIZE;
  }
}

// Creates a new active pair
__device__ inline void put_redex(Worker* worker, Ptr a_ptr, Ptr b_ptr) {
  worker->bag[worker->type] = (Wire){a_ptr, b_ptr};
}

// Gets the value of a ref; waits if busy
__device__ Ptr take(Ptr* ref) {
  Ptr got = atomicExch((u64*)ref, BSY);
  u64 K = 0;
  while (got == BSY) {
    //dbug(&K, "take");
    got = atomicExch((u64*)ref, BSY);
  }
  return got;
}

// Attempts to replace 'exp' by 'neo', until it succeeds
__device__ void replace(Ptr* ref, Ptr exp, Ptr neo) {
  Ptr got = atomicCAS((u64*)ref, exp, neo);
  u64 K = 0;
  while (got != exp) {
    //dbug(&K, "replace");
    got = atomicCAS((u64*)ref, exp, neo);
  }
}

// Atomically links the principal in 'pri_ref' towards 'dir_ptr'
// - If target is a redirection => clear it and move forwards
// - If target is a variable    => pass the node into it and halt
// - If target is a node        => form an active pair and halt
__device__ void link(Worker* worker, Net* net, Ptr* pri_ref, Ptr dir_ptr) {
  //printf("[%04X] linking node=%8X towards %8X\n", worker->gid, *pri_ref, dir_ptr);

  u64 K = 0;
  while (true) {
    //dbug(&K, "link");
    //printf("[%04X] step\n", worker->gid);

    // We must be careful to not cross boundaries. When 'trg_ptr' is a VAR, it
    // isn't owned by us. As such, we can't 'take()' it, and must peek instead.
    Ptr* trg_ref = target(net, dir_ptr);
    Ptr  trg_ptr = atomicAdd((u64*)trg_ref, 0);

    // If trg_ptr is a redirection, clear it
    if (is_red(trg_ptr)) {
      //printf("[%04X] redir\n", worker->gid);
      u64 cleared = atomicCAS((u64*)trg_ref, trg_ptr, 0);
      if (cleared == trg_ptr) {
        dir_ptr = trg_ptr;
      }
      continue;
    }

    // If trg_ptr is a var, try replacing it by the principal
    else if (is_var(trg_ptr)) {
      //printf("[%04X] var\n", worker->gid);
      // Peeks our own principal
      Ptr pri_ptr = atomicAdd((u64*)pri_ref, 0);
      // We don't own the var, so we must try replacing with a CAS
      u64 replaced = atomicCAS((u64*)trg_ref, trg_ptr, pri_ptr);
      // If it worked, we successfully moved our principal to another region
      if (replaced == trg_ptr) {
        // Collects the backwards path, which is now orphan
        trg_ref = target(net, trg_ptr);
        trg_ptr = atomicAdd((u64*)trg_ref, 0);
        u64 K2 = 0;
        while (tag(trg_ptr) >= RDR && tag(trg_ptr) <= RD2) {
          //dbug(&K2, "inner-link");
          u64 cleared = atomicCAS((u64*)trg_ref, trg_ptr, 0);
          if (cleared == trg_ptr) {
            trg_ref = target(net, trg_ptr);
            trg_ptr = atomicAdd((u64*)trg_ref, 0);
          }
        }
        // Clear our principal
        atomicCAS((u64*)pri_ref, pri_ptr, 0);
        return;
      // Otherwise, things probably changed, so we step back and try again
      } else {
        continue;
      }
    }

    // If it is a principal, two threads will reach this branch
    // The first to arrive makes a redex, the second exits
    else if (is_pri(trg_ptr) || trg_ptr == GON) {
      //printf("[%04X] leap %8X - %8X\n", worker->gid, *fst_ref, *snd_ref);
      Ptr *fst_ref = pri_ref < trg_ref ? pri_ref : trg_ref;
      Ptr *snd_ref = pri_ref < trg_ref ? trg_ref : pri_ref;
      Ptr  fst_ptr = atomicExch((u64*)fst_ref, GON);
      if (fst_ptr == GON) {
        atomicCAS((u64*)fst_ref, GON, 0);
        replace((u64*)snd_ref, GON, 0);
        return;
      } else {
        Ptr snd_ptr = atomicExch((u64*)snd_ref, GON);
        //printf("[%4X] putleap %08X %08X\n", worker->gid, fst_ptr, snd_ptr);
        put_redex(worker, fst_ptr, snd_ptr);
        return;
      }
    }

    // If it is busy, we wait
    else if (trg_ptr == BSY) {
      //printf("[%04X] waits\n", worker->gid);
      continue;
    }

    else {
      //printf("[%04X] WTF?? ~ %8X\n", worker->gid, trg_ptr);
      return;
    }
  }
}

// ScanSum
// -------

// Performs a local scan sum
__device__ int scansum(u64* arr) {
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
      u64 tmp = arr[tid + (1 << d) - 1];
      arr[tid + (1 << (d + 0)) - 1] = arr[tid + (1 << (d + 1)) - 1];
      arr[tid + (1 << (d + 1)) - 1] += tmp;
    }
    __syncthreads();
  }

  return sum;
}

// Scatter
// -------

// Local scatter
__device__ u64 local_scatter(Net* net) {
  u64   tid = threadIdx.x;
  u64   bid = blockIdx.x;
  u64   gid = bid * blockDim.x + tid;
  u64*  loc = net->gidx + bid * BLOCK_SIZE;
  Wire* bag = net->bags + bid * BLOCK_SIZE;

  // Initializes the index object
  loc[tid] = bag[tid].lft > 0 ? 1 : 0;
  __syncthreads();

  // Computes our bag len
  u64 bag_len = scansum(loc);
  __syncthreads();

  // Takes our wire
  Wire wire = bag[tid];
  bag[tid] = Wire{0,0};
  __syncthreads();

  // Moves our wire to target location
  if (wire.lft != 0) {
    u64 serial_index = loc[tid];
    u64 spread_index = (BLOCK_SIZE / bag_len) * serial_index;
    bag[spread_index] = wire;
  }
  __syncthreads();

  return bag_len;
}

// Computes redex indices on blocks (and block lengths)
__global__ void global_scatter_prepare_0(Net* net) {
  u64  tid = threadIdx.x;
  u64  bid = blockIdx.x;
  u64  gid = bid * blockDim.x + tid;
  u64* redex_indices = net->gidx + BLOCK_SIZE * bid;
  u64* block_length  = net->gidx + BLOCK_SIZE * BLOCK_SIZE * BLOCK_SIZE + bid;
  redex_indices[tid] = net->bags[gid].lft > 0 ? 1 : 0; __syncthreads();
  *block_length      = scansum(redex_indices);
  __syncthreads();
  //printf("[%04X on %d] scatter 0 | redex_index=%d block_length=[%d,%d,%d,%d,...]\n", gid, bid, redex_indices[tid], *block_length, *(block_length+1), *(block_length+2), *(block_length+3));
}

// Computes block indices on groups (and group lengths)
__global__ void global_scatter_prepare_1(Net* net) {
  u64 tid = threadIdx.x;
  u64 bid = blockIdx.x;
  u64 gid = bid * blockDim.x + tid;
  u64* block_indices = net->gidx + BLOCK_SIZE * BLOCK_SIZE * BLOCK_SIZE + BLOCK_SIZE * bid;
  u64* group_length  = net->gidx + BLOCK_SIZE * BLOCK_SIZE * BLOCK_SIZE + BLOCK_SIZE * BLOCK_SIZE + bid;
  *group_length      = scansum(block_indices);
  //printf("[%04X on %d] scatter 1 | block_index=%d group_length=%d\n", gid, bid, block_indices[tid], *group_length);
}

// Computes group indices on bag (and bag length)
__global__ void global_scatter_prepare_2(Net* net) {
  u64* group_indices = net->gidx + BLOCK_SIZE * BLOCK_SIZE * BLOCK_SIZE + BLOCK_SIZE * BLOCK_SIZE;
  u64* total_length  = &net->blen; __syncthreads();
  *total_length      = scansum(group_indices);
  //printf("[%04X] scatter 2 | group_index=%d total_length=%d\n", threadIdx.x, group_indices[threadIdx.x], *total_length);
}

// Global scatter: takes redex from bag into aux buff
__global__ void global_scatter_take(Net* net, u64 blocks) {
  u64  tid = threadIdx.x;
  u64  bid = blockIdx.x;
  u64  gid = bid * blockDim.x + tid;
  net->gmov[gid] = net->bags[gid];
  net->bags[gid] = Wire{0,0};
}

// Global scatter: moves redex to target location
__global__ void global_scatter_move(Net* net, u64 blocks) {
  u64  tid = threadIdx.x;
  u64  bid = blockIdx.x;
  u64  gid = bid * blockDim.x + tid;

  // Block and group indices
  u64* block_index = net->gidx + BLOCK_SIZE * BLOCK_SIZE * BLOCK_SIZE + (gid / BLOCK_SIZE);
  u64* group_index = net->gidx + BLOCK_SIZE * BLOCK_SIZE * BLOCK_SIZE + BLOCK_SIZE * BLOCK_SIZE + (gid / BLOCK_SIZE / BLOCK_SIZE);

  // Takes our wire
  Wire wire = net->gmov[gid];
  net->gmov[gid] = Wire{0,0};

  // Moves wire to target location
  if (wire.lft != 0) {
    u64 serial_index = net->gidx[gid+0] + (*block_index) + (*group_index);
    u64 spread_index = (blocks * BLOCK_SIZE / net->blen) * serial_index;
    net->bags[spread_index] = wire;
  }
}

// Cleans up memory used by global scatter
__global__ void global_scatter_cleanup(Net* net) {
  u64  tid = threadIdx.x;
  u64  bid = blockIdx.x;
  u64  gid = bid * blockDim.x + tid;
  u64* redex_index = net->gidx + BLOCK_SIZE * bid + tid;
  u64* block_index = net->gidx + BLOCK_SIZE * BLOCK_SIZE * BLOCK_SIZE + bid + tid;
  u64* group_index = net->gidx + BLOCK_SIZE * BLOCK_SIZE * BLOCK_SIZE + BLOCK_SIZE * BLOCK_SIZE + (bid / BLOCK_SIZE) + tid;
  *redex_index = 0;
  if (bid % BLOCK_SIZE == 0) {
    *block_index = 0;
  }
  if (bid == 0) {
    *group_index = 0;
  }
  //printf("[%04X] clean %d %d %d\n", gid, BLOCK_SIZE * bid + tid, BLOCK_SIZE * BLOCK_SIZE * BLOCK_SIZE + bid, BLOCK_SIZE * BLOCK_SIZE * BLOCK_SIZE + BLOCK_SIZE * BLOCK_SIZE + (bid / BLOCK_SIZE));
}

// Performs a global scatter
u64 do_global_scatter(Net* net, u64 prev_blocks) {
  u64 bag_length, next_blocks;

  if (prev_blocks == 0) {
    prev_blocks = BLOCK_SIZE;
  }

  // Prepares scatter
  global_scatter_prepare_0<<<prev_blocks, BLOCK_SIZE>>>(net);
  global_scatter_prepare_1<<<div(prev_blocks, BLOCK_SIZE), BLOCK_SIZE>>>(net);
  global_scatter_prepare_2<<<div(prev_blocks, BLOCK_SIZE * BLOCK_SIZE), BLOCK_SIZE>>>(net); // always == 1

  // Gets bag length
  cudaMemcpy(&bag_length, &(net->blen), sizeof(u64), cudaMemcpyDeviceToHost);

  // Computes next block count
  next_blocks = min(div(bag_length, BLOCK_SIZE / 8), BLOCK_SIZE * BLOCK_SIZE);

  // Performs the scatter
  global_scatter_take<<<prev_blocks, BLOCK_SIZE>>>(net, next_blocks);
  global_scatter_move<<<prev_blocks, BLOCK_SIZE>>>(net, next_blocks);
  global_scatter_cleanup<<<prev_blocks, BLOCK_SIZE>>>(net);

  return next_blocks;
}

// Interactions
// ------------

__device__ Ptr adjust(Worker* worker, Ptr ptr) {
  //printf("[%04X] adjust %d | %d to %x\n", worker->gid, has_loc(ptr), val(ptr), has_loc(ptr) ? locs[val(ptr)] : val(ptr));
  return mkptr(tag(ptr), has_loc(ptr) ? worker->locs[val(ptr)] : val(ptr));
}

__device__ bool deref(Worker* worker, Net* net, Book* book, Ptr* dptr, Ptr parent) {
  // Loads definition
  Term* term = NULL;
  if (dptr != NULL) {
    term = book->defs[val(*dptr)];
  }

  // Allocates needed space
  if (term != NULL) {
    //printf("[%04X] deref: %x\n", worker->gid, val(*dref));
    for (u64 i = 0; i < div(term->nlen, (u64)4); ++i) {
      u64 loc = i * 4 + worker->type;
      if (loc < term->nlen) {
        worker->locs[loc] = alloc(worker, net);
      }
    }
  }
  __syncwarp();

  // Loads dereferenced nodes, adjusted
  if (term != NULL) {
    //printf("[%04X] deref B\n", worker->gid);
    for (u64 i = 0; i < div(term->nlen, (u64)4); ++i) {
      u64 loc = i * 4 + worker->type;
      if (loc < term->nlen) {
        Node node = term->node[loc];
        replace(at(net, worker->locs[loc], P1), NEO, adjust(worker, node.ports[P1]));
        replace(at(net, worker->locs[loc], P2), NEO, adjust(worker, node.ports[P2]));
      }
    }
  }

  // Loads dereferenced redexes, adjusted
  if (term != NULL && worker->type < term->alen) {
    //printf("[%04X] deref C\n", worker->gid);
    Wire wire = term->acts[worker->type];
    wire.lft  = adjust(worker, wire.lft);
    wire.rgt  = adjust(worker, wire.rgt);
    //printf("[%4X] putdref %08X %08X\n", worker->gid, wire.lft, wire.rgt);
    put_redex(worker, wire.lft, wire.rgt);
  }

  // Loads dereferenced root, adjusted
  if (term != NULL) {
    //printf("[%04X] deref D\n", worker->gid);
    *dptr = adjust(worker, term->root);
  }
  __syncwarp();

  // Links root
  if (term != NULL && worker->type == A1) {
    Ptr* dtrg = target(net, *dptr);
    if (dtrg != NULL) {
      *dtrg = parent;
    }
  }

  return term != NULL && term->alen > 0;
}

__device__ Worker init_worker(Net* net) {
  __shared__ u64 LOCS[BLOCK_SIZE / UNIT_SIZE * MAX_TERM_SIZE]; // aux arr for deref locs
  Worker worker;
  worker.tid  = threadIdx.x;
  worker.bid  = blockIdx.x;
  worker.gid  = worker.bid * blockDim.x + worker.tid;
  worker.aloc = rng(clock() * (worker.gid + 1));
  worker.rwts = 0;
  worker.type = worker.tid % 4;
  worker.port = worker.tid % 2;
  worker.bag  = net->bags + worker.gid / UNIT_SIZE * UNIT_SIZE;
  worker.locs = LOCS + worker.tid / UNIT_SIZE * MAX_TERM_SIZE;
  return worker;
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
__global__ void global_rewrite(Net* net, Book* book, u64 blocks) {

  // Initializes local vars
  Worker worker = init_worker(net);

  // Performs a local rewrite many times; FIXME: halt dynamically
  for (u64 tick = 0; tick < REPEAT_RATE; ++tick) {

    // Performs local scatter
    u64 redexes = local_scatter(net);

    // Counts unit redexes
    u64 wlen = 0;
    u64 widx = 0;
    for (u64 i = 0; i < 4; ++i) {
      if (worker.bag[i].lft != 0) {
        wlen += 1;
        widx = i;
      }
    }

    // Gets unit redex
    Wire wire = wlen == 1 ? worker.bag[widx] : Wire{0,0};
    __syncwarp();

    // Clears unit redex
    if (wlen == 1 && worker.type == widx) {
      worker.bag[widx] = Wire{0,0};
    }

    // Gets redex endpoints
    Ptr a_ptr = worker.type <= A2 ? wire.lft : wire.rgt;
    Ptr b_ptr = worker.type <= A2 ? wire.rgt : wire.lft;

    // Dereferences
    Ptr* dptr = NULL;
    if (is_ref(a_ptr) && is_ctr(b_ptr)) {
      dptr = &a_ptr;
    }
    if (is_ref(b_ptr) && is_ctr(a_ptr)) {
      dptr = &b_ptr;
    }
    bool is_full = deref(&worker, net, book, dptr, mkptr(NIL,0));

    // Defines type of interaction
    bool rewrite = !is_full && a_ptr != 0 && b_ptr != 0;
    bool var_pri = rewrite && is_var(a_ptr) && is_pri(b_ptr) && worker.port == P1;
    bool era_ctr = rewrite && is_era(a_ptr) && is_ctr(b_ptr);
    bool ctr_era = rewrite && is_ctr(a_ptr) && is_era(b_ptr);
    bool con_con = rewrite && is_ctr(a_ptr) && is_ctr(b_ptr) && tag(a_ptr) == tag(b_ptr);
    bool con_dup = rewrite && is_ctr(a_ptr) && is_ctr(b_ptr) && tag(a_ptr) != tag(b_ptr);

    //if (is_full || rewrite) {
      //printf("[%04llx] %llx redex? | rewrite=%d is_full=%d era_ctr=%d ctr_era=%d con_con=%d con_dup=%d opx_num=%d num_opx=%d opy_num=%d num_opy=%d opx_ctr=%d ctr_opx=%d opy_ctr=%d ctr_opy=%d | %llx %llx | %x %x\n", worker.gid, tick, rewrite, is_full, era_ctr, ctr_era, con_con, con_dup, opx_num, num_opx, opy_num, num_opy, opx_ctr, ctr_opx, opy_ctr, ctr_opy, a_ptr, b_ptr, is_num(a_ptr), is_opy(b_ptr));
    //}

    // If is_full, put this redex back
    if (is_full && worker.type == B2) {
      put_redex(&worker, a_ptr, b_ptr);
    }

    // Local rewrite variables
    Ptr *ak_ref; // ref to our aux port
    Ptr *bk_ref; // ref to other aux port
    Ptr  ak_ptr; // val of our aux port
    u64  mv_tag; // tag of ptr to send to other side
    u64  mv_loc; // loc of ptr to send to other side
    Ptr  mv_ptr; // val of ptr to send to other side
    u64  y0_idx; // idx of other clone idx

    // Inc rewrite count
    if (rewrite && worker.type == A1) {
      worker.rwts += 1;
    }

    // Gets port here
    if (rewrite && (ctr_era || con_con || con_dup)) {
      ak_ref = at(net, val(a_ptr), worker.port);
      ak_ptr = take(ak_ref);
    }

    // Gets port there
    if (rewrite && (era_ctr || con_con || con_dup)) {
      bk_ref = at(net, val(b_ptr), worker.port);
    }

    // If era_ctr, send an erasure
    if (rewrite && era_ctr) {
      mv_ptr = mkptr(ERA, 0);
    }

    // If con_con, send a redirection
    if (rewrite && con_con) {
      mv_ptr = redir(ak_ptr);
    }

    // If con_dup, send clone (CON)
    if (rewrite && con_dup) {
      mv_tag = tag(a_ptr);
      mv_loc = alloc(&worker, net); // alloc a clone
      mv_ptr = mkptr(mv_tag, mv_loc); // cloned ptr to send
      worker.locs[worker.type] = mv_loc; // pass cloned index to other threads
    }
    __syncwarp();

    // If con_dup, create inner wires between clones
    if (rewrite && con_dup) {
      u64 c1_loc = worker.locs[(worker.type <= A2 ? 2 : 0) + 0];
      u64 c2_loc = worker.locs[(worker.type <= A2 ? 2 : 0) + 1];
      replace(at(net, mv_loc, P1), NEO, mkptr(worker.port == P1 ? VR1 : VR2, c1_loc));
      replace(at(net, mv_loc, P2), NEO, mkptr(worker.port == P1 ? VR1 : VR2, c2_loc));
    }
    __syncwarp();

    // Send ptr to other side
    if (rewrite && (era_ctr || con_con || con_dup )) {
      replace(bk_ref, BSY, mv_ptr);
    }

    // If var_pri, the var is a deref root, so we just inject the node
    if (rewrite && var_pri && worker.port == P1) {
      atomicExch((u64*)target(net, a_ptr), b_ptr);
    }

    // If con_con and we sent a PRI, link the PRI there, towards our port
    // If ctr_era and we have a VAR, link the ERA  here, towards that var
    // If con_dup and we have a VAR, link the CPY  here, towards that var
    if (rewrite &&
      (  con_con && is_pri(ak_ptr)
      || ctr_era && is_var(ak_ptr)
      || con_dup && is_var(ak_ptr))) {
      Ptr targ, *node;
      if (con_con) {
        node = bk_ref;
        targ = mkptr(worker.port == P1 ? RD1 : RD2, val(a_ptr));
      } else {
        node = ak_ref;
        targ = redir(ak_ptr);
      }
      link(&worker, net, node, targ);
    }

    // If we have a PRI...
    // - if ctr_era, form an active pair with the eraser we got
    // - if con_dup, form an active pair with the clone we got
    if (rewrite &&
      (  ctr_era && is_pri(ak_ptr)
      || con_dup && is_pri(ak_ptr))) {
      //printf("[%4X] ~ %8X %8X\n", worker.gid, ak_ptr, *ak_ref);
      put_redex(&worker, ak_ptr, take(ak_ref));
      atomicCAS((u64*)ak_ref, BSY, 0);
    }
    __syncwarp();

  }

  // When the work ends, sum stats
  if (worker.rwts > 0 && worker.type == A1) {
    atomicAdd((u64*)&net->rwts, worker.rwts);
  }
}

void do_global_rewrite(Net* net, Book* book, u64 blocks) {
  global_rewrite<<<blocks, BLOCK_SIZE>>>(net, book, blocks);
}

// Expand
// ------

// FIXME: optimize...
// Pushes a new head to the head vector.
__device__ void push_head(Net* net, Ptr head, u64 index) {
  while (true) {
    u64 got = atomicCAS((u64*)&net->head[index % HEAD_SIZE], 0, head);
    if (got == 0) {
      net->hlen = 1; // FIXME: count properly
      return;
    }
    ++index;
  }
}

// Resets the hlen counter
__global__ void global_hreset(Net* net) {
  net->hlen = 0;
}

// Performs a parallel expansion of unormalized heads.
__global__ void global_expand(Net* net, Book* book) {
  __shared__ Ptr* REF[BLOCK_SIZE / UNIT_SIZE]; 
  Worker worker = init_worker(net);

  Ptr  dir = mkptr(NIL, 0);
  Ptr* ref;
  if (worker.type == A1) {
    dir = atomicExch(&net->head[worker.gid / UNIT_SIZE], 0);
    ref = REF[worker.tid / UNIT_SIZE] = target(net, dir);
  }
  __syncwarp();

  if (worker.type != A1) {
    ref = REF[worker.tid / UNIT_SIZE];
  }

  Ptr ptr = mkptr(NIL, 0);
  if (ref != NULL) {
    ptr = *ref;
  }

  //if (ref != NULL || ptr != 0) {
    //printf("[%04X] expand %p | %08X\n", worker.gid, ref, ptr);
  //}

  if (worker.type == A1 && is_ctr(ptr)) {
    push_head(net, mkptr(VR1, val(ptr)), worker.gid / UNIT_SIZE * 2 + 0);
    push_head(net, mkptr(VR2, val(ptr)), worker.gid / UNIT_SIZE * 2 + 1);
  }

  if (is_ref(ptr)) {
    deref(&worker, net, book, ref, dir);
  }
  __syncwarp();

  if (worker.type == A1 && is_ref(ptr)) {
    //printf("[%04X] pushing head %llu\n", worker.gid, worker.gid / UNIT_SIZE);
    push_head(net, dir, worker.gid / UNIT_SIZE);
  }

}

u64 do_global_expand(Net* net, Book* book) {
  global_hreset<<<1, 1>>>(net);
  global_expand<<<HEAD_SIZE / BLOCK_SIZE / UNIT_SIZE, BLOCK_SIZE>>>(net, book);
  u64 head_length = 0;
  cudaMemcpy(&head_length, &(net->hlen), sizeof(u64), cudaMemcpyDeviceToHost);
  return head_length;
}

// Reduce
// ------

// Performs global parallel rewrites.
u64 do_rewrite(Net* net, Book* book, u64* prev_blocks) {
  // Scatters redexes evenly
  u64 next_blocks = *prev_blocks = do_global_scatter(net, *prev_blocks);

  // Prints debug message
  printf(">> reducing with %d blocks\n", next_blocks);

  // Performs global parallel rewrite
  if (next_blocks > 0) {
    do_global_rewrite(net, book, next_blocks);
  }

  // Synchronize and show errors using cudaGetLastError
  //cudaDeviceSynchronize();
  //cudaError_t error = cudaGetLastError();
  //if (error != cudaSuccess) {
    //printf("CUDA error: %s\n", cudaGetErrorString(error));
  //}

  return next_blocks;
}

// Performs a single head expansion.
u64 do_expand(Net* net, Book* book) {
  return do_global_expand(net, book);
}

// Reduces until there are no more redexes.
void do_reduce(Net* net, Book* book) {
  u64 prev_blocks = 0;
  while (true) {
    if (do_rewrite(net, book, &prev_blocks) == 0) {
      break;
    }
  }
}

// Reduces until normal form.
void do_normal(Net* net, Book* book) {
  while (true) {
    do_reduce(net, book);
    if (do_expand(net, book) == 0) {
      break;
    }
  }
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
  net->hlen = 0;
  net->head = (Ptr*) malloc(HEAD_SIZE * sizeof(Ptr));
  net->gidx = (u64*) malloc(GIDX_SIZE * sizeof(u64));
  net->gmov = (Wire*)malloc(GMOV_SIZE * sizeof(Wire));
  net->node = (Node*)malloc(NODE_SIZE * sizeof(Node));
  memset(net->bags, 0, BAGS_SIZE * sizeof(Wire));
  memset(net->head, 0, HEAD_SIZE * sizeof(u64));
  memset(net->gidx, 0, GIDX_SIZE * sizeof(u64));
  memset(net->gmov, 0, GMOV_SIZE * sizeof(Wire));
  memset(net->node, 0, NODE_SIZE * sizeof(Node));
  return net;
}

__host__ Net* net_to_gpu(Net* host_net) {
  // Allocate memory on the device for the Net object, and its data
  Net*  device_net;
  Wire* device_bags;
  Ptr*  device_head;
  u64*  device_gidx;
  Wire* device_gmov;
  Node* device_node;

  cudaMalloc((void**)&device_net, sizeof(Net));
  cudaMalloc((void**)&device_bags, BAGS_SIZE * sizeof(Wire));
  cudaMalloc((void**)&device_head, HEAD_SIZE * sizeof(Ptr));
  cudaMalloc((void**)&device_gidx, GIDX_SIZE * sizeof(u64));
  cudaMalloc((void**)&device_gmov, GMOV_SIZE * sizeof(Wire));
  cudaMalloc((void**)&device_node, NODE_SIZE * sizeof(Node));

  // Copy the host data to the device memory
  cudaMemcpy(device_bags, host_net->bags, BAGS_SIZE * sizeof(Wire), cudaMemcpyHostToDevice);
  cudaMemcpy(device_head, host_net->head, HEAD_SIZE * sizeof(Ptr),  cudaMemcpyHostToDevice);
  cudaMemcpy(device_gidx, host_net->gidx, GIDX_SIZE * sizeof(u64),  cudaMemcpyHostToDevice);
  cudaMemcpy(device_gmov, host_net->gmov, GMOV_SIZE * sizeof(Wire), cudaMemcpyHostToDevice);
  cudaMemcpy(device_node, host_net->node, NODE_SIZE * sizeof(Node), cudaMemcpyHostToDevice);

  // Create a temporary host Net object with device pointers
  Net temp_net  = *host_net;
  temp_net.bags = device_bags;
  temp_net.head = device_head;
  temp_net.gidx = device_gidx;
  temp_net.gmov = device_gmov;
  temp_net.node = device_node;

  // Copy the temporary host Net object to the device memory
  cudaMemcpy(device_net, &temp_net, sizeof(Net), cudaMemcpyHostToDevice);

  // Return the device pointer to the created Net object
  return device_net;
}

__host__ Net* net_to_cpu(Net* device_net) {
  // Create a new host Net object
  Net* host_net = (Net*)malloc(sizeof(Net));

  // Copy the device Net object to the host memory
  cudaMemcpy(host_net, device_net, sizeof(Net), cudaMemcpyDeviceToHost);

  // Allocate host memory for data
  host_net->bags = (Wire*)malloc(BAGS_SIZE * sizeof(Wire));
  host_net->head = (Ptr*) malloc(HEAD_SIZE * sizeof(Ptr));
  host_net->gidx = (u64*) malloc(GIDX_SIZE * sizeof(u64));
  host_net->gmov = (Wire*)malloc(GMOV_SIZE * sizeof(Wire));
  host_net->node = (Node*)malloc(NODE_SIZE * sizeof(Node));

  // Retrieve the device pointers for data
  Wire* device_bags;
  Ptr*  device_head;
  u64*  device_gidx;
  Wire* device_gmov;
  Node* device_node;
  cudaMemcpy(&device_bags, &(device_net->bags), sizeof(Wire*), cudaMemcpyDeviceToHost);
  cudaMemcpy(&device_head, &(device_net->head), sizeof(Ptr*),  cudaMemcpyDeviceToHost);
  cudaMemcpy(&device_gidx, &(device_net->gidx), sizeof(u64*),  cudaMemcpyDeviceToHost);
  cudaMemcpy(&device_gmov, &(device_net->gmov), sizeof(Wire*), cudaMemcpyDeviceToHost);
  cudaMemcpy(&device_node, &(device_net->node), sizeof(Node*), cudaMemcpyDeviceToHost);

  // Copy the device data to the host memory
  cudaMemcpy(host_net->bags, device_bags, BAGS_SIZE * sizeof(Wire), cudaMemcpyDeviceToHost);
  cudaMemcpy(host_net->head, device_head, HEAD_SIZE * sizeof(Ptr),  cudaMemcpyDeviceToHost);
  cudaMemcpy(host_net->gidx, device_gidx, GIDX_SIZE * sizeof(u64),  cudaMemcpyDeviceToHost);
  cudaMemcpy(host_net->gmov, device_gmov, GMOV_SIZE * sizeof(Wire), cudaMemcpyDeviceToHost);
  cudaMemcpy(host_net->node, device_node, NODE_SIZE * sizeof(Node), cudaMemcpyDeviceToHost);

  return host_net;
}

__host__ void net_free_on_gpu(Net* device_net) {
  // Retrieve the device pointers for data
  Wire* device_bags;
  Ptr*  device_head;
  u64*  device_gidx;
  Wire* device_gmov;
  Node* device_node;
  cudaMemcpy(&device_bags, &(device_net->bags), sizeof(Wire*), cudaMemcpyDeviceToHost);
  cudaMemcpy(&device_head, &(device_net->head), sizeof(Ptr*),  cudaMemcpyDeviceToHost);
  cudaMemcpy(&device_gidx, &(device_net->gidx), sizeof(u64*),  cudaMemcpyDeviceToHost);
  cudaMemcpy(&device_gmov, &(device_net->gmov), sizeof(Wire*), cudaMemcpyDeviceToHost);
  cudaMemcpy(&device_node, &(device_net->node), sizeof(Node*), cudaMemcpyDeviceToHost);

  // Free the device memory
  cudaFree(device_bags);
  cudaFree(device_head);
  cudaFree(device_gidx);
  cudaFree(device_gmov);
  cudaFree(device_node);
  cudaFree(device_net);
}

__host__ void net_free_on_cpu(Net* host_net) {
  free(host_net->bags);
  free(host_net->head);
  free(host_net->gidx);
  free(host_net->gmov);
  free(host_net->node);
  free(host_net);
}

// Creates a new book
__host__ __device__ inline Book* mkbook() {
  Book* book = (Book*)malloc(sizeof(Book));
  book->defs = (Term**)malloc(MAX_DEFS * sizeof(Term*));
  memset(book->defs, 0, sizeof(book->defs));
  return book;
}

__host__ Term* term_to_gpu(Term* host_term) {
  // Allocate memory on the device for the Term object, and its data
  Term* device_term;
  Wire* device_acts;
  Node* device_node;

  cudaMalloc((void**)&device_term, sizeof(Term));
  cudaMalloc((void**)&device_acts, host_term->alen * sizeof(Wire));
  cudaMalloc((void**)&device_node, host_term->nlen * sizeof(Node));

  // Copy the host data to the device memory
  cudaMemcpy(device_acts, host_term->acts, host_term->alen * sizeof(Wire), cudaMemcpyHostToDevice);
  cudaMemcpy(device_node, host_term->node, host_term->nlen * sizeof(Node), cudaMemcpyHostToDevice);

  // Create a temporary host Term object with device pointers
  Term temp_term = *host_term;
  temp_term.acts = device_acts;
  temp_term.node = device_node;

  // Copy the temporary host Term object to the device memory
  cudaMemcpy(device_term, &temp_term, sizeof(Term), cudaMemcpyHostToDevice);

  // Return the device pointer to the created Term object
  return device_term;
}

__host__ Term* term_to_cpu(Term* device_term) {
  // Create a new host Term object
  Term* host_term = (Term*)malloc(sizeof(Term));

  // Copy the device Term object to the host memory
  cudaMemcpy(host_term, device_term, sizeof(Term), cudaMemcpyDeviceToHost);

  // Allocate host memory for data
  host_term->acts = (Wire*)malloc(host_term->alen * sizeof(Wire));
  host_term->node = (Node*)malloc(host_term->nlen * sizeof(Node));

  // Retrieve the device pointers for data
  Wire* device_acts;
  Node* device_node;
  cudaMemcpy(&device_acts, &(device_term->acts), sizeof(Wire*), cudaMemcpyDeviceToHost);
  cudaMemcpy(&device_node, &(device_term->node), sizeof(Node*), cudaMemcpyDeviceToHost);

  // Copy the device data to the host memory
  cudaMemcpy(host_term->acts, device_acts, host_term->alen * sizeof(Wire), cudaMemcpyDeviceToHost);
  cudaMemcpy(host_term->node, device_node, host_term->nlen * sizeof(Node), cudaMemcpyDeviceToHost);

  return host_term;
}

__host__ Book* book_to_gpu(Book* host_book) {
  Book* device_book;
  Term** device_defs;

  cudaMalloc((void**)&device_book, sizeof(Book));
  cudaMalloc((void**)&device_defs, MAX_DEFS * sizeof(Term*));
  cudaMemset(device_defs, 0, MAX_DEFS * sizeof(Term*));

  for (u64 i = 0; i < MAX_DEFS; ++i) {
    if (host_book->defs[i] != NULL) {
      Term* device_term = term_to_gpu(host_book->defs[i]);
      cudaMemcpy(device_defs + i, &device_term, sizeof(Term*), cudaMemcpyHostToDevice);
    }
  }

  cudaMemcpy(&(device_book->defs), &device_defs, sizeof(Term*), cudaMemcpyHostToDevice);

  return device_book;
}

// opposite of book_to_gpu; same style as net_to_cpu and term_to_cpu
__host__ Book* book_to_cpu(Book* device_book) {
  // Create a new host Book object
  Book* host_book = (Book*)malloc(sizeof(Book));

  // Copy the device Book object to the host memory
  cudaMemcpy(host_book, device_book, sizeof(Book), cudaMemcpyDeviceToHost);

  // Allocate host memory for data
  host_book->defs = (Term**)malloc(MAX_DEFS * sizeof(Term*));

  // Retrieve the device pointer for data
  Term** device_defs;
  cudaMemcpy(&device_defs, &(device_book->defs), sizeof(Term**), cudaMemcpyDeviceToHost);

  // Copies device_defs into host_book->defs
  cudaMemcpy(host_book->defs, device_defs, MAX_DEFS * sizeof(Term*), cudaMemcpyDeviceToHost);

  // Copy the device data to the host memory
  for (u64 i = 0; i < MAX_DEFS; ++i) {
    if (host_book->defs[i] != NULL) {
      host_book->defs[i] = term_to_cpu(host_book->defs[i]);
    }
  }

  return host_book;
}

__host__ void book_free_on_gpu(Book* device_book) {
  // TODO
}

__host__ void book_free_on_cpu(Book* host_book) {
  // TODO
}

// Debugging
// ---------

__host__ const char* show_ptr(Ptr ptr, u64 slot) {
  static char buffer[8][20];
  if (ptr == 0) {
    strcpy(buffer[slot], "                ");
    return buffer[slot];
  } else if (ptr == BSY) {
    strcpy(buffer[slot], "[..............]");
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
      default:tag_str = tag(ptr) >= DUP ? "DUP" : "???"; break;
    }
    snprintf(buffer[slot], sizeof(buffer[slot]), "%s:%012X", tag_str, val(ptr));
    return buffer[slot];
  }
}

// Prints a net in hexadecimal, limited to a given size
void print_net(Net* net) {
  printf("Root:\n");
  printf("- %s\n", show_ptr(net->root,0));
  printf("Bags:\n");
  for (u64 i = 0; i < BAGS_SIZE; ++i) {
    Ptr a = net->bags[i].lft;
    Ptr b = net->bags[i].rgt;
    if (a != 0 || b != 0) {
      printf("- [%07X] %s %s\n", i, show_ptr(a,0), show_ptr(b,1));
    }
  }
  printf("Node:\n");
  for (u64 i = 0; i < NODE_SIZE; ++i) {
    Ptr a = net->node[i].ports[P1];
    Ptr b = net->node[i].ports[P2];
    if (a != 0 || b != 0) {
      printf("- [%07X] %s %s\n", i, show_ptr(a,0), show_ptr(b,1));
    }
  }
  printf("HLen: %u\n", net->hlen);
  printf("BLen: %u\n", net->blen);
  printf("Rwts: %u\n", net->rwts);
  printf("\n");
}

// Struct to represent a Map of entries using a simple array of (key,id) pairs
typedef struct {
  u64 keys[65536];
  u64 vals[65536];
  u64 size;
} Map;

// Function to insert a new entry into the map
__host__ void map_insert(Map* map, u64 key, u64 val) {
  map->keys[map->size] = key;
  map->vals[map->size] = val;
  map->size++;
}

// Function to lookup an id in the map by key
__host__ u64 map_lookup(Map* map, u64 key) {
  for (u64 i = 0; i < map->size; ++i) {
    if (map->keys[i] == key) {
      return map->vals[i];
    }
  }
  return map->size;
}

// Recursive function to print a term as a tree with unique variable IDs
__host__ void print_tree_go(Net* net, Ptr ptr, Map* var_ids) {
  if (is_var(ptr)) {
    u64 got = map_lookup(var_ids, ptr);
    if (got == var_ids->size) {
      u64 name = var_ids->size;
      Ptr targ = *enter(net, ptr);
      map_insert(var_ids, targ, name);
      printf("x%d", name);
    } else {
      printf("x%d", got);
    }
  } else if (is_ref(ptr)) {
    printf("{%x}", val(ptr));
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

__host__ void populate(Book* book) {
  // E
  book->defs[0x0000000f]           = (Term*) malloc(sizeof(Term));
  book->defs[0x0000000f]->root     = 0x1000000000000000;
  book->defs[0x0000000f]->alen     = 0;
  book->defs[0x0000000f]->acts     = (Wire*) malloc(0 * sizeof(Wire));
  book->defs[0x0000000f]->nlen     = 3;
  book->defs[0x0000000f]->node     = (Node*) malloc(3 * sizeof(Node));
  book->defs[0x0000000f]->node[ 0] = (Node) {0x2000000000000,0x1000000000000001};
  book->defs[0x0000000f]->node[ 1] = (Node) {0x2000000000000,0x1000000000000002};
  book->defs[0x0000000f]->node[ 2] = (Node) {0x5000000000002,0x4000000000002};
  // F
  book->defs[0x00000010]           = (Term*) malloc(sizeof(Term));
  book->defs[0x00000010]->root     = 0x1000000000000000;
  book->defs[0x00000010]->alen     = 0;
  book->defs[0x00000010]->acts     = (Wire*) malloc(0 * sizeof(Wire));
  book->defs[0x00000010]->nlen     = 2;
  book->defs[0x00000010]->node     = (Node*) malloc(2 * sizeof(Node));
  book->defs[0x00000010]->node[ 0] = (Node) {0x2000000000000,0x1000000000000001};
  book->defs[0x00000010]->node[ 1] = (Node) {0x5000000000001,0x4000000000001};
  // I
  book->defs[0x00000013]           = (Term*) malloc(sizeof(Term));
  book->defs[0x00000013]->root     = 0x1000000000000000;
  book->defs[0x00000013]->alen     = 0;
  book->defs[0x00000013]->acts     = (Wire*) malloc(0 * sizeof(Wire));
  book->defs[0x00000013]->nlen     = 5;
  book->defs[0x00000013]->node     = (Node*) malloc(5 * sizeof(Node));
  book->defs[0x00000013]->node[ 0] = (Node) {0x4000000000003,0x1000000000000001};
  book->defs[0x00000013]->node[ 1] = (Node) {0x2000000000000,0x1000000000000002};
  book->defs[0x00000013]->node[ 2] = (Node) {0x1000000000000003,0x1000000000000004};
  book->defs[0x00000013]->node[ 3] = (Node) {0x4000000000000,0x5000000000004};
  book->defs[0x00000013]->node[ 4] = (Node) {0x2000000000000,0x5000000000003};
  // O
  book->defs[0x00000019]           = (Term*) malloc(sizeof(Term));
  book->defs[0x00000019]->root     = 0x1000000000000000;
  book->defs[0x00000019]->alen     = 0;
  book->defs[0x00000019]->acts     = (Wire*) malloc(0 * sizeof(Wire));
  book->defs[0x00000019]->nlen     = 5;
  book->defs[0x00000019]->node     = (Node*) malloc(5 * sizeof(Node));
  book->defs[0x00000019]->node[ 0] = (Node) {0x4000000000002,0x1000000000000001};
  book->defs[0x00000019]->node[ 1] = (Node) {0x1000000000000002,0x1000000000000003};
  book->defs[0x00000019]->node[ 2] = (Node) {0x4000000000000,0x5000000000004};
  book->defs[0x00000019]->node[ 3] = (Node) {0x2000000000000,0x1000000000000004};
  book->defs[0x00000019]->node[ 4] = (Node) {0x2000000000000,0x5000000000002};
  // S
  book->defs[0x0000001d]           = (Term*) malloc(sizeof(Term));
  book->defs[0x0000001d]->root     = 0x1000000000000000;
  book->defs[0x0000001d]->alen     = 0;
  book->defs[0x0000001d]->acts     = (Wire*) malloc(0 * sizeof(Wire));
  book->defs[0x0000001d]->nlen     = 4;
  book->defs[0x0000001d]->node     = (Node*) malloc(4 * sizeof(Node));
  book->defs[0x0000001d]->node[ 0] = (Node) {0x4000000000002,0x1000000000000001};
  book->defs[0x0000001d]->node[ 1] = (Node) {0x1000000000000002,0x1000000000000003};
  book->defs[0x0000001d]->node[ 2] = (Node) {0x4000000000000,0x5000000000003};
  book->defs[0x0000001d]->node[ 3] = (Node) {0x2000000000000,0x5000000000002};
  // T
  book->defs[0x0000001e]           = (Term*) malloc(sizeof(Term));
  book->defs[0x0000001e]->root     = 0x1000000000000000;
  book->defs[0x0000001e]->alen     = 0;
  book->defs[0x0000001e]->acts     = (Wire*) malloc(0 * sizeof(Wire));
  book->defs[0x0000001e]->nlen     = 2;
  book->defs[0x0000001e]->node     = (Node*) malloc(2 * sizeof(Node));
  book->defs[0x0000001e]->node[ 0] = (Node) {0x5000000000001,0x1000000000000001};
  book->defs[0x0000001e]->node[ 1] = (Node) {0x2000000000000,0x4000000000000};
  // Z
  book->defs[0x00000024]           = (Term*) malloc(sizeof(Term));
  book->defs[0x00000024]->root     = 0x1000000000000000;
  book->defs[0x00000024]->alen     = 0;
  book->defs[0x00000024]->acts     = (Wire*) malloc(0 * sizeof(Wire));
  book->defs[0x00000024]->nlen     = 2;
  book->defs[0x00000024]->node     = (Node*) malloc(2 * sizeof(Node));
  book->defs[0x00000024]->node[ 0] = (Node) {0x2000000000000,0x1000000000000001};
  book->defs[0x00000024]->node[ 1] = (Node) {0x5000000000001,0x4000000000001};
  // c0
  book->defs[0x000009c1]           = (Term*) malloc(sizeof(Term));
  book->defs[0x000009c1]->root     = 0x1000000000000000;
  book->defs[0x000009c1]->alen     = 0;
  book->defs[0x000009c1]->acts     = (Wire*) malloc(0 * sizeof(Wire));
  book->defs[0x000009c1]->nlen     = 2;
  book->defs[0x000009c1]->node     = (Node*) malloc(2 * sizeof(Node));
  book->defs[0x000009c1]->node[ 0] = (Node) {0x2000000000000,0x1000000000000001};
  book->defs[0x000009c1]->node[ 1] = (Node) {0x5000000000001,0x4000000000001};
  // c1
  book->defs[0x000009c2]           = (Term*) malloc(sizeof(Term));
  book->defs[0x000009c2]->root     = 0x1000000000000000;
  book->defs[0x000009c2]->alen     = 0;
  book->defs[0x000009c2]->acts     = (Wire*) malloc(0 * sizeof(Wire));
  book->defs[0x000009c2]->nlen     = 3;
  book->defs[0x000009c2]->node     = (Node*) malloc(3 * sizeof(Node));
  book->defs[0x000009c2]->node[ 0] = (Node) {0x1000000000000001,0x1000000000000002};
  book->defs[0x000009c2]->node[ 1] = (Node) {0x4000000000002,0x5000000000002};
  book->defs[0x000009c2]->node[ 2] = (Node) {0x4000000000001,0x5000000000001};
  // c2
  book->defs[0x000009c3]           = (Term*) malloc(sizeof(Term));
  book->defs[0x000009c3]->root     = 0x1000000000000000;
  book->defs[0x000009c3]->alen     = 0;
  book->defs[0x000009c3]->acts     = (Wire*) malloc(0 * sizeof(Wire));
  book->defs[0x000009c3]->nlen     = 5;
  book->defs[0x000009c3]->node     = (Node*) malloc(5 * sizeof(Node));
  book->defs[0x000009c3]->node[ 0] = (Node) {0x1001000000000001,0x1000000000000004};
  book->defs[0x000009c3]->node[ 1] = (Node) {0x1000000000000002,0x1000000000000003};
  book->defs[0x000009c3]->node[ 2] = (Node) {0x4000000000004,0x4000000000003};
  book->defs[0x000009c3]->node[ 3] = (Node) {0x5000000000002,0x5000000000004};
  book->defs[0x000009c3]->node[ 4] = (Node) {0x4000000000002,0x5000000000003};
  // c3
  book->defs[0x000009c4]           = (Term*) malloc(sizeof(Term));
  book->defs[0x000009c4]->root     = 0x1000000000000000;
  book->defs[0x000009c4]->alen     = 0;
  book->defs[0x000009c4]->acts     = (Wire*) malloc(0 * sizeof(Wire));
  book->defs[0x000009c4]->nlen     = 7;
  book->defs[0x000009c4]->node     = (Node*) malloc(7 * sizeof(Node));
  book->defs[0x000009c4]->node[ 0] = (Node) {0x1001000000000001,0x1000000000000006};
  book->defs[0x000009c4]->node[ 1] = (Node) {0x1001000000000002,0x1000000000000005};
  book->defs[0x000009c4]->node[ 2] = (Node) {0x1000000000000003,0x1000000000000004};
  book->defs[0x000009c4]->node[ 3] = (Node) {0x4000000000006,0x4000000000004};
  book->defs[0x000009c4]->node[ 4] = (Node) {0x5000000000003,0x4000000000005};
  book->defs[0x000009c4]->node[ 5] = (Node) {0x5000000000004,0x5000000000006};
  book->defs[0x000009c4]->node[ 6] = (Node) {0x4000000000003,0x5000000000005};
  // c4
  book->defs[0x000009c5]           = (Term*) malloc(sizeof(Term));
  book->defs[0x000009c5]->root     = 0x1000000000000000;
  book->defs[0x000009c5]->alen     = 0;
  book->defs[0x000009c5]->acts     = (Wire*) malloc(0 * sizeof(Wire));
  book->defs[0x000009c5]->nlen     = 9;
  book->defs[0x000009c5]->node     = (Node*) malloc(9 * sizeof(Node));
  book->defs[0x000009c5]->node[ 0] = (Node) {0x1001000000000001,0x1000000000000008};
  book->defs[0x000009c5]->node[ 1] = (Node) {0x1001000000000002,0x1000000000000007};
  book->defs[0x000009c5]->node[ 2] = (Node) {0x1001000000000003,0x1000000000000006};
  book->defs[0x000009c5]->node[ 3] = (Node) {0x1000000000000004,0x1000000000000005};
  book->defs[0x000009c5]->node[ 4] = (Node) {0x4000000000008,0x4000000000005};
  book->defs[0x000009c5]->node[ 5] = (Node) {0x5000000000004,0x4000000000006};
  book->defs[0x000009c5]->node[ 6] = (Node) {0x5000000000005,0x4000000000007};
  book->defs[0x000009c5]->node[ 7] = (Node) {0x5000000000006,0x5000000000008};
  book->defs[0x000009c5]->node[ 8] = (Node) {0x4000000000004,0x5000000000007};
  // c5
  book->defs[0x000009c6]           = (Term*) malloc(sizeof(Term));
  book->defs[0x000009c6]->root     = 0x1000000000000000;
  book->defs[0x000009c6]->alen     = 0;
  book->defs[0x000009c6]->acts     = (Wire*) malloc(0 * sizeof(Wire));
  book->defs[0x000009c6]->nlen     = 11;
  book->defs[0x000009c6]->node     = (Node*) malloc(11 * sizeof(Node));
  book->defs[0x000009c6]->node[ 0] = (Node) {0x1001000000000001,0x100000000000000a};
  book->defs[0x000009c6]->node[ 1] = (Node) {0x1001000000000002,0x1000000000000009};
  book->defs[0x000009c6]->node[ 2] = (Node) {0x1001000000000003,0x1000000000000008};
  book->defs[0x000009c6]->node[ 3] = (Node) {0x1001000000000004,0x1000000000000007};
  book->defs[0x000009c6]->node[ 4] = (Node) {0x1000000000000005,0x1000000000000006};
  book->defs[0x000009c6]->node[ 5] = (Node) {0x400000000000a,0x4000000000006};
  book->defs[0x000009c6]->node[ 6] = (Node) {0x5000000000005,0x4000000000007};
  book->defs[0x000009c6]->node[ 7] = (Node) {0x5000000000006,0x4000000000008};
  book->defs[0x000009c6]->node[ 8] = (Node) {0x5000000000007,0x4000000000009};
  book->defs[0x000009c6]->node[ 9] = (Node) {0x5000000000008,0x500000000000a};
  book->defs[0x000009c6]->node[10] = (Node) {0x4000000000005,0x5000000000009};
  // c6
  book->defs[0x000009c7]           = (Term*) malloc(sizeof(Term));
  book->defs[0x000009c7]->root     = 0x1000000000000000;
  book->defs[0x000009c7]->alen     = 0;
  book->defs[0x000009c7]->acts     = (Wire*) malloc(0 * sizeof(Wire));
  book->defs[0x000009c7]->nlen     = 13;
  book->defs[0x000009c7]->node     = (Node*) malloc(13 * sizeof(Node));
  book->defs[0x000009c7]->node[ 0] = (Node) {0x1001000000000001,0x100000000000000c};
  book->defs[0x000009c7]->node[ 1] = (Node) {0x1001000000000002,0x100000000000000b};
  book->defs[0x000009c7]->node[ 2] = (Node) {0x1001000000000003,0x100000000000000a};
  book->defs[0x000009c7]->node[ 3] = (Node) {0x1001000000000004,0x1000000000000009};
  book->defs[0x000009c7]->node[ 4] = (Node) {0x1001000000000005,0x1000000000000008};
  book->defs[0x000009c7]->node[ 5] = (Node) {0x1000000000000006,0x1000000000000007};
  book->defs[0x000009c7]->node[ 6] = (Node) {0x400000000000c,0x4000000000007};
  book->defs[0x000009c7]->node[ 7] = (Node) {0x5000000000006,0x4000000000008};
  book->defs[0x000009c7]->node[ 8] = (Node) {0x5000000000007,0x4000000000009};
  book->defs[0x000009c7]->node[ 9] = (Node) {0x5000000000008,0x400000000000a};
  book->defs[0x000009c7]->node[10] = (Node) {0x5000000000009,0x400000000000b};
  book->defs[0x000009c7]->node[11] = (Node) {0x500000000000a,0x500000000000c};
  book->defs[0x000009c7]->node[12] = (Node) {0x4000000000006,0x500000000000b};
  // c7
  book->defs[0x000009c8]           = (Term*) malloc(sizeof(Term));
  book->defs[0x000009c8]->root     = 0x1000000000000000;
  book->defs[0x000009c8]->alen     = 0;
  book->defs[0x000009c8]->acts     = (Wire*) malloc(0 * sizeof(Wire));
  book->defs[0x000009c8]->nlen     = 15;
  book->defs[0x000009c8]->node     = (Node*) malloc(15 * sizeof(Node));
  book->defs[0x000009c8]->node[ 0] = (Node) {0x1001000000000001,0x100000000000000e};
  book->defs[0x000009c8]->node[ 1] = (Node) {0x1001000000000002,0x100000000000000d};
  book->defs[0x000009c8]->node[ 2] = (Node) {0x1001000000000003,0x100000000000000c};
  book->defs[0x000009c8]->node[ 3] = (Node) {0x1001000000000004,0x100000000000000b};
  book->defs[0x000009c8]->node[ 4] = (Node) {0x1001000000000005,0x100000000000000a};
  book->defs[0x000009c8]->node[ 5] = (Node) {0x1001000000000006,0x1000000000000009};
  book->defs[0x000009c8]->node[ 6] = (Node) {0x1000000000000007,0x1000000000000008};
  book->defs[0x000009c8]->node[ 7] = (Node) {0x400000000000e,0x4000000000008};
  book->defs[0x000009c8]->node[ 8] = (Node) {0x5000000000007,0x4000000000009};
  book->defs[0x000009c8]->node[ 9] = (Node) {0x5000000000008,0x400000000000a};
  book->defs[0x000009c8]->node[10] = (Node) {0x5000000000009,0x400000000000b};
  book->defs[0x000009c8]->node[11] = (Node) {0x500000000000a,0x400000000000c};
  book->defs[0x000009c8]->node[12] = (Node) {0x500000000000b,0x400000000000d};
  book->defs[0x000009c8]->node[13] = (Node) {0x500000000000c,0x500000000000e};
  book->defs[0x000009c8]->node[14] = (Node) {0x4000000000007,0x500000000000d};
  // c8
  book->defs[0x000009c9]           = (Term*) malloc(sizeof(Term));
  book->defs[0x000009c9]->root     = 0x1000000000000000;
  book->defs[0x000009c9]->alen     = 0;
  book->defs[0x000009c9]->acts     = (Wire*) malloc(0 * sizeof(Wire));
  book->defs[0x000009c9]->nlen     = 17;
  book->defs[0x000009c9]->node     = (Node*) malloc(17 * sizeof(Node));
  book->defs[0x000009c9]->node[ 0] = (Node) {0x1001000000000001,0x1000000000000010};
  book->defs[0x000009c9]->node[ 1] = (Node) {0x1001000000000002,0x100000000000000f};
  book->defs[0x000009c9]->node[ 2] = (Node) {0x1001000000000003,0x100000000000000e};
  book->defs[0x000009c9]->node[ 3] = (Node) {0x1001000000000004,0x100000000000000d};
  book->defs[0x000009c9]->node[ 4] = (Node) {0x1001000000000005,0x100000000000000c};
  book->defs[0x000009c9]->node[ 5] = (Node) {0x1001000000000006,0x100000000000000b};
  book->defs[0x000009c9]->node[ 6] = (Node) {0x1001000000000007,0x100000000000000a};
  book->defs[0x000009c9]->node[ 7] = (Node) {0x1000000000000008,0x1000000000000009};
  book->defs[0x000009c9]->node[ 8] = (Node) {0x4000000000010,0x4000000000009};
  book->defs[0x000009c9]->node[ 9] = (Node) {0x5000000000008,0x400000000000a};
  book->defs[0x000009c9]->node[10] = (Node) {0x5000000000009,0x400000000000b};
  book->defs[0x000009c9]->node[11] = (Node) {0x500000000000a,0x400000000000c};
  book->defs[0x000009c9]->node[12] = (Node) {0x500000000000b,0x400000000000d};
  book->defs[0x000009c9]->node[13] = (Node) {0x500000000000c,0x400000000000e};
  book->defs[0x000009c9]->node[14] = (Node) {0x500000000000d,0x400000000000f};
  book->defs[0x000009c9]->node[15] = (Node) {0x500000000000e,0x5000000000010};
  book->defs[0x000009c9]->node[16] = (Node) {0x4000000000008,0x500000000000f};
  // c9
  book->defs[0x000009ca]           = (Term*) malloc(sizeof(Term));
  book->defs[0x000009ca]->root     = 0x1000000000000000;
  book->defs[0x000009ca]->alen     = 0;
  book->defs[0x000009ca]->acts     = (Wire*) malloc(0 * sizeof(Wire));
  book->defs[0x000009ca]->nlen     = 19;
  book->defs[0x000009ca]->node     = (Node*) malloc(19 * sizeof(Node));
  book->defs[0x000009ca]->node[ 0] = (Node) {0x1001000000000001,0x1000000000000012};
  book->defs[0x000009ca]->node[ 1] = (Node) {0x1001000000000002,0x1000000000000011};
  book->defs[0x000009ca]->node[ 2] = (Node) {0x1001000000000003,0x1000000000000010};
  book->defs[0x000009ca]->node[ 3] = (Node) {0x1001000000000004,0x100000000000000f};
  book->defs[0x000009ca]->node[ 4] = (Node) {0x1001000000000005,0x100000000000000e};
  book->defs[0x000009ca]->node[ 5] = (Node) {0x1001000000000006,0x100000000000000d};
  book->defs[0x000009ca]->node[ 6] = (Node) {0x1001000000000007,0x100000000000000c};
  book->defs[0x000009ca]->node[ 7] = (Node) {0x1001000000000008,0x100000000000000b};
  book->defs[0x000009ca]->node[ 8] = (Node) {0x1000000000000009,0x100000000000000a};
  book->defs[0x000009ca]->node[ 9] = (Node) {0x4000000000012,0x400000000000a};
  book->defs[0x000009ca]->node[10] = (Node) {0x5000000000009,0x400000000000b};
  book->defs[0x000009ca]->node[11] = (Node) {0x500000000000a,0x400000000000c};
  book->defs[0x000009ca]->node[12] = (Node) {0x500000000000b,0x400000000000d};
  book->defs[0x000009ca]->node[13] = (Node) {0x500000000000c,0x400000000000e};
  book->defs[0x000009ca]->node[14] = (Node) {0x500000000000d,0x400000000000f};
  book->defs[0x000009ca]->node[15] = (Node) {0x500000000000e,0x4000000000010};
  book->defs[0x000009ca]->node[16] = (Node) {0x500000000000f,0x4000000000011};
  book->defs[0x000009ca]->node[17] = (Node) {0x5000000000010,0x5000000000012};
  book->defs[0x000009ca]->node[18] = (Node) {0x4000000000009,0x5000000000011};
  // id
  book->defs[0x00000b68]           = (Term*) malloc(sizeof(Term));
  book->defs[0x00000b68]->root     = 0x1000000000000000;
  book->defs[0x00000b68]->alen     = 0;
  book->defs[0x00000b68]->acts     = (Wire*) malloc(0 * sizeof(Wire));
  book->defs[0x00000b68]->nlen     = 1;
  book->defs[0x00000b68]->node     = (Node*) malloc(1 * sizeof(Node));
  book->defs[0x00000b68]->node[ 0] = (Node) {0x5000000000000,0x4000000000000};
  // k0
  book->defs[0x00000bc1]           = (Term*) malloc(sizeof(Term));
  book->defs[0x00000bc1]->root     = 0x1000000000000000;
  book->defs[0x00000bc1]->alen     = 0;
  book->defs[0x00000bc1]->acts     = (Wire*) malloc(0 * sizeof(Wire));
  book->defs[0x00000bc1]->nlen     = 2;
  book->defs[0x00000bc1]->node     = (Node*) malloc(2 * sizeof(Node));
  book->defs[0x00000bc1]->node[ 0] = (Node) {0x2000000000000,0x1000000000000001};
  book->defs[0x00000bc1]->node[ 1] = (Node) {0x5000000000001,0x4000000000001};
  // k1
  book->defs[0x00000bc2]           = (Term*) malloc(sizeof(Term));
  book->defs[0x00000bc2]->root     = 0x1000000000000000;
  book->defs[0x00000bc2]->alen     = 0;
  book->defs[0x00000bc2]->acts     = (Wire*) malloc(0 * sizeof(Wire));
  book->defs[0x00000bc2]->nlen     = 3;
  book->defs[0x00000bc2]->node     = (Node*) malloc(3 * sizeof(Node));
  book->defs[0x00000bc2]->node[ 0] = (Node) {0x1000000000000001,0x1000000000000002};
  book->defs[0x00000bc2]->node[ 1] = (Node) {0x4000000000002,0x5000000000002};
  book->defs[0x00000bc2]->node[ 2] = (Node) {0x4000000000001,0x5000000000001};
  // k2
  book->defs[0x00000bc3]           = (Term*) malloc(sizeof(Term));
  book->defs[0x00000bc3]->root     = 0x1000000000000000;
  book->defs[0x00000bc3]->alen     = 0;
  book->defs[0x00000bc3]->acts     = (Wire*) malloc(0 * sizeof(Wire));
  book->defs[0x00000bc3]->nlen     = 5;
  book->defs[0x00000bc3]->node     = (Node*) malloc(5 * sizeof(Node));
  book->defs[0x00000bc3]->node[ 0] = (Node) {0x1002000000000001,0x1000000000000004};
  book->defs[0x00000bc3]->node[ 1] = (Node) {0x1000000000000002,0x1000000000000003};
  book->defs[0x00000bc3]->node[ 2] = (Node) {0x4000000000004,0x4000000000003};
  book->defs[0x00000bc3]->node[ 3] = (Node) {0x5000000000002,0x5000000000004};
  book->defs[0x00000bc3]->node[ 4] = (Node) {0x4000000000002,0x5000000000003};
  // k3
  book->defs[0x00000bc4]           = (Term*) malloc(sizeof(Term));
  book->defs[0x00000bc4]->root     = 0x1000000000000000;
  book->defs[0x00000bc4]->alen     = 0;
  book->defs[0x00000bc4]->acts     = (Wire*) malloc(0 * sizeof(Wire));
  book->defs[0x00000bc4]->nlen     = 7;
  book->defs[0x00000bc4]->node     = (Node*) malloc(7 * sizeof(Node));
  book->defs[0x00000bc4]->node[ 0] = (Node) {0x1002000000000001,0x1000000000000006};
  book->defs[0x00000bc4]->node[ 1] = (Node) {0x1002000000000002,0x1000000000000005};
  book->defs[0x00000bc4]->node[ 2] = (Node) {0x1000000000000003,0x1000000000000004};
  book->defs[0x00000bc4]->node[ 3] = (Node) {0x4000000000006,0x4000000000004};
  book->defs[0x00000bc4]->node[ 4] = (Node) {0x5000000000003,0x4000000000005};
  book->defs[0x00000bc4]->node[ 5] = (Node) {0x5000000000004,0x5000000000006};
  book->defs[0x00000bc4]->node[ 6] = (Node) {0x4000000000003,0x5000000000005};
  // k4
  book->defs[0x00000bc5]           = (Term*) malloc(sizeof(Term));
  book->defs[0x00000bc5]->root     = 0x1000000000000000;
  book->defs[0x00000bc5]->alen     = 0;
  book->defs[0x00000bc5]->acts     = (Wire*) malloc(0 * sizeof(Wire));
  book->defs[0x00000bc5]->nlen     = 9;
  book->defs[0x00000bc5]->node     = (Node*) malloc(9 * sizeof(Node));
  book->defs[0x00000bc5]->node[ 0] = (Node) {0x1002000000000001,0x1000000000000008};
  book->defs[0x00000bc5]->node[ 1] = (Node) {0x1002000000000002,0x1000000000000007};
  book->defs[0x00000bc5]->node[ 2] = (Node) {0x1002000000000003,0x1000000000000006};
  book->defs[0x00000bc5]->node[ 3] = (Node) {0x1000000000000004,0x1000000000000005};
  book->defs[0x00000bc5]->node[ 4] = (Node) {0x4000000000008,0x4000000000005};
  book->defs[0x00000bc5]->node[ 5] = (Node) {0x5000000000004,0x4000000000006};
  book->defs[0x00000bc5]->node[ 6] = (Node) {0x5000000000005,0x4000000000007};
  book->defs[0x00000bc5]->node[ 7] = (Node) {0x5000000000006,0x5000000000008};
  book->defs[0x00000bc5]->node[ 8] = (Node) {0x4000000000004,0x5000000000007};
  // k5
  book->defs[0x00000bc6]           = (Term*) malloc(sizeof(Term));
  book->defs[0x00000bc6]->root     = 0x1000000000000000;
  book->defs[0x00000bc6]->alen     = 0;
  book->defs[0x00000bc6]->acts     = (Wire*) malloc(0 * sizeof(Wire));
  book->defs[0x00000bc6]->nlen     = 11;
  book->defs[0x00000bc6]->node     = (Node*) malloc(11 * sizeof(Node));
  book->defs[0x00000bc6]->node[ 0] = (Node) {0x1002000000000001,0x100000000000000a};
  book->defs[0x00000bc6]->node[ 1] = (Node) {0x1002000000000002,0x1000000000000009};
  book->defs[0x00000bc6]->node[ 2] = (Node) {0x1002000000000003,0x1000000000000008};
  book->defs[0x00000bc6]->node[ 3] = (Node) {0x1002000000000004,0x1000000000000007};
  book->defs[0x00000bc6]->node[ 4] = (Node) {0x1000000000000005,0x1000000000000006};
  book->defs[0x00000bc6]->node[ 5] = (Node) {0x400000000000a,0x4000000000006};
  book->defs[0x00000bc6]->node[ 6] = (Node) {0x5000000000005,0x4000000000007};
  book->defs[0x00000bc6]->node[ 7] = (Node) {0x5000000000006,0x4000000000008};
  book->defs[0x00000bc6]->node[ 8] = (Node) {0x5000000000007,0x4000000000009};
  book->defs[0x00000bc6]->node[ 9] = (Node) {0x5000000000008,0x500000000000a};
  book->defs[0x00000bc6]->node[10] = (Node) {0x4000000000005,0x5000000000009};
  // k6
  book->defs[0x00000bc7]           = (Term*) malloc(sizeof(Term));
  book->defs[0x00000bc7]->root     = 0x1000000000000000;
  book->defs[0x00000bc7]->alen     = 0;
  book->defs[0x00000bc7]->acts     = (Wire*) malloc(0 * sizeof(Wire));
  book->defs[0x00000bc7]->nlen     = 13;
  book->defs[0x00000bc7]->node     = (Node*) malloc(13 * sizeof(Node));
  book->defs[0x00000bc7]->node[ 0] = (Node) {0x1002000000000001,0x100000000000000c};
  book->defs[0x00000bc7]->node[ 1] = (Node) {0x1002000000000002,0x100000000000000b};
  book->defs[0x00000bc7]->node[ 2] = (Node) {0x1002000000000003,0x100000000000000a};
  book->defs[0x00000bc7]->node[ 3] = (Node) {0x1002000000000004,0x1000000000000009};
  book->defs[0x00000bc7]->node[ 4] = (Node) {0x1002000000000005,0x1000000000000008};
  book->defs[0x00000bc7]->node[ 5] = (Node) {0x1000000000000006,0x1000000000000007};
  book->defs[0x00000bc7]->node[ 6] = (Node) {0x400000000000c,0x4000000000007};
  book->defs[0x00000bc7]->node[ 7] = (Node) {0x5000000000006,0x4000000000008};
  book->defs[0x00000bc7]->node[ 8] = (Node) {0x5000000000007,0x4000000000009};
  book->defs[0x00000bc7]->node[ 9] = (Node) {0x5000000000008,0x400000000000a};
  book->defs[0x00000bc7]->node[10] = (Node) {0x5000000000009,0x400000000000b};
  book->defs[0x00000bc7]->node[11] = (Node) {0x500000000000a,0x500000000000c};
  book->defs[0x00000bc7]->node[12] = (Node) {0x4000000000006,0x500000000000b};
  // k7
  book->defs[0x00000bc8]           = (Term*) malloc(sizeof(Term));
  book->defs[0x00000bc8]->root     = 0x1000000000000000;
  book->defs[0x00000bc8]->alen     = 0;
  book->defs[0x00000bc8]->acts     = (Wire*) malloc(0 * sizeof(Wire));
  book->defs[0x00000bc8]->nlen     = 15;
  book->defs[0x00000bc8]->node     = (Node*) malloc(15 * sizeof(Node));
  book->defs[0x00000bc8]->node[ 0] = (Node) {0x1002000000000001,0x100000000000000e};
  book->defs[0x00000bc8]->node[ 1] = (Node) {0x1002000000000002,0x100000000000000d};
  book->defs[0x00000bc8]->node[ 2] = (Node) {0x1002000000000003,0x100000000000000c};
  book->defs[0x00000bc8]->node[ 3] = (Node) {0x1002000000000004,0x100000000000000b};
  book->defs[0x00000bc8]->node[ 4] = (Node) {0x1002000000000005,0x100000000000000a};
  book->defs[0x00000bc8]->node[ 5] = (Node) {0x1002000000000006,0x1000000000000009};
  book->defs[0x00000bc8]->node[ 6] = (Node) {0x1000000000000007,0x1000000000000008};
  book->defs[0x00000bc8]->node[ 7] = (Node) {0x400000000000e,0x4000000000008};
  book->defs[0x00000bc8]->node[ 8] = (Node) {0x5000000000007,0x4000000000009};
  book->defs[0x00000bc8]->node[ 9] = (Node) {0x5000000000008,0x400000000000a};
  book->defs[0x00000bc8]->node[10] = (Node) {0x5000000000009,0x400000000000b};
  book->defs[0x00000bc8]->node[11] = (Node) {0x500000000000a,0x400000000000c};
  book->defs[0x00000bc8]->node[12] = (Node) {0x500000000000b,0x400000000000d};
  book->defs[0x00000bc8]->node[13] = (Node) {0x500000000000c,0x500000000000e};
  book->defs[0x00000bc8]->node[14] = (Node) {0x4000000000007,0x500000000000d};
  // k8
  book->defs[0x00000bc9]           = (Term*) malloc(sizeof(Term));
  book->defs[0x00000bc9]->root     = 0x1000000000000000;
  book->defs[0x00000bc9]->alen     = 0;
  book->defs[0x00000bc9]->acts     = (Wire*) malloc(0 * sizeof(Wire));
  book->defs[0x00000bc9]->nlen     = 17;
  book->defs[0x00000bc9]->node     = (Node*) malloc(17 * sizeof(Node));
  book->defs[0x00000bc9]->node[ 0] = (Node) {0x1002000000000001,0x1000000000000010};
  book->defs[0x00000bc9]->node[ 1] = (Node) {0x1002000000000002,0x100000000000000f};
  book->defs[0x00000bc9]->node[ 2] = (Node) {0x1002000000000003,0x100000000000000e};
  book->defs[0x00000bc9]->node[ 3] = (Node) {0x1002000000000004,0x100000000000000d};
  book->defs[0x00000bc9]->node[ 4] = (Node) {0x1002000000000005,0x100000000000000c};
  book->defs[0x00000bc9]->node[ 5] = (Node) {0x1002000000000006,0x100000000000000b};
  book->defs[0x00000bc9]->node[ 6] = (Node) {0x1002000000000007,0x100000000000000a};
  book->defs[0x00000bc9]->node[ 7] = (Node) {0x1000000000000008,0x1000000000000009};
  book->defs[0x00000bc9]->node[ 8] = (Node) {0x4000000000010,0x4000000000009};
  book->defs[0x00000bc9]->node[ 9] = (Node) {0x5000000000008,0x400000000000a};
  book->defs[0x00000bc9]->node[10] = (Node) {0x5000000000009,0x400000000000b};
  book->defs[0x00000bc9]->node[11] = (Node) {0x500000000000a,0x400000000000c};
  book->defs[0x00000bc9]->node[12] = (Node) {0x500000000000b,0x400000000000d};
  book->defs[0x00000bc9]->node[13] = (Node) {0x500000000000c,0x400000000000e};
  book->defs[0x00000bc9]->node[14] = (Node) {0x500000000000d,0x400000000000f};
  book->defs[0x00000bc9]->node[15] = (Node) {0x500000000000e,0x5000000000010};
  book->defs[0x00000bc9]->node[16] = (Node) {0x4000000000008,0x500000000000f};
  // k9
  book->defs[0x00000bca]           = (Term*) malloc(sizeof(Term));
  book->defs[0x00000bca]->root     = 0x1000000000000000;
  book->defs[0x00000bca]->alen     = 0;
  book->defs[0x00000bca]->acts     = (Wire*) malloc(0 * sizeof(Wire));
  book->defs[0x00000bca]->nlen     = 19;
  book->defs[0x00000bca]->node     = (Node*) malloc(19 * sizeof(Node));
  book->defs[0x00000bca]->node[ 0] = (Node) {0x1002000000000001,0x1000000000000012};
  book->defs[0x00000bca]->node[ 1] = (Node) {0x1002000000000002,0x1000000000000011};
  book->defs[0x00000bca]->node[ 2] = (Node) {0x1002000000000003,0x1000000000000010};
  book->defs[0x00000bca]->node[ 3] = (Node) {0x1002000000000004,0x100000000000000f};
  book->defs[0x00000bca]->node[ 4] = (Node) {0x1002000000000005,0x100000000000000e};
  book->defs[0x00000bca]->node[ 5] = (Node) {0x1002000000000006,0x100000000000000d};
  book->defs[0x00000bca]->node[ 6] = (Node) {0x1002000000000007,0x100000000000000c};
  book->defs[0x00000bca]->node[ 7] = (Node) {0x1002000000000008,0x100000000000000b};
  book->defs[0x00000bca]->node[ 8] = (Node) {0x1000000000000009,0x100000000000000a};
  book->defs[0x00000bca]->node[ 9] = (Node) {0x4000000000012,0x400000000000a};
  book->defs[0x00000bca]->node[10] = (Node) {0x5000000000009,0x400000000000b};
  book->defs[0x00000bca]->node[11] = (Node) {0x500000000000a,0x400000000000c};
  book->defs[0x00000bca]->node[12] = (Node) {0x500000000000b,0x400000000000d};
  book->defs[0x00000bca]->node[13] = (Node) {0x500000000000c,0x400000000000e};
  book->defs[0x00000bca]->node[14] = (Node) {0x500000000000d,0x400000000000f};
  book->defs[0x00000bca]->node[15] = (Node) {0x500000000000e,0x4000000000010};
  book->defs[0x00000bca]->node[16] = (Node) {0x500000000000f,0x4000000000011};
  book->defs[0x00000bca]->node[17] = (Node) {0x5000000000010,0x5000000000012};
  book->defs[0x00000bca]->node[18] = (Node) {0x4000000000009,0x5000000000011};
  // brn
  book->defs[0x00026db2]           = (Term*) malloc(sizeof(Term));
  book->defs[0x00026db2]->root     = 0x1000000000000000;
  book->defs[0x00026db2]->alen     = 0;
  book->defs[0x00026db2]->acts     = (Wire*) malloc(0 * sizeof(Wire));
  book->defs[0x00026db2]->nlen     = 3;
  book->defs[0x00026db2]->node     = (Node*) malloc(3 * sizeof(Node));
  book->defs[0x00026db2]->node[ 0] = (Node) {0x1000000000000001,0x5000000000002};
  book->defs[0x00026db2]->node[ 1] = (Node) {0x10000009b6c9d,0x1000000000000002};
  book->defs[0x00026db2]->node[ 2] = (Node) {0x10000009b6ca4,0x5000000000000};
  // c10
  book->defs[0x00027081]           = (Term*) malloc(sizeof(Term));
  book->defs[0x00027081]->root     = 0x1000000000000000;
  book->defs[0x00027081]->alen     = 0;
  book->defs[0x00027081]->acts     = (Wire*) malloc(0 * sizeof(Wire));
  book->defs[0x00027081]->nlen     = 21;
  book->defs[0x00027081]->node     = (Node*) malloc(21 * sizeof(Node));
  book->defs[0x00027081]->node[ 0] = (Node) {0x1001000000000001,0x1000000000000014};
  book->defs[0x00027081]->node[ 1] = (Node) {0x1001000000000002,0x1000000000000013};
  book->defs[0x00027081]->node[ 2] = (Node) {0x1001000000000003,0x1000000000000012};
  book->defs[0x00027081]->node[ 3] = (Node) {0x1001000000000004,0x1000000000000011};
  book->defs[0x00027081]->node[ 4] = (Node) {0x1001000000000005,0x1000000000000010};
  book->defs[0x00027081]->node[ 5] = (Node) {0x1001000000000006,0x100000000000000f};
  book->defs[0x00027081]->node[ 6] = (Node) {0x1001000000000007,0x100000000000000e};
  book->defs[0x00027081]->node[ 7] = (Node) {0x1001000000000008,0x100000000000000d};
  book->defs[0x00027081]->node[ 8] = (Node) {0x1001000000000009,0x100000000000000c};
  book->defs[0x00027081]->node[ 9] = (Node) {0x100000000000000a,0x100000000000000b};
  book->defs[0x00027081]->node[10] = (Node) {0x4000000000014,0x400000000000b};
  book->defs[0x00027081]->node[11] = (Node) {0x500000000000a,0x400000000000c};
  book->defs[0x00027081]->node[12] = (Node) {0x500000000000b,0x400000000000d};
  book->defs[0x00027081]->node[13] = (Node) {0x500000000000c,0x400000000000e};
  book->defs[0x00027081]->node[14] = (Node) {0x500000000000d,0x400000000000f};
  book->defs[0x00027081]->node[15] = (Node) {0x500000000000e,0x4000000000010};
  book->defs[0x00027081]->node[16] = (Node) {0x500000000000f,0x4000000000011};
  book->defs[0x00027081]->node[17] = (Node) {0x5000000000010,0x4000000000012};
  book->defs[0x00027081]->node[18] = (Node) {0x5000000000011,0x4000000000013};
  book->defs[0x00027081]->node[19] = (Node) {0x5000000000012,0x5000000000014};
  book->defs[0x00027081]->node[20] = (Node) {0x400000000000a,0x5000000000013};
  // c11
  book->defs[0x00027082]           = (Term*) malloc(sizeof(Term));
  book->defs[0x00027082]->root     = 0x1000000000000000;
  book->defs[0x00027082]->alen     = 0;
  book->defs[0x00027082]->acts     = (Wire*) malloc(0 * sizeof(Wire));
  book->defs[0x00027082]->nlen     = 23;
  book->defs[0x00027082]->node     = (Node*) malloc(23 * sizeof(Node));
  book->defs[0x00027082]->node[ 0] = (Node) {0x1001000000000001,0x1000000000000016};
  book->defs[0x00027082]->node[ 1] = (Node) {0x1001000000000002,0x1000000000000015};
  book->defs[0x00027082]->node[ 2] = (Node) {0x1001000000000003,0x1000000000000014};
  book->defs[0x00027082]->node[ 3] = (Node) {0x1001000000000004,0x1000000000000013};
  book->defs[0x00027082]->node[ 4] = (Node) {0x1001000000000005,0x1000000000000012};
  book->defs[0x00027082]->node[ 5] = (Node) {0x1001000000000006,0x1000000000000011};
  book->defs[0x00027082]->node[ 6] = (Node) {0x1001000000000007,0x1000000000000010};
  book->defs[0x00027082]->node[ 7] = (Node) {0x1001000000000008,0x100000000000000f};
  book->defs[0x00027082]->node[ 8] = (Node) {0x1001000000000009,0x100000000000000e};
  book->defs[0x00027082]->node[ 9] = (Node) {0x100100000000000a,0x100000000000000d};
  book->defs[0x00027082]->node[10] = (Node) {0x100000000000000b,0x100000000000000c};
  book->defs[0x00027082]->node[11] = (Node) {0x4000000000016,0x400000000000c};
  book->defs[0x00027082]->node[12] = (Node) {0x500000000000b,0x400000000000d};
  book->defs[0x00027082]->node[13] = (Node) {0x500000000000c,0x400000000000e};
  book->defs[0x00027082]->node[14] = (Node) {0x500000000000d,0x400000000000f};
  book->defs[0x00027082]->node[15] = (Node) {0x500000000000e,0x4000000000010};
  book->defs[0x00027082]->node[16] = (Node) {0x500000000000f,0x4000000000011};
  book->defs[0x00027082]->node[17] = (Node) {0x5000000000010,0x4000000000012};
  book->defs[0x00027082]->node[18] = (Node) {0x5000000000011,0x4000000000013};
  book->defs[0x00027082]->node[19] = (Node) {0x5000000000012,0x4000000000014};
  book->defs[0x00027082]->node[20] = (Node) {0x5000000000013,0x4000000000015};
  book->defs[0x00027082]->node[21] = (Node) {0x5000000000014,0x5000000000016};
  book->defs[0x00027082]->node[22] = (Node) {0x400000000000b,0x5000000000015};
  // c12
  book->defs[0x00027083]           = (Term*) malloc(sizeof(Term));
  book->defs[0x00027083]->root     = 0x1000000000000000;
  book->defs[0x00027083]->alen     = 0;
  book->defs[0x00027083]->acts     = (Wire*) malloc(0 * sizeof(Wire));
  book->defs[0x00027083]->nlen     = 25;
  book->defs[0x00027083]->node     = (Node*) malloc(25 * sizeof(Node));
  book->defs[0x00027083]->node[ 0] = (Node) {0x1001000000000001,0x1000000000000018};
  book->defs[0x00027083]->node[ 1] = (Node) {0x1001000000000002,0x1000000000000017};
  book->defs[0x00027083]->node[ 2] = (Node) {0x1001000000000003,0x1000000000000016};
  book->defs[0x00027083]->node[ 3] = (Node) {0x1001000000000004,0x1000000000000015};
  book->defs[0x00027083]->node[ 4] = (Node) {0x1001000000000005,0x1000000000000014};
  book->defs[0x00027083]->node[ 5] = (Node) {0x1001000000000006,0x1000000000000013};
  book->defs[0x00027083]->node[ 6] = (Node) {0x1001000000000007,0x1000000000000012};
  book->defs[0x00027083]->node[ 7] = (Node) {0x1001000000000008,0x1000000000000011};
  book->defs[0x00027083]->node[ 8] = (Node) {0x1001000000000009,0x1000000000000010};
  book->defs[0x00027083]->node[ 9] = (Node) {0x100100000000000a,0x100000000000000f};
  book->defs[0x00027083]->node[10] = (Node) {0x100100000000000b,0x100000000000000e};
  book->defs[0x00027083]->node[11] = (Node) {0x100000000000000c,0x100000000000000d};
  book->defs[0x00027083]->node[12] = (Node) {0x4000000000018,0x400000000000d};
  book->defs[0x00027083]->node[13] = (Node) {0x500000000000c,0x400000000000e};
  book->defs[0x00027083]->node[14] = (Node) {0x500000000000d,0x400000000000f};
  book->defs[0x00027083]->node[15] = (Node) {0x500000000000e,0x4000000000010};
  book->defs[0x00027083]->node[16] = (Node) {0x500000000000f,0x4000000000011};
  book->defs[0x00027083]->node[17] = (Node) {0x5000000000010,0x4000000000012};
  book->defs[0x00027083]->node[18] = (Node) {0x5000000000011,0x4000000000013};
  book->defs[0x00027083]->node[19] = (Node) {0x5000000000012,0x4000000000014};
  book->defs[0x00027083]->node[20] = (Node) {0x5000000000013,0x4000000000015};
  book->defs[0x00027083]->node[21] = (Node) {0x5000000000014,0x4000000000016};
  book->defs[0x00027083]->node[22] = (Node) {0x5000000000015,0x4000000000017};
  book->defs[0x00027083]->node[23] = (Node) {0x5000000000016,0x5000000000018};
  book->defs[0x00027083]->node[24] = (Node) {0x400000000000c,0x5000000000017};
  // c13
  book->defs[0x00027084]           = (Term*) malloc(sizeof(Term));
  book->defs[0x00027084]->root     = 0x1000000000000000;
  book->defs[0x00027084]->alen     = 0;
  book->defs[0x00027084]->acts     = (Wire*) malloc(0 * sizeof(Wire));
  book->defs[0x00027084]->nlen     = 27;
  book->defs[0x00027084]->node     = (Node*) malloc(27 * sizeof(Node));
  book->defs[0x00027084]->node[ 0] = (Node) {0x1001000000000001,0x100000000000001a};
  book->defs[0x00027084]->node[ 1] = (Node) {0x1001000000000002,0x1000000000000019};
  book->defs[0x00027084]->node[ 2] = (Node) {0x1001000000000003,0x1000000000000018};
  book->defs[0x00027084]->node[ 3] = (Node) {0x1001000000000004,0x1000000000000017};
  book->defs[0x00027084]->node[ 4] = (Node) {0x1001000000000005,0x1000000000000016};
  book->defs[0x00027084]->node[ 5] = (Node) {0x1001000000000006,0x1000000000000015};
  book->defs[0x00027084]->node[ 6] = (Node) {0x1001000000000007,0x1000000000000014};
  book->defs[0x00027084]->node[ 7] = (Node) {0x1001000000000008,0x1000000000000013};
  book->defs[0x00027084]->node[ 8] = (Node) {0x1001000000000009,0x1000000000000012};
  book->defs[0x00027084]->node[ 9] = (Node) {0x100100000000000a,0x1000000000000011};
  book->defs[0x00027084]->node[10] = (Node) {0x100100000000000b,0x1000000000000010};
  book->defs[0x00027084]->node[11] = (Node) {0x100100000000000c,0x100000000000000f};
  book->defs[0x00027084]->node[12] = (Node) {0x100000000000000d,0x100000000000000e};
  book->defs[0x00027084]->node[13] = (Node) {0x400000000001a,0x400000000000e};
  book->defs[0x00027084]->node[14] = (Node) {0x500000000000d,0x400000000000f};
  book->defs[0x00027084]->node[15] = (Node) {0x500000000000e,0x4000000000010};
  book->defs[0x00027084]->node[16] = (Node) {0x500000000000f,0x4000000000011};
  book->defs[0x00027084]->node[17] = (Node) {0x5000000000010,0x4000000000012};
  book->defs[0x00027084]->node[18] = (Node) {0x5000000000011,0x4000000000013};
  book->defs[0x00027084]->node[19] = (Node) {0x5000000000012,0x4000000000014};
  book->defs[0x00027084]->node[20] = (Node) {0x5000000000013,0x4000000000015};
  book->defs[0x00027084]->node[21] = (Node) {0x5000000000014,0x4000000000016};
  book->defs[0x00027084]->node[22] = (Node) {0x5000000000015,0x4000000000017};
  book->defs[0x00027084]->node[23] = (Node) {0x5000000000016,0x4000000000018};
  book->defs[0x00027084]->node[24] = (Node) {0x5000000000017,0x4000000000019};
  book->defs[0x00027084]->node[25] = (Node) {0x5000000000018,0x500000000001a};
  book->defs[0x00027084]->node[26] = (Node) {0x400000000000d,0x5000000000019};
  // c14
  book->defs[0x00027085]           = (Term*) malloc(sizeof(Term));
  book->defs[0x00027085]->root     = 0x1000000000000000;
  book->defs[0x00027085]->alen     = 0;
  book->defs[0x00027085]->acts     = (Wire*) malloc(0 * sizeof(Wire));
  book->defs[0x00027085]->nlen     = 29;
  book->defs[0x00027085]->node     = (Node*) malloc(29 * sizeof(Node));
  book->defs[0x00027085]->node[ 0] = (Node) {0x1001000000000001,0x100000000000001c};
  book->defs[0x00027085]->node[ 1] = (Node) {0x1001000000000002,0x100000000000001b};
  book->defs[0x00027085]->node[ 2] = (Node) {0x1001000000000003,0x100000000000001a};
  book->defs[0x00027085]->node[ 3] = (Node) {0x1001000000000004,0x1000000000000019};
  book->defs[0x00027085]->node[ 4] = (Node) {0x1001000000000005,0x1000000000000018};
  book->defs[0x00027085]->node[ 5] = (Node) {0x1001000000000006,0x1000000000000017};
  book->defs[0x00027085]->node[ 6] = (Node) {0x1001000000000007,0x1000000000000016};
  book->defs[0x00027085]->node[ 7] = (Node) {0x1001000000000008,0x1000000000000015};
  book->defs[0x00027085]->node[ 8] = (Node) {0x1001000000000009,0x1000000000000014};
  book->defs[0x00027085]->node[ 9] = (Node) {0x100100000000000a,0x1000000000000013};
  book->defs[0x00027085]->node[10] = (Node) {0x100100000000000b,0x1000000000000012};
  book->defs[0x00027085]->node[11] = (Node) {0x100100000000000c,0x1000000000000011};
  book->defs[0x00027085]->node[12] = (Node) {0x100100000000000d,0x1000000000000010};
  book->defs[0x00027085]->node[13] = (Node) {0x100000000000000e,0x100000000000000f};
  book->defs[0x00027085]->node[14] = (Node) {0x400000000001c,0x400000000000f};
  book->defs[0x00027085]->node[15] = (Node) {0x500000000000e,0x4000000000010};
  book->defs[0x00027085]->node[16] = (Node) {0x500000000000f,0x4000000000011};
  book->defs[0x00027085]->node[17] = (Node) {0x5000000000010,0x4000000000012};
  book->defs[0x00027085]->node[18] = (Node) {0x5000000000011,0x4000000000013};
  book->defs[0x00027085]->node[19] = (Node) {0x5000000000012,0x4000000000014};
  book->defs[0x00027085]->node[20] = (Node) {0x5000000000013,0x4000000000015};
  book->defs[0x00027085]->node[21] = (Node) {0x5000000000014,0x4000000000016};
  book->defs[0x00027085]->node[22] = (Node) {0x5000000000015,0x4000000000017};
  book->defs[0x00027085]->node[23] = (Node) {0x5000000000016,0x4000000000018};
  book->defs[0x00027085]->node[24] = (Node) {0x5000000000017,0x4000000000019};
  book->defs[0x00027085]->node[25] = (Node) {0x5000000000018,0x400000000001a};
  book->defs[0x00027085]->node[26] = (Node) {0x5000000000019,0x400000000001b};
  book->defs[0x00027085]->node[27] = (Node) {0x500000000001a,0x500000000001c};
  book->defs[0x00027085]->node[28] = (Node) {0x400000000000e,0x500000000001b};
  // c15
  book->defs[0x00027086]           = (Term*) malloc(sizeof(Term));
  book->defs[0x00027086]->root     = 0x1000000000000000;
  book->defs[0x00027086]->alen     = 0;
  book->defs[0x00027086]->acts     = (Wire*) malloc(0 * sizeof(Wire));
  book->defs[0x00027086]->nlen     = 31;
  book->defs[0x00027086]->node     = (Node*) malloc(31 * sizeof(Node));
  book->defs[0x00027086]->node[ 0] = (Node) {0x1001000000000001,0x100000000000001e};
  book->defs[0x00027086]->node[ 1] = (Node) {0x1001000000000002,0x100000000000001d};
  book->defs[0x00027086]->node[ 2] = (Node) {0x1001000000000003,0x100000000000001c};
  book->defs[0x00027086]->node[ 3] = (Node) {0x1001000000000004,0x100000000000001b};
  book->defs[0x00027086]->node[ 4] = (Node) {0x1001000000000005,0x100000000000001a};
  book->defs[0x00027086]->node[ 5] = (Node) {0x1001000000000006,0x1000000000000019};
  book->defs[0x00027086]->node[ 6] = (Node) {0x1001000000000007,0x1000000000000018};
  book->defs[0x00027086]->node[ 7] = (Node) {0x1001000000000008,0x1000000000000017};
  book->defs[0x00027086]->node[ 8] = (Node) {0x1001000000000009,0x1000000000000016};
  book->defs[0x00027086]->node[ 9] = (Node) {0x100100000000000a,0x1000000000000015};
  book->defs[0x00027086]->node[10] = (Node) {0x100100000000000b,0x1000000000000014};
  book->defs[0x00027086]->node[11] = (Node) {0x100100000000000c,0x1000000000000013};
  book->defs[0x00027086]->node[12] = (Node) {0x100100000000000d,0x1000000000000012};
  book->defs[0x00027086]->node[13] = (Node) {0x100100000000000e,0x1000000000000011};
  book->defs[0x00027086]->node[14] = (Node) {0x100000000000000f,0x1000000000000010};
  book->defs[0x00027086]->node[15] = (Node) {0x400000000001e,0x4000000000010};
  book->defs[0x00027086]->node[16] = (Node) {0x500000000000f,0x4000000000011};
  book->defs[0x00027086]->node[17] = (Node) {0x5000000000010,0x4000000000012};
  book->defs[0x00027086]->node[18] = (Node) {0x5000000000011,0x4000000000013};
  book->defs[0x00027086]->node[19] = (Node) {0x5000000000012,0x4000000000014};
  book->defs[0x00027086]->node[20] = (Node) {0x5000000000013,0x4000000000015};
  book->defs[0x00027086]->node[21] = (Node) {0x5000000000014,0x4000000000016};
  book->defs[0x00027086]->node[22] = (Node) {0x5000000000015,0x4000000000017};
  book->defs[0x00027086]->node[23] = (Node) {0x5000000000016,0x4000000000018};
  book->defs[0x00027086]->node[24] = (Node) {0x5000000000017,0x4000000000019};
  book->defs[0x00027086]->node[25] = (Node) {0x5000000000018,0x400000000001a};
  book->defs[0x00027086]->node[26] = (Node) {0x5000000000019,0x400000000001b};
  book->defs[0x00027086]->node[27] = (Node) {0x500000000001a,0x400000000001c};
  book->defs[0x00027086]->node[28] = (Node) {0x500000000001b,0x400000000001d};
  book->defs[0x00027086]->node[29] = (Node) {0x500000000001c,0x500000000001e};
  book->defs[0x00027086]->node[30] = (Node) {0x400000000000f,0x500000000001d};
  // c16
  book->defs[0x00027087]           = (Term*) malloc(sizeof(Term));
  book->defs[0x00027087]->root     = 0x1000000000000000;
  book->defs[0x00027087]->alen     = 0;
  book->defs[0x00027087]->acts     = (Wire*) malloc(0 * sizeof(Wire));
  book->defs[0x00027087]->nlen     = 33;
  book->defs[0x00027087]->node     = (Node*) malloc(33 * sizeof(Node));
  book->defs[0x00027087]->node[ 0] = (Node) {0x1001000000000001,0x1000000000000020};
  book->defs[0x00027087]->node[ 1] = (Node) {0x1001000000000002,0x100000000000001f};
  book->defs[0x00027087]->node[ 2] = (Node) {0x1001000000000003,0x100000000000001e};
  book->defs[0x00027087]->node[ 3] = (Node) {0x1001000000000004,0x100000000000001d};
  book->defs[0x00027087]->node[ 4] = (Node) {0x1001000000000005,0x100000000000001c};
  book->defs[0x00027087]->node[ 5] = (Node) {0x1001000000000006,0x100000000000001b};
  book->defs[0x00027087]->node[ 6] = (Node) {0x1001000000000007,0x100000000000001a};
  book->defs[0x00027087]->node[ 7] = (Node) {0x1001000000000008,0x1000000000000019};
  book->defs[0x00027087]->node[ 8] = (Node) {0x1001000000000009,0x1000000000000018};
  book->defs[0x00027087]->node[ 9] = (Node) {0x100100000000000a,0x1000000000000017};
  book->defs[0x00027087]->node[10] = (Node) {0x100100000000000b,0x1000000000000016};
  book->defs[0x00027087]->node[11] = (Node) {0x100100000000000c,0x1000000000000015};
  book->defs[0x00027087]->node[12] = (Node) {0x100100000000000d,0x1000000000000014};
  book->defs[0x00027087]->node[13] = (Node) {0x100100000000000e,0x1000000000000013};
  book->defs[0x00027087]->node[14] = (Node) {0x100100000000000f,0x1000000000000012};
  book->defs[0x00027087]->node[15] = (Node) {0x1000000000000010,0x1000000000000011};
  book->defs[0x00027087]->node[16] = (Node) {0x4000000000020,0x4000000000011};
  book->defs[0x00027087]->node[17] = (Node) {0x5000000000010,0x4000000000012};
  book->defs[0x00027087]->node[18] = (Node) {0x5000000000011,0x4000000000013};
  book->defs[0x00027087]->node[19] = (Node) {0x5000000000012,0x4000000000014};
  book->defs[0x00027087]->node[20] = (Node) {0x5000000000013,0x4000000000015};
  book->defs[0x00027087]->node[21] = (Node) {0x5000000000014,0x4000000000016};
  book->defs[0x00027087]->node[22] = (Node) {0x5000000000015,0x4000000000017};
  book->defs[0x00027087]->node[23] = (Node) {0x5000000000016,0x4000000000018};
  book->defs[0x00027087]->node[24] = (Node) {0x5000000000017,0x4000000000019};
  book->defs[0x00027087]->node[25] = (Node) {0x5000000000018,0x400000000001a};
  book->defs[0x00027087]->node[26] = (Node) {0x5000000000019,0x400000000001b};
  book->defs[0x00027087]->node[27] = (Node) {0x500000000001a,0x400000000001c};
  book->defs[0x00027087]->node[28] = (Node) {0x500000000001b,0x400000000001d};
  book->defs[0x00027087]->node[29] = (Node) {0x500000000001c,0x400000000001e};
  book->defs[0x00027087]->node[30] = (Node) {0x500000000001d,0x400000000001f};
  book->defs[0x00027087]->node[31] = (Node) {0x500000000001e,0x5000000000020};
  book->defs[0x00027087]->node[32] = (Node) {0x4000000000010,0x500000000001f};
  // c17
  book->defs[0x00027088]           = (Term*) malloc(sizeof(Term));
  book->defs[0x00027088]->root     = 0x1000000000000000;
  book->defs[0x00027088]->alen     = 0;
  book->defs[0x00027088]->acts     = (Wire*) malloc(0 * sizeof(Wire));
  book->defs[0x00027088]->nlen     = 35;
  book->defs[0x00027088]->node     = (Node*) malloc(35 * sizeof(Node));
  book->defs[0x00027088]->node[ 0] = (Node) {0x1001000000000001,0x1000000000000022};
  book->defs[0x00027088]->node[ 1] = (Node) {0x1001000000000002,0x1000000000000021};
  book->defs[0x00027088]->node[ 2] = (Node) {0x1001000000000003,0x1000000000000020};
  book->defs[0x00027088]->node[ 3] = (Node) {0x1001000000000004,0x100000000000001f};
  book->defs[0x00027088]->node[ 4] = (Node) {0x1001000000000005,0x100000000000001e};
  book->defs[0x00027088]->node[ 5] = (Node) {0x1001000000000006,0x100000000000001d};
  book->defs[0x00027088]->node[ 6] = (Node) {0x1001000000000007,0x100000000000001c};
  book->defs[0x00027088]->node[ 7] = (Node) {0x1001000000000008,0x100000000000001b};
  book->defs[0x00027088]->node[ 8] = (Node) {0x1001000000000009,0x100000000000001a};
  book->defs[0x00027088]->node[ 9] = (Node) {0x100100000000000a,0x1000000000000019};
  book->defs[0x00027088]->node[10] = (Node) {0x100100000000000b,0x1000000000000018};
  book->defs[0x00027088]->node[11] = (Node) {0x100100000000000c,0x1000000000000017};
  book->defs[0x00027088]->node[12] = (Node) {0x100100000000000d,0x1000000000000016};
  book->defs[0x00027088]->node[13] = (Node) {0x100100000000000e,0x1000000000000015};
  book->defs[0x00027088]->node[14] = (Node) {0x100100000000000f,0x1000000000000014};
  book->defs[0x00027088]->node[15] = (Node) {0x1001000000000010,0x1000000000000013};
  book->defs[0x00027088]->node[16] = (Node) {0x1000000000000011,0x1000000000000012};
  book->defs[0x00027088]->node[17] = (Node) {0x4000000000022,0x4000000000012};
  book->defs[0x00027088]->node[18] = (Node) {0x5000000000011,0x4000000000013};
  book->defs[0x00027088]->node[19] = (Node) {0x5000000000012,0x4000000000014};
  book->defs[0x00027088]->node[20] = (Node) {0x5000000000013,0x4000000000015};
  book->defs[0x00027088]->node[21] = (Node) {0x5000000000014,0x4000000000016};
  book->defs[0x00027088]->node[22] = (Node) {0x5000000000015,0x4000000000017};
  book->defs[0x00027088]->node[23] = (Node) {0x5000000000016,0x4000000000018};
  book->defs[0x00027088]->node[24] = (Node) {0x5000000000017,0x4000000000019};
  book->defs[0x00027088]->node[25] = (Node) {0x5000000000018,0x400000000001a};
  book->defs[0x00027088]->node[26] = (Node) {0x5000000000019,0x400000000001b};
  book->defs[0x00027088]->node[27] = (Node) {0x500000000001a,0x400000000001c};
  book->defs[0x00027088]->node[28] = (Node) {0x500000000001b,0x400000000001d};
  book->defs[0x00027088]->node[29] = (Node) {0x500000000001c,0x400000000001e};
  book->defs[0x00027088]->node[30] = (Node) {0x500000000001d,0x400000000001f};
  book->defs[0x00027088]->node[31] = (Node) {0x500000000001e,0x4000000000020};
  book->defs[0x00027088]->node[32] = (Node) {0x500000000001f,0x4000000000021};
  book->defs[0x00027088]->node[33] = (Node) {0x5000000000020,0x5000000000022};
  book->defs[0x00027088]->node[34] = (Node) {0x4000000000011,0x5000000000021};
  // c18
  book->defs[0x00027089]           = (Term*) malloc(sizeof(Term));
  book->defs[0x00027089]->root     = 0x1000000000000000;
  book->defs[0x00027089]->alen     = 0;
  book->defs[0x00027089]->acts     = (Wire*) malloc(0 * sizeof(Wire));
  book->defs[0x00027089]->nlen     = 37;
  book->defs[0x00027089]->node     = (Node*) malloc(37 * sizeof(Node));
  book->defs[0x00027089]->node[ 0] = (Node) {0x1001000000000001,0x1000000000000024};
  book->defs[0x00027089]->node[ 1] = (Node) {0x1001000000000002,0x1000000000000023};
  book->defs[0x00027089]->node[ 2] = (Node) {0x1001000000000003,0x1000000000000022};
  book->defs[0x00027089]->node[ 3] = (Node) {0x1001000000000004,0x1000000000000021};
  book->defs[0x00027089]->node[ 4] = (Node) {0x1001000000000005,0x1000000000000020};
  book->defs[0x00027089]->node[ 5] = (Node) {0x1001000000000006,0x100000000000001f};
  book->defs[0x00027089]->node[ 6] = (Node) {0x1001000000000007,0x100000000000001e};
  book->defs[0x00027089]->node[ 7] = (Node) {0x1001000000000008,0x100000000000001d};
  book->defs[0x00027089]->node[ 8] = (Node) {0x1001000000000009,0x100000000000001c};
  book->defs[0x00027089]->node[ 9] = (Node) {0x100100000000000a,0x100000000000001b};
  book->defs[0x00027089]->node[10] = (Node) {0x100100000000000b,0x100000000000001a};
  book->defs[0x00027089]->node[11] = (Node) {0x100100000000000c,0x1000000000000019};
  book->defs[0x00027089]->node[12] = (Node) {0x100100000000000d,0x1000000000000018};
  book->defs[0x00027089]->node[13] = (Node) {0x100100000000000e,0x1000000000000017};
  book->defs[0x00027089]->node[14] = (Node) {0x100100000000000f,0x1000000000000016};
  book->defs[0x00027089]->node[15] = (Node) {0x1001000000000010,0x1000000000000015};
  book->defs[0x00027089]->node[16] = (Node) {0x1001000000000011,0x1000000000000014};
  book->defs[0x00027089]->node[17] = (Node) {0x1000000000000012,0x1000000000000013};
  book->defs[0x00027089]->node[18] = (Node) {0x4000000000024,0x4000000000013};
  book->defs[0x00027089]->node[19] = (Node) {0x5000000000012,0x4000000000014};
  book->defs[0x00027089]->node[20] = (Node) {0x5000000000013,0x4000000000015};
  book->defs[0x00027089]->node[21] = (Node) {0x5000000000014,0x4000000000016};
  book->defs[0x00027089]->node[22] = (Node) {0x5000000000015,0x4000000000017};
  book->defs[0x00027089]->node[23] = (Node) {0x5000000000016,0x4000000000018};
  book->defs[0x00027089]->node[24] = (Node) {0x5000000000017,0x4000000000019};
  book->defs[0x00027089]->node[25] = (Node) {0x5000000000018,0x400000000001a};
  book->defs[0x00027089]->node[26] = (Node) {0x5000000000019,0x400000000001b};
  book->defs[0x00027089]->node[27] = (Node) {0x500000000001a,0x400000000001c};
  book->defs[0x00027089]->node[28] = (Node) {0x500000000001b,0x400000000001d};
  book->defs[0x00027089]->node[29] = (Node) {0x500000000001c,0x400000000001e};
  book->defs[0x00027089]->node[30] = (Node) {0x500000000001d,0x400000000001f};
  book->defs[0x00027089]->node[31] = (Node) {0x500000000001e,0x4000000000020};
  book->defs[0x00027089]->node[32] = (Node) {0x500000000001f,0x4000000000021};
  book->defs[0x00027089]->node[33] = (Node) {0x5000000000020,0x4000000000022};
  book->defs[0x00027089]->node[34] = (Node) {0x5000000000021,0x4000000000023};
  book->defs[0x00027089]->node[35] = (Node) {0x5000000000022,0x5000000000024};
  book->defs[0x00027089]->node[36] = (Node) {0x4000000000012,0x5000000000023};
  // c19
  book->defs[0x0002708a]           = (Term*) malloc(sizeof(Term));
  book->defs[0x0002708a]->root     = 0x1000000000000000;
  book->defs[0x0002708a]->alen     = 0;
  book->defs[0x0002708a]->acts     = (Wire*) malloc(0 * sizeof(Wire));
  book->defs[0x0002708a]->nlen     = 39;
  book->defs[0x0002708a]->node     = (Node*) malloc(39 * sizeof(Node));
  book->defs[0x0002708a]->node[ 0] = (Node) {0x1001000000000001,0x1000000000000026};
  book->defs[0x0002708a]->node[ 1] = (Node) {0x1001000000000002,0x1000000000000025};
  book->defs[0x0002708a]->node[ 2] = (Node) {0x1001000000000003,0x1000000000000024};
  book->defs[0x0002708a]->node[ 3] = (Node) {0x1001000000000004,0x1000000000000023};
  book->defs[0x0002708a]->node[ 4] = (Node) {0x1001000000000005,0x1000000000000022};
  book->defs[0x0002708a]->node[ 5] = (Node) {0x1001000000000006,0x1000000000000021};
  book->defs[0x0002708a]->node[ 6] = (Node) {0x1001000000000007,0x1000000000000020};
  book->defs[0x0002708a]->node[ 7] = (Node) {0x1001000000000008,0x100000000000001f};
  book->defs[0x0002708a]->node[ 8] = (Node) {0x1001000000000009,0x100000000000001e};
  book->defs[0x0002708a]->node[ 9] = (Node) {0x100100000000000a,0x100000000000001d};
  book->defs[0x0002708a]->node[10] = (Node) {0x100100000000000b,0x100000000000001c};
  book->defs[0x0002708a]->node[11] = (Node) {0x100100000000000c,0x100000000000001b};
  book->defs[0x0002708a]->node[12] = (Node) {0x100100000000000d,0x100000000000001a};
  book->defs[0x0002708a]->node[13] = (Node) {0x100100000000000e,0x1000000000000019};
  book->defs[0x0002708a]->node[14] = (Node) {0x100100000000000f,0x1000000000000018};
  book->defs[0x0002708a]->node[15] = (Node) {0x1001000000000010,0x1000000000000017};
  book->defs[0x0002708a]->node[16] = (Node) {0x1001000000000011,0x1000000000000016};
  book->defs[0x0002708a]->node[17] = (Node) {0x1001000000000012,0x1000000000000015};
  book->defs[0x0002708a]->node[18] = (Node) {0x1000000000000013,0x1000000000000014};
  book->defs[0x0002708a]->node[19] = (Node) {0x4000000000026,0x4000000000014};
  book->defs[0x0002708a]->node[20] = (Node) {0x5000000000013,0x4000000000015};
  book->defs[0x0002708a]->node[21] = (Node) {0x5000000000014,0x4000000000016};
  book->defs[0x0002708a]->node[22] = (Node) {0x5000000000015,0x4000000000017};
  book->defs[0x0002708a]->node[23] = (Node) {0x5000000000016,0x4000000000018};
  book->defs[0x0002708a]->node[24] = (Node) {0x5000000000017,0x4000000000019};
  book->defs[0x0002708a]->node[25] = (Node) {0x5000000000018,0x400000000001a};
  book->defs[0x0002708a]->node[26] = (Node) {0x5000000000019,0x400000000001b};
  book->defs[0x0002708a]->node[27] = (Node) {0x500000000001a,0x400000000001c};
  book->defs[0x0002708a]->node[28] = (Node) {0x500000000001b,0x400000000001d};
  book->defs[0x0002708a]->node[29] = (Node) {0x500000000001c,0x400000000001e};
  book->defs[0x0002708a]->node[30] = (Node) {0x500000000001d,0x400000000001f};
  book->defs[0x0002708a]->node[31] = (Node) {0x500000000001e,0x4000000000020};
  book->defs[0x0002708a]->node[32] = (Node) {0x500000000001f,0x4000000000021};
  book->defs[0x0002708a]->node[33] = (Node) {0x5000000000020,0x4000000000022};
  book->defs[0x0002708a]->node[34] = (Node) {0x5000000000021,0x4000000000023};
  book->defs[0x0002708a]->node[35] = (Node) {0x5000000000022,0x4000000000024};
  book->defs[0x0002708a]->node[36] = (Node) {0x5000000000023,0x4000000000025};
  book->defs[0x0002708a]->node[37] = (Node) {0x5000000000024,0x5000000000026};
  book->defs[0x0002708a]->node[38] = (Node) {0x4000000000013,0x5000000000025};
  // c20
  book->defs[0x000270c1]           = (Term*) malloc(sizeof(Term));
  book->defs[0x000270c1]->root     = 0x1000000000000000;
  book->defs[0x000270c1]->alen     = 0;
  book->defs[0x000270c1]->acts     = (Wire*) malloc(0 * sizeof(Wire));
  book->defs[0x000270c1]->nlen     = 41;
  book->defs[0x000270c1]->node     = (Node*) malloc(41 * sizeof(Node));
  book->defs[0x000270c1]->node[ 0] = (Node) {0x1001000000000001,0x1000000000000028};
  book->defs[0x000270c1]->node[ 1] = (Node) {0x1001000000000002,0x1000000000000027};
  book->defs[0x000270c1]->node[ 2] = (Node) {0x1001000000000003,0x1000000000000026};
  book->defs[0x000270c1]->node[ 3] = (Node) {0x1001000000000004,0x1000000000000025};
  book->defs[0x000270c1]->node[ 4] = (Node) {0x1001000000000005,0x1000000000000024};
  book->defs[0x000270c1]->node[ 5] = (Node) {0x1001000000000006,0x1000000000000023};
  book->defs[0x000270c1]->node[ 6] = (Node) {0x1001000000000007,0x1000000000000022};
  book->defs[0x000270c1]->node[ 7] = (Node) {0x1001000000000008,0x1000000000000021};
  book->defs[0x000270c1]->node[ 8] = (Node) {0x1001000000000009,0x1000000000000020};
  book->defs[0x000270c1]->node[ 9] = (Node) {0x100100000000000a,0x100000000000001f};
  book->defs[0x000270c1]->node[10] = (Node) {0x100100000000000b,0x100000000000001e};
  book->defs[0x000270c1]->node[11] = (Node) {0x100100000000000c,0x100000000000001d};
  book->defs[0x000270c1]->node[12] = (Node) {0x100100000000000d,0x100000000000001c};
  book->defs[0x000270c1]->node[13] = (Node) {0x100100000000000e,0x100000000000001b};
  book->defs[0x000270c1]->node[14] = (Node) {0x100100000000000f,0x100000000000001a};
  book->defs[0x000270c1]->node[15] = (Node) {0x1001000000000010,0x1000000000000019};
  book->defs[0x000270c1]->node[16] = (Node) {0x1001000000000011,0x1000000000000018};
  book->defs[0x000270c1]->node[17] = (Node) {0x1001000000000012,0x1000000000000017};
  book->defs[0x000270c1]->node[18] = (Node) {0x1001000000000013,0x1000000000000016};
  book->defs[0x000270c1]->node[19] = (Node) {0x1000000000000014,0x1000000000000015};
  book->defs[0x000270c1]->node[20] = (Node) {0x4000000000028,0x4000000000015};
  book->defs[0x000270c1]->node[21] = (Node) {0x5000000000014,0x4000000000016};
  book->defs[0x000270c1]->node[22] = (Node) {0x5000000000015,0x4000000000017};
  book->defs[0x000270c1]->node[23] = (Node) {0x5000000000016,0x4000000000018};
  book->defs[0x000270c1]->node[24] = (Node) {0x5000000000017,0x4000000000019};
  book->defs[0x000270c1]->node[25] = (Node) {0x5000000000018,0x400000000001a};
  book->defs[0x000270c1]->node[26] = (Node) {0x5000000000019,0x400000000001b};
  book->defs[0x000270c1]->node[27] = (Node) {0x500000000001a,0x400000000001c};
  book->defs[0x000270c1]->node[28] = (Node) {0x500000000001b,0x400000000001d};
  book->defs[0x000270c1]->node[29] = (Node) {0x500000000001c,0x400000000001e};
  book->defs[0x000270c1]->node[30] = (Node) {0x500000000001d,0x400000000001f};
  book->defs[0x000270c1]->node[31] = (Node) {0x500000000001e,0x4000000000020};
  book->defs[0x000270c1]->node[32] = (Node) {0x500000000001f,0x4000000000021};
  book->defs[0x000270c1]->node[33] = (Node) {0x5000000000020,0x4000000000022};
  book->defs[0x000270c1]->node[34] = (Node) {0x5000000000021,0x4000000000023};
  book->defs[0x000270c1]->node[35] = (Node) {0x5000000000022,0x4000000000024};
  book->defs[0x000270c1]->node[36] = (Node) {0x5000000000023,0x4000000000025};
  book->defs[0x000270c1]->node[37] = (Node) {0x5000000000024,0x4000000000026};
  book->defs[0x000270c1]->node[38] = (Node) {0x5000000000025,0x4000000000027};
  book->defs[0x000270c1]->node[39] = (Node) {0x5000000000026,0x5000000000028};
  book->defs[0x000270c1]->node[40] = (Node) {0x4000000000014,0x5000000000027};
  // c21
  book->defs[0x000270c2]           = (Term*) malloc(sizeof(Term));
  book->defs[0x000270c2]->root     = 0x1000000000000000;
  book->defs[0x000270c2]->alen     = 0;
  book->defs[0x000270c2]->acts     = (Wire*) malloc(0 * sizeof(Wire));
  book->defs[0x000270c2]->nlen     = 43;
  book->defs[0x000270c2]->node     = (Node*) malloc(43 * sizeof(Node));
  book->defs[0x000270c2]->node[ 0] = (Node) {0x1001000000000001,0x100000000000002a};
  book->defs[0x000270c2]->node[ 1] = (Node) {0x1001000000000002,0x1000000000000029};
  book->defs[0x000270c2]->node[ 2] = (Node) {0x1001000000000003,0x1000000000000028};
  book->defs[0x000270c2]->node[ 3] = (Node) {0x1001000000000004,0x1000000000000027};
  book->defs[0x000270c2]->node[ 4] = (Node) {0x1001000000000005,0x1000000000000026};
  book->defs[0x000270c2]->node[ 5] = (Node) {0x1001000000000006,0x1000000000000025};
  book->defs[0x000270c2]->node[ 6] = (Node) {0x1001000000000007,0x1000000000000024};
  book->defs[0x000270c2]->node[ 7] = (Node) {0x1001000000000008,0x1000000000000023};
  book->defs[0x000270c2]->node[ 8] = (Node) {0x1001000000000009,0x1000000000000022};
  book->defs[0x000270c2]->node[ 9] = (Node) {0x100100000000000a,0x1000000000000021};
  book->defs[0x000270c2]->node[10] = (Node) {0x100100000000000b,0x1000000000000020};
  book->defs[0x000270c2]->node[11] = (Node) {0x100100000000000c,0x100000000000001f};
  book->defs[0x000270c2]->node[12] = (Node) {0x100100000000000d,0x100000000000001e};
  book->defs[0x000270c2]->node[13] = (Node) {0x100100000000000e,0x100000000000001d};
  book->defs[0x000270c2]->node[14] = (Node) {0x100100000000000f,0x100000000000001c};
  book->defs[0x000270c2]->node[15] = (Node) {0x1001000000000010,0x100000000000001b};
  book->defs[0x000270c2]->node[16] = (Node) {0x1001000000000011,0x100000000000001a};
  book->defs[0x000270c2]->node[17] = (Node) {0x1001000000000012,0x1000000000000019};
  book->defs[0x000270c2]->node[18] = (Node) {0x1001000000000013,0x1000000000000018};
  book->defs[0x000270c2]->node[19] = (Node) {0x1001000000000014,0x1000000000000017};
  book->defs[0x000270c2]->node[20] = (Node) {0x1000000000000015,0x1000000000000016};
  book->defs[0x000270c2]->node[21] = (Node) {0x400000000002a,0x4000000000016};
  book->defs[0x000270c2]->node[22] = (Node) {0x5000000000015,0x4000000000017};
  book->defs[0x000270c2]->node[23] = (Node) {0x5000000000016,0x4000000000018};
  book->defs[0x000270c2]->node[24] = (Node) {0x5000000000017,0x4000000000019};
  book->defs[0x000270c2]->node[25] = (Node) {0x5000000000018,0x400000000001a};
  book->defs[0x000270c2]->node[26] = (Node) {0x5000000000019,0x400000000001b};
  book->defs[0x000270c2]->node[27] = (Node) {0x500000000001a,0x400000000001c};
  book->defs[0x000270c2]->node[28] = (Node) {0x500000000001b,0x400000000001d};
  book->defs[0x000270c2]->node[29] = (Node) {0x500000000001c,0x400000000001e};
  book->defs[0x000270c2]->node[30] = (Node) {0x500000000001d,0x400000000001f};
  book->defs[0x000270c2]->node[31] = (Node) {0x500000000001e,0x4000000000020};
  book->defs[0x000270c2]->node[32] = (Node) {0x500000000001f,0x4000000000021};
  book->defs[0x000270c2]->node[33] = (Node) {0x5000000000020,0x4000000000022};
  book->defs[0x000270c2]->node[34] = (Node) {0x5000000000021,0x4000000000023};
  book->defs[0x000270c2]->node[35] = (Node) {0x5000000000022,0x4000000000024};
  book->defs[0x000270c2]->node[36] = (Node) {0x5000000000023,0x4000000000025};
  book->defs[0x000270c2]->node[37] = (Node) {0x5000000000024,0x4000000000026};
  book->defs[0x000270c2]->node[38] = (Node) {0x5000000000025,0x4000000000027};
  book->defs[0x000270c2]->node[39] = (Node) {0x5000000000026,0x4000000000028};
  book->defs[0x000270c2]->node[40] = (Node) {0x5000000000027,0x4000000000029};
  book->defs[0x000270c2]->node[41] = (Node) {0x5000000000028,0x500000000002a};
  book->defs[0x000270c2]->node[42] = (Node) {0x4000000000015,0x5000000000029};
  // c22
  book->defs[0x000270c3]           = (Term*) malloc(sizeof(Term));
  book->defs[0x000270c3]->root     = 0x1000000000000000;
  book->defs[0x000270c3]->alen     = 0;
  book->defs[0x000270c3]->acts     = (Wire*) malloc(0 * sizeof(Wire));
  book->defs[0x000270c3]->nlen     = 45;
  book->defs[0x000270c3]->node     = (Node*) malloc(45 * sizeof(Node));
  book->defs[0x000270c3]->node[ 0] = (Node) {0x1001000000000001,0x100000000000002c};
  book->defs[0x000270c3]->node[ 1] = (Node) {0x1001000000000002,0x100000000000002b};
  book->defs[0x000270c3]->node[ 2] = (Node) {0x1001000000000003,0x100000000000002a};
  book->defs[0x000270c3]->node[ 3] = (Node) {0x1001000000000004,0x1000000000000029};
  book->defs[0x000270c3]->node[ 4] = (Node) {0x1001000000000005,0x1000000000000028};
  book->defs[0x000270c3]->node[ 5] = (Node) {0x1001000000000006,0x1000000000000027};
  book->defs[0x000270c3]->node[ 6] = (Node) {0x1001000000000007,0x1000000000000026};
  book->defs[0x000270c3]->node[ 7] = (Node) {0x1001000000000008,0x1000000000000025};
  book->defs[0x000270c3]->node[ 8] = (Node) {0x1001000000000009,0x1000000000000024};
  book->defs[0x000270c3]->node[ 9] = (Node) {0x100100000000000a,0x1000000000000023};
  book->defs[0x000270c3]->node[10] = (Node) {0x100100000000000b,0x1000000000000022};
  book->defs[0x000270c3]->node[11] = (Node) {0x100100000000000c,0x1000000000000021};
  book->defs[0x000270c3]->node[12] = (Node) {0x100100000000000d,0x1000000000000020};
  book->defs[0x000270c3]->node[13] = (Node) {0x100100000000000e,0x100000000000001f};
  book->defs[0x000270c3]->node[14] = (Node) {0x100100000000000f,0x100000000000001e};
  book->defs[0x000270c3]->node[15] = (Node) {0x1001000000000010,0x100000000000001d};
  book->defs[0x000270c3]->node[16] = (Node) {0x1001000000000011,0x100000000000001c};
  book->defs[0x000270c3]->node[17] = (Node) {0x1001000000000012,0x100000000000001b};
  book->defs[0x000270c3]->node[18] = (Node) {0x1001000000000013,0x100000000000001a};
  book->defs[0x000270c3]->node[19] = (Node) {0x1001000000000014,0x1000000000000019};
  book->defs[0x000270c3]->node[20] = (Node) {0x1001000000000015,0x1000000000000018};
  book->defs[0x000270c3]->node[21] = (Node) {0x1000000000000016,0x1000000000000017};
  book->defs[0x000270c3]->node[22] = (Node) {0x400000000002c,0x4000000000017};
  book->defs[0x000270c3]->node[23] = (Node) {0x5000000000016,0x4000000000018};
  book->defs[0x000270c3]->node[24] = (Node) {0x5000000000017,0x4000000000019};
  book->defs[0x000270c3]->node[25] = (Node) {0x5000000000018,0x400000000001a};
  book->defs[0x000270c3]->node[26] = (Node) {0x5000000000019,0x400000000001b};
  book->defs[0x000270c3]->node[27] = (Node) {0x500000000001a,0x400000000001c};
  book->defs[0x000270c3]->node[28] = (Node) {0x500000000001b,0x400000000001d};
  book->defs[0x000270c3]->node[29] = (Node) {0x500000000001c,0x400000000001e};
  book->defs[0x000270c3]->node[30] = (Node) {0x500000000001d,0x400000000001f};
  book->defs[0x000270c3]->node[31] = (Node) {0x500000000001e,0x4000000000020};
  book->defs[0x000270c3]->node[32] = (Node) {0x500000000001f,0x4000000000021};
  book->defs[0x000270c3]->node[33] = (Node) {0x5000000000020,0x4000000000022};
  book->defs[0x000270c3]->node[34] = (Node) {0x5000000000021,0x4000000000023};
  book->defs[0x000270c3]->node[35] = (Node) {0x5000000000022,0x4000000000024};
  book->defs[0x000270c3]->node[36] = (Node) {0x5000000000023,0x4000000000025};
  book->defs[0x000270c3]->node[37] = (Node) {0x5000000000024,0x4000000000026};
  book->defs[0x000270c3]->node[38] = (Node) {0x5000000000025,0x4000000000027};
  book->defs[0x000270c3]->node[39] = (Node) {0x5000000000026,0x4000000000028};
  book->defs[0x000270c3]->node[40] = (Node) {0x5000000000027,0x4000000000029};
  book->defs[0x000270c3]->node[41] = (Node) {0x5000000000028,0x400000000002a};
  book->defs[0x000270c3]->node[42] = (Node) {0x5000000000029,0x400000000002b};
  book->defs[0x000270c3]->node[43] = (Node) {0x500000000002a,0x500000000002c};
  book->defs[0x000270c3]->node[44] = (Node) {0x4000000000016,0x500000000002b};
  // c23
  book->defs[0x000270c4]           = (Term*) malloc(sizeof(Term));
  book->defs[0x000270c4]->root     = 0x1000000000000000;
  book->defs[0x000270c4]->alen     = 0;
  book->defs[0x000270c4]->acts     = (Wire*) malloc(0 * sizeof(Wire));
  book->defs[0x000270c4]->nlen     = 47;
  book->defs[0x000270c4]->node     = (Node*) malloc(47 * sizeof(Node));
  book->defs[0x000270c4]->node[ 0] = (Node) {0x1001000000000001,0x100000000000002e};
  book->defs[0x000270c4]->node[ 1] = (Node) {0x1001000000000002,0x100000000000002d};
  book->defs[0x000270c4]->node[ 2] = (Node) {0x1001000000000003,0x100000000000002c};
  book->defs[0x000270c4]->node[ 3] = (Node) {0x1001000000000004,0x100000000000002b};
  book->defs[0x000270c4]->node[ 4] = (Node) {0x1001000000000005,0x100000000000002a};
  book->defs[0x000270c4]->node[ 5] = (Node) {0x1001000000000006,0x1000000000000029};
  book->defs[0x000270c4]->node[ 6] = (Node) {0x1001000000000007,0x1000000000000028};
  book->defs[0x000270c4]->node[ 7] = (Node) {0x1001000000000008,0x1000000000000027};
  book->defs[0x000270c4]->node[ 8] = (Node) {0x1001000000000009,0x1000000000000026};
  book->defs[0x000270c4]->node[ 9] = (Node) {0x100100000000000a,0x1000000000000025};
  book->defs[0x000270c4]->node[10] = (Node) {0x100100000000000b,0x1000000000000024};
  book->defs[0x000270c4]->node[11] = (Node) {0x100100000000000c,0x1000000000000023};
  book->defs[0x000270c4]->node[12] = (Node) {0x100100000000000d,0x1000000000000022};
  book->defs[0x000270c4]->node[13] = (Node) {0x100100000000000e,0x1000000000000021};
  book->defs[0x000270c4]->node[14] = (Node) {0x100100000000000f,0x1000000000000020};
  book->defs[0x000270c4]->node[15] = (Node) {0x1001000000000010,0x100000000000001f};
  book->defs[0x000270c4]->node[16] = (Node) {0x1001000000000011,0x100000000000001e};
  book->defs[0x000270c4]->node[17] = (Node) {0x1001000000000012,0x100000000000001d};
  book->defs[0x000270c4]->node[18] = (Node) {0x1001000000000013,0x100000000000001c};
  book->defs[0x000270c4]->node[19] = (Node) {0x1001000000000014,0x100000000000001b};
  book->defs[0x000270c4]->node[20] = (Node) {0x1001000000000015,0x100000000000001a};
  book->defs[0x000270c4]->node[21] = (Node) {0x1001000000000016,0x1000000000000019};
  book->defs[0x000270c4]->node[22] = (Node) {0x1000000000000017,0x1000000000000018};
  book->defs[0x000270c4]->node[23] = (Node) {0x400000000002e,0x4000000000018};
  book->defs[0x000270c4]->node[24] = (Node) {0x5000000000017,0x4000000000019};
  book->defs[0x000270c4]->node[25] = (Node) {0x5000000000018,0x400000000001a};
  book->defs[0x000270c4]->node[26] = (Node) {0x5000000000019,0x400000000001b};
  book->defs[0x000270c4]->node[27] = (Node) {0x500000000001a,0x400000000001c};
  book->defs[0x000270c4]->node[28] = (Node) {0x500000000001b,0x400000000001d};
  book->defs[0x000270c4]->node[29] = (Node) {0x500000000001c,0x400000000001e};
  book->defs[0x000270c4]->node[30] = (Node) {0x500000000001d,0x400000000001f};
  book->defs[0x000270c4]->node[31] = (Node) {0x500000000001e,0x4000000000020};
  book->defs[0x000270c4]->node[32] = (Node) {0x500000000001f,0x4000000000021};
  book->defs[0x000270c4]->node[33] = (Node) {0x5000000000020,0x4000000000022};
  book->defs[0x000270c4]->node[34] = (Node) {0x5000000000021,0x4000000000023};
  book->defs[0x000270c4]->node[35] = (Node) {0x5000000000022,0x4000000000024};
  book->defs[0x000270c4]->node[36] = (Node) {0x5000000000023,0x4000000000025};
  book->defs[0x000270c4]->node[37] = (Node) {0x5000000000024,0x4000000000026};
  book->defs[0x000270c4]->node[38] = (Node) {0x5000000000025,0x4000000000027};
  book->defs[0x000270c4]->node[39] = (Node) {0x5000000000026,0x4000000000028};
  book->defs[0x000270c4]->node[40] = (Node) {0x5000000000027,0x4000000000029};
  book->defs[0x000270c4]->node[41] = (Node) {0x5000000000028,0x400000000002a};
  book->defs[0x000270c4]->node[42] = (Node) {0x5000000000029,0x400000000002b};
  book->defs[0x000270c4]->node[43] = (Node) {0x500000000002a,0x400000000002c};
  book->defs[0x000270c4]->node[44] = (Node) {0x500000000002b,0x400000000002d};
  book->defs[0x000270c4]->node[45] = (Node) {0x500000000002c,0x500000000002e};
  book->defs[0x000270c4]->node[46] = (Node) {0x4000000000017,0x500000000002d};
  // c24
  book->defs[0x000270c5]           = (Term*) malloc(sizeof(Term));
  book->defs[0x000270c5]->root     = 0x1000000000000000;
  book->defs[0x000270c5]->alen     = 0;
  book->defs[0x000270c5]->acts     = (Wire*) malloc(0 * sizeof(Wire));
  book->defs[0x000270c5]->nlen     = 49;
  book->defs[0x000270c5]->node     = (Node*) malloc(49 * sizeof(Node));
  book->defs[0x000270c5]->node[ 0] = (Node) {0x1001000000000001,0x1000000000000030};
  book->defs[0x000270c5]->node[ 1] = (Node) {0x1001000000000002,0x100000000000002f};
  book->defs[0x000270c5]->node[ 2] = (Node) {0x1001000000000003,0x100000000000002e};
  book->defs[0x000270c5]->node[ 3] = (Node) {0x1001000000000004,0x100000000000002d};
  book->defs[0x000270c5]->node[ 4] = (Node) {0x1001000000000005,0x100000000000002c};
  book->defs[0x000270c5]->node[ 5] = (Node) {0x1001000000000006,0x100000000000002b};
  book->defs[0x000270c5]->node[ 6] = (Node) {0x1001000000000007,0x100000000000002a};
  book->defs[0x000270c5]->node[ 7] = (Node) {0x1001000000000008,0x1000000000000029};
  book->defs[0x000270c5]->node[ 8] = (Node) {0x1001000000000009,0x1000000000000028};
  book->defs[0x000270c5]->node[ 9] = (Node) {0x100100000000000a,0x1000000000000027};
  book->defs[0x000270c5]->node[10] = (Node) {0x100100000000000b,0x1000000000000026};
  book->defs[0x000270c5]->node[11] = (Node) {0x100100000000000c,0x1000000000000025};
  book->defs[0x000270c5]->node[12] = (Node) {0x100100000000000d,0x1000000000000024};
  book->defs[0x000270c5]->node[13] = (Node) {0x100100000000000e,0x1000000000000023};
  book->defs[0x000270c5]->node[14] = (Node) {0x100100000000000f,0x1000000000000022};
  book->defs[0x000270c5]->node[15] = (Node) {0x1001000000000010,0x1000000000000021};
  book->defs[0x000270c5]->node[16] = (Node) {0x1001000000000011,0x1000000000000020};
  book->defs[0x000270c5]->node[17] = (Node) {0x1001000000000012,0x100000000000001f};
  book->defs[0x000270c5]->node[18] = (Node) {0x1001000000000013,0x100000000000001e};
  book->defs[0x000270c5]->node[19] = (Node) {0x1001000000000014,0x100000000000001d};
  book->defs[0x000270c5]->node[20] = (Node) {0x1001000000000015,0x100000000000001c};
  book->defs[0x000270c5]->node[21] = (Node) {0x1001000000000016,0x100000000000001b};
  book->defs[0x000270c5]->node[22] = (Node) {0x1001000000000017,0x100000000000001a};
  book->defs[0x000270c5]->node[23] = (Node) {0x1000000000000018,0x1000000000000019};
  book->defs[0x000270c5]->node[24] = (Node) {0x4000000000030,0x4000000000019};
  book->defs[0x000270c5]->node[25] = (Node) {0x5000000000018,0x400000000001a};
  book->defs[0x000270c5]->node[26] = (Node) {0x5000000000019,0x400000000001b};
  book->defs[0x000270c5]->node[27] = (Node) {0x500000000001a,0x400000000001c};
  book->defs[0x000270c5]->node[28] = (Node) {0x500000000001b,0x400000000001d};
  book->defs[0x000270c5]->node[29] = (Node) {0x500000000001c,0x400000000001e};
  book->defs[0x000270c5]->node[30] = (Node) {0x500000000001d,0x400000000001f};
  book->defs[0x000270c5]->node[31] = (Node) {0x500000000001e,0x4000000000020};
  book->defs[0x000270c5]->node[32] = (Node) {0x500000000001f,0x4000000000021};
  book->defs[0x000270c5]->node[33] = (Node) {0x5000000000020,0x4000000000022};
  book->defs[0x000270c5]->node[34] = (Node) {0x5000000000021,0x4000000000023};
  book->defs[0x000270c5]->node[35] = (Node) {0x5000000000022,0x4000000000024};
  book->defs[0x000270c5]->node[36] = (Node) {0x5000000000023,0x4000000000025};
  book->defs[0x000270c5]->node[37] = (Node) {0x5000000000024,0x4000000000026};
  book->defs[0x000270c5]->node[38] = (Node) {0x5000000000025,0x4000000000027};
  book->defs[0x000270c5]->node[39] = (Node) {0x5000000000026,0x4000000000028};
  book->defs[0x000270c5]->node[40] = (Node) {0x5000000000027,0x4000000000029};
  book->defs[0x000270c5]->node[41] = (Node) {0x5000000000028,0x400000000002a};
  book->defs[0x000270c5]->node[42] = (Node) {0x5000000000029,0x400000000002b};
  book->defs[0x000270c5]->node[43] = (Node) {0x500000000002a,0x400000000002c};
  book->defs[0x000270c5]->node[44] = (Node) {0x500000000002b,0x400000000002d};
  book->defs[0x000270c5]->node[45] = (Node) {0x500000000002c,0x400000000002e};
  book->defs[0x000270c5]->node[46] = (Node) {0x500000000002d,0x400000000002f};
  book->defs[0x000270c5]->node[47] = (Node) {0x500000000002e,0x5000000000030};
  book->defs[0x000270c5]->node[48] = (Node) {0x4000000000018,0x500000000002f};
  // c_s
  book->defs[0x00027ff7]           = (Term*) malloc(sizeof(Term));
  book->defs[0x00027ff7]->root     = 0x1000000000000000;
  book->defs[0x00027ff7]->alen     = 0;
  book->defs[0x00027ff7]->acts     = (Wire*) malloc(0 * sizeof(Wire));
  book->defs[0x00027ff7]->nlen     = 7;
  book->defs[0x00027ff7]->node     = (Node*) malloc(7 * sizeof(Node));
  book->defs[0x00027ff7]->node[ 0] = (Node) {0x1000000000000001,0x1000000000000003};
  book->defs[0x00027ff7]->node[ 1] = (Node) {0x5000000000004,0x1000000000000002};
  book->defs[0x00027ff7]->node[ 2] = (Node) {0x4000000000006,0x4000000000005};
  book->defs[0x00027ff7]->node[ 3] = (Node) {0x1001000000000004,0x1000000000000006};
  book->defs[0x00027ff7]->node[ 4] = (Node) {0x1000000000000005,0x4000000000001};
  book->defs[0x00027ff7]->node[ 5] = (Node) {0x5000000000002,0x5000000000006};
  book->defs[0x00027ff7]->node[ 6] = (Node) {0x4000000000002,0x5000000000005};
  // c_z
  book->defs[0x00027ffe]           = (Term*) malloc(sizeof(Term));
  book->defs[0x00027ffe]->root     = 0x1000000000000000;
  book->defs[0x00027ffe]->alen     = 0;
  book->defs[0x00027ffe]->acts     = (Wire*) malloc(0 * sizeof(Wire));
  book->defs[0x00027ffe]->nlen     = 2;
  book->defs[0x00027ffe]->node     = (Node*) malloc(2 * sizeof(Node));
  book->defs[0x00027ffe]->node[ 0] = (Node) {0x2000000000000,0x1000000000000001};
  book->defs[0x00027ffe]->node[ 1] = (Node) {0x5000000000001,0x4000000000001};
  // dec
  book->defs[0x00028a67]           = (Term*) malloc(sizeof(Term));
  book->defs[0x00028a67]->root     = 0x1000000000000000;
  book->defs[0x00028a67]->alen     = 0;
  book->defs[0x00028a67]->acts     = (Wire*) malloc(0 * sizeof(Wire));
  book->defs[0x00028a67]->nlen     = 4;
  book->defs[0x00028a67]->node     = (Node*) malloc(4 * sizeof(Node));
  book->defs[0x00028a67]->node[ 0] = (Node) {0x1000000000000001,0x5000000000003};
  book->defs[0x00028a67]->node[ 1] = (Node) {0x1000000a299d9,0x1000000000000002};
  book->defs[0x00028a67]->node[ 2] = (Node) {0x1000000a299d3,0x1000000000000003};
  book->defs[0x00028a67]->node[ 3] = (Node) {0x100000000000f,0x5000000000000};
  // ex0
  book->defs[0x00029f01]           = (Term*) malloc(sizeof(Term));
  book->defs[0x00029f01]->root     = 0x5000000000001;
  book->defs[0x00029f01]->alen     = 1;
  book->defs[0x00029f01]->acts     = (Wire*) malloc(1 * sizeof(Wire));
  book->defs[0x00029f01]->acts[ 0] = (Wire) {0x10000000009c3,0x1000000000000000};
  book->defs[0x00029f01]->nlen     = 2;
  book->defs[0x00029f01]->node     = (Node*) malloc(2 * sizeof(Node));
  book->defs[0x00029f01]->node[ 0] = (Node) {0x100000000001d,0x1000000000000001};
  book->defs[0x00029f01]->node[ 1] = (Node) {0x1000000000024,0x3000000000000};
  // ex1
  book->defs[0x00029f02]           = (Term*) malloc(sizeof(Term));
  book->defs[0x00029f02]->root     = 0x5000000000001;
  book->defs[0x00029f02]->alen     = 1;
  book->defs[0x00029f02]->acts     = (Wire*) malloc(1 * sizeof(Wire));
  book->defs[0x00029f02]->acts[ 0] = (Wire) {0x10000000270c4,0x1000000000000000};
  book->defs[0x00029f02]->nlen     = 2;
  book->defs[0x00029f02]->node     = (Node*) malloc(2 * sizeof(Node));
  book->defs[0x00029f02]->node[ 0] = (Node) {0x100000002bff7,0x1000000000000001};
  book->defs[0x00029f02]->node[ 1] = (Node) {0x100000002bffe,0x3000000000000};
  // ex2
  book->defs[0x00029f03]           = (Term*) malloc(sizeof(Term));
  book->defs[0x00029f03]->root     = 0x5000000000000;
  book->defs[0x00029f03]->alen     = 2;
  book->defs[0x00029f03]->acts     = (Wire*) malloc(2 * sizeof(Wire));
  book->defs[0x00029f03]->acts[ 0] = (Wire) {0x1000000036e72,0x1000000000000000};
  book->defs[0x00029f03]->acts[ 1] = (Wire) {0x10000000009c7,0x1000000000000001};
  book->defs[0x00029f03]->nlen     = 3;
  book->defs[0x00029f03]->node     = (Node*) malloc(3 * sizeof(Node));
  book->defs[0x00029f03]->node[ 0] = (Node) {0x5000000000002,0x3000000000000};
  book->defs[0x00029f03]->node[ 1] = (Node) {0x1000000000013,0x1000000000000002};
  book->defs[0x00029f03]->node[ 2] = (Node) {0x100000000000f,0x4000000000000};
  // ex3
  book->defs[0x00029f04]           = (Term*) malloc(sizeof(Term));
  book->defs[0x00029f04]->root     = 0x5000000000000;
  book->defs[0x00029f04]->alen     = 2;
  book->defs[0x00029f04]->acts     = (Wire*) malloc(2 * sizeof(Wire));
  book->defs[0x00029f04]->acts[ 0] = (Wire) {0x1000000026db2,0x1000000000000000};
  book->defs[0x00029f04]->acts[ 1] = (Wire) {0x1000000027085,0x1000000000000001};
  book->defs[0x00029f04]->nlen     = 3;
  book->defs[0x00029f04]->node     = (Node*) malloc(3 * sizeof(Node));
  book->defs[0x00029f04]->node[ 0] = (Node) {0x5000000000002,0x3000000000000};
  book->defs[0x00029f04]->node[ 1] = (Node) {0x100000000001d,0x1000000000000002};
  book->defs[0x00029f04]->node[ 2] = (Node) {0x1000000000024,0x4000000000000};
  // g_s
  book->defs[0x0002bff7]           = (Term*) malloc(sizeof(Term));
  book->defs[0x0002bff7]->root     = 0x1000000000000000;
  book->defs[0x0002bff7]->alen     = 0;
  book->defs[0x0002bff7]->acts     = (Wire*) malloc(0 * sizeof(Wire));
  book->defs[0x0002bff7]->nlen     = 5;
  book->defs[0x0002bff7]->node     = (Node*) malloc(5 * sizeof(Node));
  book->defs[0x0002bff7]->node[ 0] = (Node) {0x1002000000000001,0x1000000000000002};
  book->defs[0x0002bff7]->node[ 1] = (Node) {0x4000000000003,0x4000000000004};
  book->defs[0x0002bff7]->node[ 2] = (Node) {0x1000000000000003,0x5000000000004};
  book->defs[0x0002bff7]->node[ 3] = (Node) {0x4000000000001,0x1000000000000004};
  book->defs[0x0002bff7]->node[ 4] = (Node) {0x5000000000001,0x5000000000002};
  // g_z
  book->defs[0x0002bffe]           = (Term*) malloc(sizeof(Term));
  book->defs[0x0002bffe]->root     = 0x1000000000000000;
  book->defs[0x0002bffe]->alen     = 0;
  book->defs[0x0002bffe]->acts     = (Wire*) malloc(0 * sizeof(Wire));
  book->defs[0x0002bffe]->nlen     = 1;
  book->defs[0x0002bffe]->node     = (Node*) malloc(1 * sizeof(Node));
  book->defs[0x0002bffe]->node[ 0] = (Node) {0x5000000000000,0x4000000000000};
  // k10
  book->defs[0x0002f081]           = (Term*) malloc(sizeof(Term));
  book->defs[0x0002f081]->root     = 0x1000000000000000;
  book->defs[0x0002f081]->alen     = 0;
  book->defs[0x0002f081]->acts     = (Wire*) malloc(0 * sizeof(Wire));
  book->defs[0x0002f081]->nlen     = 21;
  book->defs[0x0002f081]->node     = (Node*) malloc(21 * sizeof(Node));
  book->defs[0x0002f081]->node[ 0] = (Node) {0x1002000000000001,0x1000000000000014};
  book->defs[0x0002f081]->node[ 1] = (Node) {0x1002000000000002,0x1000000000000013};
  book->defs[0x0002f081]->node[ 2] = (Node) {0x1002000000000003,0x1000000000000012};
  book->defs[0x0002f081]->node[ 3] = (Node) {0x1002000000000004,0x1000000000000011};
  book->defs[0x0002f081]->node[ 4] = (Node) {0x1002000000000005,0x1000000000000010};
  book->defs[0x0002f081]->node[ 5] = (Node) {0x1002000000000006,0x100000000000000f};
  book->defs[0x0002f081]->node[ 6] = (Node) {0x1002000000000007,0x100000000000000e};
  book->defs[0x0002f081]->node[ 7] = (Node) {0x1002000000000008,0x100000000000000d};
  book->defs[0x0002f081]->node[ 8] = (Node) {0x1002000000000009,0x100000000000000c};
  book->defs[0x0002f081]->node[ 9] = (Node) {0x100000000000000a,0x100000000000000b};
  book->defs[0x0002f081]->node[10] = (Node) {0x4000000000014,0x400000000000b};
  book->defs[0x0002f081]->node[11] = (Node) {0x500000000000a,0x400000000000c};
  book->defs[0x0002f081]->node[12] = (Node) {0x500000000000b,0x400000000000d};
  book->defs[0x0002f081]->node[13] = (Node) {0x500000000000c,0x400000000000e};
  book->defs[0x0002f081]->node[14] = (Node) {0x500000000000d,0x400000000000f};
  book->defs[0x0002f081]->node[15] = (Node) {0x500000000000e,0x4000000000010};
  book->defs[0x0002f081]->node[16] = (Node) {0x500000000000f,0x4000000000011};
  book->defs[0x0002f081]->node[17] = (Node) {0x5000000000010,0x4000000000012};
  book->defs[0x0002f081]->node[18] = (Node) {0x5000000000011,0x4000000000013};
  book->defs[0x0002f081]->node[19] = (Node) {0x5000000000012,0x5000000000014};
  book->defs[0x0002f081]->node[20] = (Node) {0x400000000000a,0x5000000000013};
  // k11
  book->defs[0x0002f082]           = (Term*) malloc(sizeof(Term));
  book->defs[0x0002f082]->root     = 0x1000000000000000;
  book->defs[0x0002f082]->alen     = 0;
  book->defs[0x0002f082]->acts     = (Wire*) malloc(0 * sizeof(Wire));
  book->defs[0x0002f082]->nlen     = 23;
  book->defs[0x0002f082]->node     = (Node*) malloc(23 * sizeof(Node));
  book->defs[0x0002f082]->node[ 0] = (Node) {0x1002000000000001,0x1000000000000016};
  book->defs[0x0002f082]->node[ 1] = (Node) {0x1002000000000002,0x1000000000000015};
  book->defs[0x0002f082]->node[ 2] = (Node) {0x1002000000000003,0x1000000000000014};
  book->defs[0x0002f082]->node[ 3] = (Node) {0x1002000000000004,0x1000000000000013};
  book->defs[0x0002f082]->node[ 4] = (Node) {0x1002000000000005,0x1000000000000012};
  book->defs[0x0002f082]->node[ 5] = (Node) {0x1002000000000006,0x1000000000000011};
  book->defs[0x0002f082]->node[ 6] = (Node) {0x1002000000000007,0x1000000000000010};
  book->defs[0x0002f082]->node[ 7] = (Node) {0x1002000000000008,0x100000000000000f};
  book->defs[0x0002f082]->node[ 8] = (Node) {0x1002000000000009,0x100000000000000e};
  book->defs[0x0002f082]->node[ 9] = (Node) {0x100200000000000a,0x100000000000000d};
  book->defs[0x0002f082]->node[10] = (Node) {0x100000000000000b,0x100000000000000c};
  book->defs[0x0002f082]->node[11] = (Node) {0x4000000000016,0x400000000000c};
  book->defs[0x0002f082]->node[12] = (Node) {0x500000000000b,0x400000000000d};
  book->defs[0x0002f082]->node[13] = (Node) {0x500000000000c,0x400000000000e};
  book->defs[0x0002f082]->node[14] = (Node) {0x500000000000d,0x400000000000f};
  book->defs[0x0002f082]->node[15] = (Node) {0x500000000000e,0x4000000000010};
  book->defs[0x0002f082]->node[16] = (Node) {0x500000000000f,0x4000000000011};
  book->defs[0x0002f082]->node[17] = (Node) {0x5000000000010,0x4000000000012};
  book->defs[0x0002f082]->node[18] = (Node) {0x5000000000011,0x4000000000013};
  book->defs[0x0002f082]->node[19] = (Node) {0x5000000000012,0x4000000000014};
  book->defs[0x0002f082]->node[20] = (Node) {0x5000000000013,0x4000000000015};
  book->defs[0x0002f082]->node[21] = (Node) {0x5000000000014,0x5000000000016};
  book->defs[0x0002f082]->node[22] = (Node) {0x400000000000b,0x5000000000015};
  // k12
  book->defs[0x0002f083]           = (Term*) malloc(sizeof(Term));
  book->defs[0x0002f083]->root     = 0x1000000000000000;
  book->defs[0x0002f083]->alen     = 0;
  book->defs[0x0002f083]->acts     = (Wire*) malloc(0 * sizeof(Wire));
  book->defs[0x0002f083]->nlen     = 25;
  book->defs[0x0002f083]->node     = (Node*) malloc(25 * sizeof(Node));
  book->defs[0x0002f083]->node[ 0] = (Node) {0x1002000000000001,0x1000000000000018};
  book->defs[0x0002f083]->node[ 1] = (Node) {0x1002000000000002,0x1000000000000017};
  book->defs[0x0002f083]->node[ 2] = (Node) {0x1002000000000003,0x1000000000000016};
  book->defs[0x0002f083]->node[ 3] = (Node) {0x1002000000000004,0x1000000000000015};
  book->defs[0x0002f083]->node[ 4] = (Node) {0x1002000000000005,0x1000000000000014};
  book->defs[0x0002f083]->node[ 5] = (Node) {0x1002000000000006,0x1000000000000013};
  book->defs[0x0002f083]->node[ 6] = (Node) {0x1002000000000007,0x1000000000000012};
  book->defs[0x0002f083]->node[ 7] = (Node) {0x1002000000000008,0x1000000000000011};
  book->defs[0x0002f083]->node[ 8] = (Node) {0x1002000000000009,0x1000000000000010};
  book->defs[0x0002f083]->node[ 9] = (Node) {0x100200000000000a,0x100000000000000f};
  book->defs[0x0002f083]->node[10] = (Node) {0x100200000000000b,0x100000000000000e};
  book->defs[0x0002f083]->node[11] = (Node) {0x100000000000000c,0x100000000000000d};
  book->defs[0x0002f083]->node[12] = (Node) {0x4000000000018,0x400000000000d};
  book->defs[0x0002f083]->node[13] = (Node) {0x500000000000c,0x400000000000e};
  book->defs[0x0002f083]->node[14] = (Node) {0x500000000000d,0x400000000000f};
  book->defs[0x0002f083]->node[15] = (Node) {0x500000000000e,0x4000000000010};
  book->defs[0x0002f083]->node[16] = (Node) {0x500000000000f,0x4000000000011};
  book->defs[0x0002f083]->node[17] = (Node) {0x5000000000010,0x4000000000012};
  book->defs[0x0002f083]->node[18] = (Node) {0x5000000000011,0x4000000000013};
  book->defs[0x0002f083]->node[19] = (Node) {0x5000000000012,0x4000000000014};
  book->defs[0x0002f083]->node[20] = (Node) {0x5000000000013,0x4000000000015};
  book->defs[0x0002f083]->node[21] = (Node) {0x5000000000014,0x4000000000016};
  book->defs[0x0002f083]->node[22] = (Node) {0x5000000000015,0x4000000000017};
  book->defs[0x0002f083]->node[23] = (Node) {0x5000000000016,0x5000000000018};
  book->defs[0x0002f083]->node[24] = (Node) {0x400000000000c,0x5000000000017};
  // k13
  book->defs[0x0002f084]           = (Term*) malloc(sizeof(Term));
  book->defs[0x0002f084]->root     = 0x1000000000000000;
  book->defs[0x0002f084]->alen     = 0;
  book->defs[0x0002f084]->acts     = (Wire*) malloc(0 * sizeof(Wire));
  book->defs[0x0002f084]->nlen     = 27;
  book->defs[0x0002f084]->node     = (Node*) malloc(27 * sizeof(Node));
  book->defs[0x0002f084]->node[ 0] = (Node) {0x1002000000000001,0x100000000000001a};
  book->defs[0x0002f084]->node[ 1] = (Node) {0x1002000000000002,0x1000000000000019};
  book->defs[0x0002f084]->node[ 2] = (Node) {0x1002000000000003,0x1000000000000018};
  book->defs[0x0002f084]->node[ 3] = (Node) {0x1002000000000004,0x1000000000000017};
  book->defs[0x0002f084]->node[ 4] = (Node) {0x1002000000000005,0x1000000000000016};
  book->defs[0x0002f084]->node[ 5] = (Node) {0x1002000000000006,0x1000000000000015};
  book->defs[0x0002f084]->node[ 6] = (Node) {0x1002000000000007,0x1000000000000014};
  book->defs[0x0002f084]->node[ 7] = (Node) {0x1002000000000008,0x1000000000000013};
  book->defs[0x0002f084]->node[ 8] = (Node) {0x1002000000000009,0x1000000000000012};
  book->defs[0x0002f084]->node[ 9] = (Node) {0x100200000000000a,0x1000000000000011};
  book->defs[0x0002f084]->node[10] = (Node) {0x100200000000000b,0x1000000000000010};
  book->defs[0x0002f084]->node[11] = (Node) {0x100200000000000c,0x100000000000000f};
  book->defs[0x0002f084]->node[12] = (Node) {0x100000000000000d,0x100000000000000e};
  book->defs[0x0002f084]->node[13] = (Node) {0x400000000001a,0x400000000000e};
  book->defs[0x0002f084]->node[14] = (Node) {0x500000000000d,0x400000000000f};
  book->defs[0x0002f084]->node[15] = (Node) {0x500000000000e,0x4000000000010};
  book->defs[0x0002f084]->node[16] = (Node) {0x500000000000f,0x4000000000011};
  book->defs[0x0002f084]->node[17] = (Node) {0x5000000000010,0x4000000000012};
  book->defs[0x0002f084]->node[18] = (Node) {0x5000000000011,0x4000000000013};
  book->defs[0x0002f084]->node[19] = (Node) {0x5000000000012,0x4000000000014};
  book->defs[0x0002f084]->node[20] = (Node) {0x5000000000013,0x4000000000015};
  book->defs[0x0002f084]->node[21] = (Node) {0x5000000000014,0x4000000000016};
  book->defs[0x0002f084]->node[22] = (Node) {0x5000000000015,0x4000000000017};
  book->defs[0x0002f084]->node[23] = (Node) {0x5000000000016,0x4000000000018};
  book->defs[0x0002f084]->node[24] = (Node) {0x5000000000017,0x4000000000019};
  book->defs[0x0002f084]->node[25] = (Node) {0x5000000000018,0x500000000001a};
  book->defs[0x0002f084]->node[26] = (Node) {0x400000000000d,0x5000000000019};
  // k14
  book->defs[0x0002f085]           = (Term*) malloc(sizeof(Term));
  book->defs[0x0002f085]->root     = 0x1000000000000000;
  book->defs[0x0002f085]->alen     = 0;
  book->defs[0x0002f085]->acts     = (Wire*) malloc(0 * sizeof(Wire));
  book->defs[0x0002f085]->nlen     = 29;
  book->defs[0x0002f085]->node     = (Node*) malloc(29 * sizeof(Node));
  book->defs[0x0002f085]->node[ 0] = (Node) {0x1002000000000001,0x100000000000001c};
  book->defs[0x0002f085]->node[ 1] = (Node) {0x1002000000000002,0x100000000000001b};
  book->defs[0x0002f085]->node[ 2] = (Node) {0x1002000000000003,0x100000000000001a};
  book->defs[0x0002f085]->node[ 3] = (Node) {0x1002000000000004,0x1000000000000019};
  book->defs[0x0002f085]->node[ 4] = (Node) {0x1002000000000005,0x1000000000000018};
  book->defs[0x0002f085]->node[ 5] = (Node) {0x1002000000000006,0x1000000000000017};
  book->defs[0x0002f085]->node[ 6] = (Node) {0x1002000000000007,0x1000000000000016};
  book->defs[0x0002f085]->node[ 7] = (Node) {0x1002000000000008,0x1000000000000015};
  book->defs[0x0002f085]->node[ 8] = (Node) {0x1002000000000009,0x1000000000000014};
  book->defs[0x0002f085]->node[ 9] = (Node) {0x100200000000000a,0x1000000000000013};
  book->defs[0x0002f085]->node[10] = (Node) {0x100200000000000b,0x1000000000000012};
  book->defs[0x0002f085]->node[11] = (Node) {0x100200000000000c,0x1000000000000011};
  book->defs[0x0002f085]->node[12] = (Node) {0x100200000000000d,0x1000000000000010};
  book->defs[0x0002f085]->node[13] = (Node) {0x100000000000000e,0x100000000000000f};
  book->defs[0x0002f085]->node[14] = (Node) {0x400000000001c,0x400000000000f};
  book->defs[0x0002f085]->node[15] = (Node) {0x500000000000e,0x4000000000010};
  book->defs[0x0002f085]->node[16] = (Node) {0x500000000000f,0x4000000000011};
  book->defs[0x0002f085]->node[17] = (Node) {0x5000000000010,0x4000000000012};
  book->defs[0x0002f085]->node[18] = (Node) {0x5000000000011,0x4000000000013};
  book->defs[0x0002f085]->node[19] = (Node) {0x5000000000012,0x4000000000014};
  book->defs[0x0002f085]->node[20] = (Node) {0x5000000000013,0x4000000000015};
  book->defs[0x0002f085]->node[21] = (Node) {0x5000000000014,0x4000000000016};
  book->defs[0x0002f085]->node[22] = (Node) {0x5000000000015,0x4000000000017};
  book->defs[0x0002f085]->node[23] = (Node) {0x5000000000016,0x4000000000018};
  book->defs[0x0002f085]->node[24] = (Node) {0x5000000000017,0x4000000000019};
  book->defs[0x0002f085]->node[25] = (Node) {0x5000000000018,0x400000000001a};
  book->defs[0x0002f085]->node[26] = (Node) {0x5000000000019,0x400000000001b};
  book->defs[0x0002f085]->node[27] = (Node) {0x500000000001a,0x500000000001c};
  book->defs[0x0002f085]->node[28] = (Node) {0x400000000000e,0x500000000001b};
  // k15
  book->defs[0x0002f086]           = (Term*) malloc(sizeof(Term));
  book->defs[0x0002f086]->root     = 0x1000000000000000;
  book->defs[0x0002f086]->alen     = 0;
  book->defs[0x0002f086]->acts     = (Wire*) malloc(0 * sizeof(Wire));
  book->defs[0x0002f086]->nlen     = 31;
  book->defs[0x0002f086]->node     = (Node*) malloc(31 * sizeof(Node));
  book->defs[0x0002f086]->node[ 0] = (Node) {0x1002000000000001,0x100000000000001e};
  book->defs[0x0002f086]->node[ 1] = (Node) {0x1002000000000002,0x100000000000001d};
  book->defs[0x0002f086]->node[ 2] = (Node) {0x1002000000000003,0x100000000000001c};
  book->defs[0x0002f086]->node[ 3] = (Node) {0x1002000000000004,0x100000000000001b};
  book->defs[0x0002f086]->node[ 4] = (Node) {0x1002000000000005,0x100000000000001a};
  book->defs[0x0002f086]->node[ 5] = (Node) {0x1002000000000006,0x1000000000000019};
  book->defs[0x0002f086]->node[ 6] = (Node) {0x1002000000000007,0x1000000000000018};
  book->defs[0x0002f086]->node[ 7] = (Node) {0x1002000000000008,0x1000000000000017};
  book->defs[0x0002f086]->node[ 8] = (Node) {0x1002000000000009,0x1000000000000016};
  book->defs[0x0002f086]->node[ 9] = (Node) {0x100200000000000a,0x1000000000000015};
  book->defs[0x0002f086]->node[10] = (Node) {0x100200000000000b,0x1000000000000014};
  book->defs[0x0002f086]->node[11] = (Node) {0x100200000000000c,0x1000000000000013};
  book->defs[0x0002f086]->node[12] = (Node) {0x100200000000000d,0x1000000000000012};
  book->defs[0x0002f086]->node[13] = (Node) {0x100200000000000e,0x1000000000000011};
  book->defs[0x0002f086]->node[14] = (Node) {0x100000000000000f,0x1000000000000010};
  book->defs[0x0002f086]->node[15] = (Node) {0x400000000001e,0x4000000000010};
  book->defs[0x0002f086]->node[16] = (Node) {0x500000000000f,0x4000000000011};
  book->defs[0x0002f086]->node[17] = (Node) {0x5000000000010,0x4000000000012};
  book->defs[0x0002f086]->node[18] = (Node) {0x5000000000011,0x4000000000013};
  book->defs[0x0002f086]->node[19] = (Node) {0x5000000000012,0x4000000000014};
  book->defs[0x0002f086]->node[20] = (Node) {0x5000000000013,0x4000000000015};
  book->defs[0x0002f086]->node[21] = (Node) {0x5000000000014,0x4000000000016};
  book->defs[0x0002f086]->node[22] = (Node) {0x5000000000015,0x4000000000017};
  book->defs[0x0002f086]->node[23] = (Node) {0x5000000000016,0x4000000000018};
  book->defs[0x0002f086]->node[24] = (Node) {0x5000000000017,0x4000000000019};
  book->defs[0x0002f086]->node[25] = (Node) {0x5000000000018,0x400000000001a};
  book->defs[0x0002f086]->node[26] = (Node) {0x5000000000019,0x400000000001b};
  book->defs[0x0002f086]->node[27] = (Node) {0x500000000001a,0x400000000001c};
  book->defs[0x0002f086]->node[28] = (Node) {0x500000000001b,0x400000000001d};
  book->defs[0x0002f086]->node[29] = (Node) {0x500000000001c,0x500000000001e};
  book->defs[0x0002f086]->node[30] = (Node) {0x400000000000f,0x500000000001d};
  // k16
  book->defs[0x0002f087]           = (Term*) malloc(sizeof(Term));
  book->defs[0x0002f087]->root     = 0x1000000000000000;
  book->defs[0x0002f087]->alen     = 0;
  book->defs[0x0002f087]->acts     = (Wire*) malloc(0 * sizeof(Wire));
  book->defs[0x0002f087]->nlen     = 33;
  book->defs[0x0002f087]->node     = (Node*) malloc(33 * sizeof(Node));
  book->defs[0x0002f087]->node[ 0] = (Node) {0x1002000000000001,0x1000000000000020};
  book->defs[0x0002f087]->node[ 1] = (Node) {0x1002000000000002,0x100000000000001f};
  book->defs[0x0002f087]->node[ 2] = (Node) {0x1002000000000003,0x100000000000001e};
  book->defs[0x0002f087]->node[ 3] = (Node) {0x1002000000000004,0x100000000000001d};
  book->defs[0x0002f087]->node[ 4] = (Node) {0x1002000000000005,0x100000000000001c};
  book->defs[0x0002f087]->node[ 5] = (Node) {0x1002000000000006,0x100000000000001b};
  book->defs[0x0002f087]->node[ 6] = (Node) {0x1002000000000007,0x100000000000001a};
  book->defs[0x0002f087]->node[ 7] = (Node) {0x1002000000000008,0x1000000000000019};
  book->defs[0x0002f087]->node[ 8] = (Node) {0x1002000000000009,0x1000000000000018};
  book->defs[0x0002f087]->node[ 9] = (Node) {0x100200000000000a,0x1000000000000017};
  book->defs[0x0002f087]->node[10] = (Node) {0x100200000000000b,0x1000000000000016};
  book->defs[0x0002f087]->node[11] = (Node) {0x100200000000000c,0x1000000000000015};
  book->defs[0x0002f087]->node[12] = (Node) {0x100200000000000d,0x1000000000000014};
  book->defs[0x0002f087]->node[13] = (Node) {0x100200000000000e,0x1000000000000013};
  book->defs[0x0002f087]->node[14] = (Node) {0x100200000000000f,0x1000000000000012};
  book->defs[0x0002f087]->node[15] = (Node) {0x1000000000000010,0x1000000000000011};
  book->defs[0x0002f087]->node[16] = (Node) {0x4000000000020,0x4000000000011};
  book->defs[0x0002f087]->node[17] = (Node) {0x5000000000010,0x4000000000012};
  book->defs[0x0002f087]->node[18] = (Node) {0x5000000000011,0x4000000000013};
  book->defs[0x0002f087]->node[19] = (Node) {0x5000000000012,0x4000000000014};
  book->defs[0x0002f087]->node[20] = (Node) {0x5000000000013,0x4000000000015};
  book->defs[0x0002f087]->node[21] = (Node) {0x5000000000014,0x4000000000016};
  book->defs[0x0002f087]->node[22] = (Node) {0x5000000000015,0x4000000000017};
  book->defs[0x0002f087]->node[23] = (Node) {0x5000000000016,0x4000000000018};
  book->defs[0x0002f087]->node[24] = (Node) {0x5000000000017,0x4000000000019};
  book->defs[0x0002f087]->node[25] = (Node) {0x5000000000018,0x400000000001a};
  book->defs[0x0002f087]->node[26] = (Node) {0x5000000000019,0x400000000001b};
  book->defs[0x0002f087]->node[27] = (Node) {0x500000000001a,0x400000000001c};
  book->defs[0x0002f087]->node[28] = (Node) {0x500000000001b,0x400000000001d};
  book->defs[0x0002f087]->node[29] = (Node) {0x500000000001c,0x400000000001e};
  book->defs[0x0002f087]->node[30] = (Node) {0x500000000001d,0x400000000001f};
  book->defs[0x0002f087]->node[31] = (Node) {0x500000000001e,0x5000000000020};
  book->defs[0x0002f087]->node[32] = (Node) {0x4000000000010,0x500000000001f};
  // k17
  book->defs[0x0002f088]           = (Term*) malloc(sizeof(Term));
  book->defs[0x0002f088]->root     = 0x1000000000000000;
  book->defs[0x0002f088]->alen     = 0;
  book->defs[0x0002f088]->acts     = (Wire*) malloc(0 * sizeof(Wire));
  book->defs[0x0002f088]->nlen     = 35;
  book->defs[0x0002f088]->node     = (Node*) malloc(35 * sizeof(Node));
  book->defs[0x0002f088]->node[ 0] = (Node) {0x1002000000000001,0x1000000000000022};
  book->defs[0x0002f088]->node[ 1] = (Node) {0x1002000000000002,0x1000000000000021};
  book->defs[0x0002f088]->node[ 2] = (Node) {0x1002000000000003,0x1000000000000020};
  book->defs[0x0002f088]->node[ 3] = (Node) {0x1002000000000004,0x100000000000001f};
  book->defs[0x0002f088]->node[ 4] = (Node) {0x1002000000000005,0x100000000000001e};
  book->defs[0x0002f088]->node[ 5] = (Node) {0x1002000000000006,0x100000000000001d};
  book->defs[0x0002f088]->node[ 6] = (Node) {0x1002000000000007,0x100000000000001c};
  book->defs[0x0002f088]->node[ 7] = (Node) {0x1002000000000008,0x100000000000001b};
  book->defs[0x0002f088]->node[ 8] = (Node) {0x1002000000000009,0x100000000000001a};
  book->defs[0x0002f088]->node[ 9] = (Node) {0x100200000000000a,0x1000000000000019};
  book->defs[0x0002f088]->node[10] = (Node) {0x100200000000000b,0x1000000000000018};
  book->defs[0x0002f088]->node[11] = (Node) {0x100200000000000c,0x1000000000000017};
  book->defs[0x0002f088]->node[12] = (Node) {0x100200000000000d,0x1000000000000016};
  book->defs[0x0002f088]->node[13] = (Node) {0x100200000000000e,0x1000000000000015};
  book->defs[0x0002f088]->node[14] = (Node) {0x100200000000000f,0x1000000000000014};
  book->defs[0x0002f088]->node[15] = (Node) {0x1002000000000010,0x1000000000000013};
  book->defs[0x0002f088]->node[16] = (Node) {0x1000000000000011,0x1000000000000012};
  book->defs[0x0002f088]->node[17] = (Node) {0x4000000000022,0x4000000000012};
  book->defs[0x0002f088]->node[18] = (Node) {0x5000000000011,0x4000000000013};
  book->defs[0x0002f088]->node[19] = (Node) {0x5000000000012,0x4000000000014};
  book->defs[0x0002f088]->node[20] = (Node) {0x5000000000013,0x4000000000015};
  book->defs[0x0002f088]->node[21] = (Node) {0x5000000000014,0x4000000000016};
  book->defs[0x0002f088]->node[22] = (Node) {0x5000000000015,0x4000000000017};
  book->defs[0x0002f088]->node[23] = (Node) {0x5000000000016,0x4000000000018};
  book->defs[0x0002f088]->node[24] = (Node) {0x5000000000017,0x4000000000019};
  book->defs[0x0002f088]->node[25] = (Node) {0x5000000000018,0x400000000001a};
  book->defs[0x0002f088]->node[26] = (Node) {0x5000000000019,0x400000000001b};
  book->defs[0x0002f088]->node[27] = (Node) {0x500000000001a,0x400000000001c};
  book->defs[0x0002f088]->node[28] = (Node) {0x500000000001b,0x400000000001d};
  book->defs[0x0002f088]->node[29] = (Node) {0x500000000001c,0x400000000001e};
  book->defs[0x0002f088]->node[30] = (Node) {0x500000000001d,0x400000000001f};
  book->defs[0x0002f088]->node[31] = (Node) {0x500000000001e,0x4000000000020};
  book->defs[0x0002f088]->node[32] = (Node) {0x500000000001f,0x4000000000021};
  book->defs[0x0002f088]->node[33] = (Node) {0x5000000000020,0x5000000000022};
  book->defs[0x0002f088]->node[34] = (Node) {0x4000000000011,0x5000000000021};
  // k18
  book->defs[0x0002f089]           = (Term*) malloc(sizeof(Term));
  book->defs[0x0002f089]->root     = 0x1000000000000000;
  book->defs[0x0002f089]->alen     = 0;
  book->defs[0x0002f089]->acts     = (Wire*) malloc(0 * sizeof(Wire));
  book->defs[0x0002f089]->nlen     = 37;
  book->defs[0x0002f089]->node     = (Node*) malloc(37 * sizeof(Node));
  book->defs[0x0002f089]->node[ 0] = (Node) {0x1002000000000001,0x1000000000000024};
  book->defs[0x0002f089]->node[ 1] = (Node) {0x1002000000000002,0x1000000000000023};
  book->defs[0x0002f089]->node[ 2] = (Node) {0x1002000000000003,0x1000000000000022};
  book->defs[0x0002f089]->node[ 3] = (Node) {0x1002000000000004,0x1000000000000021};
  book->defs[0x0002f089]->node[ 4] = (Node) {0x1002000000000005,0x1000000000000020};
  book->defs[0x0002f089]->node[ 5] = (Node) {0x1002000000000006,0x100000000000001f};
  book->defs[0x0002f089]->node[ 6] = (Node) {0x1002000000000007,0x100000000000001e};
  book->defs[0x0002f089]->node[ 7] = (Node) {0x1002000000000008,0x100000000000001d};
  book->defs[0x0002f089]->node[ 8] = (Node) {0x1002000000000009,0x100000000000001c};
  book->defs[0x0002f089]->node[ 9] = (Node) {0x100200000000000a,0x100000000000001b};
  book->defs[0x0002f089]->node[10] = (Node) {0x100200000000000b,0x100000000000001a};
  book->defs[0x0002f089]->node[11] = (Node) {0x100200000000000c,0x1000000000000019};
  book->defs[0x0002f089]->node[12] = (Node) {0x100200000000000d,0x1000000000000018};
  book->defs[0x0002f089]->node[13] = (Node) {0x100200000000000e,0x1000000000000017};
  book->defs[0x0002f089]->node[14] = (Node) {0x100200000000000f,0x1000000000000016};
  book->defs[0x0002f089]->node[15] = (Node) {0x1002000000000010,0x1000000000000015};
  book->defs[0x0002f089]->node[16] = (Node) {0x1002000000000011,0x1000000000000014};
  book->defs[0x0002f089]->node[17] = (Node) {0x1000000000000012,0x1000000000000013};
  book->defs[0x0002f089]->node[18] = (Node) {0x4000000000024,0x4000000000013};
  book->defs[0x0002f089]->node[19] = (Node) {0x5000000000012,0x4000000000014};
  book->defs[0x0002f089]->node[20] = (Node) {0x5000000000013,0x4000000000015};
  book->defs[0x0002f089]->node[21] = (Node) {0x5000000000014,0x4000000000016};
  book->defs[0x0002f089]->node[22] = (Node) {0x5000000000015,0x4000000000017};
  book->defs[0x0002f089]->node[23] = (Node) {0x5000000000016,0x4000000000018};
  book->defs[0x0002f089]->node[24] = (Node) {0x5000000000017,0x4000000000019};
  book->defs[0x0002f089]->node[25] = (Node) {0x5000000000018,0x400000000001a};
  book->defs[0x0002f089]->node[26] = (Node) {0x5000000000019,0x400000000001b};
  book->defs[0x0002f089]->node[27] = (Node) {0x500000000001a,0x400000000001c};
  book->defs[0x0002f089]->node[28] = (Node) {0x500000000001b,0x400000000001d};
  book->defs[0x0002f089]->node[29] = (Node) {0x500000000001c,0x400000000001e};
  book->defs[0x0002f089]->node[30] = (Node) {0x500000000001d,0x400000000001f};
  book->defs[0x0002f089]->node[31] = (Node) {0x500000000001e,0x4000000000020};
  book->defs[0x0002f089]->node[32] = (Node) {0x500000000001f,0x4000000000021};
  book->defs[0x0002f089]->node[33] = (Node) {0x5000000000020,0x4000000000022};
  book->defs[0x0002f089]->node[34] = (Node) {0x5000000000021,0x4000000000023};
  book->defs[0x0002f089]->node[35] = (Node) {0x5000000000022,0x5000000000024};
  book->defs[0x0002f089]->node[36] = (Node) {0x4000000000012,0x5000000000023};
  // k19
  book->defs[0x0002f08a]           = (Term*) malloc(sizeof(Term));
  book->defs[0x0002f08a]->root     = 0x1000000000000000;
  book->defs[0x0002f08a]->alen     = 0;
  book->defs[0x0002f08a]->acts     = (Wire*) malloc(0 * sizeof(Wire));
  book->defs[0x0002f08a]->nlen     = 39;
  book->defs[0x0002f08a]->node     = (Node*) malloc(39 * sizeof(Node));
  book->defs[0x0002f08a]->node[ 0] = (Node) {0x1002000000000001,0x1000000000000026};
  book->defs[0x0002f08a]->node[ 1] = (Node) {0x1002000000000002,0x1000000000000025};
  book->defs[0x0002f08a]->node[ 2] = (Node) {0x1002000000000003,0x1000000000000024};
  book->defs[0x0002f08a]->node[ 3] = (Node) {0x1002000000000004,0x1000000000000023};
  book->defs[0x0002f08a]->node[ 4] = (Node) {0x1002000000000005,0x1000000000000022};
  book->defs[0x0002f08a]->node[ 5] = (Node) {0x1002000000000006,0x1000000000000021};
  book->defs[0x0002f08a]->node[ 6] = (Node) {0x1002000000000007,0x1000000000000020};
  book->defs[0x0002f08a]->node[ 7] = (Node) {0x1002000000000008,0x100000000000001f};
  book->defs[0x0002f08a]->node[ 8] = (Node) {0x1002000000000009,0x100000000000001e};
  book->defs[0x0002f08a]->node[ 9] = (Node) {0x100200000000000a,0x100000000000001d};
  book->defs[0x0002f08a]->node[10] = (Node) {0x100200000000000b,0x100000000000001c};
  book->defs[0x0002f08a]->node[11] = (Node) {0x100200000000000c,0x100000000000001b};
  book->defs[0x0002f08a]->node[12] = (Node) {0x100200000000000d,0x100000000000001a};
  book->defs[0x0002f08a]->node[13] = (Node) {0x100200000000000e,0x1000000000000019};
  book->defs[0x0002f08a]->node[14] = (Node) {0x100200000000000f,0x1000000000000018};
  book->defs[0x0002f08a]->node[15] = (Node) {0x1002000000000010,0x1000000000000017};
  book->defs[0x0002f08a]->node[16] = (Node) {0x1002000000000011,0x1000000000000016};
  book->defs[0x0002f08a]->node[17] = (Node) {0x1002000000000012,0x1000000000000015};
  book->defs[0x0002f08a]->node[18] = (Node) {0x1000000000000013,0x1000000000000014};
  book->defs[0x0002f08a]->node[19] = (Node) {0x4000000000026,0x4000000000014};
  book->defs[0x0002f08a]->node[20] = (Node) {0x5000000000013,0x4000000000015};
  book->defs[0x0002f08a]->node[21] = (Node) {0x5000000000014,0x4000000000016};
  book->defs[0x0002f08a]->node[22] = (Node) {0x5000000000015,0x4000000000017};
  book->defs[0x0002f08a]->node[23] = (Node) {0x5000000000016,0x4000000000018};
  book->defs[0x0002f08a]->node[24] = (Node) {0x5000000000017,0x4000000000019};
  book->defs[0x0002f08a]->node[25] = (Node) {0x5000000000018,0x400000000001a};
  book->defs[0x0002f08a]->node[26] = (Node) {0x5000000000019,0x400000000001b};
  book->defs[0x0002f08a]->node[27] = (Node) {0x500000000001a,0x400000000001c};
  book->defs[0x0002f08a]->node[28] = (Node) {0x500000000001b,0x400000000001d};
  book->defs[0x0002f08a]->node[29] = (Node) {0x500000000001c,0x400000000001e};
  book->defs[0x0002f08a]->node[30] = (Node) {0x500000000001d,0x400000000001f};
  book->defs[0x0002f08a]->node[31] = (Node) {0x500000000001e,0x4000000000020};
  book->defs[0x0002f08a]->node[32] = (Node) {0x500000000001f,0x4000000000021};
  book->defs[0x0002f08a]->node[33] = (Node) {0x5000000000020,0x4000000000022};
  book->defs[0x0002f08a]->node[34] = (Node) {0x5000000000021,0x4000000000023};
  book->defs[0x0002f08a]->node[35] = (Node) {0x5000000000022,0x4000000000024};
  book->defs[0x0002f08a]->node[36] = (Node) {0x5000000000023,0x4000000000025};
  book->defs[0x0002f08a]->node[37] = (Node) {0x5000000000024,0x5000000000026};
  book->defs[0x0002f08a]->node[38] = (Node) {0x4000000000013,0x5000000000025};
  // k20
  book->defs[0x0002f0c1]           = (Term*) malloc(sizeof(Term));
  book->defs[0x0002f0c1]->root     = 0x1000000000000000;
  book->defs[0x0002f0c1]->alen     = 0;
  book->defs[0x0002f0c1]->acts     = (Wire*) malloc(0 * sizeof(Wire));
  book->defs[0x0002f0c1]->nlen     = 41;
  book->defs[0x0002f0c1]->node     = (Node*) malloc(41 * sizeof(Node));
  book->defs[0x0002f0c1]->node[ 0] = (Node) {0x1002000000000001,0x1000000000000028};
  book->defs[0x0002f0c1]->node[ 1] = (Node) {0x1002000000000002,0x1000000000000027};
  book->defs[0x0002f0c1]->node[ 2] = (Node) {0x1002000000000003,0x1000000000000026};
  book->defs[0x0002f0c1]->node[ 3] = (Node) {0x1002000000000004,0x1000000000000025};
  book->defs[0x0002f0c1]->node[ 4] = (Node) {0x1002000000000005,0x1000000000000024};
  book->defs[0x0002f0c1]->node[ 5] = (Node) {0x1002000000000006,0x1000000000000023};
  book->defs[0x0002f0c1]->node[ 6] = (Node) {0x1002000000000007,0x1000000000000022};
  book->defs[0x0002f0c1]->node[ 7] = (Node) {0x1002000000000008,0x1000000000000021};
  book->defs[0x0002f0c1]->node[ 8] = (Node) {0x1002000000000009,0x1000000000000020};
  book->defs[0x0002f0c1]->node[ 9] = (Node) {0x100200000000000a,0x100000000000001f};
  book->defs[0x0002f0c1]->node[10] = (Node) {0x100200000000000b,0x100000000000001e};
  book->defs[0x0002f0c1]->node[11] = (Node) {0x100200000000000c,0x100000000000001d};
  book->defs[0x0002f0c1]->node[12] = (Node) {0x100200000000000d,0x100000000000001c};
  book->defs[0x0002f0c1]->node[13] = (Node) {0x100200000000000e,0x100000000000001b};
  book->defs[0x0002f0c1]->node[14] = (Node) {0x100200000000000f,0x100000000000001a};
  book->defs[0x0002f0c1]->node[15] = (Node) {0x1002000000000010,0x1000000000000019};
  book->defs[0x0002f0c1]->node[16] = (Node) {0x1002000000000011,0x1000000000000018};
  book->defs[0x0002f0c1]->node[17] = (Node) {0x1002000000000012,0x1000000000000017};
  book->defs[0x0002f0c1]->node[18] = (Node) {0x1002000000000013,0x1000000000000016};
  book->defs[0x0002f0c1]->node[19] = (Node) {0x1000000000000014,0x1000000000000015};
  book->defs[0x0002f0c1]->node[20] = (Node) {0x4000000000028,0x4000000000015};
  book->defs[0x0002f0c1]->node[21] = (Node) {0x5000000000014,0x4000000000016};
  book->defs[0x0002f0c1]->node[22] = (Node) {0x5000000000015,0x4000000000017};
  book->defs[0x0002f0c1]->node[23] = (Node) {0x5000000000016,0x4000000000018};
  book->defs[0x0002f0c1]->node[24] = (Node) {0x5000000000017,0x4000000000019};
  book->defs[0x0002f0c1]->node[25] = (Node) {0x5000000000018,0x400000000001a};
  book->defs[0x0002f0c1]->node[26] = (Node) {0x5000000000019,0x400000000001b};
  book->defs[0x0002f0c1]->node[27] = (Node) {0x500000000001a,0x400000000001c};
  book->defs[0x0002f0c1]->node[28] = (Node) {0x500000000001b,0x400000000001d};
  book->defs[0x0002f0c1]->node[29] = (Node) {0x500000000001c,0x400000000001e};
  book->defs[0x0002f0c1]->node[30] = (Node) {0x500000000001d,0x400000000001f};
  book->defs[0x0002f0c1]->node[31] = (Node) {0x500000000001e,0x4000000000020};
  book->defs[0x0002f0c1]->node[32] = (Node) {0x500000000001f,0x4000000000021};
  book->defs[0x0002f0c1]->node[33] = (Node) {0x5000000000020,0x4000000000022};
  book->defs[0x0002f0c1]->node[34] = (Node) {0x5000000000021,0x4000000000023};
  book->defs[0x0002f0c1]->node[35] = (Node) {0x5000000000022,0x4000000000024};
  book->defs[0x0002f0c1]->node[36] = (Node) {0x5000000000023,0x4000000000025};
  book->defs[0x0002f0c1]->node[37] = (Node) {0x5000000000024,0x4000000000026};
  book->defs[0x0002f0c1]->node[38] = (Node) {0x5000000000025,0x4000000000027};
  book->defs[0x0002f0c1]->node[39] = (Node) {0x5000000000026,0x5000000000028};
  book->defs[0x0002f0c1]->node[40] = (Node) {0x4000000000014,0x5000000000027};
  // k21
  book->defs[0x0002f0c2]           = (Term*) malloc(sizeof(Term));
  book->defs[0x0002f0c2]->root     = 0x1000000000000000;
  book->defs[0x0002f0c2]->alen     = 0;
  book->defs[0x0002f0c2]->acts     = (Wire*) malloc(0 * sizeof(Wire));
  book->defs[0x0002f0c2]->nlen     = 43;
  book->defs[0x0002f0c2]->node     = (Node*) malloc(43 * sizeof(Node));
  book->defs[0x0002f0c2]->node[ 0] = (Node) {0x1002000000000001,0x100000000000002a};
  book->defs[0x0002f0c2]->node[ 1] = (Node) {0x1002000000000002,0x1000000000000029};
  book->defs[0x0002f0c2]->node[ 2] = (Node) {0x1002000000000003,0x1000000000000028};
  book->defs[0x0002f0c2]->node[ 3] = (Node) {0x1002000000000004,0x1000000000000027};
  book->defs[0x0002f0c2]->node[ 4] = (Node) {0x1002000000000005,0x1000000000000026};
  book->defs[0x0002f0c2]->node[ 5] = (Node) {0x1002000000000006,0x1000000000000025};
  book->defs[0x0002f0c2]->node[ 6] = (Node) {0x1002000000000007,0x1000000000000024};
  book->defs[0x0002f0c2]->node[ 7] = (Node) {0x1002000000000008,0x1000000000000023};
  book->defs[0x0002f0c2]->node[ 8] = (Node) {0x1002000000000009,0x1000000000000022};
  book->defs[0x0002f0c2]->node[ 9] = (Node) {0x100200000000000a,0x1000000000000021};
  book->defs[0x0002f0c2]->node[10] = (Node) {0x100200000000000b,0x1000000000000020};
  book->defs[0x0002f0c2]->node[11] = (Node) {0x100200000000000c,0x100000000000001f};
  book->defs[0x0002f0c2]->node[12] = (Node) {0x100200000000000d,0x100000000000001e};
  book->defs[0x0002f0c2]->node[13] = (Node) {0x100200000000000e,0x100000000000001d};
  book->defs[0x0002f0c2]->node[14] = (Node) {0x100200000000000f,0x100000000000001c};
  book->defs[0x0002f0c2]->node[15] = (Node) {0x1002000000000010,0x100000000000001b};
  book->defs[0x0002f0c2]->node[16] = (Node) {0x1002000000000011,0x100000000000001a};
  book->defs[0x0002f0c2]->node[17] = (Node) {0x1002000000000012,0x1000000000000019};
  book->defs[0x0002f0c2]->node[18] = (Node) {0x1002000000000013,0x1000000000000018};
  book->defs[0x0002f0c2]->node[19] = (Node) {0x1002000000000014,0x1000000000000017};
  book->defs[0x0002f0c2]->node[20] = (Node) {0x1000000000000015,0x1000000000000016};
  book->defs[0x0002f0c2]->node[21] = (Node) {0x400000000002a,0x4000000000016};
  book->defs[0x0002f0c2]->node[22] = (Node) {0x5000000000015,0x4000000000017};
  book->defs[0x0002f0c2]->node[23] = (Node) {0x5000000000016,0x4000000000018};
  book->defs[0x0002f0c2]->node[24] = (Node) {0x5000000000017,0x4000000000019};
  book->defs[0x0002f0c2]->node[25] = (Node) {0x5000000000018,0x400000000001a};
  book->defs[0x0002f0c2]->node[26] = (Node) {0x5000000000019,0x400000000001b};
  book->defs[0x0002f0c2]->node[27] = (Node) {0x500000000001a,0x400000000001c};
  book->defs[0x0002f0c2]->node[28] = (Node) {0x500000000001b,0x400000000001d};
  book->defs[0x0002f0c2]->node[29] = (Node) {0x500000000001c,0x400000000001e};
  book->defs[0x0002f0c2]->node[30] = (Node) {0x500000000001d,0x400000000001f};
  book->defs[0x0002f0c2]->node[31] = (Node) {0x500000000001e,0x4000000000020};
  book->defs[0x0002f0c2]->node[32] = (Node) {0x500000000001f,0x4000000000021};
  book->defs[0x0002f0c2]->node[33] = (Node) {0x5000000000020,0x4000000000022};
  book->defs[0x0002f0c2]->node[34] = (Node) {0x5000000000021,0x4000000000023};
  book->defs[0x0002f0c2]->node[35] = (Node) {0x5000000000022,0x4000000000024};
  book->defs[0x0002f0c2]->node[36] = (Node) {0x5000000000023,0x4000000000025};
  book->defs[0x0002f0c2]->node[37] = (Node) {0x5000000000024,0x4000000000026};
  book->defs[0x0002f0c2]->node[38] = (Node) {0x5000000000025,0x4000000000027};
  book->defs[0x0002f0c2]->node[39] = (Node) {0x5000000000026,0x4000000000028};
  book->defs[0x0002f0c2]->node[40] = (Node) {0x5000000000027,0x4000000000029};
  book->defs[0x0002f0c2]->node[41] = (Node) {0x5000000000028,0x500000000002a};
  book->defs[0x0002f0c2]->node[42] = (Node) {0x4000000000015,0x5000000000029};
  // k22
  book->defs[0x0002f0c3]           = (Term*) malloc(sizeof(Term));
  book->defs[0x0002f0c3]->root     = 0x1000000000000000;
  book->defs[0x0002f0c3]->alen     = 0;
  book->defs[0x0002f0c3]->acts     = (Wire*) malloc(0 * sizeof(Wire));
  book->defs[0x0002f0c3]->nlen     = 45;
  book->defs[0x0002f0c3]->node     = (Node*) malloc(45 * sizeof(Node));
  book->defs[0x0002f0c3]->node[ 0] = (Node) {0x1002000000000001,0x100000000000002c};
  book->defs[0x0002f0c3]->node[ 1] = (Node) {0x1002000000000002,0x100000000000002b};
  book->defs[0x0002f0c3]->node[ 2] = (Node) {0x1002000000000003,0x100000000000002a};
  book->defs[0x0002f0c3]->node[ 3] = (Node) {0x1002000000000004,0x1000000000000029};
  book->defs[0x0002f0c3]->node[ 4] = (Node) {0x1002000000000005,0x1000000000000028};
  book->defs[0x0002f0c3]->node[ 5] = (Node) {0x1002000000000006,0x1000000000000027};
  book->defs[0x0002f0c3]->node[ 6] = (Node) {0x1002000000000007,0x1000000000000026};
  book->defs[0x0002f0c3]->node[ 7] = (Node) {0x1002000000000008,0x1000000000000025};
  book->defs[0x0002f0c3]->node[ 8] = (Node) {0x1002000000000009,0x1000000000000024};
  book->defs[0x0002f0c3]->node[ 9] = (Node) {0x100200000000000a,0x1000000000000023};
  book->defs[0x0002f0c3]->node[10] = (Node) {0x100200000000000b,0x1000000000000022};
  book->defs[0x0002f0c3]->node[11] = (Node) {0x100200000000000c,0x1000000000000021};
  book->defs[0x0002f0c3]->node[12] = (Node) {0x100200000000000d,0x1000000000000020};
  book->defs[0x0002f0c3]->node[13] = (Node) {0x100200000000000e,0x100000000000001f};
  book->defs[0x0002f0c3]->node[14] = (Node) {0x100200000000000f,0x100000000000001e};
  book->defs[0x0002f0c3]->node[15] = (Node) {0x1002000000000010,0x100000000000001d};
  book->defs[0x0002f0c3]->node[16] = (Node) {0x1002000000000011,0x100000000000001c};
  book->defs[0x0002f0c3]->node[17] = (Node) {0x1002000000000012,0x100000000000001b};
  book->defs[0x0002f0c3]->node[18] = (Node) {0x1002000000000013,0x100000000000001a};
  book->defs[0x0002f0c3]->node[19] = (Node) {0x1002000000000014,0x1000000000000019};
  book->defs[0x0002f0c3]->node[20] = (Node) {0x1002000000000015,0x1000000000000018};
  book->defs[0x0002f0c3]->node[21] = (Node) {0x1000000000000016,0x1000000000000017};
  book->defs[0x0002f0c3]->node[22] = (Node) {0x400000000002c,0x4000000000017};
  book->defs[0x0002f0c3]->node[23] = (Node) {0x5000000000016,0x4000000000018};
  book->defs[0x0002f0c3]->node[24] = (Node) {0x5000000000017,0x4000000000019};
  book->defs[0x0002f0c3]->node[25] = (Node) {0x5000000000018,0x400000000001a};
  book->defs[0x0002f0c3]->node[26] = (Node) {0x5000000000019,0x400000000001b};
  book->defs[0x0002f0c3]->node[27] = (Node) {0x500000000001a,0x400000000001c};
  book->defs[0x0002f0c3]->node[28] = (Node) {0x500000000001b,0x400000000001d};
  book->defs[0x0002f0c3]->node[29] = (Node) {0x500000000001c,0x400000000001e};
  book->defs[0x0002f0c3]->node[30] = (Node) {0x500000000001d,0x400000000001f};
  book->defs[0x0002f0c3]->node[31] = (Node) {0x500000000001e,0x4000000000020};
  book->defs[0x0002f0c3]->node[32] = (Node) {0x500000000001f,0x4000000000021};
  book->defs[0x0002f0c3]->node[33] = (Node) {0x5000000000020,0x4000000000022};
  book->defs[0x0002f0c3]->node[34] = (Node) {0x5000000000021,0x4000000000023};
  book->defs[0x0002f0c3]->node[35] = (Node) {0x5000000000022,0x4000000000024};
  book->defs[0x0002f0c3]->node[36] = (Node) {0x5000000000023,0x4000000000025};
  book->defs[0x0002f0c3]->node[37] = (Node) {0x5000000000024,0x4000000000026};
  book->defs[0x0002f0c3]->node[38] = (Node) {0x5000000000025,0x4000000000027};
  book->defs[0x0002f0c3]->node[39] = (Node) {0x5000000000026,0x4000000000028};
  book->defs[0x0002f0c3]->node[40] = (Node) {0x5000000000027,0x4000000000029};
  book->defs[0x0002f0c3]->node[41] = (Node) {0x5000000000028,0x400000000002a};
  book->defs[0x0002f0c3]->node[42] = (Node) {0x5000000000029,0x400000000002b};
  book->defs[0x0002f0c3]->node[43] = (Node) {0x500000000002a,0x500000000002c};
  book->defs[0x0002f0c3]->node[44] = (Node) {0x4000000000016,0x500000000002b};
  // k23
  book->defs[0x0002f0c4]           = (Term*) malloc(sizeof(Term));
  book->defs[0x0002f0c4]->root     = 0x1000000000000000;
  book->defs[0x0002f0c4]->alen     = 0;
  book->defs[0x0002f0c4]->acts     = (Wire*) malloc(0 * sizeof(Wire));
  book->defs[0x0002f0c4]->nlen     = 47;
  book->defs[0x0002f0c4]->node     = (Node*) malloc(47 * sizeof(Node));
  book->defs[0x0002f0c4]->node[ 0] = (Node) {0x1002000000000001,0x100000000000002e};
  book->defs[0x0002f0c4]->node[ 1] = (Node) {0x1002000000000002,0x100000000000002d};
  book->defs[0x0002f0c4]->node[ 2] = (Node) {0x1002000000000003,0x100000000000002c};
  book->defs[0x0002f0c4]->node[ 3] = (Node) {0x1002000000000004,0x100000000000002b};
  book->defs[0x0002f0c4]->node[ 4] = (Node) {0x1002000000000005,0x100000000000002a};
  book->defs[0x0002f0c4]->node[ 5] = (Node) {0x1002000000000006,0x1000000000000029};
  book->defs[0x0002f0c4]->node[ 6] = (Node) {0x1002000000000007,0x1000000000000028};
  book->defs[0x0002f0c4]->node[ 7] = (Node) {0x1002000000000008,0x1000000000000027};
  book->defs[0x0002f0c4]->node[ 8] = (Node) {0x1002000000000009,0x1000000000000026};
  book->defs[0x0002f0c4]->node[ 9] = (Node) {0x100200000000000a,0x1000000000000025};
  book->defs[0x0002f0c4]->node[10] = (Node) {0x100200000000000b,0x1000000000000024};
  book->defs[0x0002f0c4]->node[11] = (Node) {0x100200000000000c,0x1000000000000023};
  book->defs[0x0002f0c4]->node[12] = (Node) {0x100200000000000d,0x1000000000000022};
  book->defs[0x0002f0c4]->node[13] = (Node) {0x100200000000000e,0x1000000000000021};
  book->defs[0x0002f0c4]->node[14] = (Node) {0x100200000000000f,0x1000000000000020};
  book->defs[0x0002f0c4]->node[15] = (Node) {0x1002000000000010,0x100000000000001f};
  book->defs[0x0002f0c4]->node[16] = (Node) {0x1002000000000011,0x100000000000001e};
  book->defs[0x0002f0c4]->node[17] = (Node) {0x1002000000000012,0x100000000000001d};
  book->defs[0x0002f0c4]->node[18] = (Node) {0x1002000000000013,0x100000000000001c};
  book->defs[0x0002f0c4]->node[19] = (Node) {0x1002000000000014,0x100000000000001b};
  book->defs[0x0002f0c4]->node[20] = (Node) {0x1002000000000015,0x100000000000001a};
  book->defs[0x0002f0c4]->node[21] = (Node) {0x1002000000000016,0x1000000000000019};
  book->defs[0x0002f0c4]->node[22] = (Node) {0x1000000000000017,0x1000000000000018};
  book->defs[0x0002f0c4]->node[23] = (Node) {0x400000000002e,0x4000000000018};
  book->defs[0x0002f0c4]->node[24] = (Node) {0x5000000000017,0x4000000000019};
  book->defs[0x0002f0c4]->node[25] = (Node) {0x5000000000018,0x400000000001a};
  book->defs[0x0002f0c4]->node[26] = (Node) {0x5000000000019,0x400000000001b};
  book->defs[0x0002f0c4]->node[27] = (Node) {0x500000000001a,0x400000000001c};
  book->defs[0x0002f0c4]->node[28] = (Node) {0x500000000001b,0x400000000001d};
  book->defs[0x0002f0c4]->node[29] = (Node) {0x500000000001c,0x400000000001e};
  book->defs[0x0002f0c4]->node[30] = (Node) {0x500000000001d,0x400000000001f};
  book->defs[0x0002f0c4]->node[31] = (Node) {0x500000000001e,0x4000000000020};
  book->defs[0x0002f0c4]->node[32] = (Node) {0x500000000001f,0x4000000000021};
  book->defs[0x0002f0c4]->node[33] = (Node) {0x5000000000020,0x4000000000022};
  book->defs[0x0002f0c4]->node[34] = (Node) {0x5000000000021,0x4000000000023};
  book->defs[0x0002f0c4]->node[35] = (Node) {0x5000000000022,0x4000000000024};
  book->defs[0x0002f0c4]->node[36] = (Node) {0x5000000000023,0x4000000000025};
  book->defs[0x0002f0c4]->node[37] = (Node) {0x5000000000024,0x4000000000026};
  book->defs[0x0002f0c4]->node[38] = (Node) {0x5000000000025,0x4000000000027};
  book->defs[0x0002f0c4]->node[39] = (Node) {0x5000000000026,0x4000000000028};
  book->defs[0x0002f0c4]->node[40] = (Node) {0x5000000000027,0x4000000000029};
  book->defs[0x0002f0c4]->node[41] = (Node) {0x5000000000028,0x400000000002a};
  book->defs[0x0002f0c4]->node[42] = (Node) {0x5000000000029,0x400000000002b};
  book->defs[0x0002f0c4]->node[43] = (Node) {0x500000000002a,0x400000000002c};
  book->defs[0x0002f0c4]->node[44] = (Node) {0x500000000002b,0x400000000002d};
  book->defs[0x0002f0c4]->node[45] = (Node) {0x500000000002c,0x500000000002e};
  book->defs[0x0002f0c4]->node[46] = (Node) {0x4000000000017,0x500000000002d};
  // k24
  book->defs[0x0002f0c5]           = (Term*) malloc(sizeof(Term));
  book->defs[0x0002f0c5]->root     = 0x1000000000000000;
  book->defs[0x0002f0c5]->alen     = 0;
  book->defs[0x0002f0c5]->acts     = (Wire*) malloc(0 * sizeof(Wire));
  book->defs[0x0002f0c5]->nlen     = 49;
  book->defs[0x0002f0c5]->node     = (Node*) malloc(49 * sizeof(Node));
  book->defs[0x0002f0c5]->node[ 0] = (Node) {0x1002000000000001,0x1000000000000030};
  book->defs[0x0002f0c5]->node[ 1] = (Node) {0x1002000000000002,0x100000000000002f};
  book->defs[0x0002f0c5]->node[ 2] = (Node) {0x1002000000000003,0x100000000000002e};
  book->defs[0x0002f0c5]->node[ 3] = (Node) {0x1002000000000004,0x100000000000002d};
  book->defs[0x0002f0c5]->node[ 4] = (Node) {0x1002000000000005,0x100000000000002c};
  book->defs[0x0002f0c5]->node[ 5] = (Node) {0x1002000000000006,0x100000000000002b};
  book->defs[0x0002f0c5]->node[ 6] = (Node) {0x1002000000000007,0x100000000000002a};
  book->defs[0x0002f0c5]->node[ 7] = (Node) {0x1002000000000008,0x1000000000000029};
  book->defs[0x0002f0c5]->node[ 8] = (Node) {0x1002000000000009,0x1000000000000028};
  book->defs[0x0002f0c5]->node[ 9] = (Node) {0x100200000000000a,0x1000000000000027};
  book->defs[0x0002f0c5]->node[10] = (Node) {0x100200000000000b,0x1000000000000026};
  book->defs[0x0002f0c5]->node[11] = (Node) {0x100200000000000c,0x1000000000000025};
  book->defs[0x0002f0c5]->node[12] = (Node) {0x100200000000000d,0x1000000000000024};
  book->defs[0x0002f0c5]->node[13] = (Node) {0x100200000000000e,0x1000000000000023};
  book->defs[0x0002f0c5]->node[14] = (Node) {0x100200000000000f,0x1000000000000022};
  book->defs[0x0002f0c5]->node[15] = (Node) {0x1002000000000010,0x1000000000000021};
  book->defs[0x0002f0c5]->node[16] = (Node) {0x1002000000000011,0x1000000000000020};
  book->defs[0x0002f0c5]->node[17] = (Node) {0x1002000000000012,0x100000000000001f};
  book->defs[0x0002f0c5]->node[18] = (Node) {0x1002000000000013,0x100000000000001e};
  book->defs[0x0002f0c5]->node[19] = (Node) {0x1002000000000014,0x100000000000001d};
  book->defs[0x0002f0c5]->node[20] = (Node) {0x1002000000000015,0x100000000000001c};
  book->defs[0x0002f0c5]->node[21] = (Node) {0x1002000000000016,0x100000000000001b};
  book->defs[0x0002f0c5]->node[22] = (Node) {0x1002000000000017,0x100000000000001a};
  book->defs[0x0002f0c5]->node[23] = (Node) {0x1000000000000018,0x1000000000000019};
  book->defs[0x0002f0c5]->node[24] = (Node) {0x4000000000030,0x4000000000019};
  book->defs[0x0002f0c5]->node[25] = (Node) {0x5000000000018,0x400000000001a};
  book->defs[0x0002f0c5]->node[26] = (Node) {0x5000000000019,0x400000000001b};
  book->defs[0x0002f0c5]->node[27] = (Node) {0x500000000001a,0x400000000001c};
  book->defs[0x0002f0c5]->node[28] = (Node) {0x500000000001b,0x400000000001d};
  book->defs[0x0002f0c5]->node[29] = (Node) {0x500000000001c,0x400000000001e};
  book->defs[0x0002f0c5]->node[30] = (Node) {0x500000000001d,0x400000000001f};
  book->defs[0x0002f0c5]->node[31] = (Node) {0x500000000001e,0x4000000000020};
  book->defs[0x0002f0c5]->node[32] = (Node) {0x500000000001f,0x4000000000021};
  book->defs[0x0002f0c5]->node[33] = (Node) {0x5000000000020,0x4000000000022};
  book->defs[0x0002f0c5]->node[34] = (Node) {0x5000000000021,0x4000000000023};
  book->defs[0x0002f0c5]->node[35] = (Node) {0x5000000000022,0x4000000000024};
  book->defs[0x0002f0c5]->node[36] = (Node) {0x5000000000023,0x4000000000025};
  book->defs[0x0002f0c5]->node[37] = (Node) {0x5000000000024,0x4000000000026};
  book->defs[0x0002f0c5]->node[38] = (Node) {0x5000000000025,0x4000000000027};
  book->defs[0x0002f0c5]->node[39] = (Node) {0x5000000000026,0x4000000000028};
  book->defs[0x0002f0c5]->node[40] = (Node) {0x5000000000027,0x4000000000029};
  book->defs[0x0002f0c5]->node[41] = (Node) {0x5000000000028,0x400000000002a};
  book->defs[0x0002f0c5]->node[42] = (Node) {0x5000000000029,0x400000000002b};
  book->defs[0x0002f0c5]->node[43] = (Node) {0x500000000002a,0x400000000002c};
  book->defs[0x0002f0c5]->node[44] = (Node) {0x500000000002b,0x400000000002d};
  book->defs[0x0002f0c5]->node[45] = (Node) {0x500000000002c,0x400000000002e};
  book->defs[0x0002f0c5]->node[46] = (Node) {0x500000000002d,0x400000000002f};
  book->defs[0x0002f0c5]->node[47] = (Node) {0x500000000002e,0x5000000000030};
  book->defs[0x0002f0c5]->node[48] = (Node) {0x4000000000018,0x500000000002f};
  // low
  book->defs[0x00030cfb]           = (Term*) malloc(sizeof(Term));
  book->defs[0x00030cfb]->root     = 0x1000000000000000;
  book->defs[0x00030cfb]->alen     = 0;
  book->defs[0x00030cfb]->acts     = (Wire*) malloc(0 * sizeof(Wire));
  book->defs[0x00030cfb]->nlen     = 4;
  book->defs[0x00030cfb]->node     = (Node*) malloc(4 * sizeof(Node));
  book->defs[0x00030cfb]->node[ 0] = (Node) {0x1000000000000001,0x5000000000003};
  book->defs[0x00030cfb]->node[ 1] = (Node) {0x1000000c33ed9,0x1000000000000002};
  book->defs[0x00030cfb]->node[ 2] = (Node) {0x1000000c33ed3,0x1000000000000003};
  book->defs[0x00030cfb]->node[ 3] = (Node) {0x100000000000f,0x5000000000000};
  // nid
  book->defs[0x00032b68]           = (Term*) malloc(sizeof(Term));
  book->defs[0x00032b68]->root     = 0x1000000000000000;
  book->defs[0x00032b68]->alen     = 0;
  book->defs[0x00032b68]->acts     = (Wire*) malloc(0 * sizeof(Wire));
  book->defs[0x00032b68]->nlen     = 3;
  book->defs[0x00032b68]->node     = (Node*) malloc(3 * sizeof(Node));
  book->defs[0x00032b68]->node[ 0] = (Node) {0x1000000000000001,0x5000000000002};
  book->defs[0x00032b68]->node[ 1] = (Node) {0x1000000cada1d,0x1000000000000002};
  book->defs[0x00032b68]->node[ 2] = (Node) {0x1000000000024,0x5000000000000};
  // not
  book->defs[0x00032cf8]           = (Term*) malloc(sizeof(Term));
  book->defs[0x00032cf8]->root     = 0x1000000000000000;
  book->defs[0x00032cf8]->alen     = 0;
  book->defs[0x00032cf8]->acts     = (Wire*) malloc(0 * sizeof(Wire));
  book->defs[0x00032cf8]->nlen     = 5;
  book->defs[0x00032cf8]->node     = (Node*) malloc(5 * sizeof(Node));
  book->defs[0x00032cf8]->node[ 0] = (Node) {0x1000000000000001,0x1000000000000003};
  book->defs[0x00032cf8]->node[ 1] = (Node) {0x4000000000004,0x1000000000000002};
  book->defs[0x00032cf8]->node[ 2] = (Node) {0x4000000000003,0x5000000000004};
  book->defs[0x00032cf8]->node[ 3] = (Node) {0x4000000000002,0x1000000000000004};
  book->defs[0x00032cf8]->node[ 4] = (Node) {0x4000000000001,0x5000000000002};
  // run
  book->defs[0x00036e72]           = (Term*) malloc(sizeof(Term));
  book->defs[0x00036e72]->root     = 0x1000000000000000;
  book->defs[0x00036e72]->alen     = 0;
  book->defs[0x00036e72]->acts     = (Wire*) malloc(0 * sizeof(Wire));
  book->defs[0x00036e72]->nlen     = 4;
  book->defs[0x00036e72]->node     = (Node*) malloc(4 * sizeof(Node));
  book->defs[0x00036e72]->node[ 0] = (Node) {0x1000000000000001,0x5000000000003};
  book->defs[0x00036e72]->node[ 1] = (Node) {0x1000000db9c99,0x1000000000000002};
  book->defs[0x00036e72]->node[ 2] = (Node) {0x1000000db9c93,0x1000000000000003};
  book->defs[0x00036e72]->node[ 3] = (Node) {0x100000000000f,0x5000000000000};
  // brnS
  book->defs[0x009b6c9d]           = (Term*) malloc(sizeof(Term));
  book->defs[0x009b6c9d]->root     = 0x1000000000000000;
  book->defs[0x009b6c9d]->alen     = 2;
  book->defs[0x009b6c9d]->acts     = (Wire*) malloc(2 * sizeof(Wire));
  book->defs[0x009b6c9d]->acts[ 0] = (Wire) {0x1000000026db2,0x1000000000000003};
  book->defs[0x009b6c9d]->acts[ 1] = (Wire) {0x1000000026db2,0x1000000000000004};
  book->defs[0x009b6c9d]->nlen     = 5;
  book->defs[0x009b6c9d]->node     = (Node*) malloc(5 * sizeof(Node));
  book->defs[0x009b6c9d]->node[ 0] = (Node) {0x1001000000000001,0x1000000000000002};
  book->defs[0x009b6c9d]->node[ 1] = (Node) {0x4000000000003,0x4000000000004};
  book->defs[0x009b6c9d]->node[ 2] = (Node) {0x5000000000003,0x5000000000004};
  book->defs[0x009b6c9d]->node[ 3] = (Node) {0x4000000000001,0x4000000000002};
  book->defs[0x009b6c9d]->node[ 4] = (Node) {0x5000000000001,0x5000000000002};
  // brnZ
  book->defs[0x009b6ca4]           = (Term*) malloc(sizeof(Term));
  book->defs[0x009b6ca4]->root     = 0x5000000000000;
  book->defs[0x009b6ca4]->alen     = 2;
  book->defs[0x009b6ca4]->acts     = (Wire*) malloc(2 * sizeof(Wire));
  book->defs[0x009b6ca4]->acts[ 0] = (Wire) {0x1000000036e72,0x1000000000000000};
  book->defs[0x009b6ca4]->acts[ 1] = (Wire) {0x1000000027081,0x1000000000000001};
  book->defs[0x009b6ca4]->nlen     = 3;
  book->defs[0x009b6ca4]->node     = (Node*) malloc(3 * sizeof(Node));
  book->defs[0x009b6ca4]->node[ 0] = (Node) {0x5000000000002,0x3000000000000};
  book->defs[0x009b6ca4]->node[ 1] = (Node) {0x1000000000013,0x1000000000000002};
  book->defs[0x009b6ca4]->node[ 2] = (Node) {0x100000000000f,0x4000000000000};
  // decI
  book->defs[0x00a299d3]           = (Term*) malloc(sizeof(Term));
  book->defs[0x00a299d3]->root     = 0x1000000000000000;
  book->defs[0x00a299d3]->alen     = 1;
  book->defs[0x00a299d3]->acts     = (Wire*) malloc(1 * sizeof(Wire));
  book->defs[0x00a299d3]->acts[ 0] = (Wire) {0x1000000030cfb,0x1000000000000001};
  book->defs[0x00a299d3]->nlen     = 2;
  book->defs[0x00a299d3]->node     = (Node*) malloc(2 * sizeof(Node));
  book->defs[0x00a299d3]->node[ 0] = (Node) {0x4000000000001,0x5000000000001};
  book->defs[0x00a299d3]->node[ 1] = (Node) {0x4000000000000,0x5000000000000};
  // decO
  book->defs[0x00a299d9]           = (Term*) malloc(sizeof(Term));
  book->defs[0x00a299d9]->root     = 0x1000000000000000;
  book->defs[0x00a299d9]->alen     = 2;
  book->defs[0x00a299d9]->acts     = (Wire*) malloc(2 * sizeof(Wire));
  book->defs[0x00a299d9]->acts[ 0] = (Wire) {0x1000000000013,0x1000000000000001};
  book->defs[0x00a299d9]->acts[ 1] = (Wire) {0x1000000028a67,0x1000000000000002};
  book->defs[0x00a299d9]->nlen     = 3;
  book->defs[0x00a299d9]->node     = (Node*) malloc(3 * sizeof(Node));
  book->defs[0x00a299d9]->node[ 0] = (Node) {0x4000000000002,0x5000000000001};
  book->defs[0x00a299d9]->node[ 1] = (Node) {0x5000000000002,0x5000000000000};
  book->defs[0x00a299d9]->node[ 2] = (Node) {0x4000000000000,0x4000000000001};
  // lowI
  book->defs[0x00c33ed3]           = (Term*) malloc(sizeof(Term));
  book->defs[0x00c33ed3]->root     = 0x1000000000000000;
  book->defs[0x00c33ed3]->alen     = 2;
  book->defs[0x00c33ed3]->acts     = (Wire*) malloc(2 * sizeof(Wire));
  book->defs[0x00c33ed3]->acts[ 0] = (Wire) {0x1000000000013,0x1000000000000001};
  book->defs[0x00c33ed3]->acts[ 1] = (Wire) {0x1000000000019,0x1000000000000002};
  book->defs[0x00c33ed3]->nlen     = 3;
  book->defs[0x00c33ed3]->node     = (Node*) malloc(3 * sizeof(Node));
  book->defs[0x00c33ed3]->node[ 0] = (Node) {0x4000000000001,0x5000000000002};
  book->defs[0x00c33ed3]->node[ 1] = (Node) {0x4000000000000,0x4000000000002};
  book->defs[0x00c33ed3]->node[ 2] = (Node) {0x5000000000001,0x5000000000000};
  // lowO
  book->defs[0x00c33ed9]           = (Term*) malloc(sizeof(Term));
  book->defs[0x00c33ed9]->root     = 0x1000000000000000;
  book->defs[0x00c33ed9]->alen     = 2;
  book->defs[0x00c33ed9]->acts     = (Wire*) malloc(2 * sizeof(Wire));
  book->defs[0x00c33ed9]->acts[ 0] = (Wire) {0x1000000000019,0x1000000000000001};
  book->defs[0x00c33ed9]->acts[ 1] = (Wire) {0x1000000000019,0x1000000000000002};
  book->defs[0x00c33ed9]->nlen     = 3;
  book->defs[0x00c33ed9]->node     = (Node*) malloc(3 * sizeof(Node));
  book->defs[0x00c33ed9]->node[ 0] = (Node) {0x4000000000001,0x5000000000002};
  book->defs[0x00c33ed9]->node[ 1] = (Node) {0x4000000000000,0x4000000000002};
  book->defs[0x00c33ed9]->node[ 2] = (Node) {0x5000000000001,0x5000000000000};
  // nidS
  book->defs[0x00cada1d]           = (Term*) malloc(sizeof(Term));
  book->defs[0x00cada1d]->root     = 0x1000000000000000;
  book->defs[0x00cada1d]->alen     = 2;
  book->defs[0x00cada1d]->acts     = (Wire*) malloc(2 * sizeof(Wire));
  book->defs[0x00cada1d]->acts[ 0] = (Wire) {0x100000000001d,0x1000000000000001};
  book->defs[0x00cada1d]->acts[ 1] = (Wire) {0x1000000032b68,0x1000000000000002};
  book->defs[0x00cada1d]->nlen     = 3;
  book->defs[0x00cada1d]->node     = (Node*) malloc(3 * sizeof(Node));
  book->defs[0x00cada1d]->node[ 0] = (Node) {0x4000000000002,0x5000000000001};
  book->defs[0x00cada1d]->node[ 1] = (Node) {0x5000000000002,0x5000000000000};
  book->defs[0x00cada1d]->node[ 2] = (Node) {0x4000000000000,0x4000000000001};
  // runI
  book->defs[0x00db9c93]           = (Term*) malloc(sizeof(Term));
  book->defs[0x00db9c93]->root     = 0x1000000000000000;
  book->defs[0x00db9c93]->alen     = 3;
  book->defs[0x00db9c93]->acts     = (Wire*) malloc(3 * sizeof(Wire));
  book->defs[0x00db9c93]->acts[ 0] = (Wire) {0x1000000036e72,0x1000000000000001};
  book->defs[0x00db9c93]->acts[ 1] = (Wire) {0x1000000028a67,0x1000000000000002};
  book->defs[0x00db9c93]->acts[ 2] = (Wire) {0x1000000000013,0x1000000000000003};
  book->defs[0x00db9c93]->nlen     = 4;
  book->defs[0x00db9c93]->node     = (Node*) malloc(4 * sizeof(Node));
  book->defs[0x00db9c93]->node[ 0] = (Node) {0x4000000000003,0x5000000000001};
  book->defs[0x00db9c93]->node[ 1] = (Node) {0x5000000000002,0x5000000000000};
  book->defs[0x00db9c93]->node[ 2] = (Node) {0x5000000000003,0x4000000000001};
  book->defs[0x00db9c93]->node[ 3] = (Node) {0x4000000000000,0x4000000000002};
  // runO
  book->defs[0x00db9c99]           = (Term*) malloc(sizeof(Term));
  book->defs[0x00db9c99]->root     = 0x1000000000000000;
  book->defs[0x00db9c99]->alen     = 3;
  book->defs[0x00db9c99]->acts     = (Wire*) malloc(3 * sizeof(Wire));
  book->defs[0x00db9c99]->acts[ 0] = (Wire) {0x1000000036e72,0x1000000000000001};
  book->defs[0x00db9c99]->acts[ 1] = (Wire) {0x1000000028a67,0x1000000000000002};
  book->defs[0x00db9c99]->acts[ 2] = (Wire) {0x1000000000019,0x1000000000000003};
  book->defs[0x00db9c99]->nlen     = 4;
  book->defs[0x00db9c99]->node     = (Node*) malloc(4 * sizeof(Node));
  book->defs[0x00db9c99]->node[ 0] = (Node) {0x4000000000003,0x5000000000001};
  book->defs[0x00db9c99]->node[ 1] = (Node) {0x5000000000002,0x5000000000000};
  book->defs[0x00db9c99]->node[ 2] = (Node) {0x5000000000003,0x4000000000001};
  book->defs[0x00db9c99]->node[ 3] = (Node) {0x4000000000000,0x4000000000002};
}

__host__ void boot(Net* net, u64 ref_id) {
  net->root    = mkptr(REF, ref_id);
  net->hlen    = 1;
  net->head[0] = mkptr(VRR, 0);
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

  // Allocates net and book on CPU
  Net* cpu_net = mknet();
  Book* cpu_book = mkbook();
  populate(cpu_book);
  boot(cpu_net, 0x00029f04); // initial term

  // Prints the input net
  printf("\nINPUT\n=====\n\n");
  print_net(cpu_net);

  // Uploads net and book to GPU
  Net* gpu_net = net_to_gpu(cpu_net);
  Book* gpu_book = book_to_gpu(cpu_book);

  // Marks init time
  struct timespec start, end;
  clock_gettime(CLOCK_MONOTONIC_RAW, &start);

  // Normalizes
  do_normal(gpu_net, gpu_book);
  cudaDeviceSynchronize();

  // Gets end time
  clock_gettime(CLOCK_MONOTONIC_RAW, &end);
  uint64_t delta_time = (end.tv_sec - start.tv_sec) * 1000 + (end.tv_nsec - start.tv_nsec) / 1000000;

  // Reads result back to cpu
  Net* norm = net_to_cpu(gpu_net);

  // Prints the output
  //print_tree(norm, norm->root);
  printf("\nNORMAL ~ rewrites=%d redexes=%d\n======\n\n", norm->rwts, norm->blen);
  print_net(norm);
  printf("Time: %llu ms\n", delta_time);

  // Clears CPU memory
  net_free_on_gpu(gpu_net);
  book_free_on_gpu(gpu_book);

  // Clears GPU memory
  net_free_on_cpu(cpu_net);
  book_free_on_cpu(cpu_book);
  net_free_on_cpu(norm);

  return 0;
}
