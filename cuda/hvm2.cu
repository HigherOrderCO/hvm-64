#include <stdarg.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>

typedef uint8_t u8;
typedef uint16_t u16;
typedef uint32_t u32;
typedef unsigned long long int u64;

// Configuration
// -------------

// This code is initially optimized for nVidia RTX 4090
const u32 BLOCK_LOG2    = 8;                                     // log2 of block size
const u32 BLOCK_SIZE    = 1 << BLOCK_LOG2;                       // threads per block
const u32 UNIT_SIZE     = 4;                                     // threads per rewrite unit
const u32 NODE_SIZE     = 1 << 28;                               // max total nodes (2GB addressable)
const u32 BAGS_SIZE     = BLOCK_SIZE * BLOCK_SIZE * BLOCK_SIZE;  // size of global redex bag
const u32 GROUP_SIZE    = BLOCK_SIZE * BLOCK_SIZE;               // size os a group of bags
const u32 GIDX_SIZE     = BAGS_SIZE + GROUP_SIZE + BLOCK_SIZE;   // aux object to hold scatter indices
const u32 GMOV_SIZE     = BAGS_SIZE;                             // aux object to hold scatter indices
const u32 REPEAT_RATE   = 1;                                     // local rewrites per global rewrite
const u32 MAX_TERM_SIZE = 32;                                    // max number of nodes in a term

// Types
// -----

typedef u8  Col;
typedef u8  Tag;
typedef u32 Val;

// Core terms
const Tag VR1 = 0;
const Tag VR2 = 1;
const Tag CTR = 2;
const Tag REF = 3;

// Unit fracs
const u64 A1 = 0;
const u64 A2 = 1;
const u64 B1 = 2;
const u64 B2 = 3;

// Ports (P1 or P2)
typedef u8 Port;
const u32 P1 = 0;
const u32 P2 = 1;

// Pointers = 4-bit tag + 28-bit val
typedef u32 Ptr;

// Special values
const Ptr NIL = 0x00000000; // empty
const Ptr SPE = 0xFFFFFFF0; // special values after this
const Ptr LEK = 0xFFFFFFFC; // leak (debug)
const Ptr GON = 0xFFFFFFFD; // principal replaced on link
const Ptr NEO = 0xFFFFFFFE; // recently allocated value
const Ptr LCK = 0xFFFFFFFF; // value taken by another thread, will be replaced soon

// Wire extension
const Col EXT = 0xFF;

// Nodes are pairs of pointers
typedef struct alignas(8) {
  Col color;
  Ptr ports[2];
} Node;

// Wires are pairs of pointers
typedef struct alignas(8) {
  Ptr lft;
  Ptr rgt;
} Wire;

// Maximum number of defs in a book
const u32 MAX_DEFS = 1 << 24; // FIXME: make a proper HashMap

typedef struct {
  u32   alen;
  Wire* acts;
  u32   nlen;
  Node* node;
} Term;

// A book
typedef struct {
  Term** defs;
} Book;

// An interaction net 
typedef struct {
  u32   blen; // total bag length (redex count)
  Wire* bags; // redex bags (active pairs)
  Node* node; // memory buffer with all nodes
  u32*  gidx; // aux buffer used on scatter fns
  Wire* gmov; // aux buffer used on scatter fns
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
  u32   nloc;  // where to alloc next node
  u32   rwts;  // total rewrites this performed
  Wire* bag;   // local redex bag
  u32*  locs;  // local alloc locs
} Worker;

// Runtime
// -------

// Integer ceil division
__host__ __device__ u32 div(u32 a, u32 b) {
  return (a + b - 1) / b;
}

// Pseudorandom Number Generator
__host__ __device__ u32 rng(u32 a) {
  return a * 214013 + 2531011;
}

// Attempts to get a value
__device__ Ptr try_lock(Ptr* ref) {
  return atomicExch((u32*)ref, LCK);
}

// Gets the value of a ref; waits if locked
__device__ Ptr lock(Ptr* ref) {
  while (true) {
    Ptr got = try_lock(ref);
    if (got != LCK) {
      return got;
    }
  }
}

// Attempts to replace 'exp' by 'neo', until it succeeds
__device__ void replace(Ptr* ref, Ptr exp, Ptr neo) {
  while (true) {
    Ptr got = atomicCAS((u32*)ref, exp, neo);
    if (got == exp) {
      return;
    }
  }
}

// Creates a new pointer
__host__ __device__ inline Ptr mkptr(Tag tag, Val val) {
  return ((u32)tag << 30) | (val & 0x3FFFFFFF);
}

// Creates a new P1 variable
__host__ __device__ inline Ptr mkvr1(Val loc) {
  return mkptr(VR1, loc);
}

// Creates a new P2 variable
__host__ __device__ inline Ptr mkvr2(Val loc) {
  return mkptr(VR2, loc);
}

// Creates a new root variable
__host__ __device__ inline Ptr mkvrr() {
  return mkptr(VR2, 0);
}

// Creates a new eraser
__host__ __device__ inline Ptr mkera() {
  return mkptr(CTR, 0);
}

// Creates a new constructor
__host__ __device__ inline Ptr mkctr(Val loc) {
  return mkptr(CTR, loc);
}

// Creates a new reference
__host__ __device__ inline Ptr mkref(Val nam) {
  return mkptr(REF, nam);
}

// Gets the tag of a pointer
__host__ __device__ inline Tag tag(Ptr ptr) {
  return (Tag)(ptr >> 30);
}

// Gets the value of a pointer
__host__ __device__ inline Val val(Ptr ptr) {
  return ptr & 0x3FFFFFFF;
}

// Is this pointer a P1 variable?
__host__ __device__ inline bool is_vr1(Ptr ptr) {
  return tag(ptr) == VR1;
}

// Is this pointer a P2 variable?
__host__ __device__ inline bool is_vr2(Ptr ptr) {
  return tag(ptr) == VR2;
}

// Is this pointer a root variable?
__host__ __device__ inline bool is_vrr(Ptr ptr) {
  return tag(ptr) == VR2 && val(ptr) == 0;
}

// Is this pointer a variable?
__host__ __device__ inline bool is_var(Ptr ptr) {
  return is_vr1(ptr) || is_vr2(ptr);
}

// Is this pointer a constructor?
__host__ __device__ inline bool is_ctr(Ptr ptr) {
  return tag(ptr) == CTR && val(ptr) != 0;
}

// Is this pointer an eraser?
__host__ __device__ inline bool is_era(Ptr ptr) {
  return tag(ptr) == CTR && val(ptr) == 0;
}

// Is this pointer a number?
//__host__ __device__ inline bool is_num(Ptr ptr) {
  //return tag(ptr) == NUM;
//}

// Is this pointer a reference?
__host__ __device__ inline bool is_ref(Ptr ptr) {
  return tag(ptr) == REF && ptr < SPE;
}

// Is this pointer a main port?
__host__ __device__ inline bool is_pri(Ptr ptr) {
  return is_ctr(ptr) || is_era(ptr) || is_ref(ptr);
}

// Gets the target ref of a var or redirection pointer
__host__ __device__ inline Ptr* target(Net* net, Ptr ptr) {
  if (is_vr1(ptr)) {
    return &net->node[val(ptr)].ports[P1];
  } else if (is_vr2(ptr)) {
    return &net->node[val(ptr)].ports[P2];
  } else {
    return NULL;
  }
}

__device__ Ptr adjust(Ptr ptr, u32* locs) {
  if (is_vr1(ptr) || is_vr2(ptr) || is_ctr(ptr)) {
    //printf("adjust %x -> %x\n", val(ptr), val(mkptr(tag(ptr), (locs[val(ptr)] << 1) | port(ptr))));
    return mkptr(tag(ptr), locs[val(ptr)]);
  } else {
    return mkptr(tag(ptr), val(ptr));
  }
}

// Creates a new node
__host__ __device__ inline Node mknode(Col col, Ptr p1, Ptr p2) {
  Node node;
  node.color     = col;
  node.ports[P1] = p1;
  node.ports[P2] = p2;
  return node;
}

// Creates a nil node
__host__ __device__ inline Node Node_nil() {
  return mknode(0, NIL, NIL);
}

// Gets a reference to a node's color
__host__ __device__ inline Col* ref_color(Net* net, Val loc) {
  return &net->node[loc].color;
}

// Gets a reference to a node's port
__host__ __device__ inline Ptr* ref_port(Net* net, Val loc, Port port) {
  return &net->node[loc].ports[port];
}

// Gets a reference to a node's root
__host__ __device__ inline Ptr* ref_root(Net* net) {
  return ref_port(net, 0, P2);
}

// Is this node an extension?
__host__ __device__ inline bool is_ext(Net* net, Val loc) {
  return *ref_color(net, loc) == EXT;
}

// Allocates a new node in memory
// FIXME: use 64-bit atomic instead
__device__ inline u32 alloc(Worker* worker, Net *net) {
  while (true) {
    Col* rfc = &net->node[worker->nloc].color;
    Ptr* rf1 = &net->node[worker->nloc].ports[P1];
    Ptr* rf2 = &net->node[worker->nloc].ports[P2];
    u32  got = atomicCAS(rf1, 0, NEO);
    if (got == 0) {
      if (atomicCAS(rf2, 0, NEO) == 0) {
        *rfc = 0;
        return worker->nloc;
      }
      *rf1 = 0;
    }
    worker->nloc = (worker->nloc + 1) % NODE_SIZE;
  }
}

// Creates a new active pair
// FIXME: use 64-bit atomic instead
__device__ inline void put_redex(Worker* worker, Net* net, Ptr a_ptr, Ptr b_ptr, u32 id) {
  worker->bag[worker->frac] = (Wire){a_ptr, b_ptr};
  //if (a_ptr >= SPE || b_ptr >= SPE) {
    //printf("BAD PUT_REDEX %d %d | %d\n", a_ptr >= SPE, b_ptr >= SPE, id);
  //}
  //u32 idx = worker->rloc % BAGS_SIZE;
  //while (true) {
    //Ptr* rfl = &net->bags[idx].lft;
    //Ptr* rfr = &net->bags[idx].rgt;
    //u32  got = atomicCAS(rfl, 0, a_ptr);
    //if (got == 0) {
      //if (atomicCAS(rfr, 0, b_ptr) == 0) {
        //printf("put_redex %d\n", idx);
        //return;
      //}
      //*rfl = 0;
    //}
    //idx = (idx + 1) % BAGS_SIZE;
  //}
}

// Scatter
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
    u32 serial_index = loc[tid];
    u32 spread_index = (BLOCK_SIZE / bag_len) * serial_index;
    bag[spread_index] = wire;
  }
  __syncthreads();
}

// Computes redex indices on blocks (and block lengths)
__global__ void global_scatter_prepare_0(Net* net) {
  u32  tid = threadIdx.x;
  u32  bid = blockIdx.x;
  u32  gid = bid * blockDim.x + tid;
  u32* redex_indices = net->gidx + BLOCK_SIZE * bid;
  u32* block_length  = net->gidx + BLOCK_SIZE * BLOCK_SIZE * BLOCK_SIZE + bid;
  redex_indices[tid] = net->bags[gid].lft > 0 ? 1 : 0; __syncthreads();
  *block_length      = scansum(redex_indices);
  __syncthreads();
  //printf("[%04X on %d] scatter 0 | redex_index=%d block_length=[%d,%d,%d,%d,...]\n", gid, bid, redex_indices[tid], *block_length, *(block_length+1), *(block_length+2), *(block_length+3));
}

// Computes block indices on groups (and group lengths)
__global__ void global_scatter_prepare_1(Net* net) {
  u32 tid = threadIdx.x;
  u32 bid = blockIdx.x;
  u32 gid = bid * blockDim.x + tid;
  u32* block_indices = net->gidx + BLOCK_SIZE * BLOCK_SIZE * BLOCK_SIZE + BLOCK_SIZE * bid;
  u32* group_length  = net->gidx + BLOCK_SIZE * BLOCK_SIZE * BLOCK_SIZE + BLOCK_SIZE * BLOCK_SIZE + bid;
  *group_length      = scansum(block_indices);
  //printf("[%04X on %d] scatter 1 | block_index=%d group_length=%d\n", gid, bid, block_indices[tid], *group_length);
}

// Computes group indices on bag (and bag length)
__global__ void global_scatter_prepare_2(Net* net) {
  u32* group_indices = net->gidx + BLOCK_SIZE * BLOCK_SIZE * BLOCK_SIZE + BLOCK_SIZE * BLOCK_SIZE;
  u32* total_length  = &net->blen; __syncthreads();
  *total_length      = scansum(group_indices);
  //printf("[%04X] scatter 2 | group_index=%d total_length=%d\n", threadIdx.x, group_indices[threadIdx.x], *total_length);
}

// Global scatter: takes redex from bag into aux buff
__global__ void global_scatter_take(Net* net, u32 blocks) {
  u32  tid = threadIdx.x;
  u32  bid = blockIdx.x;
  u32  gid = bid * blockDim.x + tid;
  net->gmov[gid] = net->bags[gid];
  net->bags[gid] = Wire{0,0};
}

// Global scatter: moves redex to target location
__global__ void global_scatter_move(Net* net, u32 blocks) {
  u32  tid = threadIdx.x;
  u32  bid = blockIdx.x;
  u32  gid = bid * blockDim.x + tid;

  // Block and group indices
  u32* block_index = net->gidx + BLOCK_SIZE * BLOCK_SIZE * BLOCK_SIZE + (gid / BLOCK_SIZE);
  u32* group_index = net->gidx + BLOCK_SIZE * BLOCK_SIZE * BLOCK_SIZE + BLOCK_SIZE * BLOCK_SIZE + (gid / BLOCK_SIZE / BLOCK_SIZE);

  // Takes our wire
  Wire wire = net->gmov[gid];
  net->gmov[gid] = Wire{0,0};

  // Moves wire to target location
  if (wire.lft != 0) {
    u32 serial_index = net->gidx[gid+0] + (*block_index) + (*group_index);
    u32 spread_index = (blocks * BLOCK_SIZE / net->blen) * serial_index;
    net->bags[spread_index] = wire;
  }
}

// Cleans up memory used by global scatter
__global__ void global_scatter_cleanup(Net* net) {
  u32  tid = threadIdx.x;
  u32  bid = blockIdx.x;
  u32  gid = bid * blockDim.x + tid;
  u32* redex_index = net->gidx + BLOCK_SIZE * bid + tid;
  u32* block_index = net->gidx + BLOCK_SIZE * BLOCK_SIZE * BLOCK_SIZE + bid + tid;
  u32* group_index = net->gidx + BLOCK_SIZE * BLOCK_SIZE * BLOCK_SIZE + BLOCK_SIZE * BLOCK_SIZE + (bid / BLOCK_SIZE) + tid;
  *redex_index = 0;
  if (bid % BLOCK_SIZE == 0) {
    *block_index = 0;
  }
  if (bid == 0) {
    *group_index = 0;
  }
  //printf("[%04X] clean %d %d %d\n", gid, BLOCK_SIZE * bid + tid, BLOCK_SIZE * BLOCK_SIZE * BLOCK_SIZE + bid, BLOCK_SIZE * BLOCK_SIZE * BLOCK_SIZE + BLOCK_SIZE * BLOCK_SIZE + (bid / BLOCK_SIZE));
}

__host__ Net* net_to_host(Net* device_net);

// Performs a global scatter
u32 do_global_scatter(Net* net, u32 prev_blocks) {
  u32 bag_length, next_blocks;

  if (prev_blocks == -1) {
    prev_blocks = BLOCK_SIZE;
  }

  // Prepares scatter
  global_scatter_prepare_0<<<prev_blocks, BLOCK_SIZE>>>(net);
  global_scatter_prepare_1<<<div(prev_blocks, BLOCK_SIZE), BLOCK_SIZE>>>(net);
  global_scatter_prepare_2<<<div(prev_blocks, BLOCK_SIZE * BLOCK_SIZE), BLOCK_SIZE>>>(net); // always == 1

  // Gets bag length
  cudaMemcpy(&bag_length, &(net->blen), sizeof(u32), cudaMemcpyDeviceToHost);

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

__device__ void deref(Worker* worker, Net* net, Book* book, Ptr* dptr, u32* locs) {
  // Loads definition
  Term* term = book->defs[val(*dptr)];

  // Allocates needed space
  if (term != NULL) {
    //printf("[%04x] deref B\n", worker->gid);
    for (u32 i = 0; i < div(term->nlen, (u32)4); ++i) {
      u32 loc = i * 4 + worker->frac;
      //printf("[%04x] deref B %d %d\n", worker->gid, loc, worker->gid);
      if (loc < term->nlen) {
        locs[loc] = alloc(worker, net);
      }
    }
    //printf("[%04x] deref B end\n", worker->gid);
  }
  __syncwarp();

  // Loads dereferenced nodes, adjusted
  if (term != NULL) {
    //printf("[%04x] deref C\n", worker->gid);
    for (u32 i = 0; i < div(term->nlen, (u32)4); ++i) {
      u32 loc = i * 4 + worker->frac;
      if (loc < term->nlen) {
        Node node = term->node[loc];
        *ref_color(net, locs[loc]) = node.color;
        replace(ref_port(net, locs[loc], P1), NEO, adjust(node.ports[P1], locs));
        replace(ref_port(net, locs[loc], P2), NEO, adjust(node.ports[P2], locs));
      }
    }
    //printf("[%04x] deref C end\n", worker->gid);
  }

  // Loads dereferenced redexes, adjusted
  if (term != NULL && worker->frac < term->alen) {
    //printf("[%04x] deref D\n", worker->gid);
    //for (u32 i = 0; i < div(term->alen, (u32)4); ++i) {
      //u32 loc = i * 4 + worker->frac;
      u32 loc = worker->frac;
      if (loc < term->alen) {
        //printf("[%04X] deref C\n", worker->gid);
        Wire wire = term->acts[loc];
        wire.lft = adjust(wire.lft, locs);
        wire.rgt = adjust(wire.rgt, locs);
        //printf("[%4X] putdref %08X %08X\n", worker->gid, wire.lft, wire.rgt);
        put_redex(worker, net, wire.lft, wire.rgt, 0);
      }
    //}
    //printf("[%04x] deref D end\n", worker->gid);
  }


  // Loads dereferenced root, adjusted
  if (term != NULL) {
    //printf("[%04x] deref F\n", worker->gid);
    *dptr = adjust(term->node[0].ports[P2], locs);
  }

  __syncwarp();
}

// FIXME: remove (used for debug)
__device__ void try_replace(u32* ref, u32 exp, u32 neo) {
  u32 got = atomicCAS(ref, exp, neo);
  if (got != exp) {
    //printf("fail replace %08X != %08X\n", got, exp);
  }
}

// Atomically moves the principal in 'pri_ref' towards 'dir_ptr'
// - If target is a red => clear it and move forwards
// - If target is a var => pass the node into it and halt
// - If target is a pri => form an active pair and halt
__device__ void link(Worker* worker, Net* net, Ptr* pri_ref, Ptr dir_ptr) {
  u32 loops = 0;
  //printf("[%04X] linking node=%8X towards %8X\n", worker->gid, *pri_ref, dir_ptr);
  //printf("move A dir=%08X\n", dir_ptr);
  while (true) {
    if (++loops > 16) {
      //printf("[%04X] move fail\n", worker->gid);
      return;
    }
    // We must be careful to not cross boundaries. When 'trg_ptr' is a VAR, it
    // isn't owned by us. As such, we can't 'take()' it, and must peek instead.
    Ptr* trg_ref = target(net, dir_ptr);
    Col  trg_ext = is_ext(net, val(dir_ptr));
    Ptr  trg_ptr = atomicAdd((u32*)trg_ref, 0);
    //printf("[%04X] move B dir_ptr=%08X trg_ptr=%08X\n", worker->gid, dir_ptr, trg_ptr);

    // If trg_ptr is a redirection, clear it
    if (is_var(trg_ptr) && trg_ext) {
      //printf("[%04x] move C\n");
      u32 cleared = atomicCAS((u32*)trg_ref, trg_ptr, 0);
      if (cleared == trg_ptr) {
        dir_ptr = trg_ptr;
      }
      continue;
    }

    // If trg_ptr is a var, try replacing it by the principal
    else if (is_var(trg_ptr) && !trg_ext) {
      //printf("move D\n");
      // Peeks our own principal
      Ptr pri_ptr = atomicAdd((u32*)pri_ref, 0);
      // We don't own the var, so we must try replacing with a CAS
      u32 replaced = atomicCAS((u32*)trg_ref, trg_ptr, pri_ptr);
      // If it worked, we successfully moved our principal to another region
      if (replaced == trg_ptr) {
        //printf("move E %08x\n", pri_ptr);
        // Collects the backwards path, which is now orphan
        trg_ref = target(net, trg_ptr);
        trg_ext = is_ext(net, val(trg_ptr));
        trg_ptr = atomicAdd((u32*)trg_ref, 0);
        while (is_var(trg_ptr) && is_ext(net, val(trg_ptr))) {
          u32 cleared = atomicCAS((u32*)trg_ref, trg_ptr, 0);
          if (cleared == trg_ptr) {
            trg_ref = target(net, trg_ptr);
            trg_ptr = atomicAdd((u32*)trg_ref, 0);
          }
        }
        // Clear our principal
        try_replace(pri_ref, pri_ptr, 0);
        return;
      // Otherwise, things changed, so we step back and try again
      } else {
        //printf("move F\n");
        continue;
      }
    }

    // If it is a principal, two threads will reach this branch
    // The first to arrive makes a redex, the second clears
    else if (is_pri(trg_ptr) || trg_ptr == GON) {
      //printf("move G\n");
      Ptr *fst_ref = pri_ref < trg_ref ? pri_ref : trg_ref;
      Ptr *snd_ref = pri_ref < trg_ref ? trg_ref : pri_ref;
      Ptr  fst_ptr = atomicExch((u32*)fst_ref, GON);
      if (fst_ptr == GON) {
        try_replace(fst_ref, GON, 0);
        try_replace(snd_ref, GON, 0); // FIXME: can leak GON if we go first
        return;
      } else {
        Ptr snd_ptr = atomicExch((u32*)snd_ref, GON);
        put_redex(worker, net, fst_ptr, snd_ptr, 1);
        return;
      }
    }

    // If it is busy, we wait
    else if (trg_ptr == LCK) {
      //printf("move H\n");
      continue;
    }

    else {
      //printf("move I\n");
      return;
    }
  }
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
__global__ void global_rewrite(Net* net, Book* book, u32 blocks) {
  __shared__ u32 LOCS[BLOCK_SIZE / UNIT_SIZE * MAX_TERM_SIZE]; // aux arr for deref locs

  // Initializes local vars
  Worker worker;
  worker.tid  = threadIdx.x;
  worker.bid  = blockIdx.x;
  worker.gid  = worker.bid * blockDim.x + worker.tid;
  worker.nloc = rng(clock() * (worker.gid + 1)) % NODE_SIZE;
  worker.rwts = 0;
  worker.frac = worker.tid % 4;
  worker.port = worker.tid % 2;
  worker.bag  = (net->bags + worker.gid / UNIT_SIZE * UNIT_SIZE);
  worker.locs = LOCS + worker.tid / UNIT_SIZE * MAX_TERM_SIZE;

  // Scatters redexes
  for (u32 tick = 0; tick < REPEAT_RATE; ++tick) {

    //printf("[%d] A\n", worker.gid);

    // Performs local scatter
    local_scatter(net);

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
    if (wlen == 1 && worker.frac == widx) {
      worker.bag[widx] = Wire{0,0};
    }

    //printf("[%d] B\n", worker.gid);

    // Gets redex endpoints
    worker.a_ptr = worker.frac <= A2 ? wire.lft : wire.rgt;
    worker.b_ptr = worker.frac <= A2 ? wire.rgt : wire.lft;

    //printf("[%04x] got %d | %08x %08x\n", worker.gid, worker.gid / UNIT_SIZE, worker.a_ptr, worker.b_ptr);
    //printf("[%d] C\n", worker.gid);

    // Dereferences
    Ptr* dptr = NULL;
    if (is_ref(worker.a_ptr) && is_ctr(worker.b_ptr)) { dptr = &worker.a_ptr; }
    if (is_ref(worker.b_ptr) && is_ctr(worker.a_ptr)) { dptr = &worker.b_ptr; }
    if (dptr != NULL) { deref(&worker, net, book, dptr, worker.locs); }

    // Gets colors
    Col* a_col = is_ctr(worker.a_ptr) ? ref_color(net, val(worker.a_ptr)) : NULL;
    Col* b_col = is_ctr(worker.b_ptr) ? ref_color(net, val(worker.b_ptr)) : NULL;

    // Defines type of interaction
    bool rewrite = worker.a_ptr != 0 && worker.b_ptr != 0;
    bool var_pri = rewrite && is_var(worker.a_ptr) && is_pri(worker.b_ptr) && worker.port == P1;
    bool era_ctr = rewrite && is_era(worker.a_ptr) && is_ctr(worker.b_ptr);
    bool ctr_era = rewrite && is_ctr(worker.a_ptr) && is_era(worker.b_ptr);
    bool con_con = rewrite && is_ctr(worker.a_ptr) && is_ctr(worker.b_ptr) && *a_col == *b_col;
    bool con_dup = rewrite && is_ctr(worker.a_ptr) && is_ctr(worker.b_ptr) && *a_col != *b_col;

    if (rewrite) {
      //printf("[%04x] reduce %08x %08x | %d %d | %d %d %d %d %d\n", worker.gid, worker.a_ptr, worker.b_ptr, is_ctr(worker.a_ptr), is_ctr(worker.b_ptr), var_pri, era_ctr, ctr_era, con_con, con_dup);
    }

    //if (is_full || rewrite) {
      //printf("[%04llx] %llx redex? | rewrite=%d is_full=%d era_ctr=%d ctr_era=%d con_con=%d con_dup=%d opx_num=%d num_opx=%d opy_num=%d num_opy=%d opx_ctr=%d ctr_opx=%d opy_ctr=%d ctr_opy=%d | %llx %llx | %x %x\n", worker.gid, tick, rewrite, is_full, era_ctr, ctr_era, con_con, con_dup, opx_num, num_opx, opy_num, num_opy, opx_ctr, ctr_opx, opy_ctr, ctr_opy, worker.a_ptr, worker.b_ptr, is_num(worker.a_ptr), is_opy(worker.b_ptr));
    //}

    // Local rewrite variables
    Ptr* ak_ref; // ref to our aux port
    Ptr* bk_ref; // ref to other aux port
    Ptr  ak_ptr; // val of our aux port
    u32  mv_tag; // tag of ptr to send to other side
    u32  mv_loc; // loc of ptr to send to other side
    Col* mv_col; // col of ptr to send to other side
    Ptr  mv_ptr; // val of ptr to send to other side
    u32  y0_idx; // idx of other clone idx

    // Inc rewrite count
    if (rewrite && worker.frac == A1) {
      worker.rwts += 1;
    }

    // Gets port here
    if (rewrite && (ctr_era || con_con || con_dup)) {
      ak_ref = ref_port(net, val(worker.a_ptr), worker.port);
      ak_ptr = lock(ak_ref);
    }

    //printf("[%d] E\n", worker.gid);

    // Gets port there
    if (rewrite && (era_ctr || con_con || con_dup)) {
      bk_ref = ref_port(net, val(worker.b_ptr), worker.port);
    }

    // If era_ctr, send an erasure
    if (rewrite && era_ctr) {
      mv_ptr = mkera();
    }

    // If con_con, send a redirection
    if (rewrite && con_con) {
      mv_ptr = ak_ptr;
    }

    // If con_dup, send clone (CON)
    if (rewrite && con_dup) {
      mv_tag = tag(worker.a_ptr);
      mv_loc = alloc(&worker, net); // alloc a clone
      mv_col = ref_color(net, mv_loc); // ...
      mv_ptr = mkptr(mv_tag, mv_loc); // cloned ptr to send
      worker.locs[worker.frac] = mv_loc; // pass cloned index to other threads
    }
    __syncwarp();

    //printf("[%d] F\n", worker.gid);

    // If con_dup, create inner wires between clones
    if (rewrite && con_dup) {
      u32 c1_loc = worker.locs[(worker.frac <= A2 ? 2 : 0) + 0];
      u32 c2_loc = worker.locs[(worker.frac <= A2 ? 2 : 0) + 1];
      *mv_col = *a_col;
      replace(ref_port(net, mv_loc, P1), NEO, mkptr(worker.port == P1 ? VR1 : VR2, c1_loc));
      replace(ref_port(net, mv_loc, P2), NEO, mkptr(worker.port == P1 ? VR1 : VR2, c2_loc));
    }
    __syncwarp();

    // Flags node as EXT
    if (rewrite && (ctr_era || con_con || con_dup)) {
      *a_col = EXT;
    }

    // Send ptr to other side
    if (rewrite && (era_ctr || con_con || con_dup)) {
      replace(bk_ref, LCK, mv_ptr);
      //printf("send %llx\n", mv_ptr);
    }

    // If var_pri, the var is a deref root, so we just inject the node
    if (rewrite && var_pri && worker.port == P1) {
      atomicExch((u32*)target(net, worker.a_ptr), worker.b_ptr);
    }

    //printf("[%d] G\n", worker.gid);

    // If con_con and we sent a PRI, link the PRI there, towards our port
    // If ctr_era and we have a VAR, link the ERA  here, towards that var
    // If con_dup and we have a VAR, link the CPY  here, towards that var
    if (rewrite &&
      (  con_con && is_pri(ak_ptr)
      || ctr_era && is_var(ak_ptr)
      || con_dup && is_var(ak_ptr))) {
      Ptr targ, *node;
      node = !con_con ? ak_ref : bk_ref;
      targ = !con_con ? ak_ptr : mkptr(worker.port == P1 ? VR1 : VR2, val(worker.a_ptr));
      link(&worker, net, node, targ);
    }

    //printf("[%d] H\n", worker.gid);

    // If we have a PRI...
    // - if ctr_era, form an active pair with the eraser we got
    // - if con_dup, form an active pair with the clone we got
    if (rewrite &&
      (  ctr_era && is_pri(ak_ptr)
      || con_dup && is_pri(ak_ptr))) {
      //printf("[%4X] ~ %8X %8X\n", worker.gid, ak_ptr, *ak_ref);
      put_redex(&worker, net, ak_ptr, lock(ak_ref), 3);
      atomicCAS((u32*)ak_ref, LCK, 0);
    }
    __syncwarp();

    //printf("[%d] I\n", worker.gid);

  }

  // When the work ends, sum stats
  if (worker.rwts > 0 && worker.frac == A1) {
    atomicAdd((u32*)&net->rwts, worker.rwts);
  }

  //printf("[%d] end\n", worker.gid);
}

void do_global_rewrite(Net* net, Book* book, u32 blocks) {
  global_rewrite<<<blocks, BLOCK_SIZE>>>(net, book, blocks);
}

// Expand
// ------

// Performs a parallel expansion of tip references.
// FIXME: currently HARDCODED for perfect binary trees; must improve
__global__ void global_expand(Net* net, Book* book, u32 depth) {
  //__shared__ u32 LOCS[BLOCK_SIZE / UNIT_SIZE * MAX_TERM_SIZE];
  // Initializes local vars
  //Worker worker;
  //worker.tid  = threadIdx.x;
  //worker.bid  = blockIdx.x;
  //worker.gid  = worker.bid * blockDim.x + worker.tid;
  //worker.aloc = rng(clock() * (worker.gid + 1));
  //worker.rwts = 0;
  //worker.frac = worker.tid % 4;
  //worker.port = worker.tid % 2;
  //worker.bag  = (net->bags + worker.gid / UNIT_SIZE * UNIT_SIZE);
  //worker.locs = LOCS + worker.tid / UNIT_SIZE * MAX_TERM_SIZE;
  //u32 div = 1 << (depth - 1);
  //u32 uni = worker.gid / UNIT_SIZE;
  //u32 key = worker.gid / UNIT_SIZE;
  //Ptr dir = mkptr(VRR, 0);
  //for (u32 d = 0; d < depth; ++d) {
    //Ptr* ref = target(net, dir);
    //if (is_ctr(*ref)) {
      //dir = mkptr(key < div ? VR2 : VR1, val(*ref));
      //key = key & (~div);
      //div = div >> 1;
    //}
  //}
  //Ptr* ref = target(net, dir);
  //if (is_ref(*ref)) {
    ////if (worker.frac == A1) {
      ////printf("[%4X] expand %08X at dir=%08X\n", worker.gid, *ref, dir);
    ////}
    //deref(&worker, net, book, ref, worker.locs);
    //if (is_var(*ref)) { // FIXME: can be simplified?
      //atomicExch((u32*)target(net, *ref), dir);
      ////printf("[%4X] linking\n", worker.gid);
    //}
  //}
}

void do_global_expand(Net* net, Book* book, u32 depth) {
  //u32 block_size = UNIT_SIZE * (1 << depth);
  //u32 block_numb = 1;
  //while (block_size > 256) {
    //block_size = block_size / 2;
    //block_numb = block_numb * 2;
  //}
  //global_expand<<<block_numb, block_size>>>(net, book, depth);
}

// Reduce
// ------

// Performs a global rewrite step.
u32 do_reduce(Net* net, Book* book, u32* blocks) {
  // Scatters redexes evenly
  *blocks = do_global_scatter(net, *blocks);

  // Prints debug message
  printf(">> reducing with %d blocks\n", *blocks);

  // Performs global parallel rewrite
  if (*blocks > 0) {
    do_global_rewrite(net, book, *blocks);
  }

  // synchronize and show errors using cudaGetLastError
  cudaDeviceSynchronize();
  cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess) {
    printf("CUDA error: %s\n", cudaGetErrorString(error));
  }

  return *blocks;
}

void do_reduce_all(Net* net, Book* book) {
  //printf(">> reduce_all\n");
  u32 blocks = -1;
  while (do_reduce(net, book, &blocks) != 0) {};
}

// Host<->Device
// -------------

__host__ Net* mknet() {
  Net* net  = (Net*)malloc(sizeof(Net));
  net->rwts = 0;
  net->done = 0;
  net->blen = 0;
  net->bags = (Wire*)malloc(BAGS_SIZE * sizeof(Wire));
  net->gidx = (u32*) malloc(GIDX_SIZE * sizeof(u32));
  net->gmov = (Wire*)malloc(GMOV_SIZE * sizeof(Wire));
  net->node = (Node*)malloc(NODE_SIZE * sizeof(Node));
  memset(net->bags, 0, BAGS_SIZE * sizeof(Wire));
  memset(net->gidx, 0, GIDX_SIZE * sizeof(u32));
  memset(net->gmov, 0, GMOV_SIZE * sizeof(Wire));
  memset(net->node, 0, NODE_SIZE * sizeof(Node));
  return net;
}

__host__ Net* net_to_device(Net* host_net) {
  // Allocate memory on the device for the Net object, and its data
  Net*  device_net;
  Wire* device_bags;
  u32*  device_gidx;
  Wire* device_gmov;
  Node* device_node;

  cudaMalloc((void**)&device_net, sizeof(Net));
  cudaMalloc((void**)&device_bags, BAGS_SIZE * sizeof(Wire));
  cudaMalloc((void**)&device_gidx, GIDX_SIZE * sizeof(u32));
  cudaMalloc((void**)&device_gmov, GMOV_SIZE * sizeof(Wire));
  cudaMalloc((void**)&device_node, NODE_SIZE * sizeof(Node));

  // Copy the host data to the device memory
  cudaMemcpy(device_bags, host_net->bags, BAGS_SIZE * sizeof(Wire), cudaMemcpyHostToDevice);
  cudaMemcpy(device_gidx, host_net->gidx, GIDX_SIZE * sizeof(u32),  cudaMemcpyHostToDevice);
  cudaMemcpy(device_gmov, host_net->gmov, GMOV_SIZE * sizeof(Wire), cudaMemcpyHostToDevice);
  cudaMemcpy(device_node, host_net->node, NODE_SIZE * sizeof(Node), cudaMemcpyHostToDevice);

  // Create a temporary host Net object with device pointers
  Net temp_net  = *host_net;
  temp_net.bags = device_bags;
  temp_net.gidx = device_gidx;
  temp_net.gmov = device_gmov;
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
  host_net->gmov = (Wire*)malloc(GMOV_SIZE * sizeof(Wire));
  host_net->node = (Node*)malloc(NODE_SIZE * sizeof(Node));

  // Retrieve the device pointers for data
  Wire* device_bags;
  u32*  device_gidx;
  Wire* device_gmov;
  Node* device_node;
  cudaMemcpy(&device_bags, &(device_net->bags), sizeof(Wire*), cudaMemcpyDeviceToHost);
  cudaMemcpy(&device_gidx, &(device_net->gidx), sizeof(u32*),  cudaMemcpyDeviceToHost);
  cudaMemcpy(&device_gmov, &(device_net->gmov), sizeof(Wire*), cudaMemcpyDeviceToHost);
  cudaMemcpy(&device_node, &(device_net->node), sizeof(Node*), cudaMemcpyDeviceToHost);

  // Copy the device data to the host memory
  cudaMemcpy(host_net->bags, device_bags, BAGS_SIZE * sizeof(Wire), cudaMemcpyDeviceToHost);
  cudaMemcpy(host_net->gidx, device_gidx, GIDX_SIZE * sizeof(u32),  cudaMemcpyDeviceToHost);
  cudaMemcpy(host_net->gmov, device_gmov, GMOV_SIZE * sizeof(Wire), cudaMemcpyDeviceToHost);
  cudaMemcpy(host_net->node, device_node, NODE_SIZE * sizeof(Node), cudaMemcpyDeviceToHost);

  return host_net;
}

__host__ void net_free_on_device(Net* device_net) {
  // Retrieve the device pointers for data
  Wire* device_bags;
  u32*  device_gidx;
  Wire* device_gmov;
  Node* device_node;
  cudaMemcpy(&device_bags, &(device_net->bags), sizeof(Wire*), cudaMemcpyDeviceToHost);
  cudaMemcpy(&device_gidx, &(device_net->gidx), sizeof(u32*),  cudaMemcpyDeviceToHost);
  cudaMemcpy(&device_gmov, &(device_net->gmov), sizeof(Wire*), cudaMemcpyDeviceToHost);
  cudaMemcpy(&device_node, &(device_net->node), sizeof(Node*), cudaMemcpyDeviceToHost);

  // Free the device memory
  cudaFree(device_bags);
  cudaFree(device_gidx);
  cudaFree(device_gmov);
  cudaFree(device_node);
  cudaFree(device_net);
}

__host__ void net_free_on_host(Net* host_net) {
  free(host_net->bags);
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

__host__ Term* term_to_device(Term* host_term) {
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

__host__ Term* term_to_host(Term* device_term) {
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

__host__ Book* book_to_device(Book* host_book) {
  Book* device_book;
  Term** device_defs;

  cudaMalloc((void**)&device_book, sizeof(Book));
  cudaMalloc((void**)&device_defs, MAX_DEFS * sizeof(Term*));
  cudaMemset(device_defs, 0, MAX_DEFS * sizeof(Term*));

  for (u32 i = 0; i < MAX_DEFS; ++i) {
    if (host_book->defs[i] != NULL) {
      Term* device_term = term_to_device(host_book->defs[i]);
      cudaMemcpy(device_defs + i, &device_term, sizeof(Term*), cudaMemcpyHostToDevice);
    }
  }

  cudaMemcpy(&(device_book->defs), &device_defs, sizeof(Term*), cudaMemcpyHostToDevice);

  return device_book;
}

// opposite of book_to_device; same style as net_to_host and term_to_host
__host__ Book* book_to_host(Book* device_book) {
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
  for (u32 i = 0; i < MAX_DEFS; ++i) {
    if (host_book->defs[i] != NULL) {
      host_book->defs[i] = term_to_host(host_book->defs[i]);
    }
  }

  return host_book;
}

__host__ void book_free_on_device(Book* device_book) {
  // TODO
}

__host__ void book_free_on_host(Book* host_book) {
  // TODO
}

// Debugging
// ---------

__host__ const char* show_ptr(Ptr ptr, u32 slot) {
  static char buffer[8][20];
  if (ptr == 0) {
    strcpy(buffer[slot], "            ");
    return buffer[slot];
  } else if (ptr == LCK) {
    strcpy(buffer[slot], "[::locked::]");
    return buffer[slot];
  } else if (ptr == LEK) {
    strcpy(buffer[slot], "[::leaked::]");
    return buffer[slot];
  } else if (ptr == GON) {
    strcpy(buffer[slot], "[:::gone:::]");
    return buffer[slot];
  } else if (ptr == NEO) {
    strcpy(buffer[slot], "[:reserved:]");
    return buffer[slot];
  } else {
    const char* tag_str = NULL;
    switch (tag(ptr)) {
      case VR1: tag_str = "VR1"; break;
      case VR2: tag_str = "VR2"; break;
      case CTR: tag_str = "CTR"; break;
      case REF: tag_str = "REF"; break;
    }
    snprintf(buffer[slot], sizeof(buffer[slot]), "%s:%08X", tag_str, val(ptr));
    return buffer[slot];
  }
}

// Prints a net in hexadecimal, limited to a given size
void print_net(Net* net) {
  printf("Bags:\n");
  for (u32 i = 0; i < BAGS_SIZE; ++i) {
    Ptr a = net->bags[i].lft;
    Ptr b = net->bags[i].rgt;
    if (a != 0 || b != 0) {
      printf("- [%07X] %s %s\n", i, show_ptr(a,0), show_ptr(b,1));
    }
  }
  printf("Node:\n");
  for (u32 i = 0; i < NODE_SIZE; ++i) {
    Col k = net->node[i].color;
    Ptr a = net->node[i].ports[P1];
    Ptr b = net->node[i].ports[P2];
    if (a != 0 || b != 0) {
      printf("- [%07X] %04x %s %s\n", i, k, show_ptr(a,0), show_ptr(b,1));
    }
  }
  printf("BLen: %u\n", net->blen);
  printf("Rwts: %u\n", net->rwts);
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
// Traverses to the other side of a wire
__host__ __device__ Ptr enter(Net* net, Ptr dir) {
  Ptr ptr = *target(net, dir);
  Col ext = is_ext(net, val(dir));
  while (ext) {
    ptr = *target(net, ptr);
    ext = is_ext(net, val(ptr));
  }
  return ptr;
}

// Recursive function to print a term as a tree with unique variable IDs
__host__ void print_tree_go(Net* net, Ptr dir, Map* var_ids) {
  Ptr ptr = *target(net, dir);
  Col ext = is_ext(net, val(dir));
  if (is_var(ptr)) {
    if (ext) {
      print_tree_go(net, ptr, var_ids);
    } else {
      u32 got = map_lookup(var_ids, ptr);
      if (got == var_ids->size) {
        u32 name = var_ids->size;
        Ptr targ = *target(net, enter(net, ptr));
        map_insert(var_ids, targ, name);
        printf("x%d", name);
      } else {
        printf("x%d", got);
      }
    }
  } else if (is_ref(ptr)) {
    printf("{%x}", val(ptr));
  } else if (is_era(ptr)) {
    printf("*");
  } else {
    printf("(%d ", *ref_color(net,val(ptr)));
    print_tree_go(net, mkvr1(val(ptr)), var_ids);
    printf(" ");
    print_tree_go(net, mkvr2(val(ptr)), var_ids);
    printf(")");
  }
}

__host__ void print_tree(Net* net, Ptr dir) {
  Map var_ids = { .size = 0 };
  print_tree_go(net, dir, &var_ids);
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
  book->defs[0x0000000f]->alen     = 0;
  book->defs[0x0000000f]->acts     = (Wire*) malloc(0 * sizeof(Wire));
  book->defs[0x0000000f]->nlen     = 4;
  book->defs[0x0000000f]->node     = (Node*) malloc(4 * sizeof(Node));
  book->defs[0x0000000f]->node[ 0] = (Node) {0x0000,0x00000000,0x80000001};
  book->defs[0x0000000f]->node[ 1] = (Node) {0x0000,0x80000000,0x80000002};
  book->defs[0x0000000f]->node[ 2] = (Node) {0x0000,0x80000000,0x80000003};
  book->defs[0x0000000f]->node[ 3] = (Node) {0x0000,0x40000003,0x00000003};
  // F
  book->defs[0x00000010]           = (Term*) malloc(sizeof(Term));
  book->defs[0x00000010]->alen     = 0;
  book->defs[0x00000010]->acts     = (Wire*) malloc(0 * sizeof(Wire));
  book->defs[0x00000010]->nlen     = 3;
  book->defs[0x00000010]->node     = (Node*) malloc(3 * sizeof(Node));
  book->defs[0x00000010]->node[ 0] = (Node) {0x0000,0x00000000,0x80000001};
  book->defs[0x00000010]->node[ 1] = (Node) {0x0000,0x80000000,0x80000002};
  book->defs[0x00000010]->node[ 2] = (Node) {0x0000,0x40000002,0x00000002};
  // I
  book->defs[0x00000013]           = (Term*) malloc(sizeof(Term));
  book->defs[0x00000013]->alen     = 0;
  book->defs[0x00000013]->acts     = (Wire*) malloc(0 * sizeof(Wire));
  book->defs[0x00000013]->nlen     = 6;
  book->defs[0x00000013]->node     = (Node*) malloc(6 * sizeof(Node));
  book->defs[0x00000013]->node[ 0] = (Node) {0x0000,0x00000000,0x80000001};
  book->defs[0x00000013]->node[ 1] = (Node) {0x0000,0x00000004,0x80000002};
  book->defs[0x00000013]->node[ 2] = (Node) {0x0000,0x80000000,0x80000003};
  book->defs[0x00000013]->node[ 3] = (Node) {0x0000,0x80000004,0x80000005};
  book->defs[0x00000013]->node[ 4] = (Node) {0x0000,0x00000001,0x40000005};
  book->defs[0x00000013]->node[ 5] = (Node) {0x0000,0x80000000,0x40000004};
  // O
  book->defs[0x00000019]           = (Term*) malloc(sizeof(Term));
  book->defs[0x00000019]->alen     = 0;
  book->defs[0x00000019]->acts     = (Wire*) malloc(0 * sizeof(Wire));
  book->defs[0x00000019]->nlen     = 6;
  book->defs[0x00000019]->node     = (Node*) malloc(6 * sizeof(Node));
  book->defs[0x00000019]->node[ 0] = (Node) {0x0000,0x00000000,0x80000001};
  book->defs[0x00000019]->node[ 1] = (Node) {0x0000,0x00000003,0x80000002};
  book->defs[0x00000019]->node[ 2] = (Node) {0x0000,0x80000003,0x80000004};
  book->defs[0x00000019]->node[ 3] = (Node) {0x0000,0x00000001,0x40000005};
  book->defs[0x00000019]->node[ 4] = (Node) {0x0000,0x80000000,0x80000005};
  book->defs[0x00000019]->node[ 5] = (Node) {0x0000,0x80000000,0x40000003};
  // S
  book->defs[0x0000001d]           = (Term*) malloc(sizeof(Term));
  book->defs[0x0000001d]->alen     = 0;
  book->defs[0x0000001d]->acts     = (Wire*) malloc(0 * sizeof(Wire));
  book->defs[0x0000001d]->nlen     = 5;
  book->defs[0x0000001d]->node     = (Node*) malloc(5 * sizeof(Node));
  book->defs[0x0000001d]->node[ 0] = (Node) {0x0000,0x00000000,0x80000001};
  book->defs[0x0000001d]->node[ 1] = (Node) {0x0000,0x00000003,0x80000002};
  book->defs[0x0000001d]->node[ 2] = (Node) {0x0000,0x80000003,0x80000004};
  book->defs[0x0000001d]->node[ 3] = (Node) {0x0000,0x00000001,0x40000004};
  book->defs[0x0000001d]->node[ 4] = (Node) {0x0000,0x80000000,0x40000003};
  // T
  book->defs[0x0000001e]           = (Term*) malloc(sizeof(Term));
  book->defs[0x0000001e]->alen     = 0;
  book->defs[0x0000001e]->acts     = (Wire*) malloc(0 * sizeof(Wire));
  book->defs[0x0000001e]->nlen     = 3;
  book->defs[0x0000001e]->node     = (Node*) malloc(3 * sizeof(Node));
  book->defs[0x0000001e]->node[ 0] = (Node) {0x0000,0x00000000,0x80000001};
  book->defs[0x0000001e]->node[ 1] = (Node) {0x0000,0x40000002,0x80000002};
  book->defs[0x0000001e]->node[ 2] = (Node) {0x0000,0x80000000,0x00000001};
  // Z
  book->defs[0x00000024]           = (Term*) malloc(sizeof(Term));
  book->defs[0x00000024]->alen     = 0;
  book->defs[0x00000024]->acts     = (Wire*) malloc(0 * sizeof(Wire));
  book->defs[0x00000024]->nlen     = 3;
  book->defs[0x00000024]->node     = (Node*) malloc(3 * sizeof(Node));
  book->defs[0x00000024]->node[ 0] = (Node) {0x0000,0x00000000,0x80000001};
  book->defs[0x00000024]->node[ 1] = (Node) {0x0000,0x80000000,0x80000002};
  book->defs[0x00000024]->node[ 2] = (Node) {0x0000,0x40000002,0x00000002};
  // c0
  book->defs[0x000009c1]           = (Term*) malloc(sizeof(Term));
  book->defs[0x000009c1]->alen     = 0;
  book->defs[0x000009c1]->acts     = (Wire*) malloc(0 * sizeof(Wire));
  book->defs[0x000009c1]->nlen     = 3;
  book->defs[0x000009c1]->node     = (Node*) malloc(3 * sizeof(Node));
  book->defs[0x000009c1]->node[ 0] = (Node) {0x0000,0x00000000,0x80000001};
  book->defs[0x000009c1]->node[ 1] = (Node) {0x0000,0x80000000,0x80000002};
  book->defs[0x000009c1]->node[ 2] = (Node) {0x0000,0x40000002,0x00000002};
  // c1
  book->defs[0x000009c2]           = (Term*) malloc(sizeof(Term));
  book->defs[0x000009c2]->alen     = 0;
  book->defs[0x000009c2]->acts     = (Wire*) malloc(0 * sizeof(Wire));
  book->defs[0x000009c2]->nlen     = 4;
  book->defs[0x000009c2]->node     = (Node*) malloc(4 * sizeof(Node));
  book->defs[0x000009c2]->node[ 0] = (Node) {0x0000,0x00000000,0x80000001};
  book->defs[0x000009c2]->node[ 1] = (Node) {0x0000,0x80000002,0x80000003};
  book->defs[0x000009c2]->node[ 2] = (Node) {0x0000,0x00000003,0x40000003};
  book->defs[0x000009c2]->node[ 3] = (Node) {0x0000,0x00000002,0x40000002};
  // c2
  book->defs[0x000009c3]           = (Term*) malloc(sizeof(Term));
  book->defs[0x000009c3]->alen     = 0;
  book->defs[0x000009c3]->acts     = (Wire*) malloc(0 * sizeof(Wire));
  book->defs[0x000009c3]->nlen     = 6;
  book->defs[0x000009c3]->node     = (Node*) malloc(6 * sizeof(Node));
  book->defs[0x000009c3]->node[ 0] = (Node) {0x0000,0x00000000,0x80000001};
  book->defs[0x000009c3]->node[ 1] = (Node) {0x0000,0x80000002,0x80000005};
  book->defs[0x000009c3]->node[ 2] = (Node) {0x0001,0x80000003,0x80000004};
  book->defs[0x000009c3]->node[ 3] = (Node) {0x0000,0x00000005,0x00000004};
  book->defs[0x000009c3]->node[ 4] = (Node) {0x0000,0x40000003,0x40000005};
  book->defs[0x000009c3]->node[ 5] = (Node) {0x0000,0x00000003,0x40000004};
  // c3
  book->defs[0x000009c4]           = (Term*) malloc(sizeof(Term));
  book->defs[0x000009c4]->alen     = 0;
  book->defs[0x000009c4]->acts     = (Wire*) malloc(0 * sizeof(Wire));
  book->defs[0x000009c4]->nlen     = 8;
  book->defs[0x000009c4]->node     = (Node*) malloc(8 * sizeof(Node));
  book->defs[0x000009c4]->node[ 0] = (Node) {0x0000,0x00000000,0x80000001};
  book->defs[0x000009c4]->node[ 1] = (Node) {0x0000,0x80000002,0x80000007};
  book->defs[0x000009c4]->node[ 2] = (Node) {0x0001,0x80000003,0x80000006};
  book->defs[0x000009c4]->node[ 3] = (Node) {0x0001,0x80000004,0x80000005};
  book->defs[0x000009c4]->node[ 4] = (Node) {0x0000,0x00000007,0x00000005};
  book->defs[0x000009c4]->node[ 5] = (Node) {0x0000,0x40000004,0x00000006};
  book->defs[0x000009c4]->node[ 6] = (Node) {0x0000,0x40000005,0x40000007};
  book->defs[0x000009c4]->node[ 7] = (Node) {0x0000,0x00000004,0x40000006};
  // c4
  book->defs[0x000009c5]           = (Term*) malloc(sizeof(Term));
  book->defs[0x000009c5]->alen     = 0;
  book->defs[0x000009c5]->acts     = (Wire*) malloc(0 * sizeof(Wire));
  book->defs[0x000009c5]->nlen     = 10;
  book->defs[0x000009c5]->node     = (Node*) malloc(10 * sizeof(Node));
  book->defs[0x000009c5]->node[ 0] = (Node) {0x0000,0x00000000,0x80000001};
  book->defs[0x000009c5]->node[ 1] = (Node) {0x0000,0x80000002,0x80000009};
  book->defs[0x000009c5]->node[ 2] = (Node) {0x0001,0x80000003,0x80000008};
  book->defs[0x000009c5]->node[ 3] = (Node) {0x0001,0x80000004,0x80000007};
  book->defs[0x000009c5]->node[ 4] = (Node) {0x0001,0x80000005,0x80000006};
  book->defs[0x000009c5]->node[ 5] = (Node) {0x0000,0x00000009,0x00000006};
  book->defs[0x000009c5]->node[ 6] = (Node) {0x0000,0x40000005,0x00000007};
  book->defs[0x000009c5]->node[ 7] = (Node) {0x0000,0x40000006,0x00000008};
  book->defs[0x000009c5]->node[ 8] = (Node) {0x0000,0x40000007,0x40000009};
  book->defs[0x000009c5]->node[ 9] = (Node) {0x0000,0x00000005,0x40000008};
  // c5
  book->defs[0x000009c6]           = (Term*) malloc(sizeof(Term));
  book->defs[0x000009c6]->alen     = 0;
  book->defs[0x000009c6]->acts     = (Wire*) malloc(0 * sizeof(Wire));
  book->defs[0x000009c6]->nlen     = 12;
  book->defs[0x000009c6]->node     = (Node*) malloc(12 * sizeof(Node));
  book->defs[0x000009c6]->node[ 0] = (Node) {0x0000,0x00000000,0x80000001};
  book->defs[0x000009c6]->node[ 1] = (Node) {0x0000,0x80000002,0x8000000b};
  book->defs[0x000009c6]->node[ 2] = (Node) {0x0001,0x80000003,0x8000000a};
  book->defs[0x000009c6]->node[ 3] = (Node) {0x0001,0x80000004,0x80000009};
  book->defs[0x000009c6]->node[ 4] = (Node) {0x0001,0x80000005,0x80000008};
  book->defs[0x000009c6]->node[ 5] = (Node) {0x0001,0x80000006,0x80000007};
  book->defs[0x000009c6]->node[ 6] = (Node) {0x0000,0x0000000b,0x00000007};
  book->defs[0x000009c6]->node[ 7] = (Node) {0x0000,0x40000006,0x00000008};
  book->defs[0x000009c6]->node[ 8] = (Node) {0x0000,0x40000007,0x00000009};
  book->defs[0x000009c6]->node[ 9] = (Node) {0x0000,0x40000008,0x0000000a};
  book->defs[0x000009c6]->node[10] = (Node) {0x0000,0x40000009,0x4000000b};
  book->defs[0x000009c6]->node[11] = (Node) {0x0000,0x00000006,0x4000000a};
  // c6
  book->defs[0x000009c7]           = (Term*) malloc(sizeof(Term));
  book->defs[0x000009c7]->alen     = 0;
  book->defs[0x000009c7]->acts     = (Wire*) malloc(0 * sizeof(Wire));
  book->defs[0x000009c7]->nlen     = 14;
  book->defs[0x000009c7]->node     = (Node*) malloc(14 * sizeof(Node));
  book->defs[0x000009c7]->node[ 0] = (Node) {0x0000,0x00000000,0x80000001};
  book->defs[0x000009c7]->node[ 1] = (Node) {0x0000,0x80000002,0x8000000d};
  book->defs[0x000009c7]->node[ 2] = (Node) {0x0001,0x80000003,0x8000000c};
  book->defs[0x000009c7]->node[ 3] = (Node) {0x0001,0x80000004,0x8000000b};
  book->defs[0x000009c7]->node[ 4] = (Node) {0x0001,0x80000005,0x8000000a};
  book->defs[0x000009c7]->node[ 5] = (Node) {0x0001,0x80000006,0x80000009};
  book->defs[0x000009c7]->node[ 6] = (Node) {0x0001,0x80000007,0x80000008};
  book->defs[0x000009c7]->node[ 7] = (Node) {0x0000,0x0000000d,0x00000008};
  book->defs[0x000009c7]->node[ 8] = (Node) {0x0000,0x40000007,0x00000009};
  book->defs[0x000009c7]->node[ 9] = (Node) {0x0000,0x40000008,0x0000000a};
  book->defs[0x000009c7]->node[10] = (Node) {0x0000,0x40000009,0x0000000b};
  book->defs[0x000009c7]->node[11] = (Node) {0x0000,0x4000000a,0x0000000c};
  book->defs[0x000009c7]->node[12] = (Node) {0x0000,0x4000000b,0x4000000d};
  book->defs[0x000009c7]->node[13] = (Node) {0x0000,0x00000007,0x4000000c};
  // c7
  book->defs[0x000009c8]           = (Term*) malloc(sizeof(Term));
  book->defs[0x000009c8]->alen     = 0;
  book->defs[0x000009c8]->acts     = (Wire*) malloc(0 * sizeof(Wire));
  book->defs[0x000009c8]->nlen     = 16;
  book->defs[0x000009c8]->node     = (Node*) malloc(16 * sizeof(Node));
  book->defs[0x000009c8]->node[ 0] = (Node) {0x0000,0x00000000,0x80000001};
  book->defs[0x000009c8]->node[ 1] = (Node) {0x0000,0x80000002,0x8000000f};
  book->defs[0x000009c8]->node[ 2] = (Node) {0x0001,0x80000003,0x8000000e};
  book->defs[0x000009c8]->node[ 3] = (Node) {0x0001,0x80000004,0x8000000d};
  book->defs[0x000009c8]->node[ 4] = (Node) {0x0001,0x80000005,0x8000000c};
  book->defs[0x000009c8]->node[ 5] = (Node) {0x0001,0x80000006,0x8000000b};
  book->defs[0x000009c8]->node[ 6] = (Node) {0x0001,0x80000007,0x8000000a};
  book->defs[0x000009c8]->node[ 7] = (Node) {0x0001,0x80000008,0x80000009};
  book->defs[0x000009c8]->node[ 8] = (Node) {0x0000,0x0000000f,0x00000009};
  book->defs[0x000009c8]->node[ 9] = (Node) {0x0000,0x40000008,0x0000000a};
  book->defs[0x000009c8]->node[10] = (Node) {0x0000,0x40000009,0x0000000b};
  book->defs[0x000009c8]->node[11] = (Node) {0x0000,0x4000000a,0x0000000c};
  book->defs[0x000009c8]->node[12] = (Node) {0x0000,0x4000000b,0x0000000d};
  book->defs[0x000009c8]->node[13] = (Node) {0x0000,0x4000000c,0x0000000e};
  book->defs[0x000009c8]->node[14] = (Node) {0x0000,0x4000000d,0x4000000f};
  book->defs[0x000009c8]->node[15] = (Node) {0x0000,0x00000008,0x4000000e};
  // c8
  book->defs[0x000009c9]           = (Term*) malloc(sizeof(Term));
  book->defs[0x000009c9]->alen     = 0;
  book->defs[0x000009c9]->acts     = (Wire*) malloc(0 * sizeof(Wire));
  book->defs[0x000009c9]->nlen     = 18;
  book->defs[0x000009c9]->node     = (Node*) malloc(18 * sizeof(Node));
  book->defs[0x000009c9]->node[ 0] = (Node) {0x0000,0x00000000,0x80000001};
  book->defs[0x000009c9]->node[ 1] = (Node) {0x0000,0x80000002,0x80000011};
  book->defs[0x000009c9]->node[ 2] = (Node) {0x0001,0x80000003,0x80000010};
  book->defs[0x000009c9]->node[ 3] = (Node) {0x0001,0x80000004,0x8000000f};
  book->defs[0x000009c9]->node[ 4] = (Node) {0x0001,0x80000005,0x8000000e};
  book->defs[0x000009c9]->node[ 5] = (Node) {0x0001,0x80000006,0x8000000d};
  book->defs[0x000009c9]->node[ 6] = (Node) {0x0001,0x80000007,0x8000000c};
  book->defs[0x000009c9]->node[ 7] = (Node) {0x0001,0x80000008,0x8000000b};
  book->defs[0x000009c9]->node[ 8] = (Node) {0x0001,0x80000009,0x8000000a};
  book->defs[0x000009c9]->node[ 9] = (Node) {0x0000,0x00000011,0x0000000a};
  book->defs[0x000009c9]->node[10] = (Node) {0x0000,0x40000009,0x0000000b};
  book->defs[0x000009c9]->node[11] = (Node) {0x0000,0x4000000a,0x0000000c};
  book->defs[0x000009c9]->node[12] = (Node) {0x0000,0x4000000b,0x0000000d};
  book->defs[0x000009c9]->node[13] = (Node) {0x0000,0x4000000c,0x0000000e};
  book->defs[0x000009c9]->node[14] = (Node) {0x0000,0x4000000d,0x0000000f};
  book->defs[0x000009c9]->node[15] = (Node) {0x0000,0x4000000e,0x00000010};
  book->defs[0x000009c9]->node[16] = (Node) {0x0000,0x4000000f,0x40000011};
  book->defs[0x000009c9]->node[17] = (Node) {0x0000,0x00000009,0x40000010};
  // c9
  book->defs[0x000009ca]           = (Term*) malloc(sizeof(Term));
  book->defs[0x000009ca]->alen     = 0;
  book->defs[0x000009ca]->acts     = (Wire*) malloc(0 * sizeof(Wire));
  book->defs[0x000009ca]->nlen     = 20;
  book->defs[0x000009ca]->node     = (Node*) malloc(20 * sizeof(Node));
  book->defs[0x000009ca]->node[ 0] = (Node) {0x0000,0x00000000,0x80000001};
  book->defs[0x000009ca]->node[ 1] = (Node) {0x0000,0x80000002,0x80000013};
  book->defs[0x000009ca]->node[ 2] = (Node) {0x0001,0x80000003,0x80000012};
  book->defs[0x000009ca]->node[ 3] = (Node) {0x0001,0x80000004,0x80000011};
  book->defs[0x000009ca]->node[ 4] = (Node) {0x0001,0x80000005,0x80000010};
  book->defs[0x000009ca]->node[ 5] = (Node) {0x0001,0x80000006,0x8000000f};
  book->defs[0x000009ca]->node[ 6] = (Node) {0x0001,0x80000007,0x8000000e};
  book->defs[0x000009ca]->node[ 7] = (Node) {0x0001,0x80000008,0x8000000d};
  book->defs[0x000009ca]->node[ 8] = (Node) {0x0001,0x80000009,0x8000000c};
  book->defs[0x000009ca]->node[ 9] = (Node) {0x0001,0x8000000a,0x8000000b};
  book->defs[0x000009ca]->node[10] = (Node) {0x0000,0x00000013,0x0000000b};
  book->defs[0x000009ca]->node[11] = (Node) {0x0000,0x4000000a,0x0000000c};
  book->defs[0x000009ca]->node[12] = (Node) {0x0000,0x4000000b,0x0000000d};
  book->defs[0x000009ca]->node[13] = (Node) {0x0000,0x4000000c,0x0000000e};
  book->defs[0x000009ca]->node[14] = (Node) {0x0000,0x4000000d,0x0000000f};
  book->defs[0x000009ca]->node[15] = (Node) {0x0000,0x4000000e,0x00000010};
  book->defs[0x000009ca]->node[16] = (Node) {0x0000,0x4000000f,0x00000011};
  book->defs[0x000009ca]->node[17] = (Node) {0x0000,0x40000010,0x00000012};
  book->defs[0x000009ca]->node[18] = (Node) {0x0000,0x40000011,0x40000013};
  book->defs[0x000009ca]->node[19] = (Node) {0x0000,0x0000000a,0x40000012};
  // id
  book->defs[0x00000b68]           = (Term*) malloc(sizeof(Term));
  book->defs[0x00000b68]->alen     = 0;
  book->defs[0x00000b68]->acts     = (Wire*) malloc(0 * sizeof(Wire));
  book->defs[0x00000b68]->nlen     = 2;
  book->defs[0x00000b68]->node     = (Node*) malloc(2 * sizeof(Node));
  book->defs[0x00000b68]->node[ 0] = (Node) {0x0000,0x00000000,0x80000001};
  book->defs[0x00000b68]->node[ 1] = (Node) {0x0000,0x40000001,0x00000001};
  // k0
  book->defs[0x00000bc1]           = (Term*) malloc(sizeof(Term));
  book->defs[0x00000bc1]->alen     = 0;
  book->defs[0x00000bc1]->acts     = (Wire*) malloc(0 * sizeof(Wire));
  book->defs[0x00000bc1]->nlen     = 3;
  book->defs[0x00000bc1]->node     = (Node*) malloc(3 * sizeof(Node));
  book->defs[0x00000bc1]->node[ 0] = (Node) {0x0000,0x00000000,0x80000001};
  book->defs[0x00000bc1]->node[ 1] = (Node) {0x0000,0x80000000,0x80000002};
  book->defs[0x00000bc1]->node[ 2] = (Node) {0x0000,0x40000002,0x00000002};
  // k1
  book->defs[0x00000bc2]           = (Term*) malloc(sizeof(Term));
  book->defs[0x00000bc2]->alen     = 0;
  book->defs[0x00000bc2]->acts     = (Wire*) malloc(0 * sizeof(Wire));
  book->defs[0x00000bc2]->nlen     = 4;
  book->defs[0x00000bc2]->node     = (Node*) malloc(4 * sizeof(Node));
  book->defs[0x00000bc2]->node[ 0] = (Node) {0x0000,0x00000000,0x80000001};
  book->defs[0x00000bc2]->node[ 1] = (Node) {0x0000,0x80000002,0x80000003};
  book->defs[0x00000bc2]->node[ 2] = (Node) {0x0000,0x00000003,0x40000003};
  book->defs[0x00000bc2]->node[ 3] = (Node) {0x0000,0x00000002,0x40000002};
  // k2
  book->defs[0x00000bc3]           = (Term*) malloc(sizeof(Term));
  book->defs[0x00000bc3]->alen     = 0;
  book->defs[0x00000bc3]->acts     = (Wire*) malloc(0 * sizeof(Wire));
  book->defs[0x00000bc3]->nlen     = 6;
  book->defs[0x00000bc3]->node     = (Node*) malloc(6 * sizeof(Node));
  book->defs[0x00000bc3]->node[ 0] = (Node) {0x0000,0x00000000,0x80000001};
  book->defs[0x00000bc3]->node[ 1] = (Node) {0x0000,0x80000002,0x80000005};
  book->defs[0x00000bc3]->node[ 2] = (Node) {0x0002,0x80000003,0x80000004};
  book->defs[0x00000bc3]->node[ 3] = (Node) {0x0000,0x00000005,0x00000004};
  book->defs[0x00000bc3]->node[ 4] = (Node) {0x0000,0x40000003,0x40000005};
  book->defs[0x00000bc3]->node[ 5] = (Node) {0x0000,0x00000003,0x40000004};
  // k3
  book->defs[0x00000bc4]           = (Term*) malloc(sizeof(Term));
  book->defs[0x00000bc4]->alen     = 0;
  book->defs[0x00000bc4]->acts     = (Wire*) malloc(0 * sizeof(Wire));
  book->defs[0x00000bc4]->nlen     = 8;
  book->defs[0x00000bc4]->node     = (Node*) malloc(8 * sizeof(Node));
  book->defs[0x00000bc4]->node[ 0] = (Node) {0x0000,0x00000000,0x80000001};
  book->defs[0x00000bc4]->node[ 1] = (Node) {0x0000,0x80000002,0x80000007};
  book->defs[0x00000bc4]->node[ 2] = (Node) {0x0002,0x80000003,0x80000006};
  book->defs[0x00000bc4]->node[ 3] = (Node) {0x0002,0x80000004,0x80000005};
  book->defs[0x00000bc4]->node[ 4] = (Node) {0x0000,0x00000007,0x00000005};
  book->defs[0x00000bc4]->node[ 5] = (Node) {0x0000,0x40000004,0x00000006};
  book->defs[0x00000bc4]->node[ 6] = (Node) {0x0000,0x40000005,0x40000007};
  book->defs[0x00000bc4]->node[ 7] = (Node) {0x0000,0x00000004,0x40000006};
  // k4
  book->defs[0x00000bc5]           = (Term*) malloc(sizeof(Term));
  book->defs[0x00000bc5]->alen     = 0;
  book->defs[0x00000bc5]->acts     = (Wire*) malloc(0 * sizeof(Wire));
  book->defs[0x00000bc5]->nlen     = 10;
  book->defs[0x00000bc5]->node     = (Node*) malloc(10 * sizeof(Node));
  book->defs[0x00000bc5]->node[ 0] = (Node) {0x0000,0x00000000,0x80000001};
  book->defs[0x00000bc5]->node[ 1] = (Node) {0x0000,0x80000002,0x80000009};
  book->defs[0x00000bc5]->node[ 2] = (Node) {0x0002,0x80000003,0x80000008};
  book->defs[0x00000bc5]->node[ 3] = (Node) {0x0002,0x80000004,0x80000007};
  book->defs[0x00000bc5]->node[ 4] = (Node) {0x0002,0x80000005,0x80000006};
  book->defs[0x00000bc5]->node[ 5] = (Node) {0x0000,0x00000009,0x00000006};
  book->defs[0x00000bc5]->node[ 6] = (Node) {0x0000,0x40000005,0x00000007};
  book->defs[0x00000bc5]->node[ 7] = (Node) {0x0000,0x40000006,0x00000008};
  book->defs[0x00000bc5]->node[ 8] = (Node) {0x0000,0x40000007,0x40000009};
  book->defs[0x00000bc5]->node[ 9] = (Node) {0x0000,0x00000005,0x40000008};
  // k5
  book->defs[0x00000bc6]           = (Term*) malloc(sizeof(Term));
  book->defs[0x00000bc6]->alen     = 0;
  book->defs[0x00000bc6]->acts     = (Wire*) malloc(0 * sizeof(Wire));
  book->defs[0x00000bc6]->nlen     = 12;
  book->defs[0x00000bc6]->node     = (Node*) malloc(12 * sizeof(Node));
  book->defs[0x00000bc6]->node[ 0] = (Node) {0x0000,0x00000000,0x80000001};
  book->defs[0x00000bc6]->node[ 1] = (Node) {0x0000,0x80000002,0x8000000b};
  book->defs[0x00000bc6]->node[ 2] = (Node) {0x0002,0x80000003,0x8000000a};
  book->defs[0x00000bc6]->node[ 3] = (Node) {0x0002,0x80000004,0x80000009};
  book->defs[0x00000bc6]->node[ 4] = (Node) {0x0002,0x80000005,0x80000008};
  book->defs[0x00000bc6]->node[ 5] = (Node) {0x0002,0x80000006,0x80000007};
  book->defs[0x00000bc6]->node[ 6] = (Node) {0x0000,0x0000000b,0x00000007};
  book->defs[0x00000bc6]->node[ 7] = (Node) {0x0000,0x40000006,0x00000008};
  book->defs[0x00000bc6]->node[ 8] = (Node) {0x0000,0x40000007,0x00000009};
  book->defs[0x00000bc6]->node[ 9] = (Node) {0x0000,0x40000008,0x0000000a};
  book->defs[0x00000bc6]->node[10] = (Node) {0x0000,0x40000009,0x4000000b};
  book->defs[0x00000bc6]->node[11] = (Node) {0x0000,0x00000006,0x4000000a};
  // k6
  book->defs[0x00000bc7]           = (Term*) malloc(sizeof(Term));
  book->defs[0x00000bc7]->alen     = 0;
  book->defs[0x00000bc7]->acts     = (Wire*) malloc(0 * sizeof(Wire));
  book->defs[0x00000bc7]->nlen     = 14;
  book->defs[0x00000bc7]->node     = (Node*) malloc(14 * sizeof(Node));
  book->defs[0x00000bc7]->node[ 0] = (Node) {0x0000,0x00000000,0x80000001};
  book->defs[0x00000bc7]->node[ 1] = (Node) {0x0000,0x80000002,0x8000000d};
  book->defs[0x00000bc7]->node[ 2] = (Node) {0x0002,0x80000003,0x8000000c};
  book->defs[0x00000bc7]->node[ 3] = (Node) {0x0002,0x80000004,0x8000000b};
  book->defs[0x00000bc7]->node[ 4] = (Node) {0x0002,0x80000005,0x8000000a};
  book->defs[0x00000bc7]->node[ 5] = (Node) {0x0002,0x80000006,0x80000009};
  book->defs[0x00000bc7]->node[ 6] = (Node) {0x0002,0x80000007,0x80000008};
  book->defs[0x00000bc7]->node[ 7] = (Node) {0x0000,0x0000000d,0x00000008};
  book->defs[0x00000bc7]->node[ 8] = (Node) {0x0000,0x40000007,0x00000009};
  book->defs[0x00000bc7]->node[ 9] = (Node) {0x0000,0x40000008,0x0000000a};
  book->defs[0x00000bc7]->node[10] = (Node) {0x0000,0x40000009,0x0000000b};
  book->defs[0x00000bc7]->node[11] = (Node) {0x0000,0x4000000a,0x0000000c};
  book->defs[0x00000bc7]->node[12] = (Node) {0x0000,0x4000000b,0x4000000d};
  book->defs[0x00000bc7]->node[13] = (Node) {0x0000,0x00000007,0x4000000c};
  // k7
  book->defs[0x00000bc8]           = (Term*) malloc(sizeof(Term));
  book->defs[0x00000bc8]->alen     = 0;
  book->defs[0x00000bc8]->acts     = (Wire*) malloc(0 * sizeof(Wire));
  book->defs[0x00000bc8]->nlen     = 16;
  book->defs[0x00000bc8]->node     = (Node*) malloc(16 * sizeof(Node));
  book->defs[0x00000bc8]->node[ 0] = (Node) {0x0000,0x00000000,0x80000001};
  book->defs[0x00000bc8]->node[ 1] = (Node) {0x0000,0x80000002,0x8000000f};
  book->defs[0x00000bc8]->node[ 2] = (Node) {0x0002,0x80000003,0x8000000e};
  book->defs[0x00000bc8]->node[ 3] = (Node) {0x0002,0x80000004,0x8000000d};
  book->defs[0x00000bc8]->node[ 4] = (Node) {0x0002,0x80000005,0x8000000c};
  book->defs[0x00000bc8]->node[ 5] = (Node) {0x0002,0x80000006,0x8000000b};
  book->defs[0x00000bc8]->node[ 6] = (Node) {0x0002,0x80000007,0x8000000a};
  book->defs[0x00000bc8]->node[ 7] = (Node) {0x0002,0x80000008,0x80000009};
  book->defs[0x00000bc8]->node[ 8] = (Node) {0x0000,0x0000000f,0x00000009};
  book->defs[0x00000bc8]->node[ 9] = (Node) {0x0000,0x40000008,0x0000000a};
  book->defs[0x00000bc8]->node[10] = (Node) {0x0000,0x40000009,0x0000000b};
  book->defs[0x00000bc8]->node[11] = (Node) {0x0000,0x4000000a,0x0000000c};
  book->defs[0x00000bc8]->node[12] = (Node) {0x0000,0x4000000b,0x0000000d};
  book->defs[0x00000bc8]->node[13] = (Node) {0x0000,0x4000000c,0x0000000e};
  book->defs[0x00000bc8]->node[14] = (Node) {0x0000,0x4000000d,0x4000000f};
  book->defs[0x00000bc8]->node[15] = (Node) {0x0000,0x00000008,0x4000000e};
  // k8
  book->defs[0x00000bc9]           = (Term*) malloc(sizeof(Term));
  book->defs[0x00000bc9]->alen     = 0;
  book->defs[0x00000bc9]->acts     = (Wire*) malloc(0 * sizeof(Wire));
  book->defs[0x00000bc9]->nlen     = 18;
  book->defs[0x00000bc9]->node     = (Node*) malloc(18 * sizeof(Node));
  book->defs[0x00000bc9]->node[ 0] = (Node) {0x0000,0x00000000,0x80000001};
  book->defs[0x00000bc9]->node[ 1] = (Node) {0x0000,0x80000002,0x80000011};
  book->defs[0x00000bc9]->node[ 2] = (Node) {0x0002,0x80000003,0x80000010};
  book->defs[0x00000bc9]->node[ 3] = (Node) {0x0002,0x80000004,0x8000000f};
  book->defs[0x00000bc9]->node[ 4] = (Node) {0x0002,0x80000005,0x8000000e};
  book->defs[0x00000bc9]->node[ 5] = (Node) {0x0002,0x80000006,0x8000000d};
  book->defs[0x00000bc9]->node[ 6] = (Node) {0x0002,0x80000007,0x8000000c};
  book->defs[0x00000bc9]->node[ 7] = (Node) {0x0002,0x80000008,0x8000000b};
  book->defs[0x00000bc9]->node[ 8] = (Node) {0x0002,0x80000009,0x8000000a};
  book->defs[0x00000bc9]->node[ 9] = (Node) {0x0000,0x00000011,0x0000000a};
  book->defs[0x00000bc9]->node[10] = (Node) {0x0000,0x40000009,0x0000000b};
  book->defs[0x00000bc9]->node[11] = (Node) {0x0000,0x4000000a,0x0000000c};
  book->defs[0x00000bc9]->node[12] = (Node) {0x0000,0x4000000b,0x0000000d};
  book->defs[0x00000bc9]->node[13] = (Node) {0x0000,0x4000000c,0x0000000e};
  book->defs[0x00000bc9]->node[14] = (Node) {0x0000,0x4000000d,0x0000000f};
  book->defs[0x00000bc9]->node[15] = (Node) {0x0000,0x4000000e,0x00000010};
  book->defs[0x00000bc9]->node[16] = (Node) {0x0000,0x4000000f,0x40000011};
  book->defs[0x00000bc9]->node[17] = (Node) {0x0000,0x00000009,0x40000010};
  // k9
  book->defs[0x00000bca]           = (Term*) malloc(sizeof(Term));
  book->defs[0x00000bca]->alen     = 0;
  book->defs[0x00000bca]->acts     = (Wire*) malloc(0 * sizeof(Wire));
  book->defs[0x00000bca]->nlen     = 20;
  book->defs[0x00000bca]->node     = (Node*) malloc(20 * sizeof(Node));
  book->defs[0x00000bca]->node[ 0] = (Node) {0x0000,0x00000000,0x80000001};
  book->defs[0x00000bca]->node[ 1] = (Node) {0x0000,0x80000002,0x80000013};
  book->defs[0x00000bca]->node[ 2] = (Node) {0x0002,0x80000003,0x80000012};
  book->defs[0x00000bca]->node[ 3] = (Node) {0x0002,0x80000004,0x80000011};
  book->defs[0x00000bca]->node[ 4] = (Node) {0x0002,0x80000005,0x80000010};
  book->defs[0x00000bca]->node[ 5] = (Node) {0x0002,0x80000006,0x8000000f};
  book->defs[0x00000bca]->node[ 6] = (Node) {0x0002,0x80000007,0x8000000e};
  book->defs[0x00000bca]->node[ 7] = (Node) {0x0002,0x80000008,0x8000000d};
  book->defs[0x00000bca]->node[ 8] = (Node) {0x0002,0x80000009,0x8000000c};
  book->defs[0x00000bca]->node[ 9] = (Node) {0x0002,0x8000000a,0x8000000b};
  book->defs[0x00000bca]->node[10] = (Node) {0x0000,0x00000013,0x0000000b};
  book->defs[0x00000bca]->node[11] = (Node) {0x0000,0x4000000a,0x0000000c};
  book->defs[0x00000bca]->node[12] = (Node) {0x0000,0x4000000b,0x0000000d};
  book->defs[0x00000bca]->node[13] = (Node) {0x0000,0x4000000c,0x0000000e};
  book->defs[0x00000bca]->node[14] = (Node) {0x0000,0x4000000d,0x0000000f};
  book->defs[0x00000bca]->node[15] = (Node) {0x0000,0x4000000e,0x00000010};
  book->defs[0x00000bca]->node[16] = (Node) {0x0000,0x4000000f,0x00000011};
  book->defs[0x00000bca]->node[17] = (Node) {0x0000,0x40000010,0x00000012};
  book->defs[0x00000bca]->node[18] = (Node) {0x0000,0x40000011,0x40000013};
  book->defs[0x00000bca]->node[19] = (Node) {0x0000,0x0000000a,0x40000012};
  // brn
  book->defs[0x00026db2]           = (Term*) malloc(sizeof(Term));
  book->defs[0x00026db2]->alen     = 0;
  book->defs[0x00026db2]->acts     = (Wire*) malloc(0 * sizeof(Wire));
  book->defs[0x00026db2]->nlen     = 4;
  book->defs[0x00026db2]->node     = (Node*) malloc(4 * sizeof(Node));
  book->defs[0x00026db2]->node[ 0] = (Node) {0x0000,0x00000000,0x80000001};
  book->defs[0x00026db2]->node[ 1] = (Node) {0x0000,0x80000002,0x40000003};
  book->defs[0x00026db2]->node[ 2] = (Node) {0x0000,0xc09b6c9d,0x80000003};
  book->defs[0x00026db2]->node[ 3] = (Node) {0x0000,0xc09b6ca4,0x40000001};
  // c10
  book->defs[0x00027081]           = (Term*) malloc(sizeof(Term));
  book->defs[0x00027081]->alen     = 0;
  book->defs[0x00027081]->acts     = (Wire*) malloc(0 * sizeof(Wire));
  book->defs[0x00027081]->nlen     = 22;
  book->defs[0x00027081]->node     = (Node*) malloc(22 * sizeof(Node));
  book->defs[0x00027081]->node[ 0] = (Node) {0x0000,0x00000000,0x80000001};
  book->defs[0x00027081]->node[ 1] = (Node) {0x0000,0x80000002,0x80000015};
  book->defs[0x00027081]->node[ 2] = (Node) {0x0001,0x80000003,0x80000014};
  book->defs[0x00027081]->node[ 3] = (Node) {0x0001,0x80000004,0x80000013};
  book->defs[0x00027081]->node[ 4] = (Node) {0x0001,0x80000005,0x80000012};
  book->defs[0x00027081]->node[ 5] = (Node) {0x0001,0x80000006,0x80000011};
  book->defs[0x00027081]->node[ 6] = (Node) {0x0001,0x80000007,0x80000010};
  book->defs[0x00027081]->node[ 7] = (Node) {0x0001,0x80000008,0x8000000f};
  book->defs[0x00027081]->node[ 8] = (Node) {0x0001,0x80000009,0x8000000e};
  book->defs[0x00027081]->node[ 9] = (Node) {0x0001,0x8000000a,0x8000000d};
  book->defs[0x00027081]->node[10] = (Node) {0x0001,0x8000000b,0x8000000c};
  book->defs[0x00027081]->node[11] = (Node) {0x0000,0x00000015,0x0000000c};
  book->defs[0x00027081]->node[12] = (Node) {0x0000,0x4000000b,0x0000000d};
  book->defs[0x00027081]->node[13] = (Node) {0x0000,0x4000000c,0x0000000e};
  book->defs[0x00027081]->node[14] = (Node) {0x0000,0x4000000d,0x0000000f};
  book->defs[0x00027081]->node[15] = (Node) {0x0000,0x4000000e,0x00000010};
  book->defs[0x00027081]->node[16] = (Node) {0x0000,0x4000000f,0x00000011};
  book->defs[0x00027081]->node[17] = (Node) {0x0000,0x40000010,0x00000012};
  book->defs[0x00027081]->node[18] = (Node) {0x0000,0x40000011,0x00000013};
  book->defs[0x00027081]->node[19] = (Node) {0x0000,0x40000012,0x00000014};
  book->defs[0x00027081]->node[20] = (Node) {0x0000,0x40000013,0x40000015};
  book->defs[0x00027081]->node[21] = (Node) {0x0000,0x0000000b,0x40000014};
  // c11
  book->defs[0x00027082]           = (Term*) malloc(sizeof(Term));
  book->defs[0x00027082]->alen     = 0;
  book->defs[0x00027082]->acts     = (Wire*) malloc(0 * sizeof(Wire));
  book->defs[0x00027082]->nlen     = 24;
  book->defs[0x00027082]->node     = (Node*) malloc(24 * sizeof(Node));
  book->defs[0x00027082]->node[ 0] = (Node) {0x0000,0x00000000,0x80000001};
  book->defs[0x00027082]->node[ 1] = (Node) {0x0000,0x80000002,0x80000017};
  book->defs[0x00027082]->node[ 2] = (Node) {0x0001,0x80000003,0x80000016};
  book->defs[0x00027082]->node[ 3] = (Node) {0x0001,0x80000004,0x80000015};
  book->defs[0x00027082]->node[ 4] = (Node) {0x0001,0x80000005,0x80000014};
  book->defs[0x00027082]->node[ 5] = (Node) {0x0001,0x80000006,0x80000013};
  book->defs[0x00027082]->node[ 6] = (Node) {0x0001,0x80000007,0x80000012};
  book->defs[0x00027082]->node[ 7] = (Node) {0x0001,0x80000008,0x80000011};
  book->defs[0x00027082]->node[ 8] = (Node) {0x0001,0x80000009,0x80000010};
  book->defs[0x00027082]->node[ 9] = (Node) {0x0001,0x8000000a,0x8000000f};
  book->defs[0x00027082]->node[10] = (Node) {0x0001,0x8000000b,0x8000000e};
  book->defs[0x00027082]->node[11] = (Node) {0x0001,0x8000000c,0x8000000d};
  book->defs[0x00027082]->node[12] = (Node) {0x0000,0x00000017,0x0000000d};
  book->defs[0x00027082]->node[13] = (Node) {0x0000,0x4000000c,0x0000000e};
  book->defs[0x00027082]->node[14] = (Node) {0x0000,0x4000000d,0x0000000f};
  book->defs[0x00027082]->node[15] = (Node) {0x0000,0x4000000e,0x00000010};
  book->defs[0x00027082]->node[16] = (Node) {0x0000,0x4000000f,0x00000011};
  book->defs[0x00027082]->node[17] = (Node) {0x0000,0x40000010,0x00000012};
  book->defs[0x00027082]->node[18] = (Node) {0x0000,0x40000011,0x00000013};
  book->defs[0x00027082]->node[19] = (Node) {0x0000,0x40000012,0x00000014};
  book->defs[0x00027082]->node[20] = (Node) {0x0000,0x40000013,0x00000015};
  book->defs[0x00027082]->node[21] = (Node) {0x0000,0x40000014,0x00000016};
  book->defs[0x00027082]->node[22] = (Node) {0x0000,0x40000015,0x40000017};
  book->defs[0x00027082]->node[23] = (Node) {0x0000,0x0000000c,0x40000016};
  // c12
  book->defs[0x00027083]           = (Term*) malloc(sizeof(Term));
  book->defs[0x00027083]->alen     = 0;
  book->defs[0x00027083]->acts     = (Wire*) malloc(0 * sizeof(Wire));
  book->defs[0x00027083]->nlen     = 26;
  book->defs[0x00027083]->node     = (Node*) malloc(26 * sizeof(Node));
  book->defs[0x00027083]->node[ 0] = (Node) {0x0000,0x00000000,0x80000001};
  book->defs[0x00027083]->node[ 1] = (Node) {0x0000,0x80000002,0x80000019};
  book->defs[0x00027083]->node[ 2] = (Node) {0x0001,0x80000003,0x80000018};
  book->defs[0x00027083]->node[ 3] = (Node) {0x0001,0x80000004,0x80000017};
  book->defs[0x00027083]->node[ 4] = (Node) {0x0001,0x80000005,0x80000016};
  book->defs[0x00027083]->node[ 5] = (Node) {0x0001,0x80000006,0x80000015};
  book->defs[0x00027083]->node[ 6] = (Node) {0x0001,0x80000007,0x80000014};
  book->defs[0x00027083]->node[ 7] = (Node) {0x0001,0x80000008,0x80000013};
  book->defs[0x00027083]->node[ 8] = (Node) {0x0001,0x80000009,0x80000012};
  book->defs[0x00027083]->node[ 9] = (Node) {0x0001,0x8000000a,0x80000011};
  book->defs[0x00027083]->node[10] = (Node) {0x0001,0x8000000b,0x80000010};
  book->defs[0x00027083]->node[11] = (Node) {0x0001,0x8000000c,0x8000000f};
  book->defs[0x00027083]->node[12] = (Node) {0x0001,0x8000000d,0x8000000e};
  book->defs[0x00027083]->node[13] = (Node) {0x0000,0x00000019,0x0000000e};
  book->defs[0x00027083]->node[14] = (Node) {0x0000,0x4000000d,0x0000000f};
  book->defs[0x00027083]->node[15] = (Node) {0x0000,0x4000000e,0x00000010};
  book->defs[0x00027083]->node[16] = (Node) {0x0000,0x4000000f,0x00000011};
  book->defs[0x00027083]->node[17] = (Node) {0x0000,0x40000010,0x00000012};
  book->defs[0x00027083]->node[18] = (Node) {0x0000,0x40000011,0x00000013};
  book->defs[0x00027083]->node[19] = (Node) {0x0000,0x40000012,0x00000014};
  book->defs[0x00027083]->node[20] = (Node) {0x0000,0x40000013,0x00000015};
  book->defs[0x00027083]->node[21] = (Node) {0x0000,0x40000014,0x00000016};
  book->defs[0x00027083]->node[22] = (Node) {0x0000,0x40000015,0x00000017};
  book->defs[0x00027083]->node[23] = (Node) {0x0000,0x40000016,0x00000018};
  book->defs[0x00027083]->node[24] = (Node) {0x0000,0x40000017,0x40000019};
  book->defs[0x00027083]->node[25] = (Node) {0x0000,0x0000000d,0x40000018};
  // c13
  book->defs[0x00027084]           = (Term*) malloc(sizeof(Term));
  book->defs[0x00027084]->alen     = 0;
  book->defs[0x00027084]->acts     = (Wire*) malloc(0 * sizeof(Wire));
  book->defs[0x00027084]->nlen     = 28;
  book->defs[0x00027084]->node     = (Node*) malloc(28 * sizeof(Node));
  book->defs[0x00027084]->node[ 0] = (Node) {0x0000,0x00000000,0x80000001};
  book->defs[0x00027084]->node[ 1] = (Node) {0x0000,0x80000002,0x8000001b};
  book->defs[0x00027084]->node[ 2] = (Node) {0x0001,0x80000003,0x8000001a};
  book->defs[0x00027084]->node[ 3] = (Node) {0x0001,0x80000004,0x80000019};
  book->defs[0x00027084]->node[ 4] = (Node) {0x0001,0x80000005,0x80000018};
  book->defs[0x00027084]->node[ 5] = (Node) {0x0001,0x80000006,0x80000017};
  book->defs[0x00027084]->node[ 6] = (Node) {0x0001,0x80000007,0x80000016};
  book->defs[0x00027084]->node[ 7] = (Node) {0x0001,0x80000008,0x80000015};
  book->defs[0x00027084]->node[ 8] = (Node) {0x0001,0x80000009,0x80000014};
  book->defs[0x00027084]->node[ 9] = (Node) {0x0001,0x8000000a,0x80000013};
  book->defs[0x00027084]->node[10] = (Node) {0x0001,0x8000000b,0x80000012};
  book->defs[0x00027084]->node[11] = (Node) {0x0001,0x8000000c,0x80000011};
  book->defs[0x00027084]->node[12] = (Node) {0x0001,0x8000000d,0x80000010};
  book->defs[0x00027084]->node[13] = (Node) {0x0001,0x8000000e,0x8000000f};
  book->defs[0x00027084]->node[14] = (Node) {0x0000,0x0000001b,0x0000000f};
  book->defs[0x00027084]->node[15] = (Node) {0x0000,0x4000000e,0x00000010};
  book->defs[0x00027084]->node[16] = (Node) {0x0000,0x4000000f,0x00000011};
  book->defs[0x00027084]->node[17] = (Node) {0x0000,0x40000010,0x00000012};
  book->defs[0x00027084]->node[18] = (Node) {0x0000,0x40000011,0x00000013};
  book->defs[0x00027084]->node[19] = (Node) {0x0000,0x40000012,0x00000014};
  book->defs[0x00027084]->node[20] = (Node) {0x0000,0x40000013,0x00000015};
  book->defs[0x00027084]->node[21] = (Node) {0x0000,0x40000014,0x00000016};
  book->defs[0x00027084]->node[22] = (Node) {0x0000,0x40000015,0x00000017};
  book->defs[0x00027084]->node[23] = (Node) {0x0000,0x40000016,0x00000018};
  book->defs[0x00027084]->node[24] = (Node) {0x0000,0x40000017,0x00000019};
  book->defs[0x00027084]->node[25] = (Node) {0x0000,0x40000018,0x0000001a};
  book->defs[0x00027084]->node[26] = (Node) {0x0000,0x40000019,0x4000001b};
  book->defs[0x00027084]->node[27] = (Node) {0x0000,0x0000000e,0x4000001a};
  // c14
  book->defs[0x00027085]           = (Term*) malloc(sizeof(Term));
  book->defs[0x00027085]->alen     = 0;
  book->defs[0x00027085]->acts     = (Wire*) malloc(0 * sizeof(Wire));
  book->defs[0x00027085]->nlen     = 30;
  book->defs[0x00027085]->node     = (Node*) malloc(30 * sizeof(Node));
  book->defs[0x00027085]->node[ 0] = (Node) {0x0000,0x00000000,0x80000001};
  book->defs[0x00027085]->node[ 1] = (Node) {0x0000,0x80000002,0x8000001d};
  book->defs[0x00027085]->node[ 2] = (Node) {0x0001,0x80000003,0x8000001c};
  book->defs[0x00027085]->node[ 3] = (Node) {0x0001,0x80000004,0x8000001b};
  book->defs[0x00027085]->node[ 4] = (Node) {0x0001,0x80000005,0x8000001a};
  book->defs[0x00027085]->node[ 5] = (Node) {0x0001,0x80000006,0x80000019};
  book->defs[0x00027085]->node[ 6] = (Node) {0x0001,0x80000007,0x80000018};
  book->defs[0x00027085]->node[ 7] = (Node) {0x0001,0x80000008,0x80000017};
  book->defs[0x00027085]->node[ 8] = (Node) {0x0001,0x80000009,0x80000016};
  book->defs[0x00027085]->node[ 9] = (Node) {0x0001,0x8000000a,0x80000015};
  book->defs[0x00027085]->node[10] = (Node) {0x0001,0x8000000b,0x80000014};
  book->defs[0x00027085]->node[11] = (Node) {0x0001,0x8000000c,0x80000013};
  book->defs[0x00027085]->node[12] = (Node) {0x0001,0x8000000d,0x80000012};
  book->defs[0x00027085]->node[13] = (Node) {0x0001,0x8000000e,0x80000011};
  book->defs[0x00027085]->node[14] = (Node) {0x0001,0x8000000f,0x80000010};
  book->defs[0x00027085]->node[15] = (Node) {0x0000,0x0000001d,0x00000010};
  book->defs[0x00027085]->node[16] = (Node) {0x0000,0x4000000f,0x00000011};
  book->defs[0x00027085]->node[17] = (Node) {0x0000,0x40000010,0x00000012};
  book->defs[0x00027085]->node[18] = (Node) {0x0000,0x40000011,0x00000013};
  book->defs[0x00027085]->node[19] = (Node) {0x0000,0x40000012,0x00000014};
  book->defs[0x00027085]->node[20] = (Node) {0x0000,0x40000013,0x00000015};
  book->defs[0x00027085]->node[21] = (Node) {0x0000,0x40000014,0x00000016};
  book->defs[0x00027085]->node[22] = (Node) {0x0000,0x40000015,0x00000017};
  book->defs[0x00027085]->node[23] = (Node) {0x0000,0x40000016,0x00000018};
  book->defs[0x00027085]->node[24] = (Node) {0x0000,0x40000017,0x00000019};
  book->defs[0x00027085]->node[25] = (Node) {0x0000,0x40000018,0x0000001a};
  book->defs[0x00027085]->node[26] = (Node) {0x0000,0x40000019,0x0000001b};
  book->defs[0x00027085]->node[27] = (Node) {0x0000,0x4000001a,0x0000001c};
  book->defs[0x00027085]->node[28] = (Node) {0x0000,0x4000001b,0x4000001d};
  book->defs[0x00027085]->node[29] = (Node) {0x0000,0x0000000f,0x4000001c};
  // c15
  book->defs[0x00027086]           = (Term*) malloc(sizeof(Term));
  book->defs[0x00027086]->alen     = 0;
  book->defs[0x00027086]->acts     = (Wire*) malloc(0 * sizeof(Wire));
  book->defs[0x00027086]->nlen     = 32;
  book->defs[0x00027086]->node     = (Node*) malloc(32 * sizeof(Node));
  book->defs[0x00027086]->node[ 0] = (Node) {0x0000,0x00000000,0x80000001};
  book->defs[0x00027086]->node[ 1] = (Node) {0x0000,0x80000002,0x8000001f};
  book->defs[0x00027086]->node[ 2] = (Node) {0x0001,0x80000003,0x8000001e};
  book->defs[0x00027086]->node[ 3] = (Node) {0x0001,0x80000004,0x8000001d};
  book->defs[0x00027086]->node[ 4] = (Node) {0x0001,0x80000005,0x8000001c};
  book->defs[0x00027086]->node[ 5] = (Node) {0x0001,0x80000006,0x8000001b};
  book->defs[0x00027086]->node[ 6] = (Node) {0x0001,0x80000007,0x8000001a};
  book->defs[0x00027086]->node[ 7] = (Node) {0x0001,0x80000008,0x80000019};
  book->defs[0x00027086]->node[ 8] = (Node) {0x0001,0x80000009,0x80000018};
  book->defs[0x00027086]->node[ 9] = (Node) {0x0001,0x8000000a,0x80000017};
  book->defs[0x00027086]->node[10] = (Node) {0x0001,0x8000000b,0x80000016};
  book->defs[0x00027086]->node[11] = (Node) {0x0001,0x8000000c,0x80000015};
  book->defs[0x00027086]->node[12] = (Node) {0x0001,0x8000000d,0x80000014};
  book->defs[0x00027086]->node[13] = (Node) {0x0001,0x8000000e,0x80000013};
  book->defs[0x00027086]->node[14] = (Node) {0x0001,0x8000000f,0x80000012};
  book->defs[0x00027086]->node[15] = (Node) {0x0001,0x80000010,0x80000011};
  book->defs[0x00027086]->node[16] = (Node) {0x0000,0x0000001f,0x00000011};
  book->defs[0x00027086]->node[17] = (Node) {0x0000,0x40000010,0x00000012};
  book->defs[0x00027086]->node[18] = (Node) {0x0000,0x40000011,0x00000013};
  book->defs[0x00027086]->node[19] = (Node) {0x0000,0x40000012,0x00000014};
  book->defs[0x00027086]->node[20] = (Node) {0x0000,0x40000013,0x00000015};
  book->defs[0x00027086]->node[21] = (Node) {0x0000,0x40000014,0x00000016};
  book->defs[0x00027086]->node[22] = (Node) {0x0000,0x40000015,0x00000017};
  book->defs[0x00027086]->node[23] = (Node) {0x0000,0x40000016,0x00000018};
  book->defs[0x00027086]->node[24] = (Node) {0x0000,0x40000017,0x00000019};
  book->defs[0x00027086]->node[25] = (Node) {0x0000,0x40000018,0x0000001a};
  book->defs[0x00027086]->node[26] = (Node) {0x0000,0x40000019,0x0000001b};
  book->defs[0x00027086]->node[27] = (Node) {0x0000,0x4000001a,0x0000001c};
  book->defs[0x00027086]->node[28] = (Node) {0x0000,0x4000001b,0x0000001d};
  book->defs[0x00027086]->node[29] = (Node) {0x0000,0x4000001c,0x0000001e};
  book->defs[0x00027086]->node[30] = (Node) {0x0000,0x4000001d,0x4000001f};
  book->defs[0x00027086]->node[31] = (Node) {0x0000,0x00000010,0x4000001e};
  // c16
  book->defs[0x00027087]           = (Term*) malloc(sizeof(Term));
  book->defs[0x00027087]->alen     = 0;
  book->defs[0x00027087]->acts     = (Wire*) malloc(0 * sizeof(Wire));
  book->defs[0x00027087]->nlen     = 34;
  book->defs[0x00027087]->node     = (Node*) malloc(34 * sizeof(Node));
  book->defs[0x00027087]->node[ 0] = (Node) {0x0000,0x00000000,0x80000001};
  book->defs[0x00027087]->node[ 1] = (Node) {0x0000,0x80000002,0x80000021};
  book->defs[0x00027087]->node[ 2] = (Node) {0x0001,0x80000003,0x80000020};
  book->defs[0x00027087]->node[ 3] = (Node) {0x0001,0x80000004,0x8000001f};
  book->defs[0x00027087]->node[ 4] = (Node) {0x0001,0x80000005,0x8000001e};
  book->defs[0x00027087]->node[ 5] = (Node) {0x0001,0x80000006,0x8000001d};
  book->defs[0x00027087]->node[ 6] = (Node) {0x0001,0x80000007,0x8000001c};
  book->defs[0x00027087]->node[ 7] = (Node) {0x0001,0x80000008,0x8000001b};
  book->defs[0x00027087]->node[ 8] = (Node) {0x0001,0x80000009,0x8000001a};
  book->defs[0x00027087]->node[ 9] = (Node) {0x0001,0x8000000a,0x80000019};
  book->defs[0x00027087]->node[10] = (Node) {0x0001,0x8000000b,0x80000018};
  book->defs[0x00027087]->node[11] = (Node) {0x0001,0x8000000c,0x80000017};
  book->defs[0x00027087]->node[12] = (Node) {0x0001,0x8000000d,0x80000016};
  book->defs[0x00027087]->node[13] = (Node) {0x0001,0x8000000e,0x80000015};
  book->defs[0x00027087]->node[14] = (Node) {0x0001,0x8000000f,0x80000014};
  book->defs[0x00027087]->node[15] = (Node) {0x0001,0x80000010,0x80000013};
  book->defs[0x00027087]->node[16] = (Node) {0x0001,0x80000011,0x80000012};
  book->defs[0x00027087]->node[17] = (Node) {0x0000,0x00000021,0x00000012};
  book->defs[0x00027087]->node[18] = (Node) {0x0000,0x40000011,0x00000013};
  book->defs[0x00027087]->node[19] = (Node) {0x0000,0x40000012,0x00000014};
  book->defs[0x00027087]->node[20] = (Node) {0x0000,0x40000013,0x00000015};
  book->defs[0x00027087]->node[21] = (Node) {0x0000,0x40000014,0x00000016};
  book->defs[0x00027087]->node[22] = (Node) {0x0000,0x40000015,0x00000017};
  book->defs[0x00027087]->node[23] = (Node) {0x0000,0x40000016,0x00000018};
  book->defs[0x00027087]->node[24] = (Node) {0x0000,0x40000017,0x00000019};
  book->defs[0x00027087]->node[25] = (Node) {0x0000,0x40000018,0x0000001a};
  book->defs[0x00027087]->node[26] = (Node) {0x0000,0x40000019,0x0000001b};
  book->defs[0x00027087]->node[27] = (Node) {0x0000,0x4000001a,0x0000001c};
  book->defs[0x00027087]->node[28] = (Node) {0x0000,0x4000001b,0x0000001d};
  book->defs[0x00027087]->node[29] = (Node) {0x0000,0x4000001c,0x0000001e};
  book->defs[0x00027087]->node[30] = (Node) {0x0000,0x4000001d,0x0000001f};
  book->defs[0x00027087]->node[31] = (Node) {0x0000,0x4000001e,0x00000020};
  book->defs[0x00027087]->node[32] = (Node) {0x0000,0x4000001f,0x40000021};
  book->defs[0x00027087]->node[33] = (Node) {0x0000,0x00000011,0x40000020};
  // c17
  book->defs[0x00027088]           = (Term*) malloc(sizeof(Term));
  book->defs[0x00027088]->alen     = 0;
  book->defs[0x00027088]->acts     = (Wire*) malloc(0 * sizeof(Wire));
  book->defs[0x00027088]->nlen     = 36;
  book->defs[0x00027088]->node     = (Node*) malloc(36 * sizeof(Node));
  book->defs[0x00027088]->node[ 0] = (Node) {0x0000,0x00000000,0x80000001};
  book->defs[0x00027088]->node[ 1] = (Node) {0x0000,0x80000002,0x80000023};
  book->defs[0x00027088]->node[ 2] = (Node) {0x0001,0x80000003,0x80000022};
  book->defs[0x00027088]->node[ 3] = (Node) {0x0001,0x80000004,0x80000021};
  book->defs[0x00027088]->node[ 4] = (Node) {0x0001,0x80000005,0x80000020};
  book->defs[0x00027088]->node[ 5] = (Node) {0x0001,0x80000006,0x8000001f};
  book->defs[0x00027088]->node[ 6] = (Node) {0x0001,0x80000007,0x8000001e};
  book->defs[0x00027088]->node[ 7] = (Node) {0x0001,0x80000008,0x8000001d};
  book->defs[0x00027088]->node[ 8] = (Node) {0x0001,0x80000009,0x8000001c};
  book->defs[0x00027088]->node[ 9] = (Node) {0x0001,0x8000000a,0x8000001b};
  book->defs[0x00027088]->node[10] = (Node) {0x0001,0x8000000b,0x8000001a};
  book->defs[0x00027088]->node[11] = (Node) {0x0001,0x8000000c,0x80000019};
  book->defs[0x00027088]->node[12] = (Node) {0x0001,0x8000000d,0x80000018};
  book->defs[0x00027088]->node[13] = (Node) {0x0001,0x8000000e,0x80000017};
  book->defs[0x00027088]->node[14] = (Node) {0x0001,0x8000000f,0x80000016};
  book->defs[0x00027088]->node[15] = (Node) {0x0001,0x80000010,0x80000015};
  book->defs[0x00027088]->node[16] = (Node) {0x0001,0x80000011,0x80000014};
  book->defs[0x00027088]->node[17] = (Node) {0x0001,0x80000012,0x80000013};
  book->defs[0x00027088]->node[18] = (Node) {0x0000,0x00000023,0x00000013};
  book->defs[0x00027088]->node[19] = (Node) {0x0000,0x40000012,0x00000014};
  book->defs[0x00027088]->node[20] = (Node) {0x0000,0x40000013,0x00000015};
  book->defs[0x00027088]->node[21] = (Node) {0x0000,0x40000014,0x00000016};
  book->defs[0x00027088]->node[22] = (Node) {0x0000,0x40000015,0x00000017};
  book->defs[0x00027088]->node[23] = (Node) {0x0000,0x40000016,0x00000018};
  book->defs[0x00027088]->node[24] = (Node) {0x0000,0x40000017,0x00000019};
  book->defs[0x00027088]->node[25] = (Node) {0x0000,0x40000018,0x0000001a};
  book->defs[0x00027088]->node[26] = (Node) {0x0000,0x40000019,0x0000001b};
  book->defs[0x00027088]->node[27] = (Node) {0x0000,0x4000001a,0x0000001c};
  book->defs[0x00027088]->node[28] = (Node) {0x0000,0x4000001b,0x0000001d};
  book->defs[0x00027088]->node[29] = (Node) {0x0000,0x4000001c,0x0000001e};
  book->defs[0x00027088]->node[30] = (Node) {0x0000,0x4000001d,0x0000001f};
  book->defs[0x00027088]->node[31] = (Node) {0x0000,0x4000001e,0x00000020};
  book->defs[0x00027088]->node[32] = (Node) {0x0000,0x4000001f,0x00000021};
  book->defs[0x00027088]->node[33] = (Node) {0x0000,0x40000020,0x00000022};
  book->defs[0x00027088]->node[34] = (Node) {0x0000,0x40000021,0x40000023};
  book->defs[0x00027088]->node[35] = (Node) {0x0000,0x00000012,0x40000022};
  // c18
  book->defs[0x00027089]           = (Term*) malloc(sizeof(Term));
  book->defs[0x00027089]->alen     = 0;
  book->defs[0x00027089]->acts     = (Wire*) malloc(0 * sizeof(Wire));
  book->defs[0x00027089]->nlen     = 38;
  book->defs[0x00027089]->node     = (Node*) malloc(38 * sizeof(Node));
  book->defs[0x00027089]->node[ 0] = (Node) {0x0000,0x00000000,0x80000001};
  book->defs[0x00027089]->node[ 1] = (Node) {0x0000,0x80000002,0x80000025};
  book->defs[0x00027089]->node[ 2] = (Node) {0x0001,0x80000003,0x80000024};
  book->defs[0x00027089]->node[ 3] = (Node) {0x0001,0x80000004,0x80000023};
  book->defs[0x00027089]->node[ 4] = (Node) {0x0001,0x80000005,0x80000022};
  book->defs[0x00027089]->node[ 5] = (Node) {0x0001,0x80000006,0x80000021};
  book->defs[0x00027089]->node[ 6] = (Node) {0x0001,0x80000007,0x80000020};
  book->defs[0x00027089]->node[ 7] = (Node) {0x0001,0x80000008,0x8000001f};
  book->defs[0x00027089]->node[ 8] = (Node) {0x0001,0x80000009,0x8000001e};
  book->defs[0x00027089]->node[ 9] = (Node) {0x0001,0x8000000a,0x8000001d};
  book->defs[0x00027089]->node[10] = (Node) {0x0001,0x8000000b,0x8000001c};
  book->defs[0x00027089]->node[11] = (Node) {0x0001,0x8000000c,0x8000001b};
  book->defs[0x00027089]->node[12] = (Node) {0x0001,0x8000000d,0x8000001a};
  book->defs[0x00027089]->node[13] = (Node) {0x0001,0x8000000e,0x80000019};
  book->defs[0x00027089]->node[14] = (Node) {0x0001,0x8000000f,0x80000018};
  book->defs[0x00027089]->node[15] = (Node) {0x0001,0x80000010,0x80000017};
  book->defs[0x00027089]->node[16] = (Node) {0x0001,0x80000011,0x80000016};
  book->defs[0x00027089]->node[17] = (Node) {0x0001,0x80000012,0x80000015};
  book->defs[0x00027089]->node[18] = (Node) {0x0001,0x80000013,0x80000014};
  book->defs[0x00027089]->node[19] = (Node) {0x0000,0x00000025,0x00000014};
  book->defs[0x00027089]->node[20] = (Node) {0x0000,0x40000013,0x00000015};
  book->defs[0x00027089]->node[21] = (Node) {0x0000,0x40000014,0x00000016};
  book->defs[0x00027089]->node[22] = (Node) {0x0000,0x40000015,0x00000017};
  book->defs[0x00027089]->node[23] = (Node) {0x0000,0x40000016,0x00000018};
  book->defs[0x00027089]->node[24] = (Node) {0x0000,0x40000017,0x00000019};
  book->defs[0x00027089]->node[25] = (Node) {0x0000,0x40000018,0x0000001a};
  book->defs[0x00027089]->node[26] = (Node) {0x0000,0x40000019,0x0000001b};
  book->defs[0x00027089]->node[27] = (Node) {0x0000,0x4000001a,0x0000001c};
  book->defs[0x00027089]->node[28] = (Node) {0x0000,0x4000001b,0x0000001d};
  book->defs[0x00027089]->node[29] = (Node) {0x0000,0x4000001c,0x0000001e};
  book->defs[0x00027089]->node[30] = (Node) {0x0000,0x4000001d,0x0000001f};
  book->defs[0x00027089]->node[31] = (Node) {0x0000,0x4000001e,0x00000020};
  book->defs[0x00027089]->node[32] = (Node) {0x0000,0x4000001f,0x00000021};
  book->defs[0x00027089]->node[33] = (Node) {0x0000,0x40000020,0x00000022};
  book->defs[0x00027089]->node[34] = (Node) {0x0000,0x40000021,0x00000023};
  book->defs[0x00027089]->node[35] = (Node) {0x0000,0x40000022,0x00000024};
  book->defs[0x00027089]->node[36] = (Node) {0x0000,0x40000023,0x40000025};
  book->defs[0x00027089]->node[37] = (Node) {0x0000,0x00000013,0x40000024};
  // c19
  book->defs[0x0002708a]           = (Term*) malloc(sizeof(Term));
  book->defs[0x0002708a]->alen     = 0;
  book->defs[0x0002708a]->acts     = (Wire*) malloc(0 * sizeof(Wire));
  book->defs[0x0002708a]->nlen     = 40;
  book->defs[0x0002708a]->node     = (Node*) malloc(40 * sizeof(Node));
  book->defs[0x0002708a]->node[ 0] = (Node) {0x0000,0x00000000,0x80000001};
  book->defs[0x0002708a]->node[ 1] = (Node) {0x0000,0x80000002,0x80000027};
  book->defs[0x0002708a]->node[ 2] = (Node) {0x0001,0x80000003,0x80000026};
  book->defs[0x0002708a]->node[ 3] = (Node) {0x0001,0x80000004,0x80000025};
  book->defs[0x0002708a]->node[ 4] = (Node) {0x0001,0x80000005,0x80000024};
  book->defs[0x0002708a]->node[ 5] = (Node) {0x0001,0x80000006,0x80000023};
  book->defs[0x0002708a]->node[ 6] = (Node) {0x0001,0x80000007,0x80000022};
  book->defs[0x0002708a]->node[ 7] = (Node) {0x0001,0x80000008,0x80000021};
  book->defs[0x0002708a]->node[ 8] = (Node) {0x0001,0x80000009,0x80000020};
  book->defs[0x0002708a]->node[ 9] = (Node) {0x0001,0x8000000a,0x8000001f};
  book->defs[0x0002708a]->node[10] = (Node) {0x0001,0x8000000b,0x8000001e};
  book->defs[0x0002708a]->node[11] = (Node) {0x0001,0x8000000c,0x8000001d};
  book->defs[0x0002708a]->node[12] = (Node) {0x0001,0x8000000d,0x8000001c};
  book->defs[0x0002708a]->node[13] = (Node) {0x0001,0x8000000e,0x8000001b};
  book->defs[0x0002708a]->node[14] = (Node) {0x0001,0x8000000f,0x8000001a};
  book->defs[0x0002708a]->node[15] = (Node) {0x0001,0x80000010,0x80000019};
  book->defs[0x0002708a]->node[16] = (Node) {0x0001,0x80000011,0x80000018};
  book->defs[0x0002708a]->node[17] = (Node) {0x0001,0x80000012,0x80000017};
  book->defs[0x0002708a]->node[18] = (Node) {0x0001,0x80000013,0x80000016};
  book->defs[0x0002708a]->node[19] = (Node) {0x0001,0x80000014,0x80000015};
  book->defs[0x0002708a]->node[20] = (Node) {0x0000,0x00000027,0x00000015};
  book->defs[0x0002708a]->node[21] = (Node) {0x0000,0x40000014,0x00000016};
  book->defs[0x0002708a]->node[22] = (Node) {0x0000,0x40000015,0x00000017};
  book->defs[0x0002708a]->node[23] = (Node) {0x0000,0x40000016,0x00000018};
  book->defs[0x0002708a]->node[24] = (Node) {0x0000,0x40000017,0x00000019};
  book->defs[0x0002708a]->node[25] = (Node) {0x0000,0x40000018,0x0000001a};
  book->defs[0x0002708a]->node[26] = (Node) {0x0000,0x40000019,0x0000001b};
  book->defs[0x0002708a]->node[27] = (Node) {0x0000,0x4000001a,0x0000001c};
  book->defs[0x0002708a]->node[28] = (Node) {0x0000,0x4000001b,0x0000001d};
  book->defs[0x0002708a]->node[29] = (Node) {0x0000,0x4000001c,0x0000001e};
  book->defs[0x0002708a]->node[30] = (Node) {0x0000,0x4000001d,0x0000001f};
  book->defs[0x0002708a]->node[31] = (Node) {0x0000,0x4000001e,0x00000020};
  book->defs[0x0002708a]->node[32] = (Node) {0x0000,0x4000001f,0x00000021};
  book->defs[0x0002708a]->node[33] = (Node) {0x0000,0x40000020,0x00000022};
  book->defs[0x0002708a]->node[34] = (Node) {0x0000,0x40000021,0x00000023};
  book->defs[0x0002708a]->node[35] = (Node) {0x0000,0x40000022,0x00000024};
  book->defs[0x0002708a]->node[36] = (Node) {0x0000,0x40000023,0x00000025};
  book->defs[0x0002708a]->node[37] = (Node) {0x0000,0x40000024,0x00000026};
  book->defs[0x0002708a]->node[38] = (Node) {0x0000,0x40000025,0x40000027};
  book->defs[0x0002708a]->node[39] = (Node) {0x0000,0x00000014,0x40000026};
  // c20
  book->defs[0x000270c1]           = (Term*) malloc(sizeof(Term));
  book->defs[0x000270c1]->alen     = 0;
  book->defs[0x000270c1]->acts     = (Wire*) malloc(0 * sizeof(Wire));
  book->defs[0x000270c1]->nlen     = 42;
  book->defs[0x000270c1]->node     = (Node*) malloc(42 * sizeof(Node));
  book->defs[0x000270c1]->node[ 0] = (Node) {0x0000,0x00000000,0x80000001};
  book->defs[0x000270c1]->node[ 1] = (Node) {0x0000,0x80000002,0x80000029};
  book->defs[0x000270c1]->node[ 2] = (Node) {0x0001,0x80000003,0x80000028};
  book->defs[0x000270c1]->node[ 3] = (Node) {0x0001,0x80000004,0x80000027};
  book->defs[0x000270c1]->node[ 4] = (Node) {0x0001,0x80000005,0x80000026};
  book->defs[0x000270c1]->node[ 5] = (Node) {0x0001,0x80000006,0x80000025};
  book->defs[0x000270c1]->node[ 6] = (Node) {0x0001,0x80000007,0x80000024};
  book->defs[0x000270c1]->node[ 7] = (Node) {0x0001,0x80000008,0x80000023};
  book->defs[0x000270c1]->node[ 8] = (Node) {0x0001,0x80000009,0x80000022};
  book->defs[0x000270c1]->node[ 9] = (Node) {0x0001,0x8000000a,0x80000021};
  book->defs[0x000270c1]->node[10] = (Node) {0x0001,0x8000000b,0x80000020};
  book->defs[0x000270c1]->node[11] = (Node) {0x0001,0x8000000c,0x8000001f};
  book->defs[0x000270c1]->node[12] = (Node) {0x0001,0x8000000d,0x8000001e};
  book->defs[0x000270c1]->node[13] = (Node) {0x0001,0x8000000e,0x8000001d};
  book->defs[0x000270c1]->node[14] = (Node) {0x0001,0x8000000f,0x8000001c};
  book->defs[0x000270c1]->node[15] = (Node) {0x0001,0x80000010,0x8000001b};
  book->defs[0x000270c1]->node[16] = (Node) {0x0001,0x80000011,0x8000001a};
  book->defs[0x000270c1]->node[17] = (Node) {0x0001,0x80000012,0x80000019};
  book->defs[0x000270c1]->node[18] = (Node) {0x0001,0x80000013,0x80000018};
  book->defs[0x000270c1]->node[19] = (Node) {0x0001,0x80000014,0x80000017};
  book->defs[0x000270c1]->node[20] = (Node) {0x0001,0x80000015,0x80000016};
  book->defs[0x000270c1]->node[21] = (Node) {0x0000,0x00000029,0x00000016};
  book->defs[0x000270c1]->node[22] = (Node) {0x0000,0x40000015,0x00000017};
  book->defs[0x000270c1]->node[23] = (Node) {0x0000,0x40000016,0x00000018};
  book->defs[0x000270c1]->node[24] = (Node) {0x0000,0x40000017,0x00000019};
  book->defs[0x000270c1]->node[25] = (Node) {0x0000,0x40000018,0x0000001a};
  book->defs[0x000270c1]->node[26] = (Node) {0x0000,0x40000019,0x0000001b};
  book->defs[0x000270c1]->node[27] = (Node) {0x0000,0x4000001a,0x0000001c};
  book->defs[0x000270c1]->node[28] = (Node) {0x0000,0x4000001b,0x0000001d};
  book->defs[0x000270c1]->node[29] = (Node) {0x0000,0x4000001c,0x0000001e};
  book->defs[0x000270c1]->node[30] = (Node) {0x0000,0x4000001d,0x0000001f};
  book->defs[0x000270c1]->node[31] = (Node) {0x0000,0x4000001e,0x00000020};
  book->defs[0x000270c1]->node[32] = (Node) {0x0000,0x4000001f,0x00000021};
  book->defs[0x000270c1]->node[33] = (Node) {0x0000,0x40000020,0x00000022};
  book->defs[0x000270c1]->node[34] = (Node) {0x0000,0x40000021,0x00000023};
  book->defs[0x000270c1]->node[35] = (Node) {0x0000,0x40000022,0x00000024};
  book->defs[0x000270c1]->node[36] = (Node) {0x0000,0x40000023,0x00000025};
  book->defs[0x000270c1]->node[37] = (Node) {0x0000,0x40000024,0x00000026};
  book->defs[0x000270c1]->node[38] = (Node) {0x0000,0x40000025,0x00000027};
  book->defs[0x000270c1]->node[39] = (Node) {0x0000,0x40000026,0x00000028};
  book->defs[0x000270c1]->node[40] = (Node) {0x0000,0x40000027,0x40000029};
  book->defs[0x000270c1]->node[41] = (Node) {0x0000,0x00000015,0x40000028};
  // c21
  book->defs[0x000270c2]           = (Term*) malloc(sizeof(Term));
  book->defs[0x000270c2]->alen     = 0;
  book->defs[0x000270c2]->acts     = (Wire*) malloc(0 * sizeof(Wire));
  book->defs[0x000270c2]->nlen     = 44;
  book->defs[0x000270c2]->node     = (Node*) malloc(44 * sizeof(Node));
  book->defs[0x000270c2]->node[ 0] = (Node) {0x0000,0x00000000,0x80000001};
  book->defs[0x000270c2]->node[ 1] = (Node) {0x0000,0x80000002,0x8000002b};
  book->defs[0x000270c2]->node[ 2] = (Node) {0x0001,0x80000003,0x8000002a};
  book->defs[0x000270c2]->node[ 3] = (Node) {0x0001,0x80000004,0x80000029};
  book->defs[0x000270c2]->node[ 4] = (Node) {0x0001,0x80000005,0x80000028};
  book->defs[0x000270c2]->node[ 5] = (Node) {0x0001,0x80000006,0x80000027};
  book->defs[0x000270c2]->node[ 6] = (Node) {0x0001,0x80000007,0x80000026};
  book->defs[0x000270c2]->node[ 7] = (Node) {0x0001,0x80000008,0x80000025};
  book->defs[0x000270c2]->node[ 8] = (Node) {0x0001,0x80000009,0x80000024};
  book->defs[0x000270c2]->node[ 9] = (Node) {0x0001,0x8000000a,0x80000023};
  book->defs[0x000270c2]->node[10] = (Node) {0x0001,0x8000000b,0x80000022};
  book->defs[0x000270c2]->node[11] = (Node) {0x0001,0x8000000c,0x80000021};
  book->defs[0x000270c2]->node[12] = (Node) {0x0001,0x8000000d,0x80000020};
  book->defs[0x000270c2]->node[13] = (Node) {0x0001,0x8000000e,0x8000001f};
  book->defs[0x000270c2]->node[14] = (Node) {0x0001,0x8000000f,0x8000001e};
  book->defs[0x000270c2]->node[15] = (Node) {0x0001,0x80000010,0x8000001d};
  book->defs[0x000270c2]->node[16] = (Node) {0x0001,0x80000011,0x8000001c};
  book->defs[0x000270c2]->node[17] = (Node) {0x0001,0x80000012,0x8000001b};
  book->defs[0x000270c2]->node[18] = (Node) {0x0001,0x80000013,0x8000001a};
  book->defs[0x000270c2]->node[19] = (Node) {0x0001,0x80000014,0x80000019};
  book->defs[0x000270c2]->node[20] = (Node) {0x0001,0x80000015,0x80000018};
  book->defs[0x000270c2]->node[21] = (Node) {0x0001,0x80000016,0x80000017};
  book->defs[0x000270c2]->node[22] = (Node) {0x0000,0x0000002b,0x00000017};
  book->defs[0x000270c2]->node[23] = (Node) {0x0000,0x40000016,0x00000018};
  book->defs[0x000270c2]->node[24] = (Node) {0x0000,0x40000017,0x00000019};
  book->defs[0x000270c2]->node[25] = (Node) {0x0000,0x40000018,0x0000001a};
  book->defs[0x000270c2]->node[26] = (Node) {0x0000,0x40000019,0x0000001b};
  book->defs[0x000270c2]->node[27] = (Node) {0x0000,0x4000001a,0x0000001c};
  book->defs[0x000270c2]->node[28] = (Node) {0x0000,0x4000001b,0x0000001d};
  book->defs[0x000270c2]->node[29] = (Node) {0x0000,0x4000001c,0x0000001e};
  book->defs[0x000270c2]->node[30] = (Node) {0x0000,0x4000001d,0x0000001f};
  book->defs[0x000270c2]->node[31] = (Node) {0x0000,0x4000001e,0x00000020};
  book->defs[0x000270c2]->node[32] = (Node) {0x0000,0x4000001f,0x00000021};
  book->defs[0x000270c2]->node[33] = (Node) {0x0000,0x40000020,0x00000022};
  book->defs[0x000270c2]->node[34] = (Node) {0x0000,0x40000021,0x00000023};
  book->defs[0x000270c2]->node[35] = (Node) {0x0000,0x40000022,0x00000024};
  book->defs[0x000270c2]->node[36] = (Node) {0x0000,0x40000023,0x00000025};
  book->defs[0x000270c2]->node[37] = (Node) {0x0000,0x40000024,0x00000026};
  book->defs[0x000270c2]->node[38] = (Node) {0x0000,0x40000025,0x00000027};
  book->defs[0x000270c2]->node[39] = (Node) {0x0000,0x40000026,0x00000028};
  book->defs[0x000270c2]->node[40] = (Node) {0x0000,0x40000027,0x00000029};
  book->defs[0x000270c2]->node[41] = (Node) {0x0000,0x40000028,0x0000002a};
  book->defs[0x000270c2]->node[42] = (Node) {0x0000,0x40000029,0x4000002b};
  book->defs[0x000270c2]->node[43] = (Node) {0x0000,0x00000016,0x4000002a};
  // c22
  book->defs[0x000270c3]           = (Term*) malloc(sizeof(Term));
  book->defs[0x000270c3]->alen     = 0;
  book->defs[0x000270c3]->acts     = (Wire*) malloc(0 * sizeof(Wire));
  book->defs[0x000270c3]->nlen     = 46;
  book->defs[0x000270c3]->node     = (Node*) malloc(46 * sizeof(Node));
  book->defs[0x000270c3]->node[ 0] = (Node) {0x0000,0x00000000,0x80000001};
  book->defs[0x000270c3]->node[ 1] = (Node) {0x0000,0x80000002,0x8000002d};
  book->defs[0x000270c3]->node[ 2] = (Node) {0x0001,0x80000003,0x8000002c};
  book->defs[0x000270c3]->node[ 3] = (Node) {0x0001,0x80000004,0x8000002b};
  book->defs[0x000270c3]->node[ 4] = (Node) {0x0001,0x80000005,0x8000002a};
  book->defs[0x000270c3]->node[ 5] = (Node) {0x0001,0x80000006,0x80000029};
  book->defs[0x000270c3]->node[ 6] = (Node) {0x0001,0x80000007,0x80000028};
  book->defs[0x000270c3]->node[ 7] = (Node) {0x0001,0x80000008,0x80000027};
  book->defs[0x000270c3]->node[ 8] = (Node) {0x0001,0x80000009,0x80000026};
  book->defs[0x000270c3]->node[ 9] = (Node) {0x0001,0x8000000a,0x80000025};
  book->defs[0x000270c3]->node[10] = (Node) {0x0001,0x8000000b,0x80000024};
  book->defs[0x000270c3]->node[11] = (Node) {0x0001,0x8000000c,0x80000023};
  book->defs[0x000270c3]->node[12] = (Node) {0x0001,0x8000000d,0x80000022};
  book->defs[0x000270c3]->node[13] = (Node) {0x0001,0x8000000e,0x80000021};
  book->defs[0x000270c3]->node[14] = (Node) {0x0001,0x8000000f,0x80000020};
  book->defs[0x000270c3]->node[15] = (Node) {0x0001,0x80000010,0x8000001f};
  book->defs[0x000270c3]->node[16] = (Node) {0x0001,0x80000011,0x8000001e};
  book->defs[0x000270c3]->node[17] = (Node) {0x0001,0x80000012,0x8000001d};
  book->defs[0x000270c3]->node[18] = (Node) {0x0001,0x80000013,0x8000001c};
  book->defs[0x000270c3]->node[19] = (Node) {0x0001,0x80000014,0x8000001b};
  book->defs[0x000270c3]->node[20] = (Node) {0x0001,0x80000015,0x8000001a};
  book->defs[0x000270c3]->node[21] = (Node) {0x0001,0x80000016,0x80000019};
  book->defs[0x000270c3]->node[22] = (Node) {0x0001,0x80000017,0x80000018};
  book->defs[0x000270c3]->node[23] = (Node) {0x0000,0x0000002d,0x00000018};
  book->defs[0x000270c3]->node[24] = (Node) {0x0000,0x40000017,0x00000019};
  book->defs[0x000270c3]->node[25] = (Node) {0x0000,0x40000018,0x0000001a};
  book->defs[0x000270c3]->node[26] = (Node) {0x0000,0x40000019,0x0000001b};
  book->defs[0x000270c3]->node[27] = (Node) {0x0000,0x4000001a,0x0000001c};
  book->defs[0x000270c3]->node[28] = (Node) {0x0000,0x4000001b,0x0000001d};
  book->defs[0x000270c3]->node[29] = (Node) {0x0000,0x4000001c,0x0000001e};
  book->defs[0x000270c3]->node[30] = (Node) {0x0000,0x4000001d,0x0000001f};
  book->defs[0x000270c3]->node[31] = (Node) {0x0000,0x4000001e,0x00000020};
  book->defs[0x000270c3]->node[32] = (Node) {0x0000,0x4000001f,0x00000021};
  book->defs[0x000270c3]->node[33] = (Node) {0x0000,0x40000020,0x00000022};
  book->defs[0x000270c3]->node[34] = (Node) {0x0000,0x40000021,0x00000023};
  book->defs[0x000270c3]->node[35] = (Node) {0x0000,0x40000022,0x00000024};
  book->defs[0x000270c3]->node[36] = (Node) {0x0000,0x40000023,0x00000025};
  book->defs[0x000270c3]->node[37] = (Node) {0x0000,0x40000024,0x00000026};
  book->defs[0x000270c3]->node[38] = (Node) {0x0000,0x40000025,0x00000027};
  book->defs[0x000270c3]->node[39] = (Node) {0x0000,0x40000026,0x00000028};
  book->defs[0x000270c3]->node[40] = (Node) {0x0000,0x40000027,0x00000029};
  book->defs[0x000270c3]->node[41] = (Node) {0x0000,0x40000028,0x0000002a};
  book->defs[0x000270c3]->node[42] = (Node) {0x0000,0x40000029,0x0000002b};
  book->defs[0x000270c3]->node[43] = (Node) {0x0000,0x4000002a,0x0000002c};
  book->defs[0x000270c3]->node[44] = (Node) {0x0000,0x4000002b,0x4000002d};
  book->defs[0x000270c3]->node[45] = (Node) {0x0000,0x00000017,0x4000002c};
  // c23
  book->defs[0x000270c4]           = (Term*) malloc(sizeof(Term));
  book->defs[0x000270c4]->alen     = 0;
  book->defs[0x000270c4]->acts     = (Wire*) malloc(0 * sizeof(Wire));
  book->defs[0x000270c4]->nlen     = 48;
  book->defs[0x000270c4]->node     = (Node*) malloc(48 * sizeof(Node));
  book->defs[0x000270c4]->node[ 0] = (Node) {0x0000,0x00000000,0x80000001};
  book->defs[0x000270c4]->node[ 1] = (Node) {0x0000,0x80000002,0x8000002f};
  book->defs[0x000270c4]->node[ 2] = (Node) {0x0001,0x80000003,0x8000002e};
  book->defs[0x000270c4]->node[ 3] = (Node) {0x0001,0x80000004,0x8000002d};
  book->defs[0x000270c4]->node[ 4] = (Node) {0x0001,0x80000005,0x8000002c};
  book->defs[0x000270c4]->node[ 5] = (Node) {0x0001,0x80000006,0x8000002b};
  book->defs[0x000270c4]->node[ 6] = (Node) {0x0001,0x80000007,0x8000002a};
  book->defs[0x000270c4]->node[ 7] = (Node) {0x0001,0x80000008,0x80000029};
  book->defs[0x000270c4]->node[ 8] = (Node) {0x0001,0x80000009,0x80000028};
  book->defs[0x000270c4]->node[ 9] = (Node) {0x0001,0x8000000a,0x80000027};
  book->defs[0x000270c4]->node[10] = (Node) {0x0001,0x8000000b,0x80000026};
  book->defs[0x000270c4]->node[11] = (Node) {0x0001,0x8000000c,0x80000025};
  book->defs[0x000270c4]->node[12] = (Node) {0x0001,0x8000000d,0x80000024};
  book->defs[0x000270c4]->node[13] = (Node) {0x0001,0x8000000e,0x80000023};
  book->defs[0x000270c4]->node[14] = (Node) {0x0001,0x8000000f,0x80000022};
  book->defs[0x000270c4]->node[15] = (Node) {0x0001,0x80000010,0x80000021};
  book->defs[0x000270c4]->node[16] = (Node) {0x0001,0x80000011,0x80000020};
  book->defs[0x000270c4]->node[17] = (Node) {0x0001,0x80000012,0x8000001f};
  book->defs[0x000270c4]->node[18] = (Node) {0x0001,0x80000013,0x8000001e};
  book->defs[0x000270c4]->node[19] = (Node) {0x0001,0x80000014,0x8000001d};
  book->defs[0x000270c4]->node[20] = (Node) {0x0001,0x80000015,0x8000001c};
  book->defs[0x000270c4]->node[21] = (Node) {0x0001,0x80000016,0x8000001b};
  book->defs[0x000270c4]->node[22] = (Node) {0x0001,0x80000017,0x8000001a};
  book->defs[0x000270c4]->node[23] = (Node) {0x0001,0x80000018,0x80000019};
  book->defs[0x000270c4]->node[24] = (Node) {0x0000,0x0000002f,0x00000019};
  book->defs[0x000270c4]->node[25] = (Node) {0x0000,0x40000018,0x0000001a};
  book->defs[0x000270c4]->node[26] = (Node) {0x0000,0x40000019,0x0000001b};
  book->defs[0x000270c4]->node[27] = (Node) {0x0000,0x4000001a,0x0000001c};
  book->defs[0x000270c4]->node[28] = (Node) {0x0000,0x4000001b,0x0000001d};
  book->defs[0x000270c4]->node[29] = (Node) {0x0000,0x4000001c,0x0000001e};
  book->defs[0x000270c4]->node[30] = (Node) {0x0000,0x4000001d,0x0000001f};
  book->defs[0x000270c4]->node[31] = (Node) {0x0000,0x4000001e,0x00000020};
  book->defs[0x000270c4]->node[32] = (Node) {0x0000,0x4000001f,0x00000021};
  book->defs[0x000270c4]->node[33] = (Node) {0x0000,0x40000020,0x00000022};
  book->defs[0x000270c4]->node[34] = (Node) {0x0000,0x40000021,0x00000023};
  book->defs[0x000270c4]->node[35] = (Node) {0x0000,0x40000022,0x00000024};
  book->defs[0x000270c4]->node[36] = (Node) {0x0000,0x40000023,0x00000025};
  book->defs[0x000270c4]->node[37] = (Node) {0x0000,0x40000024,0x00000026};
  book->defs[0x000270c4]->node[38] = (Node) {0x0000,0x40000025,0x00000027};
  book->defs[0x000270c4]->node[39] = (Node) {0x0000,0x40000026,0x00000028};
  book->defs[0x000270c4]->node[40] = (Node) {0x0000,0x40000027,0x00000029};
  book->defs[0x000270c4]->node[41] = (Node) {0x0000,0x40000028,0x0000002a};
  book->defs[0x000270c4]->node[42] = (Node) {0x0000,0x40000029,0x0000002b};
  book->defs[0x000270c4]->node[43] = (Node) {0x0000,0x4000002a,0x0000002c};
  book->defs[0x000270c4]->node[44] = (Node) {0x0000,0x4000002b,0x0000002d};
  book->defs[0x000270c4]->node[45] = (Node) {0x0000,0x4000002c,0x0000002e};
  book->defs[0x000270c4]->node[46] = (Node) {0x0000,0x4000002d,0x4000002f};
  book->defs[0x000270c4]->node[47] = (Node) {0x0000,0x00000018,0x4000002e};
  // c24
  book->defs[0x000270c5]           = (Term*) malloc(sizeof(Term));
  book->defs[0x000270c5]->alen     = 0;
  book->defs[0x000270c5]->acts     = (Wire*) malloc(0 * sizeof(Wire));
  book->defs[0x000270c5]->nlen     = 50;
  book->defs[0x000270c5]->node     = (Node*) malloc(50 * sizeof(Node));
  book->defs[0x000270c5]->node[ 0] = (Node) {0x0000,0x00000000,0x80000001};
  book->defs[0x000270c5]->node[ 1] = (Node) {0x0000,0x80000002,0x80000031};
  book->defs[0x000270c5]->node[ 2] = (Node) {0x0001,0x80000003,0x80000030};
  book->defs[0x000270c5]->node[ 3] = (Node) {0x0001,0x80000004,0x8000002f};
  book->defs[0x000270c5]->node[ 4] = (Node) {0x0001,0x80000005,0x8000002e};
  book->defs[0x000270c5]->node[ 5] = (Node) {0x0001,0x80000006,0x8000002d};
  book->defs[0x000270c5]->node[ 6] = (Node) {0x0001,0x80000007,0x8000002c};
  book->defs[0x000270c5]->node[ 7] = (Node) {0x0001,0x80000008,0x8000002b};
  book->defs[0x000270c5]->node[ 8] = (Node) {0x0001,0x80000009,0x8000002a};
  book->defs[0x000270c5]->node[ 9] = (Node) {0x0001,0x8000000a,0x80000029};
  book->defs[0x000270c5]->node[10] = (Node) {0x0001,0x8000000b,0x80000028};
  book->defs[0x000270c5]->node[11] = (Node) {0x0001,0x8000000c,0x80000027};
  book->defs[0x000270c5]->node[12] = (Node) {0x0001,0x8000000d,0x80000026};
  book->defs[0x000270c5]->node[13] = (Node) {0x0001,0x8000000e,0x80000025};
  book->defs[0x000270c5]->node[14] = (Node) {0x0001,0x8000000f,0x80000024};
  book->defs[0x000270c5]->node[15] = (Node) {0x0001,0x80000010,0x80000023};
  book->defs[0x000270c5]->node[16] = (Node) {0x0001,0x80000011,0x80000022};
  book->defs[0x000270c5]->node[17] = (Node) {0x0001,0x80000012,0x80000021};
  book->defs[0x000270c5]->node[18] = (Node) {0x0001,0x80000013,0x80000020};
  book->defs[0x000270c5]->node[19] = (Node) {0x0001,0x80000014,0x8000001f};
  book->defs[0x000270c5]->node[20] = (Node) {0x0001,0x80000015,0x8000001e};
  book->defs[0x000270c5]->node[21] = (Node) {0x0001,0x80000016,0x8000001d};
  book->defs[0x000270c5]->node[22] = (Node) {0x0001,0x80000017,0x8000001c};
  book->defs[0x000270c5]->node[23] = (Node) {0x0001,0x80000018,0x8000001b};
  book->defs[0x000270c5]->node[24] = (Node) {0x0001,0x80000019,0x8000001a};
  book->defs[0x000270c5]->node[25] = (Node) {0x0000,0x00000031,0x0000001a};
  book->defs[0x000270c5]->node[26] = (Node) {0x0000,0x40000019,0x0000001b};
  book->defs[0x000270c5]->node[27] = (Node) {0x0000,0x4000001a,0x0000001c};
  book->defs[0x000270c5]->node[28] = (Node) {0x0000,0x4000001b,0x0000001d};
  book->defs[0x000270c5]->node[29] = (Node) {0x0000,0x4000001c,0x0000001e};
  book->defs[0x000270c5]->node[30] = (Node) {0x0000,0x4000001d,0x0000001f};
  book->defs[0x000270c5]->node[31] = (Node) {0x0000,0x4000001e,0x00000020};
  book->defs[0x000270c5]->node[32] = (Node) {0x0000,0x4000001f,0x00000021};
  book->defs[0x000270c5]->node[33] = (Node) {0x0000,0x40000020,0x00000022};
  book->defs[0x000270c5]->node[34] = (Node) {0x0000,0x40000021,0x00000023};
  book->defs[0x000270c5]->node[35] = (Node) {0x0000,0x40000022,0x00000024};
  book->defs[0x000270c5]->node[36] = (Node) {0x0000,0x40000023,0x00000025};
  book->defs[0x000270c5]->node[37] = (Node) {0x0000,0x40000024,0x00000026};
  book->defs[0x000270c5]->node[38] = (Node) {0x0000,0x40000025,0x00000027};
  book->defs[0x000270c5]->node[39] = (Node) {0x0000,0x40000026,0x00000028};
  book->defs[0x000270c5]->node[40] = (Node) {0x0000,0x40000027,0x00000029};
  book->defs[0x000270c5]->node[41] = (Node) {0x0000,0x40000028,0x0000002a};
  book->defs[0x000270c5]->node[42] = (Node) {0x0000,0x40000029,0x0000002b};
  book->defs[0x000270c5]->node[43] = (Node) {0x0000,0x4000002a,0x0000002c};
  book->defs[0x000270c5]->node[44] = (Node) {0x0000,0x4000002b,0x0000002d};
  book->defs[0x000270c5]->node[45] = (Node) {0x0000,0x4000002c,0x0000002e};
  book->defs[0x000270c5]->node[46] = (Node) {0x0000,0x4000002d,0x0000002f};
  book->defs[0x000270c5]->node[47] = (Node) {0x0000,0x4000002e,0x00000030};
  book->defs[0x000270c5]->node[48] = (Node) {0x0000,0x4000002f,0x40000031};
  book->defs[0x000270c5]->node[49] = (Node) {0x0000,0x00000019,0x40000030};
  // c_s
  book->defs[0x00027ff7]           = (Term*) malloc(sizeof(Term));
  book->defs[0x00027ff7]->alen     = 0;
  book->defs[0x00027ff7]->acts     = (Wire*) malloc(0 * sizeof(Wire));
  book->defs[0x00027ff7]->nlen     = 8;
  book->defs[0x00027ff7]->node     = (Node*) malloc(8 * sizeof(Node));
  book->defs[0x00027ff7]->node[ 0] = (Node) {0x0000,0x00000000,0x80000001};
  book->defs[0x00027ff7]->node[ 1] = (Node) {0x0000,0x80000002,0x80000004};
  book->defs[0x00027ff7]->node[ 2] = (Node) {0x0000,0x40000005,0x80000003};
  book->defs[0x00027ff7]->node[ 3] = (Node) {0x0000,0x00000007,0x00000006};
  book->defs[0x00027ff7]->node[ 4] = (Node) {0x0000,0x80000005,0x80000007};
  book->defs[0x00027ff7]->node[ 5] = (Node) {0x0001,0x80000006,0x00000002};
  book->defs[0x00027ff7]->node[ 6] = (Node) {0x0000,0x40000003,0x40000007};
  book->defs[0x00027ff7]->node[ 7] = (Node) {0x0000,0x00000003,0x40000006};
  // c_z
  book->defs[0x00027ffe]           = (Term*) malloc(sizeof(Term));
  book->defs[0x00027ffe]->alen     = 0;
  book->defs[0x00027ffe]->acts     = (Wire*) malloc(0 * sizeof(Wire));
  book->defs[0x00027ffe]->nlen     = 3;
  book->defs[0x00027ffe]->node     = (Node*) malloc(3 * sizeof(Node));
  book->defs[0x00027ffe]->node[ 0] = (Node) {0x0000,0x00000000,0x80000001};
  book->defs[0x00027ffe]->node[ 1] = (Node) {0x0000,0x80000000,0x80000002};
  book->defs[0x00027ffe]->node[ 2] = (Node) {0x0000,0x40000002,0x00000002};
  // dec
  book->defs[0x00028a67]           = (Term*) malloc(sizeof(Term));
  book->defs[0x00028a67]->alen     = 0;
  book->defs[0x00028a67]->acts     = (Wire*) malloc(0 * sizeof(Wire));
  book->defs[0x00028a67]->nlen     = 5;
  book->defs[0x00028a67]->node     = (Node*) malloc(5 * sizeof(Node));
  book->defs[0x00028a67]->node[ 0] = (Node) {0x0000,0x00000000,0x80000001};
  book->defs[0x00028a67]->node[ 1] = (Node) {0x0000,0x80000002,0x40000004};
  book->defs[0x00028a67]->node[ 2] = (Node) {0x0000,0xc0a299d9,0x80000003};
  book->defs[0x00028a67]->node[ 3] = (Node) {0x0000,0xc0a299d3,0x80000004};
  book->defs[0x00028a67]->node[ 4] = (Node) {0x0000,0xc000000f,0x40000001};
  // ex0
  book->defs[0x00029f01]           = (Term*) malloc(sizeof(Term));
  book->defs[0x00029f01]->alen     = 1;
  book->defs[0x00029f01]->acts     = (Wire*) malloc(1 * sizeof(Wire));
  book->defs[0x00029f01]->acts[ 0] = (Wire) {0xc00009c3,0x80000001};
  book->defs[0x00029f01]->nlen     = 2;
  book->defs[0x00029f01]->node     = (Node*) malloc(2 * sizeof(Node));
  book->defs[0x00029f01]->node[ 0] = (Node) {0x0000,0x00000000,0x40000001};
  book->defs[0x00029f01]->node[ 1] = (Node) {0x0000,0xc0000bc3,0x40000000};
  // ex1
  book->defs[0x00029f02]           = (Term*) malloc(sizeof(Term));
  book->defs[0x00029f02]->alen     = 1;
  book->defs[0x00029f02]->acts     = (Wire*) malloc(1 * sizeof(Wire));
  book->defs[0x00029f02]->acts[ 0] = (Wire) {0xc00270c4,0x80000001};
  book->defs[0x00029f02]->nlen     = 3;
  book->defs[0x00029f02]->node     = (Node*) malloc(3 * sizeof(Node));
  book->defs[0x00029f02]->node[ 0] = (Node) {0x0000,0x00000000,0x40000002};
  book->defs[0x00029f02]->node[ 1] = (Node) {0x0000,0xc002bff7,0x80000002};
  book->defs[0x00029f02]->node[ 2] = (Node) {0x0000,0xc002bffe,0x40000000};
  // ex2
  book->defs[0x00029f03]           = (Term*) malloc(sizeof(Term));
  book->defs[0x00029f03]->alen     = 2;
  book->defs[0x00029f03]->acts     = (Wire*) malloc(2 * sizeof(Wire));
  book->defs[0x00029f03]->acts[ 0] = (Wire) {0xc0036e72,0x80000001};
  book->defs[0x00029f03]->acts[ 1] = (Wire) {0xc00009c7,0x80000002};
  book->defs[0x00029f03]->nlen     = 4;
  book->defs[0x00029f03]->node     = (Node*) malloc(4 * sizeof(Node));
  book->defs[0x00029f03]->node[ 0] = (Node) {0x0000,0x00000000,0x40000001};
  book->defs[0x00029f03]->node[ 1] = (Node) {0x0000,0x40000003,0x40000000};
  book->defs[0x00029f03]->node[ 2] = (Node) {0x0000,0xc0000013,0x80000003};
  book->defs[0x00029f03]->node[ 3] = (Node) {0x0000,0xc000000f,0x00000001};
  // ex3
  book->defs[0x00029f04]           = (Term*) malloc(sizeof(Term));
  book->defs[0x00029f04]->alen     = 2;
  book->defs[0x00029f04]->acts     = (Wire*) malloc(2 * sizeof(Wire));
  book->defs[0x00029f04]->acts[ 0] = (Wire) {0xc0026db2,0x80000001};
  book->defs[0x00029f04]->acts[ 1] = (Wire) {0xc0027083,0x80000002};
  book->defs[0x00029f04]->nlen     = 4;
  book->defs[0x00029f04]->node     = (Node*) malloc(4 * sizeof(Node));
  book->defs[0x00029f04]->node[ 0] = (Node) {0x0000,0x00000000,0x40000001};
  book->defs[0x00029f04]->node[ 1] = (Node) {0x0000,0x40000003,0x40000000};
  book->defs[0x00029f04]->node[ 2] = (Node) {0x0000,0xc000001d,0x80000003};
  book->defs[0x00029f04]->node[ 3] = (Node) {0x0000,0xc0000024,0x00000001};
  // g_s
  book->defs[0x0002bff7]           = (Term*) malloc(sizeof(Term));
  book->defs[0x0002bff7]->alen     = 0;
  book->defs[0x0002bff7]->acts     = (Wire*) malloc(0 * sizeof(Wire));
  book->defs[0x0002bff7]->nlen     = 6;
  book->defs[0x0002bff7]->node     = (Node*) malloc(6 * sizeof(Node));
  book->defs[0x0002bff7]->node[ 0] = (Node) {0x0000,0x00000000,0x80000001};
  book->defs[0x0002bff7]->node[ 1] = (Node) {0x0000,0x80000002,0x80000003};
  book->defs[0x0002bff7]->node[ 2] = (Node) {0x0002,0x00000004,0x00000005};
  book->defs[0x0002bff7]->node[ 3] = (Node) {0x0000,0x80000004,0x40000005};
  book->defs[0x0002bff7]->node[ 4] = (Node) {0x0000,0x00000002,0x80000005};
  book->defs[0x0002bff7]->node[ 5] = (Node) {0x0000,0x40000002,0x40000003};
  // g_z
  book->defs[0x0002bffe]           = (Term*) malloc(sizeof(Term));
  book->defs[0x0002bffe]->alen     = 0;
  book->defs[0x0002bffe]->acts     = (Wire*) malloc(0 * sizeof(Wire));
  book->defs[0x0002bffe]->nlen     = 2;
  book->defs[0x0002bffe]->node     = (Node*) malloc(2 * sizeof(Node));
  book->defs[0x0002bffe]->node[ 0] = (Node) {0x0000,0x00000000,0x80000001};
  book->defs[0x0002bffe]->node[ 1] = (Node) {0x0000,0x40000001,0x00000001};
  // k10
  book->defs[0x0002f081]           = (Term*) malloc(sizeof(Term));
  book->defs[0x0002f081]->alen     = 0;
  book->defs[0x0002f081]->acts     = (Wire*) malloc(0 * sizeof(Wire));
  book->defs[0x0002f081]->nlen     = 22;
  book->defs[0x0002f081]->node     = (Node*) malloc(22 * sizeof(Node));
  book->defs[0x0002f081]->node[ 0] = (Node) {0x0000,0x00000000,0x80000001};
  book->defs[0x0002f081]->node[ 1] = (Node) {0x0000,0x80000002,0x80000015};
  book->defs[0x0002f081]->node[ 2] = (Node) {0x0002,0x80000003,0x80000014};
  book->defs[0x0002f081]->node[ 3] = (Node) {0x0002,0x80000004,0x80000013};
  book->defs[0x0002f081]->node[ 4] = (Node) {0x0002,0x80000005,0x80000012};
  book->defs[0x0002f081]->node[ 5] = (Node) {0x0002,0x80000006,0x80000011};
  book->defs[0x0002f081]->node[ 6] = (Node) {0x0002,0x80000007,0x80000010};
  book->defs[0x0002f081]->node[ 7] = (Node) {0x0002,0x80000008,0x8000000f};
  book->defs[0x0002f081]->node[ 8] = (Node) {0x0002,0x80000009,0x8000000e};
  book->defs[0x0002f081]->node[ 9] = (Node) {0x0002,0x8000000a,0x8000000d};
  book->defs[0x0002f081]->node[10] = (Node) {0x0002,0x8000000b,0x8000000c};
  book->defs[0x0002f081]->node[11] = (Node) {0x0000,0x00000015,0x0000000c};
  book->defs[0x0002f081]->node[12] = (Node) {0x0000,0x4000000b,0x0000000d};
  book->defs[0x0002f081]->node[13] = (Node) {0x0000,0x4000000c,0x0000000e};
  book->defs[0x0002f081]->node[14] = (Node) {0x0000,0x4000000d,0x0000000f};
  book->defs[0x0002f081]->node[15] = (Node) {0x0000,0x4000000e,0x00000010};
  book->defs[0x0002f081]->node[16] = (Node) {0x0000,0x4000000f,0x00000011};
  book->defs[0x0002f081]->node[17] = (Node) {0x0000,0x40000010,0x00000012};
  book->defs[0x0002f081]->node[18] = (Node) {0x0000,0x40000011,0x00000013};
  book->defs[0x0002f081]->node[19] = (Node) {0x0000,0x40000012,0x00000014};
  book->defs[0x0002f081]->node[20] = (Node) {0x0000,0x40000013,0x40000015};
  book->defs[0x0002f081]->node[21] = (Node) {0x0000,0x0000000b,0x40000014};
  // k11
  book->defs[0x0002f082]           = (Term*) malloc(sizeof(Term));
  book->defs[0x0002f082]->alen     = 0;
  book->defs[0x0002f082]->acts     = (Wire*) malloc(0 * sizeof(Wire));
  book->defs[0x0002f082]->nlen     = 24;
  book->defs[0x0002f082]->node     = (Node*) malloc(24 * sizeof(Node));
  book->defs[0x0002f082]->node[ 0] = (Node) {0x0000,0x00000000,0x80000001};
  book->defs[0x0002f082]->node[ 1] = (Node) {0x0000,0x80000002,0x80000017};
  book->defs[0x0002f082]->node[ 2] = (Node) {0x0002,0x80000003,0x80000016};
  book->defs[0x0002f082]->node[ 3] = (Node) {0x0002,0x80000004,0x80000015};
  book->defs[0x0002f082]->node[ 4] = (Node) {0x0002,0x80000005,0x80000014};
  book->defs[0x0002f082]->node[ 5] = (Node) {0x0002,0x80000006,0x80000013};
  book->defs[0x0002f082]->node[ 6] = (Node) {0x0002,0x80000007,0x80000012};
  book->defs[0x0002f082]->node[ 7] = (Node) {0x0002,0x80000008,0x80000011};
  book->defs[0x0002f082]->node[ 8] = (Node) {0x0002,0x80000009,0x80000010};
  book->defs[0x0002f082]->node[ 9] = (Node) {0x0002,0x8000000a,0x8000000f};
  book->defs[0x0002f082]->node[10] = (Node) {0x0002,0x8000000b,0x8000000e};
  book->defs[0x0002f082]->node[11] = (Node) {0x0002,0x8000000c,0x8000000d};
  book->defs[0x0002f082]->node[12] = (Node) {0x0000,0x00000017,0x0000000d};
  book->defs[0x0002f082]->node[13] = (Node) {0x0000,0x4000000c,0x0000000e};
  book->defs[0x0002f082]->node[14] = (Node) {0x0000,0x4000000d,0x0000000f};
  book->defs[0x0002f082]->node[15] = (Node) {0x0000,0x4000000e,0x00000010};
  book->defs[0x0002f082]->node[16] = (Node) {0x0000,0x4000000f,0x00000011};
  book->defs[0x0002f082]->node[17] = (Node) {0x0000,0x40000010,0x00000012};
  book->defs[0x0002f082]->node[18] = (Node) {0x0000,0x40000011,0x00000013};
  book->defs[0x0002f082]->node[19] = (Node) {0x0000,0x40000012,0x00000014};
  book->defs[0x0002f082]->node[20] = (Node) {0x0000,0x40000013,0x00000015};
  book->defs[0x0002f082]->node[21] = (Node) {0x0000,0x40000014,0x00000016};
  book->defs[0x0002f082]->node[22] = (Node) {0x0000,0x40000015,0x40000017};
  book->defs[0x0002f082]->node[23] = (Node) {0x0000,0x0000000c,0x40000016};
  // k12
  book->defs[0x0002f083]           = (Term*) malloc(sizeof(Term));
  book->defs[0x0002f083]->alen     = 0;
  book->defs[0x0002f083]->acts     = (Wire*) malloc(0 * sizeof(Wire));
  book->defs[0x0002f083]->nlen     = 26;
  book->defs[0x0002f083]->node     = (Node*) malloc(26 * sizeof(Node));
  book->defs[0x0002f083]->node[ 0] = (Node) {0x0000,0x00000000,0x80000001};
  book->defs[0x0002f083]->node[ 1] = (Node) {0x0000,0x80000002,0x80000019};
  book->defs[0x0002f083]->node[ 2] = (Node) {0x0002,0x80000003,0x80000018};
  book->defs[0x0002f083]->node[ 3] = (Node) {0x0002,0x80000004,0x80000017};
  book->defs[0x0002f083]->node[ 4] = (Node) {0x0002,0x80000005,0x80000016};
  book->defs[0x0002f083]->node[ 5] = (Node) {0x0002,0x80000006,0x80000015};
  book->defs[0x0002f083]->node[ 6] = (Node) {0x0002,0x80000007,0x80000014};
  book->defs[0x0002f083]->node[ 7] = (Node) {0x0002,0x80000008,0x80000013};
  book->defs[0x0002f083]->node[ 8] = (Node) {0x0002,0x80000009,0x80000012};
  book->defs[0x0002f083]->node[ 9] = (Node) {0x0002,0x8000000a,0x80000011};
  book->defs[0x0002f083]->node[10] = (Node) {0x0002,0x8000000b,0x80000010};
  book->defs[0x0002f083]->node[11] = (Node) {0x0002,0x8000000c,0x8000000f};
  book->defs[0x0002f083]->node[12] = (Node) {0x0002,0x8000000d,0x8000000e};
  book->defs[0x0002f083]->node[13] = (Node) {0x0000,0x00000019,0x0000000e};
  book->defs[0x0002f083]->node[14] = (Node) {0x0000,0x4000000d,0x0000000f};
  book->defs[0x0002f083]->node[15] = (Node) {0x0000,0x4000000e,0x00000010};
  book->defs[0x0002f083]->node[16] = (Node) {0x0000,0x4000000f,0x00000011};
  book->defs[0x0002f083]->node[17] = (Node) {0x0000,0x40000010,0x00000012};
  book->defs[0x0002f083]->node[18] = (Node) {0x0000,0x40000011,0x00000013};
  book->defs[0x0002f083]->node[19] = (Node) {0x0000,0x40000012,0x00000014};
  book->defs[0x0002f083]->node[20] = (Node) {0x0000,0x40000013,0x00000015};
  book->defs[0x0002f083]->node[21] = (Node) {0x0000,0x40000014,0x00000016};
  book->defs[0x0002f083]->node[22] = (Node) {0x0000,0x40000015,0x00000017};
  book->defs[0x0002f083]->node[23] = (Node) {0x0000,0x40000016,0x00000018};
  book->defs[0x0002f083]->node[24] = (Node) {0x0000,0x40000017,0x40000019};
  book->defs[0x0002f083]->node[25] = (Node) {0x0000,0x0000000d,0x40000018};
  // k13
  book->defs[0x0002f084]           = (Term*) malloc(sizeof(Term));
  book->defs[0x0002f084]->alen     = 0;
  book->defs[0x0002f084]->acts     = (Wire*) malloc(0 * sizeof(Wire));
  book->defs[0x0002f084]->nlen     = 28;
  book->defs[0x0002f084]->node     = (Node*) malloc(28 * sizeof(Node));
  book->defs[0x0002f084]->node[ 0] = (Node) {0x0000,0x00000000,0x80000001};
  book->defs[0x0002f084]->node[ 1] = (Node) {0x0000,0x80000002,0x8000001b};
  book->defs[0x0002f084]->node[ 2] = (Node) {0x0002,0x80000003,0x8000001a};
  book->defs[0x0002f084]->node[ 3] = (Node) {0x0002,0x80000004,0x80000019};
  book->defs[0x0002f084]->node[ 4] = (Node) {0x0002,0x80000005,0x80000018};
  book->defs[0x0002f084]->node[ 5] = (Node) {0x0002,0x80000006,0x80000017};
  book->defs[0x0002f084]->node[ 6] = (Node) {0x0002,0x80000007,0x80000016};
  book->defs[0x0002f084]->node[ 7] = (Node) {0x0002,0x80000008,0x80000015};
  book->defs[0x0002f084]->node[ 8] = (Node) {0x0002,0x80000009,0x80000014};
  book->defs[0x0002f084]->node[ 9] = (Node) {0x0002,0x8000000a,0x80000013};
  book->defs[0x0002f084]->node[10] = (Node) {0x0002,0x8000000b,0x80000012};
  book->defs[0x0002f084]->node[11] = (Node) {0x0002,0x8000000c,0x80000011};
  book->defs[0x0002f084]->node[12] = (Node) {0x0002,0x8000000d,0x80000010};
  book->defs[0x0002f084]->node[13] = (Node) {0x0002,0x8000000e,0x8000000f};
  book->defs[0x0002f084]->node[14] = (Node) {0x0000,0x0000001b,0x0000000f};
  book->defs[0x0002f084]->node[15] = (Node) {0x0000,0x4000000e,0x00000010};
  book->defs[0x0002f084]->node[16] = (Node) {0x0000,0x4000000f,0x00000011};
  book->defs[0x0002f084]->node[17] = (Node) {0x0000,0x40000010,0x00000012};
  book->defs[0x0002f084]->node[18] = (Node) {0x0000,0x40000011,0x00000013};
  book->defs[0x0002f084]->node[19] = (Node) {0x0000,0x40000012,0x00000014};
  book->defs[0x0002f084]->node[20] = (Node) {0x0000,0x40000013,0x00000015};
  book->defs[0x0002f084]->node[21] = (Node) {0x0000,0x40000014,0x00000016};
  book->defs[0x0002f084]->node[22] = (Node) {0x0000,0x40000015,0x00000017};
  book->defs[0x0002f084]->node[23] = (Node) {0x0000,0x40000016,0x00000018};
  book->defs[0x0002f084]->node[24] = (Node) {0x0000,0x40000017,0x00000019};
  book->defs[0x0002f084]->node[25] = (Node) {0x0000,0x40000018,0x0000001a};
  book->defs[0x0002f084]->node[26] = (Node) {0x0000,0x40000019,0x4000001b};
  book->defs[0x0002f084]->node[27] = (Node) {0x0000,0x0000000e,0x4000001a};
  // k14
  book->defs[0x0002f085]           = (Term*) malloc(sizeof(Term));
  book->defs[0x0002f085]->alen     = 0;
  book->defs[0x0002f085]->acts     = (Wire*) malloc(0 * sizeof(Wire));
  book->defs[0x0002f085]->nlen     = 30;
  book->defs[0x0002f085]->node     = (Node*) malloc(30 * sizeof(Node));
  book->defs[0x0002f085]->node[ 0] = (Node) {0x0000,0x00000000,0x80000001};
  book->defs[0x0002f085]->node[ 1] = (Node) {0x0000,0x80000002,0x8000001d};
  book->defs[0x0002f085]->node[ 2] = (Node) {0x0002,0x80000003,0x8000001c};
  book->defs[0x0002f085]->node[ 3] = (Node) {0x0002,0x80000004,0x8000001b};
  book->defs[0x0002f085]->node[ 4] = (Node) {0x0002,0x80000005,0x8000001a};
  book->defs[0x0002f085]->node[ 5] = (Node) {0x0002,0x80000006,0x80000019};
  book->defs[0x0002f085]->node[ 6] = (Node) {0x0002,0x80000007,0x80000018};
  book->defs[0x0002f085]->node[ 7] = (Node) {0x0002,0x80000008,0x80000017};
  book->defs[0x0002f085]->node[ 8] = (Node) {0x0002,0x80000009,0x80000016};
  book->defs[0x0002f085]->node[ 9] = (Node) {0x0002,0x8000000a,0x80000015};
  book->defs[0x0002f085]->node[10] = (Node) {0x0002,0x8000000b,0x80000014};
  book->defs[0x0002f085]->node[11] = (Node) {0x0002,0x8000000c,0x80000013};
  book->defs[0x0002f085]->node[12] = (Node) {0x0002,0x8000000d,0x80000012};
  book->defs[0x0002f085]->node[13] = (Node) {0x0002,0x8000000e,0x80000011};
  book->defs[0x0002f085]->node[14] = (Node) {0x0002,0x8000000f,0x80000010};
  book->defs[0x0002f085]->node[15] = (Node) {0x0000,0x0000001d,0x00000010};
  book->defs[0x0002f085]->node[16] = (Node) {0x0000,0x4000000f,0x00000011};
  book->defs[0x0002f085]->node[17] = (Node) {0x0000,0x40000010,0x00000012};
  book->defs[0x0002f085]->node[18] = (Node) {0x0000,0x40000011,0x00000013};
  book->defs[0x0002f085]->node[19] = (Node) {0x0000,0x40000012,0x00000014};
  book->defs[0x0002f085]->node[20] = (Node) {0x0000,0x40000013,0x00000015};
  book->defs[0x0002f085]->node[21] = (Node) {0x0000,0x40000014,0x00000016};
  book->defs[0x0002f085]->node[22] = (Node) {0x0000,0x40000015,0x00000017};
  book->defs[0x0002f085]->node[23] = (Node) {0x0000,0x40000016,0x00000018};
  book->defs[0x0002f085]->node[24] = (Node) {0x0000,0x40000017,0x00000019};
  book->defs[0x0002f085]->node[25] = (Node) {0x0000,0x40000018,0x0000001a};
  book->defs[0x0002f085]->node[26] = (Node) {0x0000,0x40000019,0x0000001b};
  book->defs[0x0002f085]->node[27] = (Node) {0x0000,0x4000001a,0x0000001c};
  book->defs[0x0002f085]->node[28] = (Node) {0x0000,0x4000001b,0x4000001d};
  book->defs[0x0002f085]->node[29] = (Node) {0x0000,0x0000000f,0x4000001c};
  // k15
  book->defs[0x0002f086]           = (Term*) malloc(sizeof(Term));
  book->defs[0x0002f086]->alen     = 0;
  book->defs[0x0002f086]->acts     = (Wire*) malloc(0 * sizeof(Wire));
  book->defs[0x0002f086]->nlen     = 32;
  book->defs[0x0002f086]->node     = (Node*) malloc(32 * sizeof(Node));
  book->defs[0x0002f086]->node[ 0] = (Node) {0x0000,0x00000000,0x80000001};
  book->defs[0x0002f086]->node[ 1] = (Node) {0x0000,0x80000002,0x8000001f};
  book->defs[0x0002f086]->node[ 2] = (Node) {0x0002,0x80000003,0x8000001e};
  book->defs[0x0002f086]->node[ 3] = (Node) {0x0002,0x80000004,0x8000001d};
  book->defs[0x0002f086]->node[ 4] = (Node) {0x0002,0x80000005,0x8000001c};
  book->defs[0x0002f086]->node[ 5] = (Node) {0x0002,0x80000006,0x8000001b};
  book->defs[0x0002f086]->node[ 6] = (Node) {0x0002,0x80000007,0x8000001a};
  book->defs[0x0002f086]->node[ 7] = (Node) {0x0002,0x80000008,0x80000019};
  book->defs[0x0002f086]->node[ 8] = (Node) {0x0002,0x80000009,0x80000018};
  book->defs[0x0002f086]->node[ 9] = (Node) {0x0002,0x8000000a,0x80000017};
  book->defs[0x0002f086]->node[10] = (Node) {0x0002,0x8000000b,0x80000016};
  book->defs[0x0002f086]->node[11] = (Node) {0x0002,0x8000000c,0x80000015};
  book->defs[0x0002f086]->node[12] = (Node) {0x0002,0x8000000d,0x80000014};
  book->defs[0x0002f086]->node[13] = (Node) {0x0002,0x8000000e,0x80000013};
  book->defs[0x0002f086]->node[14] = (Node) {0x0002,0x8000000f,0x80000012};
  book->defs[0x0002f086]->node[15] = (Node) {0x0002,0x80000010,0x80000011};
  book->defs[0x0002f086]->node[16] = (Node) {0x0000,0x0000001f,0x00000011};
  book->defs[0x0002f086]->node[17] = (Node) {0x0000,0x40000010,0x00000012};
  book->defs[0x0002f086]->node[18] = (Node) {0x0000,0x40000011,0x00000013};
  book->defs[0x0002f086]->node[19] = (Node) {0x0000,0x40000012,0x00000014};
  book->defs[0x0002f086]->node[20] = (Node) {0x0000,0x40000013,0x00000015};
  book->defs[0x0002f086]->node[21] = (Node) {0x0000,0x40000014,0x00000016};
  book->defs[0x0002f086]->node[22] = (Node) {0x0000,0x40000015,0x00000017};
  book->defs[0x0002f086]->node[23] = (Node) {0x0000,0x40000016,0x00000018};
  book->defs[0x0002f086]->node[24] = (Node) {0x0000,0x40000017,0x00000019};
  book->defs[0x0002f086]->node[25] = (Node) {0x0000,0x40000018,0x0000001a};
  book->defs[0x0002f086]->node[26] = (Node) {0x0000,0x40000019,0x0000001b};
  book->defs[0x0002f086]->node[27] = (Node) {0x0000,0x4000001a,0x0000001c};
  book->defs[0x0002f086]->node[28] = (Node) {0x0000,0x4000001b,0x0000001d};
  book->defs[0x0002f086]->node[29] = (Node) {0x0000,0x4000001c,0x0000001e};
  book->defs[0x0002f086]->node[30] = (Node) {0x0000,0x4000001d,0x4000001f};
  book->defs[0x0002f086]->node[31] = (Node) {0x0000,0x00000010,0x4000001e};
  // k16
  book->defs[0x0002f087]           = (Term*) malloc(sizeof(Term));
  book->defs[0x0002f087]->alen     = 0;
  book->defs[0x0002f087]->acts     = (Wire*) malloc(0 * sizeof(Wire));
  book->defs[0x0002f087]->nlen     = 34;
  book->defs[0x0002f087]->node     = (Node*) malloc(34 * sizeof(Node));
  book->defs[0x0002f087]->node[ 0] = (Node) {0x0000,0x00000000,0x80000001};
  book->defs[0x0002f087]->node[ 1] = (Node) {0x0000,0x80000002,0x80000021};
  book->defs[0x0002f087]->node[ 2] = (Node) {0x0002,0x80000003,0x80000020};
  book->defs[0x0002f087]->node[ 3] = (Node) {0x0002,0x80000004,0x8000001f};
  book->defs[0x0002f087]->node[ 4] = (Node) {0x0002,0x80000005,0x8000001e};
  book->defs[0x0002f087]->node[ 5] = (Node) {0x0002,0x80000006,0x8000001d};
  book->defs[0x0002f087]->node[ 6] = (Node) {0x0002,0x80000007,0x8000001c};
  book->defs[0x0002f087]->node[ 7] = (Node) {0x0002,0x80000008,0x8000001b};
  book->defs[0x0002f087]->node[ 8] = (Node) {0x0002,0x80000009,0x8000001a};
  book->defs[0x0002f087]->node[ 9] = (Node) {0x0002,0x8000000a,0x80000019};
  book->defs[0x0002f087]->node[10] = (Node) {0x0002,0x8000000b,0x80000018};
  book->defs[0x0002f087]->node[11] = (Node) {0x0002,0x8000000c,0x80000017};
  book->defs[0x0002f087]->node[12] = (Node) {0x0002,0x8000000d,0x80000016};
  book->defs[0x0002f087]->node[13] = (Node) {0x0002,0x8000000e,0x80000015};
  book->defs[0x0002f087]->node[14] = (Node) {0x0002,0x8000000f,0x80000014};
  book->defs[0x0002f087]->node[15] = (Node) {0x0002,0x80000010,0x80000013};
  book->defs[0x0002f087]->node[16] = (Node) {0x0002,0x80000011,0x80000012};
  book->defs[0x0002f087]->node[17] = (Node) {0x0000,0x00000021,0x00000012};
  book->defs[0x0002f087]->node[18] = (Node) {0x0000,0x40000011,0x00000013};
  book->defs[0x0002f087]->node[19] = (Node) {0x0000,0x40000012,0x00000014};
  book->defs[0x0002f087]->node[20] = (Node) {0x0000,0x40000013,0x00000015};
  book->defs[0x0002f087]->node[21] = (Node) {0x0000,0x40000014,0x00000016};
  book->defs[0x0002f087]->node[22] = (Node) {0x0000,0x40000015,0x00000017};
  book->defs[0x0002f087]->node[23] = (Node) {0x0000,0x40000016,0x00000018};
  book->defs[0x0002f087]->node[24] = (Node) {0x0000,0x40000017,0x00000019};
  book->defs[0x0002f087]->node[25] = (Node) {0x0000,0x40000018,0x0000001a};
  book->defs[0x0002f087]->node[26] = (Node) {0x0000,0x40000019,0x0000001b};
  book->defs[0x0002f087]->node[27] = (Node) {0x0000,0x4000001a,0x0000001c};
  book->defs[0x0002f087]->node[28] = (Node) {0x0000,0x4000001b,0x0000001d};
  book->defs[0x0002f087]->node[29] = (Node) {0x0000,0x4000001c,0x0000001e};
  book->defs[0x0002f087]->node[30] = (Node) {0x0000,0x4000001d,0x0000001f};
  book->defs[0x0002f087]->node[31] = (Node) {0x0000,0x4000001e,0x00000020};
  book->defs[0x0002f087]->node[32] = (Node) {0x0000,0x4000001f,0x40000021};
  book->defs[0x0002f087]->node[33] = (Node) {0x0000,0x00000011,0x40000020};
  // k17
  book->defs[0x0002f088]           = (Term*) malloc(sizeof(Term));
  book->defs[0x0002f088]->alen     = 0;
  book->defs[0x0002f088]->acts     = (Wire*) malloc(0 * sizeof(Wire));
  book->defs[0x0002f088]->nlen     = 36;
  book->defs[0x0002f088]->node     = (Node*) malloc(36 * sizeof(Node));
  book->defs[0x0002f088]->node[ 0] = (Node) {0x0000,0x00000000,0x80000001};
  book->defs[0x0002f088]->node[ 1] = (Node) {0x0000,0x80000002,0x80000023};
  book->defs[0x0002f088]->node[ 2] = (Node) {0x0002,0x80000003,0x80000022};
  book->defs[0x0002f088]->node[ 3] = (Node) {0x0002,0x80000004,0x80000021};
  book->defs[0x0002f088]->node[ 4] = (Node) {0x0002,0x80000005,0x80000020};
  book->defs[0x0002f088]->node[ 5] = (Node) {0x0002,0x80000006,0x8000001f};
  book->defs[0x0002f088]->node[ 6] = (Node) {0x0002,0x80000007,0x8000001e};
  book->defs[0x0002f088]->node[ 7] = (Node) {0x0002,0x80000008,0x8000001d};
  book->defs[0x0002f088]->node[ 8] = (Node) {0x0002,0x80000009,0x8000001c};
  book->defs[0x0002f088]->node[ 9] = (Node) {0x0002,0x8000000a,0x8000001b};
  book->defs[0x0002f088]->node[10] = (Node) {0x0002,0x8000000b,0x8000001a};
  book->defs[0x0002f088]->node[11] = (Node) {0x0002,0x8000000c,0x80000019};
  book->defs[0x0002f088]->node[12] = (Node) {0x0002,0x8000000d,0x80000018};
  book->defs[0x0002f088]->node[13] = (Node) {0x0002,0x8000000e,0x80000017};
  book->defs[0x0002f088]->node[14] = (Node) {0x0002,0x8000000f,0x80000016};
  book->defs[0x0002f088]->node[15] = (Node) {0x0002,0x80000010,0x80000015};
  book->defs[0x0002f088]->node[16] = (Node) {0x0002,0x80000011,0x80000014};
  book->defs[0x0002f088]->node[17] = (Node) {0x0002,0x80000012,0x80000013};
  book->defs[0x0002f088]->node[18] = (Node) {0x0000,0x00000023,0x00000013};
  book->defs[0x0002f088]->node[19] = (Node) {0x0000,0x40000012,0x00000014};
  book->defs[0x0002f088]->node[20] = (Node) {0x0000,0x40000013,0x00000015};
  book->defs[0x0002f088]->node[21] = (Node) {0x0000,0x40000014,0x00000016};
  book->defs[0x0002f088]->node[22] = (Node) {0x0000,0x40000015,0x00000017};
  book->defs[0x0002f088]->node[23] = (Node) {0x0000,0x40000016,0x00000018};
  book->defs[0x0002f088]->node[24] = (Node) {0x0000,0x40000017,0x00000019};
  book->defs[0x0002f088]->node[25] = (Node) {0x0000,0x40000018,0x0000001a};
  book->defs[0x0002f088]->node[26] = (Node) {0x0000,0x40000019,0x0000001b};
  book->defs[0x0002f088]->node[27] = (Node) {0x0000,0x4000001a,0x0000001c};
  book->defs[0x0002f088]->node[28] = (Node) {0x0000,0x4000001b,0x0000001d};
  book->defs[0x0002f088]->node[29] = (Node) {0x0000,0x4000001c,0x0000001e};
  book->defs[0x0002f088]->node[30] = (Node) {0x0000,0x4000001d,0x0000001f};
  book->defs[0x0002f088]->node[31] = (Node) {0x0000,0x4000001e,0x00000020};
  book->defs[0x0002f088]->node[32] = (Node) {0x0000,0x4000001f,0x00000021};
  book->defs[0x0002f088]->node[33] = (Node) {0x0000,0x40000020,0x00000022};
  book->defs[0x0002f088]->node[34] = (Node) {0x0000,0x40000021,0x40000023};
  book->defs[0x0002f088]->node[35] = (Node) {0x0000,0x00000012,0x40000022};
  // k18
  book->defs[0x0002f089]           = (Term*) malloc(sizeof(Term));
  book->defs[0x0002f089]->alen     = 0;
  book->defs[0x0002f089]->acts     = (Wire*) malloc(0 * sizeof(Wire));
  book->defs[0x0002f089]->nlen     = 38;
  book->defs[0x0002f089]->node     = (Node*) malloc(38 * sizeof(Node));
  book->defs[0x0002f089]->node[ 0] = (Node) {0x0000,0x00000000,0x80000001};
  book->defs[0x0002f089]->node[ 1] = (Node) {0x0000,0x80000002,0x80000025};
  book->defs[0x0002f089]->node[ 2] = (Node) {0x0002,0x80000003,0x80000024};
  book->defs[0x0002f089]->node[ 3] = (Node) {0x0002,0x80000004,0x80000023};
  book->defs[0x0002f089]->node[ 4] = (Node) {0x0002,0x80000005,0x80000022};
  book->defs[0x0002f089]->node[ 5] = (Node) {0x0002,0x80000006,0x80000021};
  book->defs[0x0002f089]->node[ 6] = (Node) {0x0002,0x80000007,0x80000020};
  book->defs[0x0002f089]->node[ 7] = (Node) {0x0002,0x80000008,0x8000001f};
  book->defs[0x0002f089]->node[ 8] = (Node) {0x0002,0x80000009,0x8000001e};
  book->defs[0x0002f089]->node[ 9] = (Node) {0x0002,0x8000000a,0x8000001d};
  book->defs[0x0002f089]->node[10] = (Node) {0x0002,0x8000000b,0x8000001c};
  book->defs[0x0002f089]->node[11] = (Node) {0x0002,0x8000000c,0x8000001b};
  book->defs[0x0002f089]->node[12] = (Node) {0x0002,0x8000000d,0x8000001a};
  book->defs[0x0002f089]->node[13] = (Node) {0x0002,0x8000000e,0x80000019};
  book->defs[0x0002f089]->node[14] = (Node) {0x0002,0x8000000f,0x80000018};
  book->defs[0x0002f089]->node[15] = (Node) {0x0002,0x80000010,0x80000017};
  book->defs[0x0002f089]->node[16] = (Node) {0x0002,0x80000011,0x80000016};
  book->defs[0x0002f089]->node[17] = (Node) {0x0002,0x80000012,0x80000015};
  book->defs[0x0002f089]->node[18] = (Node) {0x0002,0x80000013,0x80000014};
  book->defs[0x0002f089]->node[19] = (Node) {0x0000,0x00000025,0x00000014};
  book->defs[0x0002f089]->node[20] = (Node) {0x0000,0x40000013,0x00000015};
  book->defs[0x0002f089]->node[21] = (Node) {0x0000,0x40000014,0x00000016};
  book->defs[0x0002f089]->node[22] = (Node) {0x0000,0x40000015,0x00000017};
  book->defs[0x0002f089]->node[23] = (Node) {0x0000,0x40000016,0x00000018};
  book->defs[0x0002f089]->node[24] = (Node) {0x0000,0x40000017,0x00000019};
  book->defs[0x0002f089]->node[25] = (Node) {0x0000,0x40000018,0x0000001a};
  book->defs[0x0002f089]->node[26] = (Node) {0x0000,0x40000019,0x0000001b};
  book->defs[0x0002f089]->node[27] = (Node) {0x0000,0x4000001a,0x0000001c};
  book->defs[0x0002f089]->node[28] = (Node) {0x0000,0x4000001b,0x0000001d};
  book->defs[0x0002f089]->node[29] = (Node) {0x0000,0x4000001c,0x0000001e};
  book->defs[0x0002f089]->node[30] = (Node) {0x0000,0x4000001d,0x0000001f};
  book->defs[0x0002f089]->node[31] = (Node) {0x0000,0x4000001e,0x00000020};
  book->defs[0x0002f089]->node[32] = (Node) {0x0000,0x4000001f,0x00000021};
  book->defs[0x0002f089]->node[33] = (Node) {0x0000,0x40000020,0x00000022};
  book->defs[0x0002f089]->node[34] = (Node) {0x0000,0x40000021,0x00000023};
  book->defs[0x0002f089]->node[35] = (Node) {0x0000,0x40000022,0x00000024};
  book->defs[0x0002f089]->node[36] = (Node) {0x0000,0x40000023,0x40000025};
  book->defs[0x0002f089]->node[37] = (Node) {0x0000,0x00000013,0x40000024};
  // k19
  book->defs[0x0002f08a]           = (Term*) malloc(sizeof(Term));
  book->defs[0x0002f08a]->alen     = 0;
  book->defs[0x0002f08a]->acts     = (Wire*) malloc(0 * sizeof(Wire));
  book->defs[0x0002f08a]->nlen     = 40;
  book->defs[0x0002f08a]->node     = (Node*) malloc(40 * sizeof(Node));
  book->defs[0x0002f08a]->node[ 0] = (Node) {0x0000,0x00000000,0x80000001};
  book->defs[0x0002f08a]->node[ 1] = (Node) {0x0000,0x80000002,0x80000027};
  book->defs[0x0002f08a]->node[ 2] = (Node) {0x0002,0x80000003,0x80000026};
  book->defs[0x0002f08a]->node[ 3] = (Node) {0x0002,0x80000004,0x80000025};
  book->defs[0x0002f08a]->node[ 4] = (Node) {0x0002,0x80000005,0x80000024};
  book->defs[0x0002f08a]->node[ 5] = (Node) {0x0002,0x80000006,0x80000023};
  book->defs[0x0002f08a]->node[ 6] = (Node) {0x0002,0x80000007,0x80000022};
  book->defs[0x0002f08a]->node[ 7] = (Node) {0x0002,0x80000008,0x80000021};
  book->defs[0x0002f08a]->node[ 8] = (Node) {0x0002,0x80000009,0x80000020};
  book->defs[0x0002f08a]->node[ 9] = (Node) {0x0002,0x8000000a,0x8000001f};
  book->defs[0x0002f08a]->node[10] = (Node) {0x0002,0x8000000b,0x8000001e};
  book->defs[0x0002f08a]->node[11] = (Node) {0x0002,0x8000000c,0x8000001d};
  book->defs[0x0002f08a]->node[12] = (Node) {0x0002,0x8000000d,0x8000001c};
  book->defs[0x0002f08a]->node[13] = (Node) {0x0002,0x8000000e,0x8000001b};
  book->defs[0x0002f08a]->node[14] = (Node) {0x0002,0x8000000f,0x8000001a};
  book->defs[0x0002f08a]->node[15] = (Node) {0x0002,0x80000010,0x80000019};
  book->defs[0x0002f08a]->node[16] = (Node) {0x0002,0x80000011,0x80000018};
  book->defs[0x0002f08a]->node[17] = (Node) {0x0002,0x80000012,0x80000017};
  book->defs[0x0002f08a]->node[18] = (Node) {0x0002,0x80000013,0x80000016};
  book->defs[0x0002f08a]->node[19] = (Node) {0x0002,0x80000014,0x80000015};
  book->defs[0x0002f08a]->node[20] = (Node) {0x0000,0x00000027,0x00000015};
  book->defs[0x0002f08a]->node[21] = (Node) {0x0000,0x40000014,0x00000016};
  book->defs[0x0002f08a]->node[22] = (Node) {0x0000,0x40000015,0x00000017};
  book->defs[0x0002f08a]->node[23] = (Node) {0x0000,0x40000016,0x00000018};
  book->defs[0x0002f08a]->node[24] = (Node) {0x0000,0x40000017,0x00000019};
  book->defs[0x0002f08a]->node[25] = (Node) {0x0000,0x40000018,0x0000001a};
  book->defs[0x0002f08a]->node[26] = (Node) {0x0000,0x40000019,0x0000001b};
  book->defs[0x0002f08a]->node[27] = (Node) {0x0000,0x4000001a,0x0000001c};
  book->defs[0x0002f08a]->node[28] = (Node) {0x0000,0x4000001b,0x0000001d};
  book->defs[0x0002f08a]->node[29] = (Node) {0x0000,0x4000001c,0x0000001e};
  book->defs[0x0002f08a]->node[30] = (Node) {0x0000,0x4000001d,0x0000001f};
  book->defs[0x0002f08a]->node[31] = (Node) {0x0000,0x4000001e,0x00000020};
  book->defs[0x0002f08a]->node[32] = (Node) {0x0000,0x4000001f,0x00000021};
  book->defs[0x0002f08a]->node[33] = (Node) {0x0000,0x40000020,0x00000022};
  book->defs[0x0002f08a]->node[34] = (Node) {0x0000,0x40000021,0x00000023};
  book->defs[0x0002f08a]->node[35] = (Node) {0x0000,0x40000022,0x00000024};
  book->defs[0x0002f08a]->node[36] = (Node) {0x0000,0x40000023,0x00000025};
  book->defs[0x0002f08a]->node[37] = (Node) {0x0000,0x40000024,0x00000026};
  book->defs[0x0002f08a]->node[38] = (Node) {0x0000,0x40000025,0x40000027};
  book->defs[0x0002f08a]->node[39] = (Node) {0x0000,0x00000014,0x40000026};
  // k20
  book->defs[0x0002f0c1]           = (Term*) malloc(sizeof(Term));
  book->defs[0x0002f0c1]->alen     = 0;
  book->defs[0x0002f0c1]->acts     = (Wire*) malloc(0 * sizeof(Wire));
  book->defs[0x0002f0c1]->nlen     = 42;
  book->defs[0x0002f0c1]->node     = (Node*) malloc(42 * sizeof(Node));
  book->defs[0x0002f0c1]->node[ 0] = (Node) {0x0000,0x00000000,0x80000001};
  book->defs[0x0002f0c1]->node[ 1] = (Node) {0x0000,0x80000002,0x80000029};
  book->defs[0x0002f0c1]->node[ 2] = (Node) {0x0002,0x80000003,0x80000028};
  book->defs[0x0002f0c1]->node[ 3] = (Node) {0x0002,0x80000004,0x80000027};
  book->defs[0x0002f0c1]->node[ 4] = (Node) {0x0002,0x80000005,0x80000026};
  book->defs[0x0002f0c1]->node[ 5] = (Node) {0x0002,0x80000006,0x80000025};
  book->defs[0x0002f0c1]->node[ 6] = (Node) {0x0002,0x80000007,0x80000024};
  book->defs[0x0002f0c1]->node[ 7] = (Node) {0x0002,0x80000008,0x80000023};
  book->defs[0x0002f0c1]->node[ 8] = (Node) {0x0002,0x80000009,0x80000022};
  book->defs[0x0002f0c1]->node[ 9] = (Node) {0x0002,0x8000000a,0x80000021};
  book->defs[0x0002f0c1]->node[10] = (Node) {0x0002,0x8000000b,0x80000020};
  book->defs[0x0002f0c1]->node[11] = (Node) {0x0002,0x8000000c,0x8000001f};
  book->defs[0x0002f0c1]->node[12] = (Node) {0x0002,0x8000000d,0x8000001e};
  book->defs[0x0002f0c1]->node[13] = (Node) {0x0002,0x8000000e,0x8000001d};
  book->defs[0x0002f0c1]->node[14] = (Node) {0x0002,0x8000000f,0x8000001c};
  book->defs[0x0002f0c1]->node[15] = (Node) {0x0002,0x80000010,0x8000001b};
  book->defs[0x0002f0c1]->node[16] = (Node) {0x0002,0x80000011,0x8000001a};
  book->defs[0x0002f0c1]->node[17] = (Node) {0x0002,0x80000012,0x80000019};
  book->defs[0x0002f0c1]->node[18] = (Node) {0x0002,0x80000013,0x80000018};
  book->defs[0x0002f0c1]->node[19] = (Node) {0x0002,0x80000014,0x80000017};
  book->defs[0x0002f0c1]->node[20] = (Node) {0x0002,0x80000015,0x80000016};
  book->defs[0x0002f0c1]->node[21] = (Node) {0x0000,0x00000029,0x00000016};
  book->defs[0x0002f0c1]->node[22] = (Node) {0x0000,0x40000015,0x00000017};
  book->defs[0x0002f0c1]->node[23] = (Node) {0x0000,0x40000016,0x00000018};
  book->defs[0x0002f0c1]->node[24] = (Node) {0x0000,0x40000017,0x00000019};
  book->defs[0x0002f0c1]->node[25] = (Node) {0x0000,0x40000018,0x0000001a};
  book->defs[0x0002f0c1]->node[26] = (Node) {0x0000,0x40000019,0x0000001b};
  book->defs[0x0002f0c1]->node[27] = (Node) {0x0000,0x4000001a,0x0000001c};
  book->defs[0x0002f0c1]->node[28] = (Node) {0x0000,0x4000001b,0x0000001d};
  book->defs[0x0002f0c1]->node[29] = (Node) {0x0000,0x4000001c,0x0000001e};
  book->defs[0x0002f0c1]->node[30] = (Node) {0x0000,0x4000001d,0x0000001f};
  book->defs[0x0002f0c1]->node[31] = (Node) {0x0000,0x4000001e,0x00000020};
  book->defs[0x0002f0c1]->node[32] = (Node) {0x0000,0x4000001f,0x00000021};
  book->defs[0x0002f0c1]->node[33] = (Node) {0x0000,0x40000020,0x00000022};
  book->defs[0x0002f0c1]->node[34] = (Node) {0x0000,0x40000021,0x00000023};
  book->defs[0x0002f0c1]->node[35] = (Node) {0x0000,0x40000022,0x00000024};
  book->defs[0x0002f0c1]->node[36] = (Node) {0x0000,0x40000023,0x00000025};
  book->defs[0x0002f0c1]->node[37] = (Node) {0x0000,0x40000024,0x00000026};
  book->defs[0x0002f0c1]->node[38] = (Node) {0x0000,0x40000025,0x00000027};
  book->defs[0x0002f0c1]->node[39] = (Node) {0x0000,0x40000026,0x00000028};
  book->defs[0x0002f0c1]->node[40] = (Node) {0x0000,0x40000027,0x40000029};
  book->defs[0x0002f0c1]->node[41] = (Node) {0x0000,0x00000015,0x40000028};
  // k21
  book->defs[0x0002f0c2]           = (Term*) malloc(sizeof(Term));
  book->defs[0x0002f0c2]->alen     = 0;
  book->defs[0x0002f0c2]->acts     = (Wire*) malloc(0 * sizeof(Wire));
  book->defs[0x0002f0c2]->nlen     = 44;
  book->defs[0x0002f0c2]->node     = (Node*) malloc(44 * sizeof(Node));
  book->defs[0x0002f0c2]->node[ 0] = (Node) {0x0000,0x00000000,0x80000001};
  book->defs[0x0002f0c2]->node[ 1] = (Node) {0x0000,0x80000002,0x8000002b};
  book->defs[0x0002f0c2]->node[ 2] = (Node) {0x0002,0x80000003,0x8000002a};
  book->defs[0x0002f0c2]->node[ 3] = (Node) {0x0002,0x80000004,0x80000029};
  book->defs[0x0002f0c2]->node[ 4] = (Node) {0x0002,0x80000005,0x80000028};
  book->defs[0x0002f0c2]->node[ 5] = (Node) {0x0002,0x80000006,0x80000027};
  book->defs[0x0002f0c2]->node[ 6] = (Node) {0x0002,0x80000007,0x80000026};
  book->defs[0x0002f0c2]->node[ 7] = (Node) {0x0002,0x80000008,0x80000025};
  book->defs[0x0002f0c2]->node[ 8] = (Node) {0x0002,0x80000009,0x80000024};
  book->defs[0x0002f0c2]->node[ 9] = (Node) {0x0002,0x8000000a,0x80000023};
  book->defs[0x0002f0c2]->node[10] = (Node) {0x0002,0x8000000b,0x80000022};
  book->defs[0x0002f0c2]->node[11] = (Node) {0x0002,0x8000000c,0x80000021};
  book->defs[0x0002f0c2]->node[12] = (Node) {0x0002,0x8000000d,0x80000020};
  book->defs[0x0002f0c2]->node[13] = (Node) {0x0002,0x8000000e,0x8000001f};
  book->defs[0x0002f0c2]->node[14] = (Node) {0x0002,0x8000000f,0x8000001e};
  book->defs[0x0002f0c2]->node[15] = (Node) {0x0002,0x80000010,0x8000001d};
  book->defs[0x0002f0c2]->node[16] = (Node) {0x0002,0x80000011,0x8000001c};
  book->defs[0x0002f0c2]->node[17] = (Node) {0x0002,0x80000012,0x8000001b};
  book->defs[0x0002f0c2]->node[18] = (Node) {0x0002,0x80000013,0x8000001a};
  book->defs[0x0002f0c2]->node[19] = (Node) {0x0002,0x80000014,0x80000019};
  book->defs[0x0002f0c2]->node[20] = (Node) {0x0002,0x80000015,0x80000018};
  book->defs[0x0002f0c2]->node[21] = (Node) {0x0002,0x80000016,0x80000017};
  book->defs[0x0002f0c2]->node[22] = (Node) {0x0000,0x0000002b,0x00000017};
  book->defs[0x0002f0c2]->node[23] = (Node) {0x0000,0x40000016,0x00000018};
  book->defs[0x0002f0c2]->node[24] = (Node) {0x0000,0x40000017,0x00000019};
  book->defs[0x0002f0c2]->node[25] = (Node) {0x0000,0x40000018,0x0000001a};
  book->defs[0x0002f0c2]->node[26] = (Node) {0x0000,0x40000019,0x0000001b};
  book->defs[0x0002f0c2]->node[27] = (Node) {0x0000,0x4000001a,0x0000001c};
  book->defs[0x0002f0c2]->node[28] = (Node) {0x0000,0x4000001b,0x0000001d};
  book->defs[0x0002f0c2]->node[29] = (Node) {0x0000,0x4000001c,0x0000001e};
  book->defs[0x0002f0c2]->node[30] = (Node) {0x0000,0x4000001d,0x0000001f};
  book->defs[0x0002f0c2]->node[31] = (Node) {0x0000,0x4000001e,0x00000020};
  book->defs[0x0002f0c2]->node[32] = (Node) {0x0000,0x4000001f,0x00000021};
  book->defs[0x0002f0c2]->node[33] = (Node) {0x0000,0x40000020,0x00000022};
  book->defs[0x0002f0c2]->node[34] = (Node) {0x0000,0x40000021,0x00000023};
  book->defs[0x0002f0c2]->node[35] = (Node) {0x0000,0x40000022,0x00000024};
  book->defs[0x0002f0c2]->node[36] = (Node) {0x0000,0x40000023,0x00000025};
  book->defs[0x0002f0c2]->node[37] = (Node) {0x0000,0x40000024,0x00000026};
  book->defs[0x0002f0c2]->node[38] = (Node) {0x0000,0x40000025,0x00000027};
  book->defs[0x0002f0c2]->node[39] = (Node) {0x0000,0x40000026,0x00000028};
  book->defs[0x0002f0c2]->node[40] = (Node) {0x0000,0x40000027,0x00000029};
  book->defs[0x0002f0c2]->node[41] = (Node) {0x0000,0x40000028,0x0000002a};
  book->defs[0x0002f0c2]->node[42] = (Node) {0x0000,0x40000029,0x4000002b};
  book->defs[0x0002f0c2]->node[43] = (Node) {0x0000,0x00000016,0x4000002a};
  // k22
  book->defs[0x0002f0c3]           = (Term*) malloc(sizeof(Term));
  book->defs[0x0002f0c3]->alen     = 0;
  book->defs[0x0002f0c3]->acts     = (Wire*) malloc(0 * sizeof(Wire));
  book->defs[0x0002f0c3]->nlen     = 46;
  book->defs[0x0002f0c3]->node     = (Node*) malloc(46 * sizeof(Node));
  book->defs[0x0002f0c3]->node[ 0] = (Node) {0x0000,0x00000000,0x80000001};
  book->defs[0x0002f0c3]->node[ 1] = (Node) {0x0000,0x80000002,0x8000002d};
  book->defs[0x0002f0c3]->node[ 2] = (Node) {0x0002,0x80000003,0x8000002c};
  book->defs[0x0002f0c3]->node[ 3] = (Node) {0x0002,0x80000004,0x8000002b};
  book->defs[0x0002f0c3]->node[ 4] = (Node) {0x0002,0x80000005,0x8000002a};
  book->defs[0x0002f0c3]->node[ 5] = (Node) {0x0002,0x80000006,0x80000029};
  book->defs[0x0002f0c3]->node[ 6] = (Node) {0x0002,0x80000007,0x80000028};
  book->defs[0x0002f0c3]->node[ 7] = (Node) {0x0002,0x80000008,0x80000027};
  book->defs[0x0002f0c3]->node[ 8] = (Node) {0x0002,0x80000009,0x80000026};
  book->defs[0x0002f0c3]->node[ 9] = (Node) {0x0002,0x8000000a,0x80000025};
  book->defs[0x0002f0c3]->node[10] = (Node) {0x0002,0x8000000b,0x80000024};
  book->defs[0x0002f0c3]->node[11] = (Node) {0x0002,0x8000000c,0x80000023};
  book->defs[0x0002f0c3]->node[12] = (Node) {0x0002,0x8000000d,0x80000022};
  book->defs[0x0002f0c3]->node[13] = (Node) {0x0002,0x8000000e,0x80000021};
  book->defs[0x0002f0c3]->node[14] = (Node) {0x0002,0x8000000f,0x80000020};
  book->defs[0x0002f0c3]->node[15] = (Node) {0x0002,0x80000010,0x8000001f};
  book->defs[0x0002f0c3]->node[16] = (Node) {0x0002,0x80000011,0x8000001e};
  book->defs[0x0002f0c3]->node[17] = (Node) {0x0002,0x80000012,0x8000001d};
  book->defs[0x0002f0c3]->node[18] = (Node) {0x0002,0x80000013,0x8000001c};
  book->defs[0x0002f0c3]->node[19] = (Node) {0x0002,0x80000014,0x8000001b};
  book->defs[0x0002f0c3]->node[20] = (Node) {0x0002,0x80000015,0x8000001a};
  book->defs[0x0002f0c3]->node[21] = (Node) {0x0002,0x80000016,0x80000019};
  book->defs[0x0002f0c3]->node[22] = (Node) {0x0002,0x80000017,0x80000018};
  book->defs[0x0002f0c3]->node[23] = (Node) {0x0000,0x0000002d,0x00000018};
  book->defs[0x0002f0c3]->node[24] = (Node) {0x0000,0x40000017,0x00000019};
  book->defs[0x0002f0c3]->node[25] = (Node) {0x0000,0x40000018,0x0000001a};
  book->defs[0x0002f0c3]->node[26] = (Node) {0x0000,0x40000019,0x0000001b};
  book->defs[0x0002f0c3]->node[27] = (Node) {0x0000,0x4000001a,0x0000001c};
  book->defs[0x0002f0c3]->node[28] = (Node) {0x0000,0x4000001b,0x0000001d};
  book->defs[0x0002f0c3]->node[29] = (Node) {0x0000,0x4000001c,0x0000001e};
  book->defs[0x0002f0c3]->node[30] = (Node) {0x0000,0x4000001d,0x0000001f};
  book->defs[0x0002f0c3]->node[31] = (Node) {0x0000,0x4000001e,0x00000020};
  book->defs[0x0002f0c3]->node[32] = (Node) {0x0000,0x4000001f,0x00000021};
  book->defs[0x0002f0c3]->node[33] = (Node) {0x0000,0x40000020,0x00000022};
  book->defs[0x0002f0c3]->node[34] = (Node) {0x0000,0x40000021,0x00000023};
  book->defs[0x0002f0c3]->node[35] = (Node) {0x0000,0x40000022,0x00000024};
  book->defs[0x0002f0c3]->node[36] = (Node) {0x0000,0x40000023,0x00000025};
  book->defs[0x0002f0c3]->node[37] = (Node) {0x0000,0x40000024,0x00000026};
  book->defs[0x0002f0c3]->node[38] = (Node) {0x0000,0x40000025,0x00000027};
  book->defs[0x0002f0c3]->node[39] = (Node) {0x0000,0x40000026,0x00000028};
  book->defs[0x0002f0c3]->node[40] = (Node) {0x0000,0x40000027,0x00000029};
  book->defs[0x0002f0c3]->node[41] = (Node) {0x0000,0x40000028,0x0000002a};
  book->defs[0x0002f0c3]->node[42] = (Node) {0x0000,0x40000029,0x0000002b};
  book->defs[0x0002f0c3]->node[43] = (Node) {0x0000,0x4000002a,0x0000002c};
  book->defs[0x0002f0c3]->node[44] = (Node) {0x0000,0x4000002b,0x4000002d};
  book->defs[0x0002f0c3]->node[45] = (Node) {0x0000,0x00000017,0x4000002c};
  // k23
  book->defs[0x0002f0c4]           = (Term*) malloc(sizeof(Term));
  book->defs[0x0002f0c4]->alen     = 0;
  book->defs[0x0002f0c4]->acts     = (Wire*) malloc(0 * sizeof(Wire));
  book->defs[0x0002f0c4]->nlen     = 48;
  book->defs[0x0002f0c4]->node     = (Node*) malloc(48 * sizeof(Node));
  book->defs[0x0002f0c4]->node[ 0] = (Node) {0x0000,0x00000000,0x80000001};
  book->defs[0x0002f0c4]->node[ 1] = (Node) {0x0000,0x80000002,0x8000002f};
  book->defs[0x0002f0c4]->node[ 2] = (Node) {0x0002,0x80000003,0x8000002e};
  book->defs[0x0002f0c4]->node[ 3] = (Node) {0x0002,0x80000004,0x8000002d};
  book->defs[0x0002f0c4]->node[ 4] = (Node) {0x0002,0x80000005,0x8000002c};
  book->defs[0x0002f0c4]->node[ 5] = (Node) {0x0002,0x80000006,0x8000002b};
  book->defs[0x0002f0c4]->node[ 6] = (Node) {0x0002,0x80000007,0x8000002a};
  book->defs[0x0002f0c4]->node[ 7] = (Node) {0x0002,0x80000008,0x80000029};
  book->defs[0x0002f0c4]->node[ 8] = (Node) {0x0002,0x80000009,0x80000028};
  book->defs[0x0002f0c4]->node[ 9] = (Node) {0x0002,0x8000000a,0x80000027};
  book->defs[0x0002f0c4]->node[10] = (Node) {0x0002,0x8000000b,0x80000026};
  book->defs[0x0002f0c4]->node[11] = (Node) {0x0002,0x8000000c,0x80000025};
  book->defs[0x0002f0c4]->node[12] = (Node) {0x0002,0x8000000d,0x80000024};
  book->defs[0x0002f0c4]->node[13] = (Node) {0x0002,0x8000000e,0x80000023};
  book->defs[0x0002f0c4]->node[14] = (Node) {0x0002,0x8000000f,0x80000022};
  book->defs[0x0002f0c4]->node[15] = (Node) {0x0002,0x80000010,0x80000021};
  book->defs[0x0002f0c4]->node[16] = (Node) {0x0002,0x80000011,0x80000020};
  book->defs[0x0002f0c4]->node[17] = (Node) {0x0002,0x80000012,0x8000001f};
  book->defs[0x0002f0c4]->node[18] = (Node) {0x0002,0x80000013,0x8000001e};
  book->defs[0x0002f0c4]->node[19] = (Node) {0x0002,0x80000014,0x8000001d};
  book->defs[0x0002f0c4]->node[20] = (Node) {0x0002,0x80000015,0x8000001c};
  book->defs[0x0002f0c4]->node[21] = (Node) {0x0002,0x80000016,0x8000001b};
  book->defs[0x0002f0c4]->node[22] = (Node) {0x0002,0x80000017,0x8000001a};
  book->defs[0x0002f0c4]->node[23] = (Node) {0x0002,0x80000018,0x80000019};
  book->defs[0x0002f0c4]->node[24] = (Node) {0x0000,0x0000002f,0x00000019};
  book->defs[0x0002f0c4]->node[25] = (Node) {0x0000,0x40000018,0x0000001a};
  book->defs[0x0002f0c4]->node[26] = (Node) {0x0000,0x40000019,0x0000001b};
  book->defs[0x0002f0c4]->node[27] = (Node) {0x0000,0x4000001a,0x0000001c};
  book->defs[0x0002f0c4]->node[28] = (Node) {0x0000,0x4000001b,0x0000001d};
  book->defs[0x0002f0c4]->node[29] = (Node) {0x0000,0x4000001c,0x0000001e};
  book->defs[0x0002f0c4]->node[30] = (Node) {0x0000,0x4000001d,0x0000001f};
  book->defs[0x0002f0c4]->node[31] = (Node) {0x0000,0x4000001e,0x00000020};
  book->defs[0x0002f0c4]->node[32] = (Node) {0x0000,0x4000001f,0x00000021};
  book->defs[0x0002f0c4]->node[33] = (Node) {0x0000,0x40000020,0x00000022};
  book->defs[0x0002f0c4]->node[34] = (Node) {0x0000,0x40000021,0x00000023};
  book->defs[0x0002f0c4]->node[35] = (Node) {0x0000,0x40000022,0x00000024};
  book->defs[0x0002f0c4]->node[36] = (Node) {0x0000,0x40000023,0x00000025};
  book->defs[0x0002f0c4]->node[37] = (Node) {0x0000,0x40000024,0x00000026};
  book->defs[0x0002f0c4]->node[38] = (Node) {0x0000,0x40000025,0x00000027};
  book->defs[0x0002f0c4]->node[39] = (Node) {0x0000,0x40000026,0x00000028};
  book->defs[0x0002f0c4]->node[40] = (Node) {0x0000,0x40000027,0x00000029};
  book->defs[0x0002f0c4]->node[41] = (Node) {0x0000,0x40000028,0x0000002a};
  book->defs[0x0002f0c4]->node[42] = (Node) {0x0000,0x40000029,0x0000002b};
  book->defs[0x0002f0c4]->node[43] = (Node) {0x0000,0x4000002a,0x0000002c};
  book->defs[0x0002f0c4]->node[44] = (Node) {0x0000,0x4000002b,0x0000002d};
  book->defs[0x0002f0c4]->node[45] = (Node) {0x0000,0x4000002c,0x0000002e};
  book->defs[0x0002f0c4]->node[46] = (Node) {0x0000,0x4000002d,0x4000002f};
  book->defs[0x0002f0c4]->node[47] = (Node) {0x0000,0x00000018,0x4000002e};
  // k24
  book->defs[0x0002f0c5]           = (Term*) malloc(sizeof(Term));
  book->defs[0x0002f0c5]->alen     = 0;
  book->defs[0x0002f0c5]->acts     = (Wire*) malloc(0 * sizeof(Wire));
  book->defs[0x0002f0c5]->nlen     = 50;
  book->defs[0x0002f0c5]->node     = (Node*) malloc(50 * sizeof(Node));
  book->defs[0x0002f0c5]->node[ 0] = (Node) {0x0000,0x00000000,0x80000001};
  book->defs[0x0002f0c5]->node[ 1] = (Node) {0x0000,0x80000002,0x80000031};
  book->defs[0x0002f0c5]->node[ 2] = (Node) {0x0002,0x80000003,0x80000030};
  book->defs[0x0002f0c5]->node[ 3] = (Node) {0x0002,0x80000004,0x8000002f};
  book->defs[0x0002f0c5]->node[ 4] = (Node) {0x0002,0x80000005,0x8000002e};
  book->defs[0x0002f0c5]->node[ 5] = (Node) {0x0002,0x80000006,0x8000002d};
  book->defs[0x0002f0c5]->node[ 6] = (Node) {0x0002,0x80000007,0x8000002c};
  book->defs[0x0002f0c5]->node[ 7] = (Node) {0x0002,0x80000008,0x8000002b};
  book->defs[0x0002f0c5]->node[ 8] = (Node) {0x0002,0x80000009,0x8000002a};
  book->defs[0x0002f0c5]->node[ 9] = (Node) {0x0002,0x8000000a,0x80000029};
  book->defs[0x0002f0c5]->node[10] = (Node) {0x0002,0x8000000b,0x80000028};
  book->defs[0x0002f0c5]->node[11] = (Node) {0x0002,0x8000000c,0x80000027};
  book->defs[0x0002f0c5]->node[12] = (Node) {0x0002,0x8000000d,0x80000026};
  book->defs[0x0002f0c5]->node[13] = (Node) {0x0002,0x8000000e,0x80000025};
  book->defs[0x0002f0c5]->node[14] = (Node) {0x0002,0x8000000f,0x80000024};
  book->defs[0x0002f0c5]->node[15] = (Node) {0x0002,0x80000010,0x80000023};
  book->defs[0x0002f0c5]->node[16] = (Node) {0x0002,0x80000011,0x80000022};
  book->defs[0x0002f0c5]->node[17] = (Node) {0x0002,0x80000012,0x80000021};
  book->defs[0x0002f0c5]->node[18] = (Node) {0x0002,0x80000013,0x80000020};
  book->defs[0x0002f0c5]->node[19] = (Node) {0x0002,0x80000014,0x8000001f};
  book->defs[0x0002f0c5]->node[20] = (Node) {0x0002,0x80000015,0x8000001e};
  book->defs[0x0002f0c5]->node[21] = (Node) {0x0002,0x80000016,0x8000001d};
  book->defs[0x0002f0c5]->node[22] = (Node) {0x0002,0x80000017,0x8000001c};
  book->defs[0x0002f0c5]->node[23] = (Node) {0x0002,0x80000018,0x8000001b};
  book->defs[0x0002f0c5]->node[24] = (Node) {0x0002,0x80000019,0x8000001a};
  book->defs[0x0002f0c5]->node[25] = (Node) {0x0000,0x00000031,0x0000001a};
  book->defs[0x0002f0c5]->node[26] = (Node) {0x0000,0x40000019,0x0000001b};
  book->defs[0x0002f0c5]->node[27] = (Node) {0x0000,0x4000001a,0x0000001c};
  book->defs[0x0002f0c5]->node[28] = (Node) {0x0000,0x4000001b,0x0000001d};
  book->defs[0x0002f0c5]->node[29] = (Node) {0x0000,0x4000001c,0x0000001e};
  book->defs[0x0002f0c5]->node[30] = (Node) {0x0000,0x4000001d,0x0000001f};
  book->defs[0x0002f0c5]->node[31] = (Node) {0x0000,0x4000001e,0x00000020};
  book->defs[0x0002f0c5]->node[32] = (Node) {0x0000,0x4000001f,0x00000021};
  book->defs[0x0002f0c5]->node[33] = (Node) {0x0000,0x40000020,0x00000022};
  book->defs[0x0002f0c5]->node[34] = (Node) {0x0000,0x40000021,0x00000023};
  book->defs[0x0002f0c5]->node[35] = (Node) {0x0000,0x40000022,0x00000024};
  book->defs[0x0002f0c5]->node[36] = (Node) {0x0000,0x40000023,0x00000025};
  book->defs[0x0002f0c5]->node[37] = (Node) {0x0000,0x40000024,0x00000026};
  book->defs[0x0002f0c5]->node[38] = (Node) {0x0000,0x40000025,0x00000027};
  book->defs[0x0002f0c5]->node[39] = (Node) {0x0000,0x40000026,0x00000028};
  book->defs[0x0002f0c5]->node[40] = (Node) {0x0000,0x40000027,0x00000029};
  book->defs[0x0002f0c5]->node[41] = (Node) {0x0000,0x40000028,0x0000002a};
  book->defs[0x0002f0c5]->node[42] = (Node) {0x0000,0x40000029,0x0000002b};
  book->defs[0x0002f0c5]->node[43] = (Node) {0x0000,0x4000002a,0x0000002c};
  book->defs[0x0002f0c5]->node[44] = (Node) {0x0000,0x4000002b,0x0000002d};
  book->defs[0x0002f0c5]->node[45] = (Node) {0x0000,0x4000002c,0x0000002e};
  book->defs[0x0002f0c5]->node[46] = (Node) {0x0000,0x4000002d,0x0000002f};
  book->defs[0x0002f0c5]->node[47] = (Node) {0x0000,0x4000002e,0x00000030};
  book->defs[0x0002f0c5]->node[48] = (Node) {0x0000,0x4000002f,0x40000031};
  book->defs[0x0002f0c5]->node[49] = (Node) {0x0000,0x00000019,0x40000030};
  // low
  book->defs[0x00030cfb]           = (Term*) malloc(sizeof(Term));
  book->defs[0x00030cfb]->alen     = 0;
  book->defs[0x00030cfb]->acts     = (Wire*) malloc(0 * sizeof(Wire));
  book->defs[0x00030cfb]->nlen     = 5;
  book->defs[0x00030cfb]->node     = (Node*) malloc(5 * sizeof(Node));
  book->defs[0x00030cfb]->node[ 0] = (Node) {0x0000,0x00000000,0x80000001};
  book->defs[0x00030cfb]->node[ 1] = (Node) {0x0000,0x80000002,0x40000004};
  book->defs[0x00030cfb]->node[ 2] = (Node) {0x0000,0xc0c33ed9,0x80000003};
  book->defs[0x00030cfb]->node[ 3] = (Node) {0x0000,0xc0c33ed3,0x80000004};
  book->defs[0x00030cfb]->node[ 4] = (Node) {0x0000,0xc000000f,0x40000001};
  // nid
  book->defs[0x00032b68]           = (Term*) malloc(sizeof(Term));
  book->defs[0x00032b68]->alen     = 0;
  book->defs[0x00032b68]->acts     = (Wire*) malloc(0 * sizeof(Wire));
  book->defs[0x00032b68]->nlen     = 4;
  book->defs[0x00032b68]->node     = (Node*) malloc(4 * sizeof(Node));
  book->defs[0x00032b68]->node[ 0] = (Node) {0x0000,0x00000000,0x80000001};
  book->defs[0x00032b68]->node[ 1] = (Node) {0x0000,0x80000002,0x40000003};
  book->defs[0x00032b68]->node[ 2] = (Node) {0x0000,0xc0cada1d,0x80000003};
  book->defs[0x00032b68]->node[ 3] = (Node) {0x0000,0xc0000024,0x40000001};
  // not
  book->defs[0x00032cf8]           = (Term*) malloc(sizeof(Term));
  book->defs[0x00032cf8]->alen     = 0;
  book->defs[0x00032cf8]->acts     = (Wire*) malloc(0 * sizeof(Wire));
  book->defs[0x00032cf8]->nlen     = 6;
  book->defs[0x00032cf8]->node     = (Node*) malloc(6 * sizeof(Node));
  book->defs[0x00032cf8]->node[ 0] = (Node) {0x0000,0x00000000,0x80000001};
  book->defs[0x00032cf8]->node[ 1] = (Node) {0x0000,0x80000002,0x80000004};
  book->defs[0x00032cf8]->node[ 2] = (Node) {0x0000,0x00000005,0x80000003};
  book->defs[0x00032cf8]->node[ 3] = (Node) {0x0000,0x00000004,0x40000005};
  book->defs[0x00032cf8]->node[ 4] = (Node) {0x0000,0x00000003,0x80000005};
  book->defs[0x00032cf8]->node[ 5] = (Node) {0x0000,0x00000002,0x40000003};
  // run
  book->defs[0x00036e72]           = (Term*) malloc(sizeof(Term));
  book->defs[0x00036e72]->alen     = 0;
  book->defs[0x00036e72]->acts     = (Wire*) malloc(0 * sizeof(Wire));
  book->defs[0x00036e72]->nlen     = 5;
  book->defs[0x00036e72]->node     = (Node*) malloc(5 * sizeof(Node));
  book->defs[0x00036e72]->node[ 0] = (Node) {0x0000,0x00000000,0x80000001};
  book->defs[0x00036e72]->node[ 1] = (Node) {0x0000,0x80000002,0x40000004};
  book->defs[0x00036e72]->node[ 2] = (Node) {0x0000,0xc0db9c99,0x80000003};
  book->defs[0x00036e72]->node[ 3] = (Node) {0x0000,0xc0db9c93,0x80000004};
  book->defs[0x00036e72]->node[ 4] = (Node) {0x0000,0xc000000f,0x40000001};
  // brnS
  book->defs[0x009b6c9d]           = (Term*) malloc(sizeof(Term));
  book->defs[0x009b6c9d]->alen     = 2;
  book->defs[0x009b6c9d]->acts     = (Wire*) malloc(2 * sizeof(Wire));
  book->defs[0x009b6c9d]->acts[ 0] = (Wire) {0xc0026db2,0x80000004};
  book->defs[0x009b6c9d]->acts[ 1] = (Wire) {0xc0026db2,0x80000005};
  book->defs[0x009b6c9d]->nlen     = 6;
  book->defs[0x009b6c9d]->node     = (Node*) malloc(6 * sizeof(Node));
  book->defs[0x009b6c9d]->node[ 0] = (Node) {0x0000,0x00000000,0x80000001};
  book->defs[0x009b6c9d]->node[ 1] = (Node) {0x0000,0x80000002,0x80000003};
  book->defs[0x009b6c9d]->node[ 2] = (Node) {0x0001,0x00000004,0x00000005};
  book->defs[0x009b6c9d]->node[ 3] = (Node) {0x0000,0x40000004,0x40000005};
  book->defs[0x009b6c9d]->node[ 4] = (Node) {0x0000,0x00000002,0x00000003};
  book->defs[0x009b6c9d]->node[ 5] = (Node) {0x0000,0x40000002,0x40000003};
  // brnZ
  book->defs[0x009b6ca4]           = (Term*) malloc(sizeof(Term));
  book->defs[0x009b6ca4]->alen     = 2;
  book->defs[0x009b6ca4]->acts     = (Wire*) malloc(2 * sizeof(Wire));
  book->defs[0x009b6ca4]->acts[ 0] = (Wire) {0xc0036e72,0x80000001};
  book->defs[0x009b6ca4]->acts[ 1] = (Wire) {0xc0027081,0x80000002};
  book->defs[0x009b6ca4]->nlen     = 4;
  book->defs[0x009b6ca4]->node     = (Node*) malloc(4 * sizeof(Node));
  book->defs[0x009b6ca4]->node[ 0] = (Node) {0x0000,0x00000000,0x40000001};
  book->defs[0x009b6ca4]->node[ 1] = (Node) {0x0000,0x40000003,0x40000000};
  book->defs[0x009b6ca4]->node[ 2] = (Node) {0x0000,0xc0000013,0x80000003};
  book->defs[0x009b6ca4]->node[ 3] = (Node) {0x0000,0xc000000f,0x00000001};
  // decI
  book->defs[0x00a299d3]           = (Term*) malloc(sizeof(Term));
  book->defs[0x00a299d3]->alen     = 1;
  book->defs[0x00a299d3]->acts     = (Wire*) malloc(1 * sizeof(Wire));
  book->defs[0x00a299d3]->acts[ 0] = (Wire) {0xc0030cfb,0x80000002};
  book->defs[0x00a299d3]->nlen     = 3;
  book->defs[0x00a299d3]->node     = (Node*) malloc(3 * sizeof(Node));
  book->defs[0x00a299d3]->node[ 0] = (Node) {0x0000,0x00000000,0x80000001};
  book->defs[0x00a299d3]->node[ 1] = (Node) {0x0000,0x00000002,0x40000002};
  book->defs[0x00a299d3]->node[ 2] = (Node) {0x0000,0x00000001,0x40000001};
  // decO
  book->defs[0x00a299d9]           = (Term*) malloc(sizeof(Term));
  book->defs[0x00a299d9]->alen     = 2;
  book->defs[0x00a299d9]->acts     = (Wire*) malloc(2 * sizeof(Wire));
  book->defs[0x00a299d9]->acts[ 0] = (Wire) {0xc0000013,0x80000002};
  book->defs[0x00a299d9]->acts[ 1] = (Wire) {0xc0028a67,0x80000003};
  book->defs[0x00a299d9]->nlen     = 4;
  book->defs[0x00a299d9]->node     = (Node*) malloc(4 * sizeof(Node));
  book->defs[0x00a299d9]->node[ 0] = (Node) {0x0000,0x00000000,0x80000001};
  book->defs[0x00a299d9]->node[ 1] = (Node) {0x0000,0x00000003,0x40000002};
  book->defs[0x00a299d9]->node[ 2] = (Node) {0x0000,0x40000003,0x40000001};
  book->defs[0x00a299d9]->node[ 3] = (Node) {0x0000,0x00000001,0x00000002};
  // lowI
  book->defs[0x00c33ed3]           = (Term*) malloc(sizeof(Term));
  book->defs[0x00c33ed3]->alen     = 2;
  book->defs[0x00c33ed3]->acts     = (Wire*) malloc(2 * sizeof(Wire));
  book->defs[0x00c33ed3]->acts[ 0] = (Wire) {0xc0000013,0x80000002};
  book->defs[0x00c33ed3]->acts[ 1] = (Wire) {0xc0000019,0x80000003};
  book->defs[0x00c33ed3]->nlen     = 4;
  book->defs[0x00c33ed3]->node     = (Node*) malloc(4 * sizeof(Node));
  book->defs[0x00c33ed3]->node[ 0] = (Node) {0x0000,0x00000000,0x80000001};
  book->defs[0x00c33ed3]->node[ 1] = (Node) {0x0000,0x00000002,0x40000003};
  book->defs[0x00c33ed3]->node[ 2] = (Node) {0x0000,0x00000001,0x00000003};
  book->defs[0x00c33ed3]->node[ 3] = (Node) {0x0000,0x40000002,0x40000001};
  // lowO
  book->defs[0x00c33ed9]           = (Term*) malloc(sizeof(Term));
  book->defs[0x00c33ed9]->alen     = 2;
  book->defs[0x00c33ed9]->acts     = (Wire*) malloc(2 * sizeof(Wire));
  book->defs[0x00c33ed9]->acts[ 0] = (Wire) {0xc0000019,0x80000002};
  book->defs[0x00c33ed9]->acts[ 1] = (Wire) {0xc0000019,0x80000003};
  book->defs[0x00c33ed9]->nlen     = 4;
  book->defs[0x00c33ed9]->node     = (Node*) malloc(4 * sizeof(Node));
  book->defs[0x00c33ed9]->node[ 0] = (Node) {0x0000,0x00000000,0x80000001};
  book->defs[0x00c33ed9]->node[ 1] = (Node) {0x0000,0x00000002,0x40000003};
  book->defs[0x00c33ed9]->node[ 2] = (Node) {0x0000,0x00000001,0x00000003};
  book->defs[0x00c33ed9]->node[ 3] = (Node) {0x0000,0x40000002,0x40000001};
  // nidS
  book->defs[0x00cada1d]           = (Term*) malloc(sizeof(Term));
  book->defs[0x00cada1d]->alen     = 2;
  book->defs[0x00cada1d]->acts     = (Wire*) malloc(2 * sizeof(Wire));
  book->defs[0x00cada1d]->acts[ 0] = (Wire) {0xc000001d,0x80000002};
  book->defs[0x00cada1d]->acts[ 1] = (Wire) {0xc0032b68,0x80000003};
  book->defs[0x00cada1d]->nlen     = 4;
  book->defs[0x00cada1d]->node     = (Node*) malloc(4 * sizeof(Node));
  book->defs[0x00cada1d]->node[ 0] = (Node) {0x0000,0x00000000,0x80000001};
  book->defs[0x00cada1d]->node[ 1] = (Node) {0x0000,0x00000003,0x40000002};
  book->defs[0x00cada1d]->node[ 2] = (Node) {0x0000,0x40000003,0x40000001};
  book->defs[0x00cada1d]->node[ 3] = (Node) {0x0000,0x00000001,0x00000002};
  // runI
  book->defs[0x00db9c93]           = (Term*) malloc(sizeof(Term));
  book->defs[0x00db9c93]->alen     = 3;
  book->defs[0x00db9c93]->acts     = (Wire*) malloc(3 * sizeof(Wire));
  book->defs[0x00db9c93]->acts[ 0] = (Wire) {0xc0036e72,0x80000002};
  book->defs[0x00db9c93]->acts[ 1] = (Wire) {0xc0028a67,0x80000003};
  book->defs[0x00db9c93]->acts[ 2] = (Wire) {0xc0000013,0x80000004};
  book->defs[0x00db9c93]->nlen     = 5;
  book->defs[0x00db9c93]->node     = (Node*) malloc(5 * sizeof(Node));
  book->defs[0x00db9c93]->node[ 0] = (Node) {0x0000,0x00000000,0x80000001};
  book->defs[0x00db9c93]->node[ 1] = (Node) {0x0000,0x00000004,0x40000002};
  book->defs[0x00db9c93]->node[ 2] = (Node) {0x0000,0x40000003,0x40000001};
  book->defs[0x00db9c93]->node[ 3] = (Node) {0x0000,0x40000004,0x00000002};
  book->defs[0x00db9c93]->node[ 4] = (Node) {0x0000,0x00000001,0x00000003};
  // runO
  book->defs[0x00db9c99]           = (Term*) malloc(sizeof(Term));
  book->defs[0x00db9c99]->alen     = 3;
  book->defs[0x00db9c99]->acts     = (Wire*) malloc(3 * sizeof(Wire));
  book->defs[0x00db9c99]->acts[ 0] = (Wire) {0xc0036e72,0x80000002};
  book->defs[0x00db9c99]->acts[ 1] = (Wire) {0xc0028a67,0x80000003};
  book->defs[0x00db9c99]->acts[ 2] = (Wire) {0xc0000019,0x80000004};
  book->defs[0x00db9c99]->nlen     = 5;
  book->defs[0x00db9c99]->node     = (Node*) malloc(5 * sizeof(Node));
  book->defs[0x00db9c99]->node[ 0] = (Node) {0x0000,0x00000000,0x80000001};
  book->defs[0x00db9c99]->node[ 1] = (Node) {0x0000,0x00000004,0x40000002};
  book->defs[0x00db9c99]->node[ 2] = (Node) {0x0000,0x40000003,0x40000001};
  book->defs[0x00db9c99]->node[ 3] = (Node) {0x0000,0x40000004,0x00000002};
  book->defs[0x00db9c99]->node[ 4] = (Node) {0x0000,0x00000001,0x00000003};
}

__host__ void boot(Net* net, Book* book, u32 id) {
  net->blen = book->defs[id]->alen;
  for (u32 i = 0; i < book->defs[id]->alen; ++i) {
    net->bags[i] = book->defs[id]->acts[i];
    printf("bag [%x] %x %x\n", i, book->defs[id]->acts[i].lft, book->defs[id]->acts[i].rgt);
  }
  for (u32 i = 0; i < book->defs[id]->nlen; ++i) {
    net->node[i] = book->defs[id]->node[i];
    printf("nod [%x] %x %x\n", i, book->defs[id]->node[i].ports[0], book->defs[id]->node[i].ports[1]);
  }
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

  // Allocates the initial book on device
  Book* h_book = mkbook();
  populate(h_book);

  // Boots the net with an initial term
  boot(h_net, h_book, 0x00029f04);

  // Prints the initial net
  printf("\nINPUT\n=====\n\n");
  print_net(h_net);

  // Sends the net from host to device
  Net* d_net = net_to_device(h_net);

  // Sends the book from host to device
  Book* d_book = book_to_device(h_book);
  Book* H_book = book_to_host(d_book);

  // Gets start time
  struct timespec start, end;
  u32 rwts;
  clock_gettime(CLOCK_MONOTONIC_RAW, &start);

  // Normalizes
  u32 blocks = 1;
  //do_reduce(d_net, d_book, &blocks);
  //do_reduce(d_net, d_book, &blocks);
  //do_reduce(d_net, d_book, &blocks);
  //do_reduce(d_net, d_book, &blocks);
  //do_reduce(d_net, d_book, &blocks);
  //do_reduce(d_net, d_book, &blocks);
  //do_reduce(d_net, d_book, &blocks);
  //do_reduce(d_net, d_book, &blocks);
  do_reduce_all(d_net, d_book);
  //do_global_expand(d_net, d_book, 14);
  //do_reduce_all(d_net, d_book);

  cudaMemcpy(&rwts, &(d_net->rwts), sizeof(u32), cudaMemcpyDeviceToHost);
  cudaDeviceSynchronize();

  // Prints stats
  clock_gettime(CLOCK_MONOTONIC_RAW, &end);
  uint64_t delta_time = (end.tv_sec - start.tv_sec) * 1000 + (end.tv_nsec - start.tv_nsec) / 1000000;
  printf("time: %llu ms\n", delta_time);
  printf("rwts: %llu\n", rwts);

  // Reads the normalized net from device to host
  Net* norm = net_to_host(d_net);

  // Prints the normal form (raw data)
  printf("\nNORMAL | rwts=%d blen=%d\n======\n\n", norm->rwts, norm->blen);
  //print_net(norm);
  //print_tree(norm, mkvrr());

  // ----
  
  // Free device memory
  //net_free_on_device(d_net);
  //book_free_on_device(b_book);

  // Free host memory
  //net_free_on_host(h_net);
  //book_free_on_host(h_book);
  //net_free_on_host(norm);

  return 0;
}
