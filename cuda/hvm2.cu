#include <stdarg.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>

typedef uint8_t u8;
typedef uint32_t u32;
typedef uint64_t u64;
typedef unsigned long long int a64;

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

// Maximum number of defs in a book
const u32 MAX_DEFS = 1 << 24;

typedef struct {
  Ptr   root;
  u32   alen;
  Wire* acts;
  u32   nlen;
  Node* node;
  u32*  locs;
} Term;

// A book
typedef struct {
  Term** defs;
} Book;

// An interaction net 
typedef struct {
  Ptr   root; // root wire
  u32   blen; // total bag length (redex count)
  Wire* bags; // redex bags (active pairs)
  Node* node; // memory buffer with all nodes
  u32*  gidx; // aux buffer used on scatter fns
  Wire* gmov; // aux buffer used on scatter fns
  u32   pbks; // last blocks count used
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
  printf(tag);
  printf("\n");
}

__device__ __host__ bool dbug(u32* K, const char* tag) {
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
__host__ __device__ u32 div(u32 a, u32 b) {
  return (a + b - 1) / b;
}

// Pseudorandom Number Generator
__host__ __device__ u32 rng(u32 a) {
  return a * 214013 + 2531011;
}

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
  u32 fail = 0;
  while (true) {
    //dbug(&K, "alloc");
    u32  idx = (worker->aloc * 4 + worker->frac) % NODE_SIZE;
    a64* ref = &((a64*)net->node)[idx];
    u64  got = atomicCAS(ref, 0, ((u64)NEO << 32) | ((u64)NEO)); // Wire{NEO,NEO}
    if (got == 0) {
      //printf("[%d] alloc at %d\n", worker->gid, idx);
      return idx;
    } else {
      //worker->aloc = ++fail % 16 == 0 ? rng(worker->aloc) : worker->aloc + 1;
      worker->aloc = (worker->aloc + 1) % NODE_SIZE;
      if (++fail > 256) {
        printf("[%d] can't alloc\n", worker->gid);
      }
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

// Atomically links the node in 'nod_ref' towards 'dir_ptr'
// - If target is a redirection => clear it and move forwards
// - If target is a variable    => pass the node into it and halt
// - If target is a node        => form an active pair and halt
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
  //printf("[%d on %d] scatter 0 | redex_index=%d block_length=[%d,%d,%d,%d,...]\n", gid, bid, redex_indices[tid], *block_length, *(block_length+1), *(block_length+2), *(block_length+3));
}

// Computes block indices on groups (and group lengths)
__global__ void global_scatter_prepare_1(Net* net) {
  u32 tid = threadIdx.x;
  u32 bid = blockIdx.x;
  u32 gid = bid * blockDim.x + tid;
  u32* block_indices = net->gidx + BLOCK_SIZE * BLOCK_SIZE * BLOCK_SIZE + BLOCK_SIZE * bid;
  u32* group_length  = net->gidx + BLOCK_SIZE * BLOCK_SIZE * BLOCK_SIZE + BLOCK_SIZE * BLOCK_SIZE + bid;
  *group_length      = scansum(block_indices);
  //printf("[%d on %d] scatter 1 | block_index=%d group_length=%d\n", gid, bid, block_indices[tid], *group_length);
}

// Computes group indices on bag (and bag length)
__global__ void global_scatter_prepare_2(Net* net) {
  u32* group_indices = net->gidx + BLOCK_SIZE * BLOCK_SIZE * BLOCK_SIZE + BLOCK_SIZE * BLOCK_SIZE;
  u32* total_length  = &net->blen; __syncthreads();
  *total_length      = scansum(group_indices);
  //printf("[%d] scatter 2 | group_index=%d total_length=%d\n", threadIdx.x, group_indices[threadIdx.x], *total_length);
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
  //printf("[%d] clean %d %d %d\n", gid, BLOCK_SIZE * bid + tid, BLOCK_SIZE * BLOCK_SIZE * BLOCK_SIZE + bid, BLOCK_SIZE * BLOCK_SIZE * BLOCK_SIZE + BLOCK_SIZE * BLOCK_SIZE + (bid / BLOCK_SIZE));
}

__host__ Net* net_to_host(Net* device_net);

// Performs a global scatter
void do_global_scatter(Net* net, u32 prev_blocks, u32 next_blocks) {
  global_scatter_prepare_0<<<prev_blocks, BLOCK_SIZE>>>(net);
  global_scatter_prepare_1<<<div(prev_blocks, BLOCK_SIZE), BLOCK_SIZE>>>(net);
  global_scatter_prepare_2<<<div(prev_blocks, BLOCK_SIZE * BLOCK_SIZE), BLOCK_SIZE>>>(net); // always == 1
  global_scatter_take<<<prev_blocks, BLOCK_SIZE>>>(net, next_blocks);
  global_scatter_move<<<prev_blocks, BLOCK_SIZE>>>(net, next_blocks);
  global_scatter_cleanup<<<prev_blocks, BLOCK_SIZE>>>(net);
}

// Interactions
// ------------

__device__ Ptr adjust(Worker* worker, Ptr ptr, u32* locs) {
  //printf("[%d] adjust %d | %d to %x\n", worker->gid, tag(ptr) >= VR1, val(ptr), tag(ptr) >= VR1 ? locs[val(ptr)] : val(ptr));
  return mkptr(tag(ptr), tag(ptr) >= VR1 ? locs[val(ptr)] : val(ptr));
}

__device__ void deref(Net* net, Worker* worker, Term* term, Ptr* dref) {
  // Allocates needed space
  if (term != NULL) {
    for (u32 i = 0; i < div(term->nlen, (u32)4); ++i) {
      u32 j = i * 4 + worker->frac;
      if (j < term->nlen) {
        term->locs[j] = alloc(worker, net);
      }
    }
  }
  __syncwarp();
      
  // Loads dereferenced nodes, adjusted
  if (term != NULL) {
    for (u32 i = 0; i < div(term->nlen, (u32)4); ++i) {
      u32 j = i * 4 + worker->frac;
      if (j < term->nlen) {
        Node node = term->node[j];
        net->node[term->locs[j]].ports[P1] = adjust(worker, node.ports[P1], term->locs);
        net->node[term->locs[j]].ports[P2] = adjust(worker, node.ports[P2], term->locs);
      }
    }
  }

  // Loads dereferenced redexes, adjusted
  if (term != NULL && worker->frac < term->alen) {
    Wire wire = term->acts[worker->frac];
    wire.lft = adjust(worker, wire.lft, term->locs);
    wire.rgt = adjust(worker, wire.rgt, term->locs);
    put_redex(worker, wire.lft, wire.rgt);
  }

  // Loads dereferenced root, adjusted
  if (term != NULL) {
    *dref = adjust(worker, term->root, term->locs);
  }
  __syncwarp();
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
  __shared__ u32 XLOC[BLOCK_SIZE]; // aux arr for clone locs

  // Initializes local vars
  Worker worker;
  worker.tid  = threadIdx.x;
  worker.bid  = blockIdx.x;
  worker.gid  = worker.bid * blockDim.x + worker.tid;
  worker.aloc = rng(clock() * (worker.gid + 1));
  worker.rwts = 0;
  worker.frac = worker.tid % 4;
  worker.port = worker.tid % 2;
  worker.bag  = (net->bags + worker.gid / UNIT_SIZE * UNIT_SIZE);

  // Scatters redexes
  for (u32 repeat = 0; repeat < 8; ++repeat) {
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
    __syncwarp();
    if (wlen == 1) {
      worker.bag[widx] = (Wire){0,0};
    }

    // Reads redex ptrs
    worker.a_ptr = worker.frac <= A2 ? wire.lft : wire.rgt;
    worker.b_ptr = worker.frac <= A2 ? wire.rgt : wire.lft;

    // Dereferences term
    Ptr*  dptr = tag(worker.a_ptr) == REF ? &worker.a_ptr : tag(worker.b_ptr) == REF ? &worker.b_ptr : NULL;
    Term* term = dptr != NULL ? book->defs[val(*dptr)] : NULL;
    deref(net, &worker, term, dptr);

    // Checks rewrite type
    bool wait = term != NULL && term->alen > 0;
    bool rdex = !wait && wlen == 1;
    bool anni = rdex && tag(worker.a_ptr) == tag(worker.b_ptr);
    bool comm = rdex && tag(worker.a_ptr) != tag(worker.b_ptr);

    // Local variables
    Ptr *ak_ref; // ref to our aux port
    Ptr *bk_ref; // ref to other aux port
    Ptr  ak_ptr; // val of our aux port
    u32  xk_loc; // loc of ptr to send to other side
    Ptr  xk_ptr; // val of ptr to send to other side
    u32  y0_idx; // idx of other clone idx

    //if (worker.gid < 32) {
      //printf("[%d] A\n", worker.gid);
    //}

    // Prints message
    if (rdex && worker.frac == A1) {
      //printf("[%04X] redex: %8X ~ %8X | %d\n", worker.gid, worker.a_ptr, worker.b_ptr, comm ? 1 : 0);
      worker.rwts += 1;
    }

    //if (worker.gid < 32) {
      //printf("[%d] B\n", worker.gid);
    //}

    // If we dereferenced redexes, we can't reduce on this step
    if (wait) {
      put_redex(&worker, worker.a_ptr, worker.b_ptr);
    }

    // Gets relevant ptrs and refs
    if (rdex && (comm || anni)) {
      ak_ref = at(net, val(worker.a_ptr), worker.port);
      bk_ref = at(net, val(worker.b_ptr), worker.port);
      ak_ptr = take(ak_ref);
    }

    // If anni, send a redirection
    if (rdex && anni) {
      xk_ptr = redir(ak_ptr); // redirection ptr to send
    }

    //if (worker.gid < 32) {
      //printf("[%d] C\n", worker.gid);
    //}

    // If comm, send a clone
    if (rdex && comm) {
      xk_loc = alloc(&worker, net); // alloc a clone
      xk_ptr = mkptr(tag(worker.a_ptr),xk_loc); // cloned node ptr to send
      XLOC[worker.tid] = xk_loc; // send cloned index to other threads
    }

    // Receive cloned indices from local threads
    __syncwarp();

    //if (worker.gid < 32) {
      //printf("[%d] D\n", worker.gid);
    //}

    // If comm, create inner wires between clones
    if (rdex && comm) {
      const u32 ADD[4] = {2, 1, -2, -3}; // deltas to get the other clone index
      const u32 VRK    = worker.port == P1 ? VR1 : VR2; // type of inner wire var
      replace(10, at(net, xk_loc, P1), NEO, mkptr(VRK, XLOC[worker.tid + ADD[worker.frac] + 0]));
      replace(20, at(net, xk_loc, P2), NEO, mkptr(VRK, XLOC[worker.tid + ADD[worker.frac] + 1]));
    }
    __syncwarp();

    // Send ptr to other side
    if (rdex && (comm || anni)) {
      replace(30, bk_ref, BSY, xk_ptr);
    }

    //if (worker.gid < 32) {
      //printf("[%d] E\n", worker.gid);
    //}

    // If anni and we sent a NOD, link the node there, towards our port
    // If comm and we have a VAR, link the clone here, towards that var
    if (rdex && (anni && !var(ak_ptr) || comm && var(ak_ptr))) {
      u32  RDK  = worker.port == P1 ? RD1 : RD2;
      Ptr *self = comm ? ak_ref        : bk_ref;
      Ptr  targ = comm ? redir(ak_ptr) : mkptr(RDK, val(worker.a_ptr)); 
      link(&worker, net, self, targ);
    }

    //if (ref0 || ref1) {
      //printf("[%d] E2 %d %d\n", worker.gid, rdex, (comm && !var(ak_ptr)));
    //}

    // If comm and we have a NOD, form an active pair with the clone we got
    if (rdex && (comm && !var(ak_ptr))) {
      put_redex(&worker, ak_ptr, take(ak_ref));
      atomicCAS((u32*)ak_ref, BSY, 0);
    }

    //if (worker.gid < 32) {
      //printf("[%d] F\n", worker.gid);
    //}

    //if (ref0 || ref1) {
      //printf("[%d] E3 %d %d\n", worker.gid);
    //}

    __syncwarp();

    //if (worker.gid < 32) {
      //printf("[%d] G\n", worker.gid);
    //}

    //if (ref0 || ref1) {
      //printf("[%d] E4 %d %d\n", worker.gid);
    //}

    //if (ref0 || ref1) {
      //printf("[%d] F\n", worker.gid);
    //}

  }

  //local_scatter(net);

  // When the work ends, sum stats
  if (worker.rwts > 0 && worker.frac == A1) {
    atomicAdd(&net->rwts, worker.rwts);
  }
}

void do_global_rewrite(Net* net, Book* book, u32 blocks) {
  global_rewrite<<<blocks, BLOCK_SIZE>>>(net, book, blocks);
}

// Reduce
// ------

// Performs a global rewrite step.
u32 do_reduce(Net* net, Book* book) {
  // Gets the total number of redexes
  u32 bag_length;
  u32 prev_blocks;
  cudaMemcpy(&bag_length, &(net->blen), sizeof(u32), cudaMemcpyDeviceToHost);
  cudaMemcpy(&prev_blocks, &(net->pbks), sizeof(u32), cudaMemcpyDeviceToHost);

  // Blocks get stuck when they're 1/4 full.
  // We give them room to grow 2x before that.
  u32 next_blocks = div(bag_length, BLOCK_SIZE / 8);
  if (next_blocks > BLOCK_SIZE * BLOCK_SIZE) {
    next_blocks = BLOCK_SIZE * BLOCK_SIZE;
  }

  // Prints debug message
  printf(">> reducing %d redexes with %d blocks\n", bag_length, next_blocks);

  // Stores next_blocks on net.pbks
  cudaMemcpy(&(net->pbks), &next_blocks, sizeof(u32), cudaMemcpyHostToDevice);
  
  // Scatters redexes evenly
  do_global_scatter(net, prev_blocks, next_blocks);

  // Performs global parallel rewrite
  do_global_rewrite(net, book, next_blocks);

  return bag_length;
}

void do_normalize(Net* net, Book* book) {
  u32 count = 0;
  while (do_reduce(net, book) != 0 && ++count < 10) {};
}

// Host<->Device
// -------------

__host__ Net* mknet() {
  Net* net  = (Net*)malloc(sizeof(Net));
  net->root = mkptr(NIL, 0);
  net->rwts = 0;
  net->pbks = 0;
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
  u32*  device_locs;

  cudaMalloc((void**)&device_term, sizeof(Term));
  cudaMalloc((void**)&device_acts, host_term->alen * sizeof(Wire));
  cudaMalloc((void**)&device_node, host_term->nlen * sizeof(Node));
  cudaMalloc((void**)&device_locs, host_term->nlen * sizeof(u32));

  // Copy the host data to the device memory
  cudaMemcpy(device_acts, host_term->acts, host_term->alen * sizeof(Wire), cudaMemcpyHostToDevice);
  cudaMemcpy(device_node, host_term->node, host_term->nlen * sizeof(Node), cudaMemcpyHostToDevice);
  cudaMemcpy(device_locs, host_term->locs, host_term->nlen * sizeof(u32),  cudaMemcpyHostToDevice);

  // Create a temporary host Term object with device pointers
  Term temp_term = *host_term;
  temp_term.acts = device_acts;
  temp_term.node = device_node;
  temp_term.locs = device_locs;

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
  host_term->locs = (u32*) malloc(host_term->nlen * sizeof(u32));

  // Retrieve the device pointers for data
  Wire* device_acts;
  Node* device_node;
  u32*  device_locs;
  cudaMemcpy(&device_acts, &(device_term->acts), sizeof(Wire*), cudaMemcpyDeviceToHost);
  cudaMemcpy(&device_node, &(device_term->node), sizeof(Node*), cudaMemcpyDeviceToHost);
  cudaMemcpy(&device_locs, &(device_term->locs), sizeof(u32*),  cudaMemcpyDeviceToHost);

  // Copy the device data to the host memory
  cudaMemcpy(host_term->acts, device_acts, host_term->alen * sizeof(Wire), cudaMemcpyDeviceToHost);
  cudaMemcpy(host_term->node, device_node, host_term->nlen * sizeof(Node), cudaMemcpyDeviceToHost);
  cudaMemcpy(host_term->locs, device_locs, host_term->nlen * sizeof(u32),  cudaMemcpyDeviceToHost);

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

__host__ void populate(Book* book) {
  // E
  book->defs[0x0000000f]           = (Term*) malloc(sizeof(Term));
  book->defs[0x0000000f]->root     = 0xa0000000;
  book->defs[0x0000000f]->alen     = 0;
  book->defs[0x0000000f]->acts     = (Wire*) malloc(0 * sizeof(Wire));
  book->defs[0x0000000f]->nlen     = 3;
  book->defs[0x0000000f]->node     = (Node*) malloc(3 * sizeof(Node));
  book->defs[0x0000000f]->node[ 0] = (Node) {0x30000000,0xa0000001};
  book->defs[0x0000000f]->node[ 1] = (Node) {0x30000000,0xa0000002};
  book->defs[0x0000000f]->node[ 2] = (Node) {0x60000002,0x50000002};
  book->defs[0x0000000f]->locs     = (u32*) malloc(3 * sizeof(u32));
  // F
  book->defs[0x00000010]           = (Term*) malloc(sizeof(Term));
  book->defs[0x00000010]->root     = 0xa0000000;
  book->defs[0x00000010]->alen     = 0;
  book->defs[0x00000010]->acts     = (Wire*) malloc(0 * sizeof(Wire));
  book->defs[0x00000010]->nlen     = 2;
  book->defs[0x00000010]->node     = (Node*) malloc(2 * sizeof(Node));
  book->defs[0x00000010]->node[ 0] = (Node) {0x30000000,0xa0000001};
  book->defs[0x00000010]->node[ 1] = (Node) {0x60000001,0x50000001};
  book->defs[0x00000010]->locs     = (u32*) malloc(2 * sizeof(u32));
  // I
  book->defs[0x00000013]           = (Term*) malloc(sizeof(Term));
  book->defs[0x00000013]->root     = 0xa0000000;
  book->defs[0x00000013]->alen     = 0;
  book->defs[0x00000013]->acts     = (Wire*) malloc(0 * sizeof(Wire));
  book->defs[0x00000013]->nlen     = 5;
  book->defs[0x00000013]->node     = (Node*) malloc(5 * sizeof(Node));
  book->defs[0x00000013]->node[ 0] = (Node) {0x50000003,0xa0000001};
  book->defs[0x00000013]->node[ 1] = (Node) {0x30000000,0xa0000002};
  book->defs[0x00000013]->node[ 2] = (Node) {0xa0000003,0xa0000004};
  book->defs[0x00000013]->node[ 3] = (Node) {0x50000000,0x60000004};
  book->defs[0x00000013]->node[ 4] = (Node) {0x30000000,0x60000003};
  book->defs[0x00000013]->locs     = (u32*) malloc(5 * sizeof(u32));
  // O
  book->defs[0x00000019]           = (Term*) malloc(sizeof(Term));
  book->defs[0x00000019]->root     = 0xa0000000;
  book->defs[0x00000019]->alen     = 0;
  book->defs[0x00000019]->acts     = (Wire*) malloc(0 * sizeof(Wire));
  book->defs[0x00000019]->nlen     = 5;
  book->defs[0x00000019]->node     = (Node*) malloc(5 * sizeof(Node));
  book->defs[0x00000019]->node[ 0] = (Node) {0x50000002,0xa0000001};
  book->defs[0x00000019]->node[ 1] = (Node) {0xa0000002,0xa0000003};
  book->defs[0x00000019]->node[ 2] = (Node) {0x50000000,0x60000004};
  book->defs[0x00000019]->node[ 3] = (Node) {0x30000000,0xa0000004};
  book->defs[0x00000019]->node[ 4] = (Node) {0x30000000,0x60000002};
  book->defs[0x00000019]->locs     = (u32*) malloc(5 * sizeof(u32));
  // S
  book->defs[0x0000001d]           = (Term*) malloc(sizeof(Term));
  book->defs[0x0000001d]->root     = 0xa0000000;
  book->defs[0x0000001d]->alen     = 0;
  book->defs[0x0000001d]->acts     = (Wire*) malloc(0 * sizeof(Wire));
  book->defs[0x0000001d]->nlen     = 4;
  book->defs[0x0000001d]->node     = (Node*) malloc(4 * sizeof(Node));
  book->defs[0x0000001d]->node[ 0] = (Node) {0x50000002,0xa0000001};
  book->defs[0x0000001d]->node[ 1] = (Node) {0xa0000002,0xa0000003};
  book->defs[0x0000001d]->node[ 2] = (Node) {0x50000000,0x60000003};
  book->defs[0x0000001d]->node[ 3] = (Node) {0x30000000,0x60000002};
  book->defs[0x0000001d]->locs     = (u32*) malloc(4 * sizeof(u32));
  // T
  book->defs[0x0000001e]           = (Term*) malloc(sizeof(Term));
  book->defs[0x0000001e]->root     = 0xa0000000;
  book->defs[0x0000001e]->alen     = 0;
  book->defs[0x0000001e]->acts     = (Wire*) malloc(0 * sizeof(Wire));
  book->defs[0x0000001e]->nlen     = 2;
  book->defs[0x0000001e]->node     = (Node*) malloc(2 * sizeof(Node));
  book->defs[0x0000001e]->node[ 0] = (Node) {0x60000001,0xa0000001};
  book->defs[0x0000001e]->node[ 1] = (Node) {0x30000000,0x50000000};
  book->defs[0x0000001e]->locs     = (u32*) malloc(2 * sizeof(u32));
  // Z
  book->defs[0x00000024]           = (Term*) malloc(sizeof(Term));
  book->defs[0x00000024]->root     = 0xa0000000;
  book->defs[0x00000024]->alen     = 0;
  book->defs[0x00000024]->acts     = (Wire*) malloc(0 * sizeof(Wire));
  book->defs[0x00000024]->nlen     = 2;
  book->defs[0x00000024]->node     = (Node*) malloc(2 * sizeof(Node));
  book->defs[0x00000024]->node[ 0] = (Node) {0x30000000,0xa0000001};
  book->defs[0x00000024]->node[ 1] = (Node) {0x60000001,0x50000001};
  book->defs[0x00000024]->locs     = (u32*) malloc(2 * sizeof(u32));
  // c0
  book->defs[0x000009c1]           = (Term*) malloc(sizeof(Term));
  book->defs[0x000009c1]->root     = 0xa0000000;
  book->defs[0x000009c1]->alen     = 0;
  book->defs[0x000009c1]->acts     = (Wire*) malloc(0 * sizeof(Wire));
  book->defs[0x000009c1]->nlen     = 2;
  book->defs[0x000009c1]->node     = (Node*) malloc(2 * sizeof(Node));
  book->defs[0x000009c1]->node[ 0] = (Node) {0x30000000,0xa0000001};
  book->defs[0x000009c1]->node[ 1] = (Node) {0x60000001,0x50000001};
  book->defs[0x000009c1]->locs     = (u32*) malloc(2 * sizeof(u32));
  // c1
  book->defs[0x000009c2]           = (Term*) malloc(sizeof(Term));
  book->defs[0x000009c2]->root     = 0xa0000000;
  book->defs[0x000009c2]->alen     = 0;
  book->defs[0x000009c2]->acts     = (Wire*) malloc(0 * sizeof(Wire));
  book->defs[0x000009c2]->nlen     = 3;
  book->defs[0x000009c2]->node     = (Node*) malloc(3 * sizeof(Node));
  book->defs[0x000009c2]->node[ 0] = (Node) {0xa0000001,0xa0000002};
  book->defs[0x000009c2]->node[ 1] = (Node) {0x50000002,0x60000002};
  book->defs[0x000009c2]->node[ 2] = (Node) {0x50000001,0x60000001};
  book->defs[0x000009c2]->locs     = (u32*) malloc(3 * sizeof(u32));
  // c2
  book->defs[0x000009c3]           = (Term*) malloc(sizeof(Term));
  book->defs[0x000009c3]->root     = 0xa0000000;
  book->defs[0x000009c3]->alen     = 0;
  book->defs[0x000009c3]->acts     = (Wire*) malloc(0 * sizeof(Wire));
  book->defs[0x000009c3]->nlen     = 5;
  book->defs[0x000009c3]->node     = (Node*) malloc(5 * sizeof(Node));
  book->defs[0x000009c3]->node[ 0] = (Node) {0xb0000001,0xa0000004};
  book->defs[0x000009c3]->node[ 1] = (Node) {0xa0000002,0xa0000003};
  book->defs[0x000009c3]->node[ 2] = (Node) {0x50000004,0x50000003};
  book->defs[0x000009c3]->node[ 3] = (Node) {0x60000002,0x60000004};
  book->defs[0x000009c3]->node[ 4] = (Node) {0x50000002,0x60000003};
  book->defs[0x000009c3]->locs     = (u32*) malloc(5 * sizeof(u32));
  // c3
  book->defs[0x000009c4]           = (Term*) malloc(sizeof(Term));
  book->defs[0x000009c4]->root     = 0xa0000000;
  book->defs[0x000009c4]->alen     = 0;
  book->defs[0x000009c4]->acts     = (Wire*) malloc(0 * sizeof(Wire));
  book->defs[0x000009c4]->nlen     = 7;
  book->defs[0x000009c4]->node     = (Node*) malloc(7 * sizeof(Node));
  book->defs[0x000009c4]->node[ 0] = (Node) {0xb0000001,0xa0000006};
  book->defs[0x000009c4]->node[ 1] = (Node) {0xb0000002,0xa0000005};
  book->defs[0x000009c4]->node[ 2] = (Node) {0xa0000003,0xa0000004};
  book->defs[0x000009c4]->node[ 3] = (Node) {0x50000006,0x50000004};
  book->defs[0x000009c4]->node[ 4] = (Node) {0x60000003,0x50000005};
  book->defs[0x000009c4]->node[ 5] = (Node) {0x60000004,0x60000006};
  book->defs[0x000009c4]->node[ 6] = (Node) {0x50000003,0x60000005};
  book->defs[0x000009c4]->locs     = (u32*) malloc(7 * sizeof(u32));
  // c4
  book->defs[0x000009c5]           = (Term*) malloc(sizeof(Term));
  book->defs[0x000009c5]->root     = 0xa0000000;
  book->defs[0x000009c5]->alen     = 0;
  book->defs[0x000009c5]->acts     = (Wire*) malloc(0 * sizeof(Wire));
  book->defs[0x000009c5]->nlen     = 9;
  book->defs[0x000009c5]->node     = (Node*) malloc(9 * sizeof(Node));
  book->defs[0x000009c5]->node[ 0] = (Node) {0xb0000001,0xa0000008};
  book->defs[0x000009c5]->node[ 1] = (Node) {0xb0000002,0xa0000007};
  book->defs[0x000009c5]->node[ 2] = (Node) {0xb0000003,0xa0000006};
  book->defs[0x000009c5]->node[ 3] = (Node) {0xa0000004,0xa0000005};
  book->defs[0x000009c5]->node[ 4] = (Node) {0x50000008,0x50000005};
  book->defs[0x000009c5]->node[ 5] = (Node) {0x60000004,0x50000006};
  book->defs[0x000009c5]->node[ 6] = (Node) {0x60000005,0x50000007};
  book->defs[0x000009c5]->node[ 7] = (Node) {0x60000006,0x60000008};
  book->defs[0x000009c5]->node[ 8] = (Node) {0x50000004,0x60000007};
  book->defs[0x000009c5]->locs     = (u32*) malloc(9 * sizeof(u32));
  // c5
  book->defs[0x000009c6]           = (Term*) malloc(sizeof(Term));
  book->defs[0x000009c6]->root     = 0xa0000000;
  book->defs[0x000009c6]->alen     = 0;
  book->defs[0x000009c6]->acts     = (Wire*) malloc(0 * sizeof(Wire));
  book->defs[0x000009c6]->nlen     = 11;
  book->defs[0x000009c6]->node     = (Node*) malloc(11 * sizeof(Node));
  book->defs[0x000009c6]->node[ 0] = (Node) {0xb0000001,0xa000000a};
  book->defs[0x000009c6]->node[ 1] = (Node) {0xb0000002,0xa0000009};
  book->defs[0x000009c6]->node[ 2] = (Node) {0xb0000003,0xa0000008};
  book->defs[0x000009c6]->node[ 3] = (Node) {0xb0000004,0xa0000007};
  book->defs[0x000009c6]->node[ 4] = (Node) {0xa0000005,0xa0000006};
  book->defs[0x000009c6]->node[ 5] = (Node) {0x5000000a,0x50000006};
  book->defs[0x000009c6]->node[ 6] = (Node) {0x60000005,0x50000007};
  book->defs[0x000009c6]->node[ 7] = (Node) {0x60000006,0x50000008};
  book->defs[0x000009c6]->node[ 8] = (Node) {0x60000007,0x50000009};
  book->defs[0x000009c6]->node[ 9] = (Node) {0x60000008,0x6000000a};
  book->defs[0x000009c6]->node[10] = (Node) {0x50000005,0x60000009};
  book->defs[0x000009c6]->locs     = (u32*) malloc(11 * sizeof(u32));
  // c6
  book->defs[0x000009c7]           = (Term*) malloc(sizeof(Term));
  book->defs[0x000009c7]->root     = 0xa0000000;
  book->defs[0x000009c7]->alen     = 0;
  book->defs[0x000009c7]->acts     = (Wire*) malloc(0 * sizeof(Wire));
  book->defs[0x000009c7]->nlen     = 13;
  book->defs[0x000009c7]->node     = (Node*) malloc(13 * sizeof(Node));
  book->defs[0x000009c7]->node[ 0] = (Node) {0xb0000001,0xa000000c};
  book->defs[0x000009c7]->node[ 1] = (Node) {0xb0000002,0xa000000b};
  book->defs[0x000009c7]->node[ 2] = (Node) {0xb0000003,0xa000000a};
  book->defs[0x000009c7]->node[ 3] = (Node) {0xb0000004,0xa0000009};
  book->defs[0x000009c7]->node[ 4] = (Node) {0xb0000005,0xa0000008};
  book->defs[0x000009c7]->node[ 5] = (Node) {0xa0000006,0xa0000007};
  book->defs[0x000009c7]->node[ 6] = (Node) {0x5000000c,0x50000007};
  book->defs[0x000009c7]->node[ 7] = (Node) {0x60000006,0x50000008};
  book->defs[0x000009c7]->node[ 8] = (Node) {0x60000007,0x50000009};
  book->defs[0x000009c7]->node[ 9] = (Node) {0x60000008,0x5000000a};
  book->defs[0x000009c7]->node[10] = (Node) {0x60000009,0x5000000b};
  book->defs[0x000009c7]->node[11] = (Node) {0x6000000a,0x6000000c};
  book->defs[0x000009c7]->node[12] = (Node) {0x50000006,0x6000000b};
  book->defs[0x000009c7]->locs     = (u32*) malloc(13 * sizeof(u32));
  // c7
  book->defs[0x000009c8]           = (Term*) malloc(sizeof(Term));
  book->defs[0x000009c8]->root     = 0xa0000000;
  book->defs[0x000009c8]->alen     = 0;
  book->defs[0x000009c8]->acts     = (Wire*) malloc(0 * sizeof(Wire));
  book->defs[0x000009c8]->nlen     = 15;
  book->defs[0x000009c8]->node     = (Node*) malloc(15 * sizeof(Node));
  book->defs[0x000009c8]->node[ 0] = (Node) {0xb0000001,0xa000000e};
  book->defs[0x000009c8]->node[ 1] = (Node) {0xb0000002,0xa000000d};
  book->defs[0x000009c8]->node[ 2] = (Node) {0xb0000003,0xa000000c};
  book->defs[0x000009c8]->node[ 3] = (Node) {0xb0000004,0xa000000b};
  book->defs[0x000009c8]->node[ 4] = (Node) {0xb0000005,0xa000000a};
  book->defs[0x000009c8]->node[ 5] = (Node) {0xb0000006,0xa0000009};
  book->defs[0x000009c8]->node[ 6] = (Node) {0xa0000007,0xa0000008};
  book->defs[0x000009c8]->node[ 7] = (Node) {0x5000000e,0x50000008};
  book->defs[0x000009c8]->node[ 8] = (Node) {0x60000007,0x50000009};
  book->defs[0x000009c8]->node[ 9] = (Node) {0x60000008,0x5000000a};
  book->defs[0x000009c8]->node[10] = (Node) {0x60000009,0x5000000b};
  book->defs[0x000009c8]->node[11] = (Node) {0x6000000a,0x5000000c};
  book->defs[0x000009c8]->node[12] = (Node) {0x6000000b,0x5000000d};
  book->defs[0x000009c8]->node[13] = (Node) {0x6000000c,0x6000000e};
  book->defs[0x000009c8]->node[14] = (Node) {0x50000007,0x6000000d};
  book->defs[0x000009c8]->locs     = (u32*) malloc(15 * sizeof(u32));
  // c8
  book->defs[0x000009c9]           = (Term*) malloc(sizeof(Term));
  book->defs[0x000009c9]->root     = 0xa0000000;
  book->defs[0x000009c9]->alen     = 0;
  book->defs[0x000009c9]->acts     = (Wire*) malloc(0 * sizeof(Wire));
  book->defs[0x000009c9]->nlen     = 17;
  book->defs[0x000009c9]->node     = (Node*) malloc(17 * sizeof(Node));
  book->defs[0x000009c9]->node[ 0] = (Node) {0xb0000001,0xa0000010};
  book->defs[0x000009c9]->node[ 1] = (Node) {0xb0000002,0xa000000f};
  book->defs[0x000009c9]->node[ 2] = (Node) {0xb0000003,0xa000000e};
  book->defs[0x000009c9]->node[ 3] = (Node) {0xb0000004,0xa000000d};
  book->defs[0x000009c9]->node[ 4] = (Node) {0xb0000005,0xa000000c};
  book->defs[0x000009c9]->node[ 5] = (Node) {0xb0000006,0xa000000b};
  book->defs[0x000009c9]->node[ 6] = (Node) {0xb0000007,0xa000000a};
  book->defs[0x000009c9]->node[ 7] = (Node) {0xa0000008,0xa0000009};
  book->defs[0x000009c9]->node[ 8] = (Node) {0x50000010,0x50000009};
  book->defs[0x000009c9]->node[ 9] = (Node) {0x60000008,0x5000000a};
  book->defs[0x000009c9]->node[10] = (Node) {0x60000009,0x5000000b};
  book->defs[0x000009c9]->node[11] = (Node) {0x6000000a,0x5000000c};
  book->defs[0x000009c9]->node[12] = (Node) {0x6000000b,0x5000000d};
  book->defs[0x000009c9]->node[13] = (Node) {0x6000000c,0x5000000e};
  book->defs[0x000009c9]->node[14] = (Node) {0x6000000d,0x5000000f};
  book->defs[0x000009c9]->node[15] = (Node) {0x6000000e,0x60000010};
  book->defs[0x000009c9]->node[16] = (Node) {0x50000008,0x6000000f};
  book->defs[0x000009c9]->locs     = (u32*) malloc(17 * sizeof(u32));
  // c9
  book->defs[0x000009ca]           = (Term*) malloc(sizeof(Term));
  book->defs[0x000009ca]->root     = 0xa0000000;
  book->defs[0x000009ca]->alen     = 0;
  book->defs[0x000009ca]->acts     = (Wire*) malloc(0 * sizeof(Wire));
  book->defs[0x000009ca]->nlen     = 19;
  book->defs[0x000009ca]->node     = (Node*) malloc(19 * sizeof(Node));
  book->defs[0x000009ca]->node[ 0] = (Node) {0xb0000001,0xa0000012};
  book->defs[0x000009ca]->node[ 1] = (Node) {0xb0000002,0xa0000011};
  book->defs[0x000009ca]->node[ 2] = (Node) {0xb0000003,0xa0000010};
  book->defs[0x000009ca]->node[ 3] = (Node) {0xb0000004,0xa000000f};
  book->defs[0x000009ca]->node[ 4] = (Node) {0xb0000005,0xa000000e};
  book->defs[0x000009ca]->node[ 5] = (Node) {0xb0000006,0xa000000d};
  book->defs[0x000009ca]->node[ 6] = (Node) {0xb0000007,0xa000000c};
  book->defs[0x000009ca]->node[ 7] = (Node) {0xb0000008,0xa000000b};
  book->defs[0x000009ca]->node[ 8] = (Node) {0xa0000009,0xa000000a};
  book->defs[0x000009ca]->node[ 9] = (Node) {0x50000012,0x5000000a};
  book->defs[0x000009ca]->node[10] = (Node) {0x60000009,0x5000000b};
  book->defs[0x000009ca]->node[11] = (Node) {0x6000000a,0x5000000c};
  book->defs[0x000009ca]->node[12] = (Node) {0x6000000b,0x5000000d};
  book->defs[0x000009ca]->node[13] = (Node) {0x6000000c,0x5000000e};
  book->defs[0x000009ca]->node[14] = (Node) {0x6000000d,0x5000000f};
  book->defs[0x000009ca]->node[15] = (Node) {0x6000000e,0x50000010};
  book->defs[0x000009ca]->node[16] = (Node) {0x6000000f,0x50000011};
  book->defs[0x000009ca]->node[17] = (Node) {0x60000010,0x60000012};
  book->defs[0x000009ca]->node[18] = (Node) {0x50000009,0x60000011};
  book->defs[0x000009ca]->locs     = (u32*) malloc(19 * sizeof(u32));
  // id
  book->defs[0x00000b68]           = (Term*) malloc(sizeof(Term));
  book->defs[0x00000b68]->root     = 0xa0000000;
  book->defs[0x00000b68]->alen     = 0;
  book->defs[0x00000b68]->acts     = (Wire*) malloc(0 * sizeof(Wire));
  book->defs[0x00000b68]->nlen     = 1;
  book->defs[0x00000b68]->node     = (Node*) malloc(1 * sizeof(Node));
  book->defs[0x00000b68]->node[ 0] = (Node) {0x60000000,0x50000000};
  book->defs[0x00000b68]->locs     = (u32*) malloc(1 * sizeof(u32));
  // k0
  book->defs[0x00000bc1]           = (Term*) malloc(sizeof(Term));
  book->defs[0x00000bc1]->root     = 0xa0000000;
  book->defs[0x00000bc1]->alen     = 0;
  book->defs[0x00000bc1]->acts     = (Wire*) malloc(0 * sizeof(Wire));
  book->defs[0x00000bc1]->nlen     = 2;
  book->defs[0x00000bc1]->node     = (Node*) malloc(2 * sizeof(Node));
  book->defs[0x00000bc1]->node[ 0] = (Node) {0x30000000,0xa0000001};
  book->defs[0x00000bc1]->node[ 1] = (Node) {0x60000001,0x50000001};
  book->defs[0x00000bc1]->locs     = (u32*) malloc(2 * sizeof(u32));
  // k1
  book->defs[0x00000bc2]           = (Term*) malloc(sizeof(Term));
  book->defs[0x00000bc2]->root     = 0xa0000000;
  book->defs[0x00000bc2]->alen     = 0;
  book->defs[0x00000bc2]->acts     = (Wire*) malloc(0 * sizeof(Wire));
  book->defs[0x00000bc2]->nlen     = 3;
  book->defs[0x00000bc2]->node     = (Node*) malloc(3 * sizeof(Node));
  book->defs[0x00000bc2]->node[ 0] = (Node) {0xa0000001,0xa0000002};
  book->defs[0x00000bc2]->node[ 1] = (Node) {0x50000002,0x60000002};
  book->defs[0x00000bc2]->node[ 2] = (Node) {0x50000001,0x60000001};
  book->defs[0x00000bc2]->locs     = (u32*) malloc(3 * sizeof(u32));
  // k2
  book->defs[0x00000bc3]           = (Term*) malloc(sizeof(Term));
  book->defs[0x00000bc3]->root     = 0xa0000000;
  book->defs[0x00000bc3]->alen     = 0;
  book->defs[0x00000bc3]->acts     = (Wire*) malloc(0 * sizeof(Wire));
  book->defs[0x00000bc3]->nlen     = 5;
  book->defs[0x00000bc3]->node     = (Node*) malloc(5 * sizeof(Node));
  book->defs[0x00000bc3]->node[ 0] = (Node) {0xc0000001,0xa0000004};
  book->defs[0x00000bc3]->node[ 1] = (Node) {0xa0000002,0xa0000003};
  book->defs[0x00000bc3]->node[ 2] = (Node) {0x50000004,0x50000003};
  book->defs[0x00000bc3]->node[ 3] = (Node) {0x60000002,0x60000004};
  book->defs[0x00000bc3]->node[ 4] = (Node) {0x50000002,0x60000003};
  book->defs[0x00000bc3]->locs     = (u32*) malloc(5 * sizeof(u32));
  // k3
  book->defs[0x00000bc4]           = (Term*) malloc(sizeof(Term));
  book->defs[0x00000bc4]->root     = 0xa0000000;
  book->defs[0x00000bc4]->alen     = 0;
  book->defs[0x00000bc4]->acts     = (Wire*) malloc(0 * sizeof(Wire));
  book->defs[0x00000bc4]->nlen     = 7;
  book->defs[0x00000bc4]->node     = (Node*) malloc(7 * sizeof(Node));
  book->defs[0x00000bc4]->node[ 0] = (Node) {0xc0000001,0xa0000006};
  book->defs[0x00000bc4]->node[ 1] = (Node) {0xc0000002,0xa0000005};
  book->defs[0x00000bc4]->node[ 2] = (Node) {0xa0000003,0xa0000004};
  book->defs[0x00000bc4]->node[ 3] = (Node) {0x50000006,0x50000004};
  book->defs[0x00000bc4]->node[ 4] = (Node) {0x60000003,0x50000005};
  book->defs[0x00000bc4]->node[ 5] = (Node) {0x60000004,0x60000006};
  book->defs[0x00000bc4]->node[ 6] = (Node) {0x50000003,0x60000005};
  book->defs[0x00000bc4]->locs     = (u32*) malloc(7 * sizeof(u32));
  // k4
  book->defs[0x00000bc5]           = (Term*) malloc(sizeof(Term));
  book->defs[0x00000bc5]->root     = 0xa0000000;
  book->defs[0x00000bc5]->alen     = 0;
  book->defs[0x00000bc5]->acts     = (Wire*) malloc(0 * sizeof(Wire));
  book->defs[0x00000bc5]->nlen     = 9;
  book->defs[0x00000bc5]->node     = (Node*) malloc(9 * sizeof(Node));
  book->defs[0x00000bc5]->node[ 0] = (Node) {0xc0000001,0xa0000008};
  book->defs[0x00000bc5]->node[ 1] = (Node) {0xc0000002,0xa0000007};
  book->defs[0x00000bc5]->node[ 2] = (Node) {0xc0000003,0xa0000006};
  book->defs[0x00000bc5]->node[ 3] = (Node) {0xa0000004,0xa0000005};
  book->defs[0x00000bc5]->node[ 4] = (Node) {0x50000008,0x50000005};
  book->defs[0x00000bc5]->node[ 5] = (Node) {0x60000004,0x50000006};
  book->defs[0x00000bc5]->node[ 6] = (Node) {0x60000005,0x50000007};
  book->defs[0x00000bc5]->node[ 7] = (Node) {0x60000006,0x60000008};
  book->defs[0x00000bc5]->node[ 8] = (Node) {0x50000004,0x60000007};
  book->defs[0x00000bc5]->locs     = (u32*) malloc(9 * sizeof(u32));
  // k5
  book->defs[0x00000bc6]           = (Term*) malloc(sizeof(Term));
  book->defs[0x00000bc6]->root     = 0xa0000000;
  book->defs[0x00000bc6]->alen     = 0;
  book->defs[0x00000bc6]->acts     = (Wire*) malloc(0 * sizeof(Wire));
  book->defs[0x00000bc6]->nlen     = 11;
  book->defs[0x00000bc6]->node     = (Node*) malloc(11 * sizeof(Node));
  book->defs[0x00000bc6]->node[ 0] = (Node) {0xc0000001,0xa000000a};
  book->defs[0x00000bc6]->node[ 1] = (Node) {0xc0000002,0xa0000009};
  book->defs[0x00000bc6]->node[ 2] = (Node) {0xc0000003,0xa0000008};
  book->defs[0x00000bc6]->node[ 3] = (Node) {0xc0000004,0xa0000007};
  book->defs[0x00000bc6]->node[ 4] = (Node) {0xa0000005,0xa0000006};
  book->defs[0x00000bc6]->node[ 5] = (Node) {0x5000000a,0x50000006};
  book->defs[0x00000bc6]->node[ 6] = (Node) {0x60000005,0x50000007};
  book->defs[0x00000bc6]->node[ 7] = (Node) {0x60000006,0x50000008};
  book->defs[0x00000bc6]->node[ 8] = (Node) {0x60000007,0x50000009};
  book->defs[0x00000bc6]->node[ 9] = (Node) {0x60000008,0x6000000a};
  book->defs[0x00000bc6]->node[10] = (Node) {0x50000005,0x60000009};
  book->defs[0x00000bc6]->locs     = (u32*) malloc(11 * sizeof(u32));
  // k6
  book->defs[0x00000bc7]           = (Term*) malloc(sizeof(Term));
  book->defs[0x00000bc7]->root     = 0xa0000000;
  book->defs[0x00000bc7]->alen     = 0;
  book->defs[0x00000bc7]->acts     = (Wire*) malloc(0 * sizeof(Wire));
  book->defs[0x00000bc7]->nlen     = 13;
  book->defs[0x00000bc7]->node     = (Node*) malloc(13 * sizeof(Node));
  book->defs[0x00000bc7]->node[ 0] = (Node) {0xc0000001,0xa000000c};
  book->defs[0x00000bc7]->node[ 1] = (Node) {0xc0000002,0xa000000b};
  book->defs[0x00000bc7]->node[ 2] = (Node) {0xc0000003,0xa000000a};
  book->defs[0x00000bc7]->node[ 3] = (Node) {0xc0000004,0xa0000009};
  book->defs[0x00000bc7]->node[ 4] = (Node) {0xc0000005,0xa0000008};
  book->defs[0x00000bc7]->node[ 5] = (Node) {0xa0000006,0xa0000007};
  book->defs[0x00000bc7]->node[ 6] = (Node) {0x5000000c,0x50000007};
  book->defs[0x00000bc7]->node[ 7] = (Node) {0x60000006,0x50000008};
  book->defs[0x00000bc7]->node[ 8] = (Node) {0x60000007,0x50000009};
  book->defs[0x00000bc7]->node[ 9] = (Node) {0x60000008,0x5000000a};
  book->defs[0x00000bc7]->node[10] = (Node) {0x60000009,0x5000000b};
  book->defs[0x00000bc7]->node[11] = (Node) {0x6000000a,0x6000000c};
  book->defs[0x00000bc7]->node[12] = (Node) {0x50000006,0x6000000b};
  book->defs[0x00000bc7]->locs     = (u32*) malloc(13 * sizeof(u32));
  // k7
  book->defs[0x00000bc8]           = (Term*) malloc(sizeof(Term));
  book->defs[0x00000bc8]->root     = 0xa0000000;
  book->defs[0x00000bc8]->alen     = 0;
  book->defs[0x00000bc8]->acts     = (Wire*) malloc(0 * sizeof(Wire));
  book->defs[0x00000bc8]->nlen     = 15;
  book->defs[0x00000bc8]->node     = (Node*) malloc(15 * sizeof(Node));
  book->defs[0x00000bc8]->node[ 0] = (Node) {0xc0000001,0xa000000e};
  book->defs[0x00000bc8]->node[ 1] = (Node) {0xc0000002,0xa000000d};
  book->defs[0x00000bc8]->node[ 2] = (Node) {0xc0000003,0xa000000c};
  book->defs[0x00000bc8]->node[ 3] = (Node) {0xc0000004,0xa000000b};
  book->defs[0x00000bc8]->node[ 4] = (Node) {0xc0000005,0xa000000a};
  book->defs[0x00000bc8]->node[ 5] = (Node) {0xc0000006,0xa0000009};
  book->defs[0x00000bc8]->node[ 6] = (Node) {0xa0000007,0xa0000008};
  book->defs[0x00000bc8]->node[ 7] = (Node) {0x5000000e,0x50000008};
  book->defs[0x00000bc8]->node[ 8] = (Node) {0x60000007,0x50000009};
  book->defs[0x00000bc8]->node[ 9] = (Node) {0x60000008,0x5000000a};
  book->defs[0x00000bc8]->node[10] = (Node) {0x60000009,0x5000000b};
  book->defs[0x00000bc8]->node[11] = (Node) {0x6000000a,0x5000000c};
  book->defs[0x00000bc8]->node[12] = (Node) {0x6000000b,0x5000000d};
  book->defs[0x00000bc8]->node[13] = (Node) {0x6000000c,0x6000000e};
  book->defs[0x00000bc8]->node[14] = (Node) {0x50000007,0x6000000d};
  book->defs[0x00000bc8]->locs     = (u32*) malloc(15 * sizeof(u32));
  // k8
  book->defs[0x00000bc9]           = (Term*) malloc(sizeof(Term));
  book->defs[0x00000bc9]->root     = 0xa0000000;
  book->defs[0x00000bc9]->alen     = 0;
  book->defs[0x00000bc9]->acts     = (Wire*) malloc(0 * sizeof(Wire));
  book->defs[0x00000bc9]->nlen     = 17;
  book->defs[0x00000bc9]->node     = (Node*) malloc(17 * sizeof(Node));
  book->defs[0x00000bc9]->node[ 0] = (Node) {0xc0000001,0xa0000010};
  book->defs[0x00000bc9]->node[ 1] = (Node) {0xc0000002,0xa000000f};
  book->defs[0x00000bc9]->node[ 2] = (Node) {0xc0000003,0xa000000e};
  book->defs[0x00000bc9]->node[ 3] = (Node) {0xc0000004,0xa000000d};
  book->defs[0x00000bc9]->node[ 4] = (Node) {0xc0000005,0xa000000c};
  book->defs[0x00000bc9]->node[ 5] = (Node) {0xc0000006,0xa000000b};
  book->defs[0x00000bc9]->node[ 6] = (Node) {0xc0000007,0xa000000a};
  book->defs[0x00000bc9]->node[ 7] = (Node) {0xa0000008,0xa0000009};
  book->defs[0x00000bc9]->node[ 8] = (Node) {0x50000010,0x50000009};
  book->defs[0x00000bc9]->node[ 9] = (Node) {0x60000008,0x5000000a};
  book->defs[0x00000bc9]->node[10] = (Node) {0x60000009,0x5000000b};
  book->defs[0x00000bc9]->node[11] = (Node) {0x6000000a,0x5000000c};
  book->defs[0x00000bc9]->node[12] = (Node) {0x6000000b,0x5000000d};
  book->defs[0x00000bc9]->node[13] = (Node) {0x6000000c,0x5000000e};
  book->defs[0x00000bc9]->node[14] = (Node) {0x6000000d,0x5000000f};
  book->defs[0x00000bc9]->node[15] = (Node) {0x6000000e,0x60000010};
  book->defs[0x00000bc9]->node[16] = (Node) {0x50000008,0x6000000f};
  book->defs[0x00000bc9]->locs     = (u32*) malloc(17 * sizeof(u32));
  // k9
  book->defs[0x00000bca]           = (Term*) malloc(sizeof(Term));
  book->defs[0x00000bca]->root     = 0xa0000000;
  book->defs[0x00000bca]->alen     = 0;
  book->defs[0x00000bca]->acts     = (Wire*) malloc(0 * sizeof(Wire));
  book->defs[0x00000bca]->nlen     = 19;
  book->defs[0x00000bca]->node     = (Node*) malloc(19 * sizeof(Node));
  book->defs[0x00000bca]->node[ 0] = (Node) {0xc0000001,0xa0000012};
  book->defs[0x00000bca]->node[ 1] = (Node) {0xc0000002,0xa0000011};
  book->defs[0x00000bca]->node[ 2] = (Node) {0xc0000003,0xa0000010};
  book->defs[0x00000bca]->node[ 3] = (Node) {0xc0000004,0xa000000f};
  book->defs[0x00000bca]->node[ 4] = (Node) {0xc0000005,0xa000000e};
  book->defs[0x00000bca]->node[ 5] = (Node) {0xc0000006,0xa000000d};
  book->defs[0x00000bca]->node[ 6] = (Node) {0xc0000007,0xa000000c};
  book->defs[0x00000bca]->node[ 7] = (Node) {0xc0000008,0xa000000b};
  book->defs[0x00000bca]->node[ 8] = (Node) {0xa0000009,0xa000000a};
  book->defs[0x00000bca]->node[ 9] = (Node) {0x50000012,0x5000000a};
  book->defs[0x00000bca]->node[10] = (Node) {0x60000009,0x5000000b};
  book->defs[0x00000bca]->node[11] = (Node) {0x6000000a,0x5000000c};
  book->defs[0x00000bca]->node[12] = (Node) {0x6000000b,0x5000000d};
  book->defs[0x00000bca]->node[13] = (Node) {0x6000000c,0x5000000e};
  book->defs[0x00000bca]->node[14] = (Node) {0x6000000d,0x5000000f};
  book->defs[0x00000bca]->node[15] = (Node) {0x6000000e,0x50000010};
  book->defs[0x00000bca]->node[16] = (Node) {0x6000000f,0x50000011};
  book->defs[0x00000bca]->node[17] = (Node) {0x60000010,0x60000012};
  book->defs[0x00000bca]->node[18] = (Node) {0x50000009,0x60000011};
  book->defs[0x00000bca]->locs     = (u32*) malloc(19 * sizeof(u32));
  // c10
  book->defs[0x00027081]           = (Term*) malloc(sizeof(Term));
  book->defs[0x00027081]->root     = 0xa0000000;
  book->defs[0x00027081]->alen     = 0;
  book->defs[0x00027081]->acts     = (Wire*) malloc(0 * sizeof(Wire));
  book->defs[0x00027081]->nlen     = 21;
  book->defs[0x00027081]->node     = (Node*) malloc(21 * sizeof(Node));
  book->defs[0x00027081]->node[ 0] = (Node) {0xb0000001,0xa0000014};
  book->defs[0x00027081]->node[ 1] = (Node) {0xb0000002,0xa0000013};
  book->defs[0x00027081]->node[ 2] = (Node) {0xb0000003,0xa0000012};
  book->defs[0x00027081]->node[ 3] = (Node) {0xb0000004,0xa0000011};
  book->defs[0x00027081]->node[ 4] = (Node) {0xb0000005,0xa0000010};
  book->defs[0x00027081]->node[ 5] = (Node) {0xb0000006,0xa000000f};
  book->defs[0x00027081]->node[ 6] = (Node) {0xb0000007,0xa000000e};
  book->defs[0x00027081]->node[ 7] = (Node) {0xb0000008,0xa000000d};
  book->defs[0x00027081]->node[ 8] = (Node) {0xb0000009,0xa000000c};
  book->defs[0x00027081]->node[ 9] = (Node) {0xa000000a,0xa000000b};
  book->defs[0x00027081]->node[10] = (Node) {0x50000014,0x5000000b};
  book->defs[0x00027081]->node[11] = (Node) {0x6000000a,0x5000000c};
  book->defs[0x00027081]->node[12] = (Node) {0x6000000b,0x5000000d};
  book->defs[0x00027081]->node[13] = (Node) {0x6000000c,0x5000000e};
  book->defs[0x00027081]->node[14] = (Node) {0x6000000d,0x5000000f};
  book->defs[0x00027081]->node[15] = (Node) {0x6000000e,0x50000010};
  book->defs[0x00027081]->node[16] = (Node) {0x6000000f,0x50000011};
  book->defs[0x00027081]->node[17] = (Node) {0x60000010,0x50000012};
  book->defs[0x00027081]->node[18] = (Node) {0x60000011,0x50000013};
  book->defs[0x00027081]->node[19] = (Node) {0x60000012,0x60000014};
  book->defs[0x00027081]->node[20] = (Node) {0x5000000a,0x60000013};
  book->defs[0x00027081]->locs     = (u32*) malloc(21 * sizeof(u32));
  // c11
  book->defs[0x00027082]           = (Term*) malloc(sizeof(Term));
  book->defs[0x00027082]->root     = 0xa0000000;
  book->defs[0x00027082]->alen     = 0;
  book->defs[0x00027082]->acts     = (Wire*) malloc(0 * sizeof(Wire));
  book->defs[0x00027082]->nlen     = 23;
  book->defs[0x00027082]->node     = (Node*) malloc(23 * sizeof(Node));
  book->defs[0x00027082]->node[ 0] = (Node) {0xb0000001,0xa0000016};
  book->defs[0x00027082]->node[ 1] = (Node) {0xb0000002,0xa0000015};
  book->defs[0x00027082]->node[ 2] = (Node) {0xb0000003,0xa0000014};
  book->defs[0x00027082]->node[ 3] = (Node) {0xb0000004,0xa0000013};
  book->defs[0x00027082]->node[ 4] = (Node) {0xb0000005,0xa0000012};
  book->defs[0x00027082]->node[ 5] = (Node) {0xb0000006,0xa0000011};
  book->defs[0x00027082]->node[ 6] = (Node) {0xb0000007,0xa0000010};
  book->defs[0x00027082]->node[ 7] = (Node) {0xb0000008,0xa000000f};
  book->defs[0x00027082]->node[ 8] = (Node) {0xb0000009,0xa000000e};
  book->defs[0x00027082]->node[ 9] = (Node) {0xb000000a,0xa000000d};
  book->defs[0x00027082]->node[10] = (Node) {0xa000000b,0xa000000c};
  book->defs[0x00027082]->node[11] = (Node) {0x50000016,0x5000000c};
  book->defs[0x00027082]->node[12] = (Node) {0x6000000b,0x5000000d};
  book->defs[0x00027082]->node[13] = (Node) {0x6000000c,0x5000000e};
  book->defs[0x00027082]->node[14] = (Node) {0x6000000d,0x5000000f};
  book->defs[0x00027082]->node[15] = (Node) {0x6000000e,0x50000010};
  book->defs[0x00027082]->node[16] = (Node) {0x6000000f,0x50000011};
  book->defs[0x00027082]->node[17] = (Node) {0x60000010,0x50000012};
  book->defs[0x00027082]->node[18] = (Node) {0x60000011,0x50000013};
  book->defs[0x00027082]->node[19] = (Node) {0x60000012,0x50000014};
  book->defs[0x00027082]->node[20] = (Node) {0x60000013,0x50000015};
  book->defs[0x00027082]->node[21] = (Node) {0x60000014,0x60000016};
  book->defs[0x00027082]->node[22] = (Node) {0x5000000b,0x60000015};
  book->defs[0x00027082]->locs     = (u32*) malloc(23 * sizeof(u32));
  // c12
  book->defs[0x00027083]           = (Term*) malloc(sizeof(Term));
  book->defs[0x00027083]->root     = 0xa0000000;
  book->defs[0x00027083]->alen     = 0;
  book->defs[0x00027083]->acts     = (Wire*) malloc(0 * sizeof(Wire));
  book->defs[0x00027083]->nlen     = 25;
  book->defs[0x00027083]->node     = (Node*) malloc(25 * sizeof(Node));
  book->defs[0x00027083]->node[ 0] = (Node) {0xb0000001,0xa0000018};
  book->defs[0x00027083]->node[ 1] = (Node) {0xb0000002,0xa0000017};
  book->defs[0x00027083]->node[ 2] = (Node) {0xb0000003,0xa0000016};
  book->defs[0x00027083]->node[ 3] = (Node) {0xb0000004,0xa0000015};
  book->defs[0x00027083]->node[ 4] = (Node) {0xb0000005,0xa0000014};
  book->defs[0x00027083]->node[ 5] = (Node) {0xb0000006,0xa0000013};
  book->defs[0x00027083]->node[ 6] = (Node) {0xb0000007,0xa0000012};
  book->defs[0x00027083]->node[ 7] = (Node) {0xb0000008,0xa0000011};
  book->defs[0x00027083]->node[ 8] = (Node) {0xb0000009,0xa0000010};
  book->defs[0x00027083]->node[ 9] = (Node) {0xb000000a,0xa000000f};
  book->defs[0x00027083]->node[10] = (Node) {0xb000000b,0xa000000e};
  book->defs[0x00027083]->node[11] = (Node) {0xa000000c,0xa000000d};
  book->defs[0x00027083]->node[12] = (Node) {0x50000018,0x5000000d};
  book->defs[0x00027083]->node[13] = (Node) {0x6000000c,0x5000000e};
  book->defs[0x00027083]->node[14] = (Node) {0x6000000d,0x5000000f};
  book->defs[0x00027083]->node[15] = (Node) {0x6000000e,0x50000010};
  book->defs[0x00027083]->node[16] = (Node) {0x6000000f,0x50000011};
  book->defs[0x00027083]->node[17] = (Node) {0x60000010,0x50000012};
  book->defs[0x00027083]->node[18] = (Node) {0x60000011,0x50000013};
  book->defs[0x00027083]->node[19] = (Node) {0x60000012,0x50000014};
  book->defs[0x00027083]->node[20] = (Node) {0x60000013,0x50000015};
  book->defs[0x00027083]->node[21] = (Node) {0x60000014,0x50000016};
  book->defs[0x00027083]->node[22] = (Node) {0x60000015,0x50000017};
  book->defs[0x00027083]->node[23] = (Node) {0x60000016,0x60000018};
  book->defs[0x00027083]->node[24] = (Node) {0x5000000c,0x60000017};
  book->defs[0x00027083]->locs     = (u32*) malloc(25 * sizeof(u32));
  // c13
  book->defs[0x00027084]           = (Term*) malloc(sizeof(Term));
  book->defs[0x00027084]->root     = 0xa0000000;
  book->defs[0x00027084]->alen     = 0;
  book->defs[0x00027084]->acts     = (Wire*) malloc(0 * sizeof(Wire));
  book->defs[0x00027084]->nlen     = 27;
  book->defs[0x00027084]->node     = (Node*) malloc(27 * sizeof(Node));
  book->defs[0x00027084]->node[ 0] = (Node) {0xb0000001,0xa000001a};
  book->defs[0x00027084]->node[ 1] = (Node) {0xb0000002,0xa0000019};
  book->defs[0x00027084]->node[ 2] = (Node) {0xb0000003,0xa0000018};
  book->defs[0x00027084]->node[ 3] = (Node) {0xb0000004,0xa0000017};
  book->defs[0x00027084]->node[ 4] = (Node) {0xb0000005,0xa0000016};
  book->defs[0x00027084]->node[ 5] = (Node) {0xb0000006,0xa0000015};
  book->defs[0x00027084]->node[ 6] = (Node) {0xb0000007,0xa0000014};
  book->defs[0x00027084]->node[ 7] = (Node) {0xb0000008,0xa0000013};
  book->defs[0x00027084]->node[ 8] = (Node) {0xb0000009,0xa0000012};
  book->defs[0x00027084]->node[ 9] = (Node) {0xb000000a,0xa0000011};
  book->defs[0x00027084]->node[10] = (Node) {0xb000000b,0xa0000010};
  book->defs[0x00027084]->node[11] = (Node) {0xb000000c,0xa000000f};
  book->defs[0x00027084]->node[12] = (Node) {0xa000000d,0xa000000e};
  book->defs[0x00027084]->node[13] = (Node) {0x5000001a,0x5000000e};
  book->defs[0x00027084]->node[14] = (Node) {0x6000000d,0x5000000f};
  book->defs[0x00027084]->node[15] = (Node) {0x6000000e,0x50000010};
  book->defs[0x00027084]->node[16] = (Node) {0x6000000f,0x50000011};
  book->defs[0x00027084]->node[17] = (Node) {0x60000010,0x50000012};
  book->defs[0x00027084]->node[18] = (Node) {0x60000011,0x50000013};
  book->defs[0x00027084]->node[19] = (Node) {0x60000012,0x50000014};
  book->defs[0x00027084]->node[20] = (Node) {0x60000013,0x50000015};
  book->defs[0x00027084]->node[21] = (Node) {0x60000014,0x50000016};
  book->defs[0x00027084]->node[22] = (Node) {0x60000015,0x50000017};
  book->defs[0x00027084]->node[23] = (Node) {0x60000016,0x50000018};
  book->defs[0x00027084]->node[24] = (Node) {0x60000017,0x50000019};
  book->defs[0x00027084]->node[25] = (Node) {0x60000018,0x6000001a};
  book->defs[0x00027084]->node[26] = (Node) {0x5000000d,0x60000019};
  book->defs[0x00027084]->locs     = (u32*) malloc(27 * sizeof(u32));
  // c14
  book->defs[0x00027085]           = (Term*) malloc(sizeof(Term));
  book->defs[0x00027085]->root     = 0xa0000000;
  book->defs[0x00027085]->alen     = 0;
  book->defs[0x00027085]->acts     = (Wire*) malloc(0 * sizeof(Wire));
  book->defs[0x00027085]->nlen     = 29;
  book->defs[0x00027085]->node     = (Node*) malloc(29 * sizeof(Node));
  book->defs[0x00027085]->node[ 0] = (Node) {0xb0000001,0xa000001c};
  book->defs[0x00027085]->node[ 1] = (Node) {0xb0000002,0xa000001b};
  book->defs[0x00027085]->node[ 2] = (Node) {0xb0000003,0xa000001a};
  book->defs[0x00027085]->node[ 3] = (Node) {0xb0000004,0xa0000019};
  book->defs[0x00027085]->node[ 4] = (Node) {0xb0000005,0xa0000018};
  book->defs[0x00027085]->node[ 5] = (Node) {0xb0000006,0xa0000017};
  book->defs[0x00027085]->node[ 6] = (Node) {0xb0000007,0xa0000016};
  book->defs[0x00027085]->node[ 7] = (Node) {0xb0000008,0xa0000015};
  book->defs[0x00027085]->node[ 8] = (Node) {0xb0000009,0xa0000014};
  book->defs[0x00027085]->node[ 9] = (Node) {0xb000000a,0xa0000013};
  book->defs[0x00027085]->node[10] = (Node) {0xb000000b,0xa0000012};
  book->defs[0x00027085]->node[11] = (Node) {0xb000000c,0xa0000011};
  book->defs[0x00027085]->node[12] = (Node) {0xb000000d,0xa0000010};
  book->defs[0x00027085]->node[13] = (Node) {0xa000000e,0xa000000f};
  book->defs[0x00027085]->node[14] = (Node) {0x5000001c,0x5000000f};
  book->defs[0x00027085]->node[15] = (Node) {0x6000000e,0x50000010};
  book->defs[0x00027085]->node[16] = (Node) {0x6000000f,0x50000011};
  book->defs[0x00027085]->node[17] = (Node) {0x60000010,0x50000012};
  book->defs[0x00027085]->node[18] = (Node) {0x60000011,0x50000013};
  book->defs[0x00027085]->node[19] = (Node) {0x60000012,0x50000014};
  book->defs[0x00027085]->node[20] = (Node) {0x60000013,0x50000015};
  book->defs[0x00027085]->node[21] = (Node) {0x60000014,0x50000016};
  book->defs[0x00027085]->node[22] = (Node) {0x60000015,0x50000017};
  book->defs[0x00027085]->node[23] = (Node) {0x60000016,0x50000018};
  book->defs[0x00027085]->node[24] = (Node) {0x60000017,0x50000019};
  book->defs[0x00027085]->node[25] = (Node) {0x60000018,0x5000001a};
  book->defs[0x00027085]->node[26] = (Node) {0x60000019,0x5000001b};
  book->defs[0x00027085]->node[27] = (Node) {0x6000001a,0x6000001c};
  book->defs[0x00027085]->node[28] = (Node) {0x5000000e,0x6000001b};
  book->defs[0x00027085]->locs     = (u32*) malloc(29 * sizeof(u32));
  // c15
  book->defs[0x00027086]           = (Term*) malloc(sizeof(Term));
  book->defs[0x00027086]->root     = 0xa0000000;
  book->defs[0x00027086]->alen     = 0;
  book->defs[0x00027086]->acts     = (Wire*) malloc(0 * sizeof(Wire));
  book->defs[0x00027086]->nlen     = 31;
  book->defs[0x00027086]->node     = (Node*) malloc(31 * sizeof(Node));
  book->defs[0x00027086]->node[ 0] = (Node) {0xb0000001,0xa000001e};
  book->defs[0x00027086]->node[ 1] = (Node) {0xb0000002,0xa000001d};
  book->defs[0x00027086]->node[ 2] = (Node) {0xb0000003,0xa000001c};
  book->defs[0x00027086]->node[ 3] = (Node) {0xb0000004,0xa000001b};
  book->defs[0x00027086]->node[ 4] = (Node) {0xb0000005,0xa000001a};
  book->defs[0x00027086]->node[ 5] = (Node) {0xb0000006,0xa0000019};
  book->defs[0x00027086]->node[ 6] = (Node) {0xb0000007,0xa0000018};
  book->defs[0x00027086]->node[ 7] = (Node) {0xb0000008,0xa0000017};
  book->defs[0x00027086]->node[ 8] = (Node) {0xb0000009,0xa0000016};
  book->defs[0x00027086]->node[ 9] = (Node) {0xb000000a,0xa0000015};
  book->defs[0x00027086]->node[10] = (Node) {0xb000000b,0xa0000014};
  book->defs[0x00027086]->node[11] = (Node) {0xb000000c,0xa0000013};
  book->defs[0x00027086]->node[12] = (Node) {0xb000000d,0xa0000012};
  book->defs[0x00027086]->node[13] = (Node) {0xb000000e,0xa0000011};
  book->defs[0x00027086]->node[14] = (Node) {0xa000000f,0xa0000010};
  book->defs[0x00027086]->node[15] = (Node) {0x5000001e,0x50000010};
  book->defs[0x00027086]->node[16] = (Node) {0x6000000f,0x50000011};
  book->defs[0x00027086]->node[17] = (Node) {0x60000010,0x50000012};
  book->defs[0x00027086]->node[18] = (Node) {0x60000011,0x50000013};
  book->defs[0x00027086]->node[19] = (Node) {0x60000012,0x50000014};
  book->defs[0x00027086]->node[20] = (Node) {0x60000013,0x50000015};
  book->defs[0x00027086]->node[21] = (Node) {0x60000014,0x50000016};
  book->defs[0x00027086]->node[22] = (Node) {0x60000015,0x50000017};
  book->defs[0x00027086]->node[23] = (Node) {0x60000016,0x50000018};
  book->defs[0x00027086]->node[24] = (Node) {0x60000017,0x50000019};
  book->defs[0x00027086]->node[25] = (Node) {0x60000018,0x5000001a};
  book->defs[0x00027086]->node[26] = (Node) {0x60000019,0x5000001b};
  book->defs[0x00027086]->node[27] = (Node) {0x6000001a,0x5000001c};
  book->defs[0x00027086]->node[28] = (Node) {0x6000001b,0x5000001d};
  book->defs[0x00027086]->node[29] = (Node) {0x6000001c,0x6000001e};
  book->defs[0x00027086]->node[30] = (Node) {0x5000000f,0x6000001d};
  book->defs[0x00027086]->locs     = (u32*) malloc(31 * sizeof(u32));
  // c16
  book->defs[0x00027087]           = (Term*) malloc(sizeof(Term));
  book->defs[0x00027087]->root     = 0xa0000000;
  book->defs[0x00027087]->alen     = 0;
  book->defs[0x00027087]->acts     = (Wire*) malloc(0 * sizeof(Wire));
  book->defs[0x00027087]->nlen     = 33;
  book->defs[0x00027087]->node     = (Node*) malloc(33 * sizeof(Node));
  book->defs[0x00027087]->node[ 0] = (Node) {0xb0000001,0xa0000020};
  book->defs[0x00027087]->node[ 1] = (Node) {0xb0000002,0xa000001f};
  book->defs[0x00027087]->node[ 2] = (Node) {0xb0000003,0xa000001e};
  book->defs[0x00027087]->node[ 3] = (Node) {0xb0000004,0xa000001d};
  book->defs[0x00027087]->node[ 4] = (Node) {0xb0000005,0xa000001c};
  book->defs[0x00027087]->node[ 5] = (Node) {0xb0000006,0xa000001b};
  book->defs[0x00027087]->node[ 6] = (Node) {0xb0000007,0xa000001a};
  book->defs[0x00027087]->node[ 7] = (Node) {0xb0000008,0xa0000019};
  book->defs[0x00027087]->node[ 8] = (Node) {0xb0000009,0xa0000018};
  book->defs[0x00027087]->node[ 9] = (Node) {0xb000000a,0xa0000017};
  book->defs[0x00027087]->node[10] = (Node) {0xb000000b,0xa0000016};
  book->defs[0x00027087]->node[11] = (Node) {0xb000000c,0xa0000015};
  book->defs[0x00027087]->node[12] = (Node) {0xb000000d,0xa0000014};
  book->defs[0x00027087]->node[13] = (Node) {0xb000000e,0xa0000013};
  book->defs[0x00027087]->node[14] = (Node) {0xb000000f,0xa0000012};
  book->defs[0x00027087]->node[15] = (Node) {0xa0000010,0xa0000011};
  book->defs[0x00027087]->node[16] = (Node) {0x50000020,0x50000011};
  book->defs[0x00027087]->node[17] = (Node) {0x60000010,0x50000012};
  book->defs[0x00027087]->node[18] = (Node) {0x60000011,0x50000013};
  book->defs[0x00027087]->node[19] = (Node) {0x60000012,0x50000014};
  book->defs[0x00027087]->node[20] = (Node) {0x60000013,0x50000015};
  book->defs[0x00027087]->node[21] = (Node) {0x60000014,0x50000016};
  book->defs[0x00027087]->node[22] = (Node) {0x60000015,0x50000017};
  book->defs[0x00027087]->node[23] = (Node) {0x60000016,0x50000018};
  book->defs[0x00027087]->node[24] = (Node) {0x60000017,0x50000019};
  book->defs[0x00027087]->node[25] = (Node) {0x60000018,0x5000001a};
  book->defs[0x00027087]->node[26] = (Node) {0x60000019,0x5000001b};
  book->defs[0x00027087]->node[27] = (Node) {0x6000001a,0x5000001c};
  book->defs[0x00027087]->node[28] = (Node) {0x6000001b,0x5000001d};
  book->defs[0x00027087]->node[29] = (Node) {0x6000001c,0x5000001e};
  book->defs[0x00027087]->node[30] = (Node) {0x6000001d,0x5000001f};
  book->defs[0x00027087]->node[31] = (Node) {0x6000001e,0x60000020};
  book->defs[0x00027087]->node[32] = (Node) {0x50000010,0x6000001f};
  book->defs[0x00027087]->locs     = (u32*) malloc(33 * sizeof(u32));
  // c17
  book->defs[0x00027088]           = (Term*) malloc(sizeof(Term));
  book->defs[0x00027088]->root     = 0xa0000000;
  book->defs[0x00027088]->alen     = 0;
  book->defs[0x00027088]->acts     = (Wire*) malloc(0 * sizeof(Wire));
  book->defs[0x00027088]->nlen     = 35;
  book->defs[0x00027088]->node     = (Node*) malloc(35 * sizeof(Node));
  book->defs[0x00027088]->node[ 0] = (Node) {0xb0000001,0xa0000022};
  book->defs[0x00027088]->node[ 1] = (Node) {0xb0000002,0xa0000021};
  book->defs[0x00027088]->node[ 2] = (Node) {0xb0000003,0xa0000020};
  book->defs[0x00027088]->node[ 3] = (Node) {0xb0000004,0xa000001f};
  book->defs[0x00027088]->node[ 4] = (Node) {0xb0000005,0xa000001e};
  book->defs[0x00027088]->node[ 5] = (Node) {0xb0000006,0xa000001d};
  book->defs[0x00027088]->node[ 6] = (Node) {0xb0000007,0xa000001c};
  book->defs[0x00027088]->node[ 7] = (Node) {0xb0000008,0xa000001b};
  book->defs[0x00027088]->node[ 8] = (Node) {0xb0000009,0xa000001a};
  book->defs[0x00027088]->node[ 9] = (Node) {0xb000000a,0xa0000019};
  book->defs[0x00027088]->node[10] = (Node) {0xb000000b,0xa0000018};
  book->defs[0x00027088]->node[11] = (Node) {0xb000000c,0xa0000017};
  book->defs[0x00027088]->node[12] = (Node) {0xb000000d,0xa0000016};
  book->defs[0x00027088]->node[13] = (Node) {0xb000000e,0xa0000015};
  book->defs[0x00027088]->node[14] = (Node) {0xb000000f,0xa0000014};
  book->defs[0x00027088]->node[15] = (Node) {0xb0000010,0xa0000013};
  book->defs[0x00027088]->node[16] = (Node) {0xa0000011,0xa0000012};
  book->defs[0x00027088]->node[17] = (Node) {0x50000022,0x50000012};
  book->defs[0x00027088]->node[18] = (Node) {0x60000011,0x50000013};
  book->defs[0x00027088]->node[19] = (Node) {0x60000012,0x50000014};
  book->defs[0x00027088]->node[20] = (Node) {0x60000013,0x50000015};
  book->defs[0x00027088]->node[21] = (Node) {0x60000014,0x50000016};
  book->defs[0x00027088]->node[22] = (Node) {0x60000015,0x50000017};
  book->defs[0x00027088]->node[23] = (Node) {0x60000016,0x50000018};
  book->defs[0x00027088]->node[24] = (Node) {0x60000017,0x50000019};
  book->defs[0x00027088]->node[25] = (Node) {0x60000018,0x5000001a};
  book->defs[0x00027088]->node[26] = (Node) {0x60000019,0x5000001b};
  book->defs[0x00027088]->node[27] = (Node) {0x6000001a,0x5000001c};
  book->defs[0x00027088]->node[28] = (Node) {0x6000001b,0x5000001d};
  book->defs[0x00027088]->node[29] = (Node) {0x6000001c,0x5000001e};
  book->defs[0x00027088]->node[30] = (Node) {0x6000001d,0x5000001f};
  book->defs[0x00027088]->node[31] = (Node) {0x6000001e,0x50000020};
  book->defs[0x00027088]->node[32] = (Node) {0x6000001f,0x50000021};
  book->defs[0x00027088]->node[33] = (Node) {0x60000020,0x60000022};
  book->defs[0x00027088]->node[34] = (Node) {0x50000011,0x60000021};
  book->defs[0x00027088]->locs     = (u32*) malloc(35 * sizeof(u32));
  // c18
  book->defs[0x00027089]           = (Term*) malloc(sizeof(Term));
  book->defs[0x00027089]->root     = 0xa0000000;
  book->defs[0x00027089]->alen     = 0;
  book->defs[0x00027089]->acts     = (Wire*) malloc(0 * sizeof(Wire));
  book->defs[0x00027089]->nlen     = 37;
  book->defs[0x00027089]->node     = (Node*) malloc(37 * sizeof(Node));
  book->defs[0x00027089]->node[ 0] = (Node) {0xb0000001,0xa0000024};
  book->defs[0x00027089]->node[ 1] = (Node) {0xb0000002,0xa0000023};
  book->defs[0x00027089]->node[ 2] = (Node) {0xb0000003,0xa0000022};
  book->defs[0x00027089]->node[ 3] = (Node) {0xb0000004,0xa0000021};
  book->defs[0x00027089]->node[ 4] = (Node) {0xb0000005,0xa0000020};
  book->defs[0x00027089]->node[ 5] = (Node) {0xb0000006,0xa000001f};
  book->defs[0x00027089]->node[ 6] = (Node) {0xb0000007,0xa000001e};
  book->defs[0x00027089]->node[ 7] = (Node) {0xb0000008,0xa000001d};
  book->defs[0x00027089]->node[ 8] = (Node) {0xb0000009,0xa000001c};
  book->defs[0x00027089]->node[ 9] = (Node) {0xb000000a,0xa000001b};
  book->defs[0x00027089]->node[10] = (Node) {0xb000000b,0xa000001a};
  book->defs[0x00027089]->node[11] = (Node) {0xb000000c,0xa0000019};
  book->defs[0x00027089]->node[12] = (Node) {0xb000000d,0xa0000018};
  book->defs[0x00027089]->node[13] = (Node) {0xb000000e,0xa0000017};
  book->defs[0x00027089]->node[14] = (Node) {0xb000000f,0xa0000016};
  book->defs[0x00027089]->node[15] = (Node) {0xb0000010,0xa0000015};
  book->defs[0x00027089]->node[16] = (Node) {0xb0000011,0xa0000014};
  book->defs[0x00027089]->node[17] = (Node) {0xa0000012,0xa0000013};
  book->defs[0x00027089]->node[18] = (Node) {0x50000024,0x50000013};
  book->defs[0x00027089]->node[19] = (Node) {0x60000012,0x50000014};
  book->defs[0x00027089]->node[20] = (Node) {0x60000013,0x50000015};
  book->defs[0x00027089]->node[21] = (Node) {0x60000014,0x50000016};
  book->defs[0x00027089]->node[22] = (Node) {0x60000015,0x50000017};
  book->defs[0x00027089]->node[23] = (Node) {0x60000016,0x50000018};
  book->defs[0x00027089]->node[24] = (Node) {0x60000017,0x50000019};
  book->defs[0x00027089]->node[25] = (Node) {0x60000018,0x5000001a};
  book->defs[0x00027089]->node[26] = (Node) {0x60000019,0x5000001b};
  book->defs[0x00027089]->node[27] = (Node) {0x6000001a,0x5000001c};
  book->defs[0x00027089]->node[28] = (Node) {0x6000001b,0x5000001d};
  book->defs[0x00027089]->node[29] = (Node) {0x6000001c,0x5000001e};
  book->defs[0x00027089]->node[30] = (Node) {0x6000001d,0x5000001f};
  book->defs[0x00027089]->node[31] = (Node) {0x6000001e,0x50000020};
  book->defs[0x00027089]->node[32] = (Node) {0x6000001f,0x50000021};
  book->defs[0x00027089]->node[33] = (Node) {0x60000020,0x50000022};
  book->defs[0x00027089]->node[34] = (Node) {0x60000021,0x50000023};
  book->defs[0x00027089]->node[35] = (Node) {0x60000022,0x60000024};
  book->defs[0x00027089]->node[36] = (Node) {0x50000012,0x60000023};
  book->defs[0x00027089]->locs     = (u32*) malloc(37 * sizeof(u32));
  // c19
  book->defs[0x0002708a]           = (Term*) malloc(sizeof(Term));
  book->defs[0x0002708a]->root     = 0xa0000000;
  book->defs[0x0002708a]->alen     = 0;
  book->defs[0x0002708a]->acts     = (Wire*) malloc(0 * sizeof(Wire));
  book->defs[0x0002708a]->nlen     = 39;
  book->defs[0x0002708a]->node     = (Node*) malloc(39 * sizeof(Node));
  book->defs[0x0002708a]->node[ 0] = (Node) {0xb0000001,0xa0000026};
  book->defs[0x0002708a]->node[ 1] = (Node) {0xb0000002,0xa0000025};
  book->defs[0x0002708a]->node[ 2] = (Node) {0xb0000003,0xa0000024};
  book->defs[0x0002708a]->node[ 3] = (Node) {0xb0000004,0xa0000023};
  book->defs[0x0002708a]->node[ 4] = (Node) {0xb0000005,0xa0000022};
  book->defs[0x0002708a]->node[ 5] = (Node) {0xb0000006,0xa0000021};
  book->defs[0x0002708a]->node[ 6] = (Node) {0xb0000007,0xa0000020};
  book->defs[0x0002708a]->node[ 7] = (Node) {0xb0000008,0xa000001f};
  book->defs[0x0002708a]->node[ 8] = (Node) {0xb0000009,0xa000001e};
  book->defs[0x0002708a]->node[ 9] = (Node) {0xb000000a,0xa000001d};
  book->defs[0x0002708a]->node[10] = (Node) {0xb000000b,0xa000001c};
  book->defs[0x0002708a]->node[11] = (Node) {0xb000000c,0xa000001b};
  book->defs[0x0002708a]->node[12] = (Node) {0xb000000d,0xa000001a};
  book->defs[0x0002708a]->node[13] = (Node) {0xb000000e,0xa0000019};
  book->defs[0x0002708a]->node[14] = (Node) {0xb000000f,0xa0000018};
  book->defs[0x0002708a]->node[15] = (Node) {0xb0000010,0xa0000017};
  book->defs[0x0002708a]->node[16] = (Node) {0xb0000011,0xa0000016};
  book->defs[0x0002708a]->node[17] = (Node) {0xb0000012,0xa0000015};
  book->defs[0x0002708a]->node[18] = (Node) {0xa0000013,0xa0000014};
  book->defs[0x0002708a]->node[19] = (Node) {0x50000026,0x50000014};
  book->defs[0x0002708a]->node[20] = (Node) {0x60000013,0x50000015};
  book->defs[0x0002708a]->node[21] = (Node) {0x60000014,0x50000016};
  book->defs[0x0002708a]->node[22] = (Node) {0x60000015,0x50000017};
  book->defs[0x0002708a]->node[23] = (Node) {0x60000016,0x50000018};
  book->defs[0x0002708a]->node[24] = (Node) {0x60000017,0x50000019};
  book->defs[0x0002708a]->node[25] = (Node) {0x60000018,0x5000001a};
  book->defs[0x0002708a]->node[26] = (Node) {0x60000019,0x5000001b};
  book->defs[0x0002708a]->node[27] = (Node) {0x6000001a,0x5000001c};
  book->defs[0x0002708a]->node[28] = (Node) {0x6000001b,0x5000001d};
  book->defs[0x0002708a]->node[29] = (Node) {0x6000001c,0x5000001e};
  book->defs[0x0002708a]->node[30] = (Node) {0x6000001d,0x5000001f};
  book->defs[0x0002708a]->node[31] = (Node) {0x6000001e,0x50000020};
  book->defs[0x0002708a]->node[32] = (Node) {0x6000001f,0x50000021};
  book->defs[0x0002708a]->node[33] = (Node) {0x60000020,0x50000022};
  book->defs[0x0002708a]->node[34] = (Node) {0x60000021,0x50000023};
  book->defs[0x0002708a]->node[35] = (Node) {0x60000022,0x50000024};
  book->defs[0x0002708a]->node[36] = (Node) {0x60000023,0x50000025};
  book->defs[0x0002708a]->node[37] = (Node) {0x60000024,0x60000026};
  book->defs[0x0002708a]->node[38] = (Node) {0x50000013,0x60000025};
  book->defs[0x0002708a]->locs     = (u32*) malloc(39 * sizeof(u32));
  // c20
  book->defs[0x000270c1]           = (Term*) malloc(sizeof(Term));
  book->defs[0x000270c1]->root     = 0xa0000000;
  book->defs[0x000270c1]->alen     = 0;
  book->defs[0x000270c1]->acts     = (Wire*) malloc(0 * sizeof(Wire));
  book->defs[0x000270c1]->nlen     = 41;
  book->defs[0x000270c1]->node     = (Node*) malloc(41 * sizeof(Node));
  book->defs[0x000270c1]->node[ 0] = (Node) {0xb0000001,0xa0000028};
  book->defs[0x000270c1]->node[ 1] = (Node) {0xb0000002,0xa0000027};
  book->defs[0x000270c1]->node[ 2] = (Node) {0xb0000003,0xa0000026};
  book->defs[0x000270c1]->node[ 3] = (Node) {0xb0000004,0xa0000025};
  book->defs[0x000270c1]->node[ 4] = (Node) {0xb0000005,0xa0000024};
  book->defs[0x000270c1]->node[ 5] = (Node) {0xb0000006,0xa0000023};
  book->defs[0x000270c1]->node[ 6] = (Node) {0xb0000007,0xa0000022};
  book->defs[0x000270c1]->node[ 7] = (Node) {0xb0000008,0xa0000021};
  book->defs[0x000270c1]->node[ 8] = (Node) {0xb0000009,0xa0000020};
  book->defs[0x000270c1]->node[ 9] = (Node) {0xb000000a,0xa000001f};
  book->defs[0x000270c1]->node[10] = (Node) {0xb000000b,0xa000001e};
  book->defs[0x000270c1]->node[11] = (Node) {0xb000000c,0xa000001d};
  book->defs[0x000270c1]->node[12] = (Node) {0xb000000d,0xa000001c};
  book->defs[0x000270c1]->node[13] = (Node) {0xb000000e,0xa000001b};
  book->defs[0x000270c1]->node[14] = (Node) {0xb000000f,0xa000001a};
  book->defs[0x000270c1]->node[15] = (Node) {0xb0000010,0xa0000019};
  book->defs[0x000270c1]->node[16] = (Node) {0xb0000011,0xa0000018};
  book->defs[0x000270c1]->node[17] = (Node) {0xb0000012,0xa0000017};
  book->defs[0x000270c1]->node[18] = (Node) {0xb0000013,0xa0000016};
  book->defs[0x000270c1]->node[19] = (Node) {0xa0000014,0xa0000015};
  book->defs[0x000270c1]->node[20] = (Node) {0x50000028,0x50000015};
  book->defs[0x000270c1]->node[21] = (Node) {0x60000014,0x50000016};
  book->defs[0x000270c1]->node[22] = (Node) {0x60000015,0x50000017};
  book->defs[0x000270c1]->node[23] = (Node) {0x60000016,0x50000018};
  book->defs[0x000270c1]->node[24] = (Node) {0x60000017,0x50000019};
  book->defs[0x000270c1]->node[25] = (Node) {0x60000018,0x5000001a};
  book->defs[0x000270c1]->node[26] = (Node) {0x60000019,0x5000001b};
  book->defs[0x000270c1]->node[27] = (Node) {0x6000001a,0x5000001c};
  book->defs[0x000270c1]->node[28] = (Node) {0x6000001b,0x5000001d};
  book->defs[0x000270c1]->node[29] = (Node) {0x6000001c,0x5000001e};
  book->defs[0x000270c1]->node[30] = (Node) {0x6000001d,0x5000001f};
  book->defs[0x000270c1]->node[31] = (Node) {0x6000001e,0x50000020};
  book->defs[0x000270c1]->node[32] = (Node) {0x6000001f,0x50000021};
  book->defs[0x000270c1]->node[33] = (Node) {0x60000020,0x50000022};
  book->defs[0x000270c1]->node[34] = (Node) {0x60000021,0x50000023};
  book->defs[0x000270c1]->node[35] = (Node) {0x60000022,0x50000024};
  book->defs[0x000270c1]->node[36] = (Node) {0x60000023,0x50000025};
  book->defs[0x000270c1]->node[37] = (Node) {0x60000024,0x50000026};
  book->defs[0x000270c1]->node[38] = (Node) {0x60000025,0x50000027};
  book->defs[0x000270c1]->node[39] = (Node) {0x60000026,0x60000028};
  book->defs[0x000270c1]->node[40] = (Node) {0x50000014,0x60000027};
  book->defs[0x000270c1]->locs     = (u32*) malloc(41 * sizeof(u32));
  // c21
  book->defs[0x000270c2]           = (Term*) malloc(sizeof(Term));
  book->defs[0x000270c2]->root     = 0xa0000000;
  book->defs[0x000270c2]->alen     = 0;
  book->defs[0x000270c2]->acts     = (Wire*) malloc(0 * sizeof(Wire));
  book->defs[0x000270c2]->nlen     = 43;
  book->defs[0x000270c2]->node     = (Node*) malloc(43 * sizeof(Node));
  book->defs[0x000270c2]->node[ 0] = (Node) {0xb0000001,0xa000002a};
  book->defs[0x000270c2]->node[ 1] = (Node) {0xb0000002,0xa0000029};
  book->defs[0x000270c2]->node[ 2] = (Node) {0xb0000003,0xa0000028};
  book->defs[0x000270c2]->node[ 3] = (Node) {0xb0000004,0xa0000027};
  book->defs[0x000270c2]->node[ 4] = (Node) {0xb0000005,0xa0000026};
  book->defs[0x000270c2]->node[ 5] = (Node) {0xb0000006,0xa0000025};
  book->defs[0x000270c2]->node[ 6] = (Node) {0xb0000007,0xa0000024};
  book->defs[0x000270c2]->node[ 7] = (Node) {0xb0000008,0xa0000023};
  book->defs[0x000270c2]->node[ 8] = (Node) {0xb0000009,0xa0000022};
  book->defs[0x000270c2]->node[ 9] = (Node) {0xb000000a,0xa0000021};
  book->defs[0x000270c2]->node[10] = (Node) {0xb000000b,0xa0000020};
  book->defs[0x000270c2]->node[11] = (Node) {0xb000000c,0xa000001f};
  book->defs[0x000270c2]->node[12] = (Node) {0xb000000d,0xa000001e};
  book->defs[0x000270c2]->node[13] = (Node) {0xb000000e,0xa000001d};
  book->defs[0x000270c2]->node[14] = (Node) {0xb000000f,0xa000001c};
  book->defs[0x000270c2]->node[15] = (Node) {0xb0000010,0xa000001b};
  book->defs[0x000270c2]->node[16] = (Node) {0xb0000011,0xa000001a};
  book->defs[0x000270c2]->node[17] = (Node) {0xb0000012,0xa0000019};
  book->defs[0x000270c2]->node[18] = (Node) {0xb0000013,0xa0000018};
  book->defs[0x000270c2]->node[19] = (Node) {0xb0000014,0xa0000017};
  book->defs[0x000270c2]->node[20] = (Node) {0xa0000015,0xa0000016};
  book->defs[0x000270c2]->node[21] = (Node) {0x5000002a,0x50000016};
  book->defs[0x000270c2]->node[22] = (Node) {0x60000015,0x50000017};
  book->defs[0x000270c2]->node[23] = (Node) {0x60000016,0x50000018};
  book->defs[0x000270c2]->node[24] = (Node) {0x60000017,0x50000019};
  book->defs[0x000270c2]->node[25] = (Node) {0x60000018,0x5000001a};
  book->defs[0x000270c2]->node[26] = (Node) {0x60000019,0x5000001b};
  book->defs[0x000270c2]->node[27] = (Node) {0x6000001a,0x5000001c};
  book->defs[0x000270c2]->node[28] = (Node) {0x6000001b,0x5000001d};
  book->defs[0x000270c2]->node[29] = (Node) {0x6000001c,0x5000001e};
  book->defs[0x000270c2]->node[30] = (Node) {0x6000001d,0x5000001f};
  book->defs[0x000270c2]->node[31] = (Node) {0x6000001e,0x50000020};
  book->defs[0x000270c2]->node[32] = (Node) {0x6000001f,0x50000021};
  book->defs[0x000270c2]->node[33] = (Node) {0x60000020,0x50000022};
  book->defs[0x000270c2]->node[34] = (Node) {0x60000021,0x50000023};
  book->defs[0x000270c2]->node[35] = (Node) {0x60000022,0x50000024};
  book->defs[0x000270c2]->node[36] = (Node) {0x60000023,0x50000025};
  book->defs[0x000270c2]->node[37] = (Node) {0x60000024,0x50000026};
  book->defs[0x000270c2]->node[38] = (Node) {0x60000025,0x50000027};
  book->defs[0x000270c2]->node[39] = (Node) {0x60000026,0x50000028};
  book->defs[0x000270c2]->node[40] = (Node) {0x60000027,0x50000029};
  book->defs[0x000270c2]->node[41] = (Node) {0x60000028,0x6000002a};
  book->defs[0x000270c2]->node[42] = (Node) {0x50000015,0x60000029};
  book->defs[0x000270c2]->locs     = (u32*) malloc(43 * sizeof(u32));
  // c22
  book->defs[0x000270c3]           = (Term*) malloc(sizeof(Term));
  book->defs[0x000270c3]->root     = 0xa0000000;
  book->defs[0x000270c3]->alen     = 0;
  book->defs[0x000270c3]->acts     = (Wire*) malloc(0 * sizeof(Wire));
  book->defs[0x000270c3]->nlen     = 45;
  book->defs[0x000270c3]->node     = (Node*) malloc(45 * sizeof(Node));
  book->defs[0x000270c3]->node[ 0] = (Node) {0xb0000001,0xa000002c};
  book->defs[0x000270c3]->node[ 1] = (Node) {0xb0000002,0xa000002b};
  book->defs[0x000270c3]->node[ 2] = (Node) {0xb0000003,0xa000002a};
  book->defs[0x000270c3]->node[ 3] = (Node) {0xb0000004,0xa0000029};
  book->defs[0x000270c3]->node[ 4] = (Node) {0xb0000005,0xa0000028};
  book->defs[0x000270c3]->node[ 5] = (Node) {0xb0000006,0xa0000027};
  book->defs[0x000270c3]->node[ 6] = (Node) {0xb0000007,0xa0000026};
  book->defs[0x000270c3]->node[ 7] = (Node) {0xb0000008,0xa0000025};
  book->defs[0x000270c3]->node[ 8] = (Node) {0xb0000009,0xa0000024};
  book->defs[0x000270c3]->node[ 9] = (Node) {0xb000000a,0xa0000023};
  book->defs[0x000270c3]->node[10] = (Node) {0xb000000b,0xa0000022};
  book->defs[0x000270c3]->node[11] = (Node) {0xb000000c,0xa0000021};
  book->defs[0x000270c3]->node[12] = (Node) {0xb000000d,0xa0000020};
  book->defs[0x000270c3]->node[13] = (Node) {0xb000000e,0xa000001f};
  book->defs[0x000270c3]->node[14] = (Node) {0xb000000f,0xa000001e};
  book->defs[0x000270c3]->node[15] = (Node) {0xb0000010,0xa000001d};
  book->defs[0x000270c3]->node[16] = (Node) {0xb0000011,0xa000001c};
  book->defs[0x000270c3]->node[17] = (Node) {0xb0000012,0xa000001b};
  book->defs[0x000270c3]->node[18] = (Node) {0xb0000013,0xa000001a};
  book->defs[0x000270c3]->node[19] = (Node) {0xb0000014,0xa0000019};
  book->defs[0x000270c3]->node[20] = (Node) {0xb0000015,0xa0000018};
  book->defs[0x000270c3]->node[21] = (Node) {0xa0000016,0xa0000017};
  book->defs[0x000270c3]->node[22] = (Node) {0x5000002c,0x50000017};
  book->defs[0x000270c3]->node[23] = (Node) {0x60000016,0x50000018};
  book->defs[0x000270c3]->node[24] = (Node) {0x60000017,0x50000019};
  book->defs[0x000270c3]->node[25] = (Node) {0x60000018,0x5000001a};
  book->defs[0x000270c3]->node[26] = (Node) {0x60000019,0x5000001b};
  book->defs[0x000270c3]->node[27] = (Node) {0x6000001a,0x5000001c};
  book->defs[0x000270c3]->node[28] = (Node) {0x6000001b,0x5000001d};
  book->defs[0x000270c3]->node[29] = (Node) {0x6000001c,0x5000001e};
  book->defs[0x000270c3]->node[30] = (Node) {0x6000001d,0x5000001f};
  book->defs[0x000270c3]->node[31] = (Node) {0x6000001e,0x50000020};
  book->defs[0x000270c3]->node[32] = (Node) {0x6000001f,0x50000021};
  book->defs[0x000270c3]->node[33] = (Node) {0x60000020,0x50000022};
  book->defs[0x000270c3]->node[34] = (Node) {0x60000021,0x50000023};
  book->defs[0x000270c3]->node[35] = (Node) {0x60000022,0x50000024};
  book->defs[0x000270c3]->node[36] = (Node) {0x60000023,0x50000025};
  book->defs[0x000270c3]->node[37] = (Node) {0x60000024,0x50000026};
  book->defs[0x000270c3]->node[38] = (Node) {0x60000025,0x50000027};
  book->defs[0x000270c3]->node[39] = (Node) {0x60000026,0x50000028};
  book->defs[0x000270c3]->node[40] = (Node) {0x60000027,0x50000029};
  book->defs[0x000270c3]->node[41] = (Node) {0x60000028,0x5000002a};
  book->defs[0x000270c3]->node[42] = (Node) {0x60000029,0x5000002b};
  book->defs[0x000270c3]->node[43] = (Node) {0x6000002a,0x6000002c};
  book->defs[0x000270c3]->node[44] = (Node) {0x50000016,0x6000002b};
  book->defs[0x000270c3]->locs     = (u32*) malloc(45 * sizeof(u32));
  // c23
  book->defs[0x000270c4]           = (Term*) malloc(sizeof(Term));
  book->defs[0x000270c4]->root     = 0xa0000000;
  book->defs[0x000270c4]->alen     = 0;
  book->defs[0x000270c4]->acts     = (Wire*) malloc(0 * sizeof(Wire));
  book->defs[0x000270c4]->nlen     = 47;
  book->defs[0x000270c4]->node     = (Node*) malloc(47 * sizeof(Node));
  book->defs[0x000270c4]->node[ 0] = (Node) {0xb0000001,0xa000002e};
  book->defs[0x000270c4]->node[ 1] = (Node) {0xb0000002,0xa000002d};
  book->defs[0x000270c4]->node[ 2] = (Node) {0xb0000003,0xa000002c};
  book->defs[0x000270c4]->node[ 3] = (Node) {0xb0000004,0xa000002b};
  book->defs[0x000270c4]->node[ 4] = (Node) {0xb0000005,0xa000002a};
  book->defs[0x000270c4]->node[ 5] = (Node) {0xb0000006,0xa0000029};
  book->defs[0x000270c4]->node[ 6] = (Node) {0xb0000007,0xa0000028};
  book->defs[0x000270c4]->node[ 7] = (Node) {0xb0000008,0xa0000027};
  book->defs[0x000270c4]->node[ 8] = (Node) {0xb0000009,0xa0000026};
  book->defs[0x000270c4]->node[ 9] = (Node) {0xb000000a,0xa0000025};
  book->defs[0x000270c4]->node[10] = (Node) {0xb000000b,0xa0000024};
  book->defs[0x000270c4]->node[11] = (Node) {0xb000000c,0xa0000023};
  book->defs[0x000270c4]->node[12] = (Node) {0xb000000d,0xa0000022};
  book->defs[0x000270c4]->node[13] = (Node) {0xb000000e,0xa0000021};
  book->defs[0x000270c4]->node[14] = (Node) {0xb000000f,0xa0000020};
  book->defs[0x000270c4]->node[15] = (Node) {0xb0000010,0xa000001f};
  book->defs[0x000270c4]->node[16] = (Node) {0xb0000011,0xa000001e};
  book->defs[0x000270c4]->node[17] = (Node) {0xb0000012,0xa000001d};
  book->defs[0x000270c4]->node[18] = (Node) {0xb0000013,0xa000001c};
  book->defs[0x000270c4]->node[19] = (Node) {0xb0000014,0xa000001b};
  book->defs[0x000270c4]->node[20] = (Node) {0xb0000015,0xa000001a};
  book->defs[0x000270c4]->node[21] = (Node) {0xb0000016,0xa0000019};
  book->defs[0x000270c4]->node[22] = (Node) {0xa0000017,0xa0000018};
  book->defs[0x000270c4]->node[23] = (Node) {0x5000002e,0x50000018};
  book->defs[0x000270c4]->node[24] = (Node) {0x60000017,0x50000019};
  book->defs[0x000270c4]->node[25] = (Node) {0x60000018,0x5000001a};
  book->defs[0x000270c4]->node[26] = (Node) {0x60000019,0x5000001b};
  book->defs[0x000270c4]->node[27] = (Node) {0x6000001a,0x5000001c};
  book->defs[0x000270c4]->node[28] = (Node) {0x6000001b,0x5000001d};
  book->defs[0x000270c4]->node[29] = (Node) {0x6000001c,0x5000001e};
  book->defs[0x000270c4]->node[30] = (Node) {0x6000001d,0x5000001f};
  book->defs[0x000270c4]->node[31] = (Node) {0x6000001e,0x50000020};
  book->defs[0x000270c4]->node[32] = (Node) {0x6000001f,0x50000021};
  book->defs[0x000270c4]->node[33] = (Node) {0x60000020,0x50000022};
  book->defs[0x000270c4]->node[34] = (Node) {0x60000021,0x50000023};
  book->defs[0x000270c4]->node[35] = (Node) {0x60000022,0x50000024};
  book->defs[0x000270c4]->node[36] = (Node) {0x60000023,0x50000025};
  book->defs[0x000270c4]->node[37] = (Node) {0x60000024,0x50000026};
  book->defs[0x000270c4]->node[38] = (Node) {0x60000025,0x50000027};
  book->defs[0x000270c4]->node[39] = (Node) {0x60000026,0x50000028};
  book->defs[0x000270c4]->node[40] = (Node) {0x60000027,0x50000029};
  book->defs[0x000270c4]->node[41] = (Node) {0x60000028,0x5000002a};
  book->defs[0x000270c4]->node[42] = (Node) {0x60000029,0x5000002b};
  book->defs[0x000270c4]->node[43] = (Node) {0x6000002a,0x5000002c};
  book->defs[0x000270c4]->node[44] = (Node) {0x6000002b,0x5000002d};
  book->defs[0x000270c4]->node[45] = (Node) {0x6000002c,0x6000002e};
  book->defs[0x000270c4]->node[46] = (Node) {0x50000017,0x6000002d};
  book->defs[0x000270c4]->locs     = (u32*) malloc(47 * sizeof(u32));
  // c24
  book->defs[0x000270c5]           = (Term*) malloc(sizeof(Term));
  book->defs[0x000270c5]->root     = 0xa0000000;
  book->defs[0x000270c5]->alen     = 0;
  book->defs[0x000270c5]->acts     = (Wire*) malloc(0 * sizeof(Wire));
  book->defs[0x000270c5]->nlen     = 49;
  book->defs[0x000270c5]->node     = (Node*) malloc(49 * sizeof(Node));
  book->defs[0x000270c5]->node[ 0] = (Node) {0xb0000001,0xa0000030};
  book->defs[0x000270c5]->node[ 1] = (Node) {0xb0000002,0xa000002f};
  book->defs[0x000270c5]->node[ 2] = (Node) {0xb0000003,0xa000002e};
  book->defs[0x000270c5]->node[ 3] = (Node) {0xb0000004,0xa000002d};
  book->defs[0x000270c5]->node[ 4] = (Node) {0xb0000005,0xa000002c};
  book->defs[0x000270c5]->node[ 5] = (Node) {0xb0000006,0xa000002b};
  book->defs[0x000270c5]->node[ 6] = (Node) {0xb0000007,0xa000002a};
  book->defs[0x000270c5]->node[ 7] = (Node) {0xb0000008,0xa0000029};
  book->defs[0x000270c5]->node[ 8] = (Node) {0xb0000009,0xa0000028};
  book->defs[0x000270c5]->node[ 9] = (Node) {0xb000000a,0xa0000027};
  book->defs[0x000270c5]->node[10] = (Node) {0xb000000b,0xa0000026};
  book->defs[0x000270c5]->node[11] = (Node) {0xb000000c,0xa0000025};
  book->defs[0x000270c5]->node[12] = (Node) {0xb000000d,0xa0000024};
  book->defs[0x000270c5]->node[13] = (Node) {0xb000000e,0xa0000023};
  book->defs[0x000270c5]->node[14] = (Node) {0xb000000f,0xa0000022};
  book->defs[0x000270c5]->node[15] = (Node) {0xb0000010,0xa0000021};
  book->defs[0x000270c5]->node[16] = (Node) {0xb0000011,0xa0000020};
  book->defs[0x000270c5]->node[17] = (Node) {0xb0000012,0xa000001f};
  book->defs[0x000270c5]->node[18] = (Node) {0xb0000013,0xa000001e};
  book->defs[0x000270c5]->node[19] = (Node) {0xb0000014,0xa000001d};
  book->defs[0x000270c5]->node[20] = (Node) {0xb0000015,0xa000001c};
  book->defs[0x000270c5]->node[21] = (Node) {0xb0000016,0xa000001b};
  book->defs[0x000270c5]->node[22] = (Node) {0xb0000017,0xa000001a};
  book->defs[0x000270c5]->node[23] = (Node) {0xa0000018,0xa0000019};
  book->defs[0x000270c5]->node[24] = (Node) {0x50000030,0x50000019};
  book->defs[0x000270c5]->node[25] = (Node) {0x60000018,0x5000001a};
  book->defs[0x000270c5]->node[26] = (Node) {0x60000019,0x5000001b};
  book->defs[0x000270c5]->node[27] = (Node) {0x6000001a,0x5000001c};
  book->defs[0x000270c5]->node[28] = (Node) {0x6000001b,0x5000001d};
  book->defs[0x000270c5]->node[29] = (Node) {0x6000001c,0x5000001e};
  book->defs[0x000270c5]->node[30] = (Node) {0x6000001d,0x5000001f};
  book->defs[0x000270c5]->node[31] = (Node) {0x6000001e,0x50000020};
  book->defs[0x000270c5]->node[32] = (Node) {0x6000001f,0x50000021};
  book->defs[0x000270c5]->node[33] = (Node) {0x60000020,0x50000022};
  book->defs[0x000270c5]->node[34] = (Node) {0x60000021,0x50000023};
  book->defs[0x000270c5]->node[35] = (Node) {0x60000022,0x50000024};
  book->defs[0x000270c5]->node[36] = (Node) {0x60000023,0x50000025};
  book->defs[0x000270c5]->node[37] = (Node) {0x60000024,0x50000026};
  book->defs[0x000270c5]->node[38] = (Node) {0x60000025,0x50000027};
  book->defs[0x000270c5]->node[39] = (Node) {0x60000026,0x50000028};
  book->defs[0x000270c5]->node[40] = (Node) {0x60000027,0x50000029};
  book->defs[0x000270c5]->node[41] = (Node) {0x60000028,0x5000002a};
  book->defs[0x000270c5]->node[42] = (Node) {0x60000029,0x5000002b};
  book->defs[0x000270c5]->node[43] = (Node) {0x6000002a,0x5000002c};
  book->defs[0x000270c5]->node[44] = (Node) {0x6000002b,0x5000002d};
  book->defs[0x000270c5]->node[45] = (Node) {0x6000002c,0x5000002e};
  book->defs[0x000270c5]->node[46] = (Node) {0x6000002d,0x5000002f};
  book->defs[0x000270c5]->node[47] = (Node) {0x6000002e,0x60000030};
  book->defs[0x000270c5]->node[48] = (Node) {0x50000018,0x6000002f};
  book->defs[0x000270c5]->locs     = (u32*) malloc(49 * sizeof(u32));
  // c_s
  book->defs[0x00027ff7]           = (Term*) malloc(sizeof(Term));
  book->defs[0x00027ff7]->root     = 0xa0000000;
  book->defs[0x00027ff7]->alen     = 0;
  book->defs[0x00027ff7]->acts     = (Wire*) malloc(0 * sizeof(Wire));
  book->defs[0x00027ff7]->nlen     = 7;
  book->defs[0x00027ff7]->node     = (Node*) malloc(7 * sizeof(Node));
  book->defs[0x00027ff7]->node[ 0] = (Node) {0xa0000001,0xa0000003};
  book->defs[0x00027ff7]->node[ 1] = (Node) {0x60000004,0xa0000002};
  book->defs[0x00027ff7]->node[ 2] = (Node) {0x50000006,0x50000005};
  book->defs[0x00027ff7]->node[ 3] = (Node) {0xb0000004,0xa0000006};
  book->defs[0x00027ff7]->node[ 4] = (Node) {0xa0000005,0x50000001};
  book->defs[0x00027ff7]->node[ 5] = (Node) {0x60000002,0x60000006};
  book->defs[0x00027ff7]->node[ 6] = (Node) {0x50000002,0x60000005};
  book->defs[0x00027ff7]->locs     = (u32*) malloc(7 * sizeof(u32));
  // c_z
  book->defs[0x00027ffe]           = (Term*) malloc(sizeof(Term));
  book->defs[0x00027ffe]->root     = 0xa0000000;
  book->defs[0x00027ffe]->alen     = 0;
  book->defs[0x00027ffe]->acts     = (Wire*) malloc(0 * sizeof(Wire));
  book->defs[0x00027ffe]->nlen     = 2;
  book->defs[0x00027ffe]->node     = (Node*) malloc(2 * sizeof(Node));
  book->defs[0x00027ffe]->node[ 0] = (Node) {0x30000000,0xa0000001};
  book->defs[0x00027ffe]->node[ 1] = (Node) {0x60000001,0x50000001};
  book->defs[0x00027ffe]->locs     = (u32*) malloc(2 * sizeof(u32));
  // dec
  book->defs[0x00028a67]           = (Term*) malloc(sizeof(Term));
  book->defs[0x00028a67]->root     = 0xa0000000;
  book->defs[0x00028a67]->alen     = 0;
  book->defs[0x00028a67]->acts     = (Wire*) malloc(0 * sizeof(Wire));
  book->defs[0x00028a67]->nlen     = 4;
  book->defs[0x00028a67]->node     = (Node*) malloc(4 * sizeof(Node));
  book->defs[0x00028a67]->node[ 0] = (Node) {0xa0000001,0x60000003};
  book->defs[0x00028a67]->node[ 1] = (Node) {0x10a299d9,0xa0000002};
  book->defs[0x00028a67]->node[ 2] = (Node) {0x10a299d3,0xa0000003};
  book->defs[0x00028a67]->node[ 3] = (Node) {0x1000000f,0x60000000};
  book->defs[0x00028a67]->locs     = (u32*) malloc(4 * sizeof(u32));
  // ex0
  book->defs[0x00029f01]           = (Term*) malloc(sizeof(Term));
  book->defs[0x00029f01]->root     = 0x60000001;
  book->defs[0x00029f01]->alen     = 1;
  book->defs[0x00029f01]->acts     = (Wire*) malloc(1 * sizeof(Wire));
  book->defs[0x00029f01]->acts[ 0] = (Wire) {0x100009c2,0xa0000000};
  book->defs[0x00029f01]->nlen     = 2;
  book->defs[0x00029f01]->node     = (Node*) malloc(2 * sizeof(Node));
  book->defs[0x00029f01]->node[ 0] = (Node) {0x1000001d,0xa0000001};
  book->defs[0x00029f01]->node[ 1] = (Node) {0x10000024,0x40000000};
  book->defs[0x00029f01]->locs     = (u32*) malloc(2 * sizeof(u32));
  // ex1
  book->defs[0x00029f02]           = (Term*) malloc(sizeof(Term));
  book->defs[0x00029f02]->root     = 0x60000001;
  book->defs[0x00029f02]->alen     = 1;
  book->defs[0x00029f02]->acts     = (Wire*) malloc(1 * sizeof(Wire));
  book->defs[0x00029f02]->acts[ 0] = (Wire) {0x100270c3,0xa0000000};
  book->defs[0x00029f02]->nlen     = 2;
  book->defs[0x00029f02]->node     = (Node*) malloc(2 * sizeof(Node));
  book->defs[0x00029f02]->node[ 0] = (Node) {0x1002bff7,0xa0000001};
  book->defs[0x00029f02]->node[ 1] = (Node) {0x1002bffe,0x40000000};
  book->defs[0x00029f02]->locs     = (u32*) malloc(2 * sizeof(u32));
  // ex2
  book->defs[0x00029f03]           = (Term*) malloc(sizeof(Term));
  book->defs[0x00029f03]->root     = 0x60000000;
  book->defs[0x00029f03]->alen     = 2;
  book->defs[0x00029f03]->acts     = (Wire*) malloc(2 * sizeof(Wire));
  book->defs[0x00029f03]->acts[ 0] = (Wire) {0x10036e72,0xa0000000};
  book->defs[0x00029f03]->acts[ 1] = (Wire) {0x100270c1,0xa0000001};
  book->defs[0x00029f03]->nlen     = 3;
  book->defs[0x00029f03]->node     = (Node*) malloc(3 * sizeof(Node));
  book->defs[0x00029f03]->node[ 0] = (Node) {0x60000002,0x40000000};
  book->defs[0x00029f03]->node[ 1] = (Node) {0x10000013,0xa0000002};
  book->defs[0x00029f03]->node[ 2] = (Node) {0x1000000f,0x50000000};
  book->defs[0x00029f03]->locs     = (u32*) malloc(3 * sizeof(u32));
  // g_s
  book->defs[0x0002bff7]           = (Term*) malloc(sizeof(Term));
  book->defs[0x0002bff7]->root     = 0xa0000000;
  book->defs[0x0002bff7]->alen     = 0;
  book->defs[0x0002bff7]->acts     = (Wire*) malloc(0 * sizeof(Wire));
  book->defs[0x0002bff7]->nlen     = 3;
  book->defs[0x0002bff7]->node     = (Node*) malloc(3 * sizeof(Node));
  book->defs[0x0002bff7]->node[ 0] = (Node) {0xe0000001,0x60000002};
  book->defs[0x0002bff7]->node[ 1] = (Node) {0xa0000002,0x50000002};
  book->defs[0x0002bff7]->node[ 2] = (Node) {0x60000001,0x60000000};
  book->defs[0x0002bff7]->locs     = (u32*) malloc(3 * sizeof(u32));
  // g_z
  book->defs[0x0002bffe]           = (Term*) malloc(sizeof(Term));
  book->defs[0x0002bffe]->root     = 0xa0000000;
  book->defs[0x0002bffe]->alen     = 0;
  book->defs[0x0002bffe]->acts     = (Wire*) malloc(0 * sizeof(Wire));
  book->defs[0x0002bffe]->nlen     = 1;
  book->defs[0x0002bffe]->node     = (Node*) malloc(1 * sizeof(Node));
  book->defs[0x0002bffe]->node[ 0] = (Node) {0x60000000,0x50000000};
  book->defs[0x0002bffe]->locs     = (u32*) malloc(1 * sizeof(u32));
  // k10
  book->defs[0x0002f081]           = (Term*) malloc(sizeof(Term));
  book->defs[0x0002f081]->root     = 0xa0000000;
  book->defs[0x0002f081]->alen     = 0;
  book->defs[0x0002f081]->acts     = (Wire*) malloc(0 * sizeof(Wire));
  book->defs[0x0002f081]->nlen     = 21;
  book->defs[0x0002f081]->node     = (Node*) malloc(21 * sizeof(Node));
  book->defs[0x0002f081]->node[ 0] = (Node) {0xc0000001,0xa0000014};
  book->defs[0x0002f081]->node[ 1] = (Node) {0xc0000002,0xa0000013};
  book->defs[0x0002f081]->node[ 2] = (Node) {0xc0000003,0xa0000012};
  book->defs[0x0002f081]->node[ 3] = (Node) {0xc0000004,0xa0000011};
  book->defs[0x0002f081]->node[ 4] = (Node) {0xc0000005,0xa0000010};
  book->defs[0x0002f081]->node[ 5] = (Node) {0xc0000006,0xa000000f};
  book->defs[0x0002f081]->node[ 6] = (Node) {0xc0000007,0xa000000e};
  book->defs[0x0002f081]->node[ 7] = (Node) {0xc0000008,0xa000000d};
  book->defs[0x0002f081]->node[ 8] = (Node) {0xc0000009,0xa000000c};
  book->defs[0x0002f081]->node[ 9] = (Node) {0xa000000a,0xa000000b};
  book->defs[0x0002f081]->node[10] = (Node) {0x50000014,0x5000000b};
  book->defs[0x0002f081]->node[11] = (Node) {0x6000000a,0x5000000c};
  book->defs[0x0002f081]->node[12] = (Node) {0x6000000b,0x5000000d};
  book->defs[0x0002f081]->node[13] = (Node) {0x6000000c,0x5000000e};
  book->defs[0x0002f081]->node[14] = (Node) {0x6000000d,0x5000000f};
  book->defs[0x0002f081]->node[15] = (Node) {0x6000000e,0x50000010};
  book->defs[0x0002f081]->node[16] = (Node) {0x6000000f,0x50000011};
  book->defs[0x0002f081]->node[17] = (Node) {0x60000010,0x50000012};
  book->defs[0x0002f081]->node[18] = (Node) {0x60000011,0x50000013};
  book->defs[0x0002f081]->node[19] = (Node) {0x60000012,0x60000014};
  book->defs[0x0002f081]->node[20] = (Node) {0x5000000a,0x60000013};
  book->defs[0x0002f081]->locs     = (u32*) malloc(21 * sizeof(u32));
  // k11
  book->defs[0x0002f082]           = (Term*) malloc(sizeof(Term));
  book->defs[0x0002f082]->root     = 0xa0000000;
  book->defs[0x0002f082]->alen     = 0;
  book->defs[0x0002f082]->acts     = (Wire*) malloc(0 * sizeof(Wire));
  book->defs[0x0002f082]->nlen     = 23;
  book->defs[0x0002f082]->node     = (Node*) malloc(23 * sizeof(Node));
  book->defs[0x0002f082]->node[ 0] = (Node) {0xc0000001,0xa0000016};
  book->defs[0x0002f082]->node[ 1] = (Node) {0xc0000002,0xa0000015};
  book->defs[0x0002f082]->node[ 2] = (Node) {0xc0000003,0xa0000014};
  book->defs[0x0002f082]->node[ 3] = (Node) {0xc0000004,0xa0000013};
  book->defs[0x0002f082]->node[ 4] = (Node) {0xc0000005,0xa0000012};
  book->defs[0x0002f082]->node[ 5] = (Node) {0xc0000006,0xa0000011};
  book->defs[0x0002f082]->node[ 6] = (Node) {0xc0000007,0xa0000010};
  book->defs[0x0002f082]->node[ 7] = (Node) {0xc0000008,0xa000000f};
  book->defs[0x0002f082]->node[ 8] = (Node) {0xc0000009,0xa000000e};
  book->defs[0x0002f082]->node[ 9] = (Node) {0xc000000a,0xa000000d};
  book->defs[0x0002f082]->node[10] = (Node) {0xa000000b,0xa000000c};
  book->defs[0x0002f082]->node[11] = (Node) {0x50000016,0x5000000c};
  book->defs[0x0002f082]->node[12] = (Node) {0x6000000b,0x5000000d};
  book->defs[0x0002f082]->node[13] = (Node) {0x6000000c,0x5000000e};
  book->defs[0x0002f082]->node[14] = (Node) {0x6000000d,0x5000000f};
  book->defs[0x0002f082]->node[15] = (Node) {0x6000000e,0x50000010};
  book->defs[0x0002f082]->node[16] = (Node) {0x6000000f,0x50000011};
  book->defs[0x0002f082]->node[17] = (Node) {0x60000010,0x50000012};
  book->defs[0x0002f082]->node[18] = (Node) {0x60000011,0x50000013};
  book->defs[0x0002f082]->node[19] = (Node) {0x60000012,0x50000014};
  book->defs[0x0002f082]->node[20] = (Node) {0x60000013,0x50000015};
  book->defs[0x0002f082]->node[21] = (Node) {0x60000014,0x60000016};
  book->defs[0x0002f082]->node[22] = (Node) {0x5000000b,0x60000015};
  book->defs[0x0002f082]->locs     = (u32*) malloc(23 * sizeof(u32));
  // k12
  book->defs[0x0002f083]           = (Term*) malloc(sizeof(Term));
  book->defs[0x0002f083]->root     = 0xa0000000;
  book->defs[0x0002f083]->alen     = 0;
  book->defs[0x0002f083]->acts     = (Wire*) malloc(0 * sizeof(Wire));
  book->defs[0x0002f083]->nlen     = 25;
  book->defs[0x0002f083]->node     = (Node*) malloc(25 * sizeof(Node));
  book->defs[0x0002f083]->node[ 0] = (Node) {0xc0000001,0xa0000018};
  book->defs[0x0002f083]->node[ 1] = (Node) {0xc0000002,0xa0000017};
  book->defs[0x0002f083]->node[ 2] = (Node) {0xc0000003,0xa0000016};
  book->defs[0x0002f083]->node[ 3] = (Node) {0xc0000004,0xa0000015};
  book->defs[0x0002f083]->node[ 4] = (Node) {0xc0000005,0xa0000014};
  book->defs[0x0002f083]->node[ 5] = (Node) {0xc0000006,0xa0000013};
  book->defs[0x0002f083]->node[ 6] = (Node) {0xc0000007,0xa0000012};
  book->defs[0x0002f083]->node[ 7] = (Node) {0xc0000008,0xa0000011};
  book->defs[0x0002f083]->node[ 8] = (Node) {0xc0000009,0xa0000010};
  book->defs[0x0002f083]->node[ 9] = (Node) {0xc000000a,0xa000000f};
  book->defs[0x0002f083]->node[10] = (Node) {0xc000000b,0xa000000e};
  book->defs[0x0002f083]->node[11] = (Node) {0xa000000c,0xa000000d};
  book->defs[0x0002f083]->node[12] = (Node) {0x50000018,0x5000000d};
  book->defs[0x0002f083]->node[13] = (Node) {0x6000000c,0x5000000e};
  book->defs[0x0002f083]->node[14] = (Node) {0x6000000d,0x5000000f};
  book->defs[0x0002f083]->node[15] = (Node) {0x6000000e,0x50000010};
  book->defs[0x0002f083]->node[16] = (Node) {0x6000000f,0x50000011};
  book->defs[0x0002f083]->node[17] = (Node) {0x60000010,0x50000012};
  book->defs[0x0002f083]->node[18] = (Node) {0x60000011,0x50000013};
  book->defs[0x0002f083]->node[19] = (Node) {0x60000012,0x50000014};
  book->defs[0x0002f083]->node[20] = (Node) {0x60000013,0x50000015};
  book->defs[0x0002f083]->node[21] = (Node) {0x60000014,0x50000016};
  book->defs[0x0002f083]->node[22] = (Node) {0x60000015,0x50000017};
  book->defs[0x0002f083]->node[23] = (Node) {0x60000016,0x60000018};
  book->defs[0x0002f083]->node[24] = (Node) {0x5000000c,0x60000017};
  book->defs[0x0002f083]->locs     = (u32*) malloc(25 * sizeof(u32));
  // k13
  book->defs[0x0002f084]           = (Term*) malloc(sizeof(Term));
  book->defs[0x0002f084]->root     = 0xa0000000;
  book->defs[0x0002f084]->alen     = 0;
  book->defs[0x0002f084]->acts     = (Wire*) malloc(0 * sizeof(Wire));
  book->defs[0x0002f084]->nlen     = 27;
  book->defs[0x0002f084]->node     = (Node*) malloc(27 * sizeof(Node));
  book->defs[0x0002f084]->node[ 0] = (Node) {0xc0000001,0xa000001a};
  book->defs[0x0002f084]->node[ 1] = (Node) {0xc0000002,0xa0000019};
  book->defs[0x0002f084]->node[ 2] = (Node) {0xc0000003,0xa0000018};
  book->defs[0x0002f084]->node[ 3] = (Node) {0xc0000004,0xa0000017};
  book->defs[0x0002f084]->node[ 4] = (Node) {0xc0000005,0xa0000016};
  book->defs[0x0002f084]->node[ 5] = (Node) {0xc0000006,0xa0000015};
  book->defs[0x0002f084]->node[ 6] = (Node) {0xc0000007,0xa0000014};
  book->defs[0x0002f084]->node[ 7] = (Node) {0xc0000008,0xa0000013};
  book->defs[0x0002f084]->node[ 8] = (Node) {0xc0000009,0xa0000012};
  book->defs[0x0002f084]->node[ 9] = (Node) {0xc000000a,0xa0000011};
  book->defs[0x0002f084]->node[10] = (Node) {0xc000000b,0xa0000010};
  book->defs[0x0002f084]->node[11] = (Node) {0xc000000c,0xa000000f};
  book->defs[0x0002f084]->node[12] = (Node) {0xa000000d,0xa000000e};
  book->defs[0x0002f084]->node[13] = (Node) {0x5000001a,0x5000000e};
  book->defs[0x0002f084]->node[14] = (Node) {0x6000000d,0x5000000f};
  book->defs[0x0002f084]->node[15] = (Node) {0x6000000e,0x50000010};
  book->defs[0x0002f084]->node[16] = (Node) {0x6000000f,0x50000011};
  book->defs[0x0002f084]->node[17] = (Node) {0x60000010,0x50000012};
  book->defs[0x0002f084]->node[18] = (Node) {0x60000011,0x50000013};
  book->defs[0x0002f084]->node[19] = (Node) {0x60000012,0x50000014};
  book->defs[0x0002f084]->node[20] = (Node) {0x60000013,0x50000015};
  book->defs[0x0002f084]->node[21] = (Node) {0x60000014,0x50000016};
  book->defs[0x0002f084]->node[22] = (Node) {0x60000015,0x50000017};
  book->defs[0x0002f084]->node[23] = (Node) {0x60000016,0x50000018};
  book->defs[0x0002f084]->node[24] = (Node) {0x60000017,0x50000019};
  book->defs[0x0002f084]->node[25] = (Node) {0x60000018,0x6000001a};
  book->defs[0x0002f084]->node[26] = (Node) {0x5000000d,0x60000019};
  book->defs[0x0002f084]->locs     = (u32*) malloc(27 * sizeof(u32));
  // k14
  book->defs[0x0002f085]           = (Term*) malloc(sizeof(Term));
  book->defs[0x0002f085]->root     = 0xa0000000;
  book->defs[0x0002f085]->alen     = 0;
  book->defs[0x0002f085]->acts     = (Wire*) malloc(0 * sizeof(Wire));
  book->defs[0x0002f085]->nlen     = 29;
  book->defs[0x0002f085]->node     = (Node*) malloc(29 * sizeof(Node));
  book->defs[0x0002f085]->node[ 0] = (Node) {0xc0000001,0xa000001c};
  book->defs[0x0002f085]->node[ 1] = (Node) {0xc0000002,0xa000001b};
  book->defs[0x0002f085]->node[ 2] = (Node) {0xc0000003,0xa000001a};
  book->defs[0x0002f085]->node[ 3] = (Node) {0xc0000004,0xa0000019};
  book->defs[0x0002f085]->node[ 4] = (Node) {0xc0000005,0xa0000018};
  book->defs[0x0002f085]->node[ 5] = (Node) {0xc0000006,0xa0000017};
  book->defs[0x0002f085]->node[ 6] = (Node) {0xc0000007,0xa0000016};
  book->defs[0x0002f085]->node[ 7] = (Node) {0xc0000008,0xa0000015};
  book->defs[0x0002f085]->node[ 8] = (Node) {0xc0000009,0xa0000014};
  book->defs[0x0002f085]->node[ 9] = (Node) {0xc000000a,0xa0000013};
  book->defs[0x0002f085]->node[10] = (Node) {0xc000000b,0xa0000012};
  book->defs[0x0002f085]->node[11] = (Node) {0xc000000c,0xa0000011};
  book->defs[0x0002f085]->node[12] = (Node) {0xc000000d,0xa0000010};
  book->defs[0x0002f085]->node[13] = (Node) {0xa000000e,0xa000000f};
  book->defs[0x0002f085]->node[14] = (Node) {0x5000001c,0x5000000f};
  book->defs[0x0002f085]->node[15] = (Node) {0x6000000e,0x50000010};
  book->defs[0x0002f085]->node[16] = (Node) {0x6000000f,0x50000011};
  book->defs[0x0002f085]->node[17] = (Node) {0x60000010,0x50000012};
  book->defs[0x0002f085]->node[18] = (Node) {0x60000011,0x50000013};
  book->defs[0x0002f085]->node[19] = (Node) {0x60000012,0x50000014};
  book->defs[0x0002f085]->node[20] = (Node) {0x60000013,0x50000015};
  book->defs[0x0002f085]->node[21] = (Node) {0x60000014,0x50000016};
  book->defs[0x0002f085]->node[22] = (Node) {0x60000015,0x50000017};
  book->defs[0x0002f085]->node[23] = (Node) {0x60000016,0x50000018};
  book->defs[0x0002f085]->node[24] = (Node) {0x60000017,0x50000019};
  book->defs[0x0002f085]->node[25] = (Node) {0x60000018,0x5000001a};
  book->defs[0x0002f085]->node[26] = (Node) {0x60000019,0x5000001b};
  book->defs[0x0002f085]->node[27] = (Node) {0x6000001a,0x6000001c};
  book->defs[0x0002f085]->node[28] = (Node) {0x5000000e,0x6000001b};
  book->defs[0x0002f085]->locs     = (u32*) malloc(29 * sizeof(u32));
  // k15
  book->defs[0x0002f086]           = (Term*) malloc(sizeof(Term));
  book->defs[0x0002f086]->root     = 0xa0000000;
  book->defs[0x0002f086]->alen     = 0;
  book->defs[0x0002f086]->acts     = (Wire*) malloc(0 * sizeof(Wire));
  book->defs[0x0002f086]->nlen     = 31;
  book->defs[0x0002f086]->node     = (Node*) malloc(31 * sizeof(Node));
  book->defs[0x0002f086]->node[ 0] = (Node) {0xc0000001,0xa000001e};
  book->defs[0x0002f086]->node[ 1] = (Node) {0xc0000002,0xa000001d};
  book->defs[0x0002f086]->node[ 2] = (Node) {0xc0000003,0xa000001c};
  book->defs[0x0002f086]->node[ 3] = (Node) {0xc0000004,0xa000001b};
  book->defs[0x0002f086]->node[ 4] = (Node) {0xc0000005,0xa000001a};
  book->defs[0x0002f086]->node[ 5] = (Node) {0xc0000006,0xa0000019};
  book->defs[0x0002f086]->node[ 6] = (Node) {0xc0000007,0xa0000018};
  book->defs[0x0002f086]->node[ 7] = (Node) {0xc0000008,0xa0000017};
  book->defs[0x0002f086]->node[ 8] = (Node) {0xc0000009,0xa0000016};
  book->defs[0x0002f086]->node[ 9] = (Node) {0xc000000a,0xa0000015};
  book->defs[0x0002f086]->node[10] = (Node) {0xc000000b,0xa0000014};
  book->defs[0x0002f086]->node[11] = (Node) {0xc000000c,0xa0000013};
  book->defs[0x0002f086]->node[12] = (Node) {0xc000000d,0xa0000012};
  book->defs[0x0002f086]->node[13] = (Node) {0xc000000e,0xa0000011};
  book->defs[0x0002f086]->node[14] = (Node) {0xa000000f,0xa0000010};
  book->defs[0x0002f086]->node[15] = (Node) {0x5000001e,0x50000010};
  book->defs[0x0002f086]->node[16] = (Node) {0x6000000f,0x50000011};
  book->defs[0x0002f086]->node[17] = (Node) {0x60000010,0x50000012};
  book->defs[0x0002f086]->node[18] = (Node) {0x60000011,0x50000013};
  book->defs[0x0002f086]->node[19] = (Node) {0x60000012,0x50000014};
  book->defs[0x0002f086]->node[20] = (Node) {0x60000013,0x50000015};
  book->defs[0x0002f086]->node[21] = (Node) {0x60000014,0x50000016};
  book->defs[0x0002f086]->node[22] = (Node) {0x60000015,0x50000017};
  book->defs[0x0002f086]->node[23] = (Node) {0x60000016,0x50000018};
  book->defs[0x0002f086]->node[24] = (Node) {0x60000017,0x50000019};
  book->defs[0x0002f086]->node[25] = (Node) {0x60000018,0x5000001a};
  book->defs[0x0002f086]->node[26] = (Node) {0x60000019,0x5000001b};
  book->defs[0x0002f086]->node[27] = (Node) {0x6000001a,0x5000001c};
  book->defs[0x0002f086]->node[28] = (Node) {0x6000001b,0x5000001d};
  book->defs[0x0002f086]->node[29] = (Node) {0x6000001c,0x6000001e};
  book->defs[0x0002f086]->node[30] = (Node) {0x5000000f,0x6000001d};
  book->defs[0x0002f086]->locs     = (u32*) malloc(31 * sizeof(u32));
  // k16
  book->defs[0x0002f087]           = (Term*) malloc(sizeof(Term));
  book->defs[0x0002f087]->root     = 0xa0000000;
  book->defs[0x0002f087]->alen     = 0;
  book->defs[0x0002f087]->acts     = (Wire*) malloc(0 * sizeof(Wire));
  book->defs[0x0002f087]->nlen     = 33;
  book->defs[0x0002f087]->node     = (Node*) malloc(33 * sizeof(Node));
  book->defs[0x0002f087]->node[ 0] = (Node) {0xc0000001,0xa0000020};
  book->defs[0x0002f087]->node[ 1] = (Node) {0xc0000002,0xa000001f};
  book->defs[0x0002f087]->node[ 2] = (Node) {0xc0000003,0xa000001e};
  book->defs[0x0002f087]->node[ 3] = (Node) {0xc0000004,0xa000001d};
  book->defs[0x0002f087]->node[ 4] = (Node) {0xc0000005,0xa000001c};
  book->defs[0x0002f087]->node[ 5] = (Node) {0xc0000006,0xa000001b};
  book->defs[0x0002f087]->node[ 6] = (Node) {0xc0000007,0xa000001a};
  book->defs[0x0002f087]->node[ 7] = (Node) {0xc0000008,0xa0000019};
  book->defs[0x0002f087]->node[ 8] = (Node) {0xc0000009,0xa0000018};
  book->defs[0x0002f087]->node[ 9] = (Node) {0xc000000a,0xa0000017};
  book->defs[0x0002f087]->node[10] = (Node) {0xc000000b,0xa0000016};
  book->defs[0x0002f087]->node[11] = (Node) {0xc000000c,0xa0000015};
  book->defs[0x0002f087]->node[12] = (Node) {0xc000000d,0xa0000014};
  book->defs[0x0002f087]->node[13] = (Node) {0xc000000e,0xa0000013};
  book->defs[0x0002f087]->node[14] = (Node) {0xc000000f,0xa0000012};
  book->defs[0x0002f087]->node[15] = (Node) {0xa0000010,0xa0000011};
  book->defs[0x0002f087]->node[16] = (Node) {0x50000020,0x50000011};
  book->defs[0x0002f087]->node[17] = (Node) {0x60000010,0x50000012};
  book->defs[0x0002f087]->node[18] = (Node) {0x60000011,0x50000013};
  book->defs[0x0002f087]->node[19] = (Node) {0x60000012,0x50000014};
  book->defs[0x0002f087]->node[20] = (Node) {0x60000013,0x50000015};
  book->defs[0x0002f087]->node[21] = (Node) {0x60000014,0x50000016};
  book->defs[0x0002f087]->node[22] = (Node) {0x60000015,0x50000017};
  book->defs[0x0002f087]->node[23] = (Node) {0x60000016,0x50000018};
  book->defs[0x0002f087]->node[24] = (Node) {0x60000017,0x50000019};
  book->defs[0x0002f087]->node[25] = (Node) {0x60000018,0x5000001a};
  book->defs[0x0002f087]->node[26] = (Node) {0x60000019,0x5000001b};
  book->defs[0x0002f087]->node[27] = (Node) {0x6000001a,0x5000001c};
  book->defs[0x0002f087]->node[28] = (Node) {0x6000001b,0x5000001d};
  book->defs[0x0002f087]->node[29] = (Node) {0x6000001c,0x5000001e};
  book->defs[0x0002f087]->node[30] = (Node) {0x6000001d,0x5000001f};
  book->defs[0x0002f087]->node[31] = (Node) {0x6000001e,0x60000020};
  book->defs[0x0002f087]->node[32] = (Node) {0x50000010,0x6000001f};
  book->defs[0x0002f087]->locs     = (u32*) malloc(33 * sizeof(u32));
  // k17
  book->defs[0x0002f088]           = (Term*) malloc(sizeof(Term));
  book->defs[0x0002f088]->root     = 0xa0000000;
  book->defs[0x0002f088]->alen     = 0;
  book->defs[0x0002f088]->acts     = (Wire*) malloc(0 * sizeof(Wire));
  book->defs[0x0002f088]->nlen     = 35;
  book->defs[0x0002f088]->node     = (Node*) malloc(35 * sizeof(Node));
  book->defs[0x0002f088]->node[ 0] = (Node) {0xc0000001,0xa0000022};
  book->defs[0x0002f088]->node[ 1] = (Node) {0xc0000002,0xa0000021};
  book->defs[0x0002f088]->node[ 2] = (Node) {0xc0000003,0xa0000020};
  book->defs[0x0002f088]->node[ 3] = (Node) {0xc0000004,0xa000001f};
  book->defs[0x0002f088]->node[ 4] = (Node) {0xc0000005,0xa000001e};
  book->defs[0x0002f088]->node[ 5] = (Node) {0xc0000006,0xa000001d};
  book->defs[0x0002f088]->node[ 6] = (Node) {0xc0000007,0xa000001c};
  book->defs[0x0002f088]->node[ 7] = (Node) {0xc0000008,0xa000001b};
  book->defs[0x0002f088]->node[ 8] = (Node) {0xc0000009,0xa000001a};
  book->defs[0x0002f088]->node[ 9] = (Node) {0xc000000a,0xa0000019};
  book->defs[0x0002f088]->node[10] = (Node) {0xc000000b,0xa0000018};
  book->defs[0x0002f088]->node[11] = (Node) {0xc000000c,0xa0000017};
  book->defs[0x0002f088]->node[12] = (Node) {0xc000000d,0xa0000016};
  book->defs[0x0002f088]->node[13] = (Node) {0xc000000e,0xa0000015};
  book->defs[0x0002f088]->node[14] = (Node) {0xc000000f,0xa0000014};
  book->defs[0x0002f088]->node[15] = (Node) {0xc0000010,0xa0000013};
  book->defs[0x0002f088]->node[16] = (Node) {0xa0000011,0xa0000012};
  book->defs[0x0002f088]->node[17] = (Node) {0x50000022,0x50000012};
  book->defs[0x0002f088]->node[18] = (Node) {0x60000011,0x50000013};
  book->defs[0x0002f088]->node[19] = (Node) {0x60000012,0x50000014};
  book->defs[0x0002f088]->node[20] = (Node) {0x60000013,0x50000015};
  book->defs[0x0002f088]->node[21] = (Node) {0x60000014,0x50000016};
  book->defs[0x0002f088]->node[22] = (Node) {0x60000015,0x50000017};
  book->defs[0x0002f088]->node[23] = (Node) {0x60000016,0x50000018};
  book->defs[0x0002f088]->node[24] = (Node) {0x60000017,0x50000019};
  book->defs[0x0002f088]->node[25] = (Node) {0x60000018,0x5000001a};
  book->defs[0x0002f088]->node[26] = (Node) {0x60000019,0x5000001b};
  book->defs[0x0002f088]->node[27] = (Node) {0x6000001a,0x5000001c};
  book->defs[0x0002f088]->node[28] = (Node) {0x6000001b,0x5000001d};
  book->defs[0x0002f088]->node[29] = (Node) {0x6000001c,0x5000001e};
  book->defs[0x0002f088]->node[30] = (Node) {0x6000001d,0x5000001f};
  book->defs[0x0002f088]->node[31] = (Node) {0x6000001e,0x50000020};
  book->defs[0x0002f088]->node[32] = (Node) {0x6000001f,0x50000021};
  book->defs[0x0002f088]->node[33] = (Node) {0x60000020,0x60000022};
  book->defs[0x0002f088]->node[34] = (Node) {0x50000011,0x60000021};
  book->defs[0x0002f088]->locs     = (u32*) malloc(35 * sizeof(u32));
  // k18
  book->defs[0x0002f089]           = (Term*) malloc(sizeof(Term));
  book->defs[0x0002f089]->root     = 0xa0000000;
  book->defs[0x0002f089]->alen     = 0;
  book->defs[0x0002f089]->acts     = (Wire*) malloc(0 * sizeof(Wire));
  book->defs[0x0002f089]->nlen     = 37;
  book->defs[0x0002f089]->node     = (Node*) malloc(37 * sizeof(Node));
  book->defs[0x0002f089]->node[ 0] = (Node) {0xc0000001,0xa0000024};
  book->defs[0x0002f089]->node[ 1] = (Node) {0xc0000002,0xa0000023};
  book->defs[0x0002f089]->node[ 2] = (Node) {0xc0000003,0xa0000022};
  book->defs[0x0002f089]->node[ 3] = (Node) {0xc0000004,0xa0000021};
  book->defs[0x0002f089]->node[ 4] = (Node) {0xc0000005,0xa0000020};
  book->defs[0x0002f089]->node[ 5] = (Node) {0xc0000006,0xa000001f};
  book->defs[0x0002f089]->node[ 6] = (Node) {0xc0000007,0xa000001e};
  book->defs[0x0002f089]->node[ 7] = (Node) {0xc0000008,0xa000001d};
  book->defs[0x0002f089]->node[ 8] = (Node) {0xc0000009,0xa000001c};
  book->defs[0x0002f089]->node[ 9] = (Node) {0xc000000a,0xa000001b};
  book->defs[0x0002f089]->node[10] = (Node) {0xc000000b,0xa000001a};
  book->defs[0x0002f089]->node[11] = (Node) {0xc000000c,0xa0000019};
  book->defs[0x0002f089]->node[12] = (Node) {0xc000000d,0xa0000018};
  book->defs[0x0002f089]->node[13] = (Node) {0xc000000e,0xa0000017};
  book->defs[0x0002f089]->node[14] = (Node) {0xc000000f,0xa0000016};
  book->defs[0x0002f089]->node[15] = (Node) {0xc0000010,0xa0000015};
  book->defs[0x0002f089]->node[16] = (Node) {0xc0000011,0xa0000014};
  book->defs[0x0002f089]->node[17] = (Node) {0xa0000012,0xa0000013};
  book->defs[0x0002f089]->node[18] = (Node) {0x50000024,0x50000013};
  book->defs[0x0002f089]->node[19] = (Node) {0x60000012,0x50000014};
  book->defs[0x0002f089]->node[20] = (Node) {0x60000013,0x50000015};
  book->defs[0x0002f089]->node[21] = (Node) {0x60000014,0x50000016};
  book->defs[0x0002f089]->node[22] = (Node) {0x60000015,0x50000017};
  book->defs[0x0002f089]->node[23] = (Node) {0x60000016,0x50000018};
  book->defs[0x0002f089]->node[24] = (Node) {0x60000017,0x50000019};
  book->defs[0x0002f089]->node[25] = (Node) {0x60000018,0x5000001a};
  book->defs[0x0002f089]->node[26] = (Node) {0x60000019,0x5000001b};
  book->defs[0x0002f089]->node[27] = (Node) {0x6000001a,0x5000001c};
  book->defs[0x0002f089]->node[28] = (Node) {0x6000001b,0x5000001d};
  book->defs[0x0002f089]->node[29] = (Node) {0x6000001c,0x5000001e};
  book->defs[0x0002f089]->node[30] = (Node) {0x6000001d,0x5000001f};
  book->defs[0x0002f089]->node[31] = (Node) {0x6000001e,0x50000020};
  book->defs[0x0002f089]->node[32] = (Node) {0x6000001f,0x50000021};
  book->defs[0x0002f089]->node[33] = (Node) {0x60000020,0x50000022};
  book->defs[0x0002f089]->node[34] = (Node) {0x60000021,0x50000023};
  book->defs[0x0002f089]->node[35] = (Node) {0x60000022,0x60000024};
  book->defs[0x0002f089]->node[36] = (Node) {0x50000012,0x60000023};
  book->defs[0x0002f089]->locs     = (u32*) malloc(37 * sizeof(u32));
  // k19
  book->defs[0x0002f08a]           = (Term*) malloc(sizeof(Term));
  book->defs[0x0002f08a]->root     = 0xa0000000;
  book->defs[0x0002f08a]->alen     = 0;
  book->defs[0x0002f08a]->acts     = (Wire*) malloc(0 * sizeof(Wire));
  book->defs[0x0002f08a]->nlen     = 39;
  book->defs[0x0002f08a]->node     = (Node*) malloc(39 * sizeof(Node));
  book->defs[0x0002f08a]->node[ 0] = (Node) {0xc0000001,0xa0000026};
  book->defs[0x0002f08a]->node[ 1] = (Node) {0xc0000002,0xa0000025};
  book->defs[0x0002f08a]->node[ 2] = (Node) {0xc0000003,0xa0000024};
  book->defs[0x0002f08a]->node[ 3] = (Node) {0xc0000004,0xa0000023};
  book->defs[0x0002f08a]->node[ 4] = (Node) {0xc0000005,0xa0000022};
  book->defs[0x0002f08a]->node[ 5] = (Node) {0xc0000006,0xa0000021};
  book->defs[0x0002f08a]->node[ 6] = (Node) {0xc0000007,0xa0000020};
  book->defs[0x0002f08a]->node[ 7] = (Node) {0xc0000008,0xa000001f};
  book->defs[0x0002f08a]->node[ 8] = (Node) {0xc0000009,0xa000001e};
  book->defs[0x0002f08a]->node[ 9] = (Node) {0xc000000a,0xa000001d};
  book->defs[0x0002f08a]->node[10] = (Node) {0xc000000b,0xa000001c};
  book->defs[0x0002f08a]->node[11] = (Node) {0xc000000c,0xa000001b};
  book->defs[0x0002f08a]->node[12] = (Node) {0xc000000d,0xa000001a};
  book->defs[0x0002f08a]->node[13] = (Node) {0xc000000e,0xa0000019};
  book->defs[0x0002f08a]->node[14] = (Node) {0xc000000f,0xa0000018};
  book->defs[0x0002f08a]->node[15] = (Node) {0xc0000010,0xa0000017};
  book->defs[0x0002f08a]->node[16] = (Node) {0xc0000011,0xa0000016};
  book->defs[0x0002f08a]->node[17] = (Node) {0xc0000012,0xa0000015};
  book->defs[0x0002f08a]->node[18] = (Node) {0xa0000013,0xa0000014};
  book->defs[0x0002f08a]->node[19] = (Node) {0x50000026,0x50000014};
  book->defs[0x0002f08a]->node[20] = (Node) {0x60000013,0x50000015};
  book->defs[0x0002f08a]->node[21] = (Node) {0x60000014,0x50000016};
  book->defs[0x0002f08a]->node[22] = (Node) {0x60000015,0x50000017};
  book->defs[0x0002f08a]->node[23] = (Node) {0x60000016,0x50000018};
  book->defs[0x0002f08a]->node[24] = (Node) {0x60000017,0x50000019};
  book->defs[0x0002f08a]->node[25] = (Node) {0x60000018,0x5000001a};
  book->defs[0x0002f08a]->node[26] = (Node) {0x60000019,0x5000001b};
  book->defs[0x0002f08a]->node[27] = (Node) {0x6000001a,0x5000001c};
  book->defs[0x0002f08a]->node[28] = (Node) {0x6000001b,0x5000001d};
  book->defs[0x0002f08a]->node[29] = (Node) {0x6000001c,0x5000001e};
  book->defs[0x0002f08a]->node[30] = (Node) {0x6000001d,0x5000001f};
  book->defs[0x0002f08a]->node[31] = (Node) {0x6000001e,0x50000020};
  book->defs[0x0002f08a]->node[32] = (Node) {0x6000001f,0x50000021};
  book->defs[0x0002f08a]->node[33] = (Node) {0x60000020,0x50000022};
  book->defs[0x0002f08a]->node[34] = (Node) {0x60000021,0x50000023};
  book->defs[0x0002f08a]->node[35] = (Node) {0x60000022,0x50000024};
  book->defs[0x0002f08a]->node[36] = (Node) {0x60000023,0x50000025};
  book->defs[0x0002f08a]->node[37] = (Node) {0x60000024,0x60000026};
  book->defs[0x0002f08a]->node[38] = (Node) {0x50000013,0x60000025};
  book->defs[0x0002f08a]->locs     = (u32*) malloc(39 * sizeof(u32));
  // k20
  book->defs[0x0002f0c1]           = (Term*) malloc(sizeof(Term));
  book->defs[0x0002f0c1]->root     = 0xa0000000;
  book->defs[0x0002f0c1]->alen     = 0;
  book->defs[0x0002f0c1]->acts     = (Wire*) malloc(0 * sizeof(Wire));
  book->defs[0x0002f0c1]->nlen     = 41;
  book->defs[0x0002f0c1]->node     = (Node*) malloc(41 * sizeof(Node));
  book->defs[0x0002f0c1]->node[ 0] = (Node) {0xc0000001,0xa0000028};
  book->defs[0x0002f0c1]->node[ 1] = (Node) {0xc0000002,0xa0000027};
  book->defs[0x0002f0c1]->node[ 2] = (Node) {0xc0000003,0xa0000026};
  book->defs[0x0002f0c1]->node[ 3] = (Node) {0xc0000004,0xa0000025};
  book->defs[0x0002f0c1]->node[ 4] = (Node) {0xc0000005,0xa0000024};
  book->defs[0x0002f0c1]->node[ 5] = (Node) {0xc0000006,0xa0000023};
  book->defs[0x0002f0c1]->node[ 6] = (Node) {0xc0000007,0xa0000022};
  book->defs[0x0002f0c1]->node[ 7] = (Node) {0xc0000008,0xa0000021};
  book->defs[0x0002f0c1]->node[ 8] = (Node) {0xc0000009,0xa0000020};
  book->defs[0x0002f0c1]->node[ 9] = (Node) {0xc000000a,0xa000001f};
  book->defs[0x0002f0c1]->node[10] = (Node) {0xc000000b,0xa000001e};
  book->defs[0x0002f0c1]->node[11] = (Node) {0xc000000c,0xa000001d};
  book->defs[0x0002f0c1]->node[12] = (Node) {0xc000000d,0xa000001c};
  book->defs[0x0002f0c1]->node[13] = (Node) {0xc000000e,0xa000001b};
  book->defs[0x0002f0c1]->node[14] = (Node) {0xc000000f,0xa000001a};
  book->defs[0x0002f0c1]->node[15] = (Node) {0xc0000010,0xa0000019};
  book->defs[0x0002f0c1]->node[16] = (Node) {0xc0000011,0xa0000018};
  book->defs[0x0002f0c1]->node[17] = (Node) {0xc0000012,0xa0000017};
  book->defs[0x0002f0c1]->node[18] = (Node) {0xc0000013,0xa0000016};
  book->defs[0x0002f0c1]->node[19] = (Node) {0xa0000014,0xa0000015};
  book->defs[0x0002f0c1]->node[20] = (Node) {0x50000028,0x50000015};
  book->defs[0x0002f0c1]->node[21] = (Node) {0x60000014,0x50000016};
  book->defs[0x0002f0c1]->node[22] = (Node) {0x60000015,0x50000017};
  book->defs[0x0002f0c1]->node[23] = (Node) {0x60000016,0x50000018};
  book->defs[0x0002f0c1]->node[24] = (Node) {0x60000017,0x50000019};
  book->defs[0x0002f0c1]->node[25] = (Node) {0x60000018,0x5000001a};
  book->defs[0x0002f0c1]->node[26] = (Node) {0x60000019,0x5000001b};
  book->defs[0x0002f0c1]->node[27] = (Node) {0x6000001a,0x5000001c};
  book->defs[0x0002f0c1]->node[28] = (Node) {0x6000001b,0x5000001d};
  book->defs[0x0002f0c1]->node[29] = (Node) {0x6000001c,0x5000001e};
  book->defs[0x0002f0c1]->node[30] = (Node) {0x6000001d,0x5000001f};
  book->defs[0x0002f0c1]->node[31] = (Node) {0x6000001e,0x50000020};
  book->defs[0x0002f0c1]->node[32] = (Node) {0x6000001f,0x50000021};
  book->defs[0x0002f0c1]->node[33] = (Node) {0x60000020,0x50000022};
  book->defs[0x0002f0c1]->node[34] = (Node) {0x60000021,0x50000023};
  book->defs[0x0002f0c1]->node[35] = (Node) {0x60000022,0x50000024};
  book->defs[0x0002f0c1]->node[36] = (Node) {0x60000023,0x50000025};
  book->defs[0x0002f0c1]->node[37] = (Node) {0x60000024,0x50000026};
  book->defs[0x0002f0c1]->node[38] = (Node) {0x60000025,0x50000027};
  book->defs[0x0002f0c1]->node[39] = (Node) {0x60000026,0x60000028};
  book->defs[0x0002f0c1]->node[40] = (Node) {0x50000014,0x60000027};
  book->defs[0x0002f0c1]->locs     = (u32*) malloc(41 * sizeof(u32));
  // k21
  book->defs[0x0002f0c2]           = (Term*) malloc(sizeof(Term));
  book->defs[0x0002f0c2]->root     = 0xa0000000;
  book->defs[0x0002f0c2]->alen     = 0;
  book->defs[0x0002f0c2]->acts     = (Wire*) malloc(0 * sizeof(Wire));
  book->defs[0x0002f0c2]->nlen     = 43;
  book->defs[0x0002f0c2]->node     = (Node*) malloc(43 * sizeof(Node));
  book->defs[0x0002f0c2]->node[ 0] = (Node) {0xc0000001,0xa000002a};
  book->defs[0x0002f0c2]->node[ 1] = (Node) {0xc0000002,0xa0000029};
  book->defs[0x0002f0c2]->node[ 2] = (Node) {0xc0000003,0xa0000028};
  book->defs[0x0002f0c2]->node[ 3] = (Node) {0xc0000004,0xa0000027};
  book->defs[0x0002f0c2]->node[ 4] = (Node) {0xc0000005,0xa0000026};
  book->defs[0x0002f0c2]->node[ 5] = (Node) {0xc0000006,0xa0000025};
  book->defs[0x0002f0c2]->node[ 6] = (Node) {0xc0000007,0xa0000024};
  book->defs[0x0002f0c2]->node[ 7] = (Node) {0xc0000008,0xa0000023};
  book->defs[0x0002f0c2]->node[ 8] = (Node) {0xc0000009,0xa0000022};
  book->defs[0x0002f0c2]->node[ 9] = (Node) {0xc000000a,0xa0000021};
  book->defs[0x0002f0c2]->node[10] = (Node) {0xc000000b,0xa0000020};
  book->defs[0x0002f0c2]->node[11] = (Node) {0xc000000c,0xa000001f};
  book->defs[0x0002f0c2]->node[12] = (Node) {0xc000000d,0xa000001e};
  book->defs[0x0002f0c2]->node[13] = (Node) {0xc000000e,0xa000001d};
  book->defs[0x0002f0c2]->node[14] = (Node) {0xc000000f,0xa000001c};
  book->defs[0x0002f0c2]->node[15] = (Node) {0xc0000010,0xa000001b};
  book->defs[0x0002f0c2]->node[16] = (Node) {0xc0000011,0xa000001a};
  book->defs[0x0002f0c2]->node[17] = (Node) {0xc0000012,0xa0000019};
  book->defs[0x0002f0c2]->node[18] = (Node) {0xc0000013,0xa0000018};
  book->defs[0x0002f0c2]->node[19] = (Node) {0xc0000014,0xa0000017};
  book->defs[0x0002f0c2]->node[20] = (Node) {0xa0000015,0xa0000016};
  book->defs[0x0002f0c2]->node[21] = (Node) {0x5000002a,0x50000016};
  book->defs[0x0002f0c2]->node[22] = (Node) {0x60000015,0x50000017};
  book->defs[0x0002f0c2]->node[23] = (Node) {0x60000016,0x50000018};
  book->defs[0x0002f0c2]->node[24] = (Node) {0x60000017,0x50000019};
  book->defs[0x0002f0c2]->node[25] = (Node) {0x60000018,0x5000001a};
  book->defs[0x0002f0c2]->node[26] = (Node) {0x60000019,0x5000001b};
  book->defs[0x0002f0c2]->node[27] = (Node) {0x6000001a,0x5000001c};
  book->defs[0x0002f0c2]->node[28] = (Node) {0x6000001b,0x5000001d};
  book->defs[0x0002f0c2]->node[29] = (Node) {0x6000001c,0x5000001e};
  book->defs[0x0002f0c2]->node[30] = (Node) {0x6000001d,0x5000001f};
  book->defs[0x0002f0c2]->node[31] = (Node) {0x6000001e,0x50000020};
  book->defs[0x0002f0c2]->node[32] = (Node) {0x6000001f,0x50000021};
  book->defs[0x0002f0c2]->node[33] = (Node) {0x60000020,0x50000022};
  book->defs[0x0002f0c2]->node[34] = (Node) {0x60000021,0x50000023};
  book->defs[0x0002f0c2]->node[35] = (Node) {0x60000022,0x50000024};
  book->defs[0x0002f0c2]->node[36] = (Node) {0x60000023,0x50000025};
  book->defs[0x0002f0c2]->node[37] = (Node) {0x60000024,0x50000026};
  book->defs[0x0002f0c2]->node[38] = (Node) {0x60000025,0x50000027};
  book->defs[0x0002f0c2]->node[39] = (Node) {0x60000026,0x50000028};
  book->defs[0x0002f0c2]->node[40] = (Node) {0x60000027,0x50000029};
  book->defs[0x0002f0c2]->node[41] = (Node) {0x60000028,0x6000002a};
  book->defs[0x0002f0c2]->node[42] = (Node) {0x50000015,0x60000029};
  book->defs[0x0002f0c2]->locs     = (u32*) malloc(43 * sizeof(u32));
  // k22
  book->defs[0x0002f0c3]           = (Term*) malloc(sizeof(Term));
  book->defs[0x0002f0c3]->root     = 0xa0000000;
  book->defs[0x0002f0c3]->alen     = 0;
  book->defs[0x0002f0c3]->acts     = (Wire*) malloc(0 * sizeof(Wire));
  book->defs[0x0002f0c3]->nlen     = 45;
  book->defs[0x0002f0c3]->node     = (Node*) malloc(45 * sizeof(Node));
  book->defs[0x0002f0c3]->node[ 0] = (Node) {0xc0000001,0xa000002c};
  book->defs[0x0002f0c3]->node[ 1] = (Node) {0xc0000002,0xa000002b};
  book->defs[0x0002f0c3]->node[ 2] = (Node) {0xc0000003,0xa000002a};
  book->defs[0x0002f0c3]->node[ 3] = (Node) {0xc0000004,0xa0000029};
  book->defs[0x0002f0c3]->node[ 4] = (Node) {0xc0000005,0xa0000028};
  book->defs[0x0002f0c3]->node[ 5] = (Node) {0xc0000006,0xa0000027};
  book->defs[0x0002f0c3]->node[ 6] = (Node) {0xc0000007,0xa0000026};
  book->defs[0x0002f0c3]->node[ 7] = (Node) {0xc0000008,0xa0000025};
  book->defs[0x0002f0c3]->node[ 8] = (Node) {0xc0000009,0xa0000024};
  book->defs[0x0002f0c3]->node[ 9] = (Node) {0xc000000a,0xa0000023};
  book->defs[0x0002f0c3]->node[10] = (Node) {0xc000000b,0xa0000022};
  book->defs[0x0002f0c3]->node[11] = (Node) {0xc000000c,0xa0000021};
  book->defs[0x0002f0c3]->node[12] = (Node) {0xc000000d,0xa0000020};
  book->defs[0x0002f0c3]->node[13] = (Node) {0xc000000e,0xa000001f};
  book->defs[0x0002f0c3]->node[14] = (Node) {0xc000000f,0xa000001e};
  book->defs[0x0002f0c3]->node[15] = (Node) {0xc0000010,0xa000001d};
  book->defs[0x0002f0c3]->node[16] = (Node) {0xc0000011,0xa000001c};
  book->defs[0x0002f0c3]->node[17] = (Node) {0xc0000012,0xa000001b};
  book->defs[0x0002f0c3]->node[18] = (Node) {0xc0000013,0xa000001a};
  book->defs[0x0002f0c3]->node[19] = (Node) {0xc0000014,0xa0000019};
  book->defs[0x0002f0c3]->node[20] = (Node) {0xc0000015,0xa0000018};
  book->defs[0x0002f0c3]->node[21] = (Node) {0xa0000016,0xa0000017};
  book->defs[0x0002f0c3]->node[22] = (Node) {0x5000002c,0x50000017};
  book->defs[0x0002f0c3]->node[23] = (Node) {0x60000016,0x50000018};
  book->defs[0x0002f0c3]->node[24] = (Node) {0x60000017,0x50000019};
  book->defs[0x0002f0c3]->node[25] = (Node) {0x60000018,0x5000001a};
  book->defs[0x0002f0c3]->node[26] = (Node) {0x60000019,0x5000001b};
  book->defs[0x0002f0c3]->node[27] = (Node) {0x6000001a,0x5000001c};
  book->defs[0x0002f0c3]->node[28] = (Node) {0x6000001b,0x5000001d};
  book->defs[0x0002f0c3]->node[29] = (Node) {0x6000001c,0x5000001e};
  book->defs[0x0002f0c3]->node[30] = (Node) {0x6000001d,0x5000001f};
  book->defs[0x0002f0c3]->node[31] = (Node) {0x6000001e,0x50000020};
  book->defs[0x0002f0c3]->node[32] = (Node) {0x6000001f,0x50000021};
  book->defs[0x0002f0c3]->node[33] = (Node) {0x60000020,0x50000022};
  book->defs[0x0002f0c3]->node[34] = (Node) {0x60000021,0x50000023};
  book->defs[0x0002f0c3]->node[35] = (Node) {0x60000022,0x50000024};
  book->defs[0x0002f0c3]->node[36] = (Node) {0x60000023,0x50000025};
  book->defs[0x0002f0c3]->node[37] = (Node) {0x60000024,0x50000026};
  book->defs[0x0002f0c3]->node[38] = (Node) {0x60000025,0x50000027};
  book->defs[0x0002f0c3]->node[39] = (Node) {0x60000026,0x50000028};
  book->defs[0x0002f0c3]->node[40] = (Node) {0x60000027,0x50000029};
  book->defs[0x0002f0c3]->node[41] = (Node) {0x60000028,0x5000002a};
  book->defs[0x0002f0c3]->node[42] = (Node) {0x60000029,0x5000002b};
  book->defs[0x0002f0c3]->node[43] = (Node) {0x6000002a,0x6000002c};
  book->defs[0x0002f0c3]->node[44] = (Node) {0x50000016,0x6000002b};
  book->defs[0x0002f0c3]->locs     = (u32*) malloc(45 * sizeof(u32));
  // k23
  book->defs[0x0002f0c4]           = (Term*) malloc(sizeof(Term));
  book->defs[0x0002f0c4]->root     = 0xa0000000;
  book->defs[0x0002f0c4]->alen     = 0;
  book->defs[0x0002f0c4]->acts     = (Wire*) malloc(0 * sizeof(Wire));
  book->defs[0x0002f0c4]->nlen     = 47;
  book->defs[0x0002f0c4]->node     = (Node*) malloc(47 * sizeof(Node));
  book->defs[0x0002f0c4]->node[ 0] = (Node) {0xc0000001,0xa000002e};
  book->defs[0x0002f0c4]->node[ 1] = (Node) {0xc0000002,0xa000002d};
  book->defs[0x0002f0c4]->node[ 2] = (Node) {0xc0000003,0xa000002c};
  book->defs[0x0002f0c4]->node[ 3] = (Node) {0xc0000004,0xa000002b};
  book->defs[0x0002f0c4]->node[ 4] = (Node) {0xc0000005,0xa000002a};
  book->defs[0x0002f0c4]->node[ 5] = (Node) {0xc0000006,0xa0000029};
  book->defs[0x0002f0c4]->node[ 6] = (Node) {0xc0000007,0xa0000028};
  book->defs[0x0002f0c4]->node[ 7] = (Node) {0xc0000008,0xa0000027};
  book->defs[0x0002f0c4]->node[ 8] = (Node) {0xc0000009,0xa0000026};
  book->defs[0x0002f0c4]->node[ 9] = (Node) {0xc000000a,0xa0000025};
  book->defs[0x0002f0c4]->node[10] = (Node) {0xc000000b,0xa0000024};
  book->defs[0x0002f0c4]->node[11] = (Node) {0xc000000c,0xa0000023};
  book->defs[0x0002f0c4]->node[12] = (Node) {0xc000000d,0xa0000022};
  book->defs[0x0002f0c4]->node[13] = (Node) {0xc000000e,0xa0000021};
  book->defs[0x0002f0c4]->node[14] = (Node) {0xc000000f,0xa0000020};
  book->defs[0x0002f0c4]->node[15] = (Node) {0xc0000010,0xa000001f};
  book->defs[0x0002f0c4]->node[16] = (Node) {0xc0000011,0xa000001e};
  book->defs[0x0002f0c4]->node[17] = (Node) {0xc0000012,0xa000001d};
  book->defs[0x0002f0c4]->node[18] = (Node) {0xc0000013,0xa000001c};
  book->defs[0x0002f0c4]->node[19] = (Node) {0xc0000014,0xa000001b};
  book->defs[0x0002f0c4]->node[20] = (Node) {0xc0000015,0xa000001a};
  book->defs[0x0002f0c4]->node[21] = (Node) {0xc0000016,0xa0000019};
  book->defs[0x0002f0c4]->node[22] = (Node) {0xa0000017,0xa0000018};
  book->defs[0x0002f0c4]->node[23] = (Node) {0x5000002e,0x50000018};
  book->defs[0x0002f0c4]->node[24] = (Node) {0x60000017,0x50000019};
  book->defs[0x0002f0c4]->node[25] = (Node) {0x60000018,0x5000001a};
  book->defs[0x0002f0c4]->node[26] = (Node) {0x60000019,0x5000001b};
  book->defs[0x0002f0c4]->node[27] = (Node) {0x6000001a,0x5000001c};
  book->defs[0x0002f0c4]->node[28] = (Node) {0x6000001b,0x5000001d};
  book->defs[0x0002f0c4]->node[29] = (Node) {0x6000001c,0x5000001e};
  book->defs[0x0002f0c4]->node[30] = (Node) {0x6000001d,0x5000001f};
  book->defs[0x0002f0c4]->node[31] = (Node) {0x6000001e,0x50000020};
  book->defs[0x0002f0c4]->node[32] = (Node) {0x6000001f,0x50000021};
  book->defs[0x0002f0c4]->node[33] = (Node) {0x60000020,0x50000022};
  book->defs[0x0002f0c4]->node[34] = (Node) {0x60000021,0x50000023};
  book->defs[0x0002f0c4]->node[35] = (Node) {0x60000022,0x50000024};
  book->defs[0x0002f0c4]->node[36] = (Node) {0x60000023,0x50000025};
  book->defs[0x0002f0c4]->node[37] = (Node) {0x60000024,0x50000026};
  book->defs[0x0002f0c4]->node[38] = (Node) {0x60000025,0x50000027};
  book->defs[0x0002f0c4]->node[39] = (Node) {0x60000026,0x50000028};
  book->defs[0x0002f0c4]->node[40] = (Node) {0x60000027,0x50000029};
  book->defs[0x0002f0c4]->node[41] = (Node) {0x60000028,0x5000002a};
  book->defs[0x0002f0c4]->node[42] = (Node) {0x60000029,0x5000002b};
  book->defs[0x0002f0c4]->node[43] = (Node) {0x6000002a,0x5000002c};
  book->defs[0x0002f0c4]->node[44] = (Node) {0x6000002b,0x5000002d};
  book->defs[0x0002f0c4]->node[45] = (Node) {0x6000002c,0x6000002e};
  book->defs[0x0002f0c4]->node[46] = (Node) {0x50000017,0x6000002d};
  book->defs[0x0002f0c4]->locs     = (u32*) malloc(47 * sizeof(u32));
  // k24
  book->defs[0x0002f0c5]           = (Term*) malloc(sizeof(Term));
  book->defs[0x0002f0c5]->root     = 0xa0000000;
  book->defs[0x0002f0c5]->alen     = 0;
  book->defs[0x0002f0c5]->acts     = (Wire*) malloc(0 * sizeof(Wire));
  book->defs[0x0002f0c5]->nlen     = 49;
  book->defs[0x0002f0c5]->node     = (Node*) malloc(49 * sizeof(Node));
  book->defs[0x0002f0c5]->node[ 0] = (Node) {0xc0000001,0xa0000030};
  book->defs[0x0002f0c5]->node[ 1] = (Node) {0xc0000002,0xa000002f};
  book->defs[0x0002f0c5]->node[ 2] = (Node) {0xc0000003,0xa000002e};
  book->defs[0x0002f0c5]->node[ 3] = (Node) {0xc0000004,0xa000002d};
  book->defs[0x0002f0c5]->node[ 4] = (Node) {0xc0000005,0xa000002c};
  book->defs[0x0002f0c5]->node[ 5] = (Node) {0xc0000006,0xa000002b};
  book->defs[0x0002f0c5]->node[ 6] = (Node) {0xc0000007,0xa000002a};
  book->defs[0x0002f0c5]->node[ 7] = (Node) {0xc0000008,0xa0000029};
  book->defs[0x0002f0c5]->node[ 8] = (Node) {0xc0000009,0xa0000028};
  book->defs[0x0002f0c5]->node[ 9] = (Node) {0xc000000a,0xa0000027};
  book->defs[0x0002f0c5]->node[10] = (Node) {0xc000000b,0xa0000026};
  book->defs[0x0002f0c5]->node[11] = (Node) {0xc000000c,0xa0000025};
  book->defs[0x0002f0c5]->node[12] = (Node) {0xc000000d,0xa0000024};
  book->defs[0x0002f0c5]->node[13] = (Node) {0xc000000e,0xa0000023};
  book->defs[0x0002f0c5]->node[14] = (Node) {0xc000000f,0xa0000022};
  book->defs[0x0002f0c5]->node[15] = (Node) {0xc0000010,0xa0000021};
  book->defs[0x0002f0c5]->node[16] = (Node) {0xc0000011,0xa0000020};
  book->defs[0x0002f0c5]->node[17] = (Node) {0xc0000012,0xa000001f};
  book->defs[0x0002f0c5]->node[18] = (Node) {0xc0000013,0xa000001e};
  book->defs[0x0002f0c5]->node[19] = (Node) {0xc0000014,0xa000001d};
  book->defs[0x0002f0c5]->node[20] = (Node) {0xc0000015,0xa000001c};
  book->defs[0x0002f0c5]->node[21] = (Node) {0xc0000016,0xa000001b};
  book->defs[0x0002f0c5]->node[22] = (Node) {0xc0000017,0xa000001a};
  book->defs[0x0002f0c5]->node[23] = (Node) {0xa0000018,0xa0000019};
  book->defs[0x0002f0c5]->node[24] = (Node) {0x50000030,0x50000019};
  book->defs[0x0002f0c5]->node[25] = (Node) {0x60000018,0x5000001a};
  book->defs[0x0002f0c5]->node[26] = (Node) {0x60000019,0x5000001b};
  book->defs[0x0002f0c5]->node[27] = (Node) {0x6000001a,0x5000001c};
  book->defs[0x0002f0c5]->node[28] = (Node) {0x6000001b,0x5000001d};
  book->defs[0x0002f0c5]->node[29] = (Node) {0x6000001c,0x5000001e};
  book->defs[0x0002f0c5]->node[30] = (Node) {0x6000001d,0x5000001f};
  book->defs[0x0002f0c5]->node[31] = (Node) {0x6000001e,0x50000020};
  book->defs[0x0002f0c5]->node[32] = (Node) {0x6000001f,0x50000021};
  book->defs[0x0002f0c5]->node[33] = (Node) {0x60000020,0x50000022};
  book->defs[0x0002f0c5]->node[34] = (Node) {0x60000021,0x50000023};
  book->defs[0x0002f0c5]->node[35] = (Node) {0x60000022,0x50000024};
  book->defs[0x0002f0c5]->node[36] = (Node) {0x60000023,0x50000025};
  book->defs[0x0002f0c5]->node[37] = (Node) {0x60000024,0x50000026};
  book->defs[0x0002f0c5]->node[38] = (Node) {0x60000025,0x50000027};
  book->defs[0x0002f0c5]->node[39] = (Node) {0x60000026,0x50000028};
  book->defs[0x0002f0c5]->node[40] = (Node) {0x60000027,0x50000029};
  book->defs[0x0002f0c5]->node[41] = (Node) {0x60000028,0x5000002a};
  book->defs[0x0002f0c5]->node[42] = (Node) {0x60000029,0x5000002b};
  book->defs[0x0002f0c5]->node[43] = (Node) {0x6000002a,0x5000002c};
  book->defs[0x0002f0c5]->node[44] = (Node) {0x6000002b,0x5000002d};
  book->defs[0x0002f0c5]->node[45] = (Node) {0x6000002c,0x5000002e};
  book->defs[0x0002f0c5]->node[46] = (Node) {0x6000002d,0x5000002f};
  book->defs[0x0002f0c5]->node[47] = (Node) {0x6000002e,0x60000030};
  book->defs[0x0002f0c5]->node[48] = (Node) {0x50000018,0x6000002f};
  book->defs[0x0002f0c5]->locs     = (u32*) malloc(49 * sizeof(u32));
  // low
  book->defs[0x00030cfb]           = (Term*) malloc(sizeof(Term));
  book->defs[0x00030cfb]->root     = 0xa0000000;
  book->defs[0x00030cfb]->alen     = 0;
  book->defs[0x00030cfb]->acts     = (Wire*) malloc(0 * sizeof(Wire));
  book->defs[0x00030cfb]->nlen     = 4;
  book->defs[0x00030cfb]->node     = (Node*) malloc(4 * sizeof(Node));
  book->defs[0x00030cfb]->node[ 0] = (Node) {0xa0000001,0x60000003};
  book->defs[0x00030cfb]->node[ 1] = (Node) {0x10c33ed9,0xa0000002};
  book->defs[0x00030cfb]->node[ 2] = (Node) {0x10c33ed3,0xa0000003};
  book->defs[0x00030cfb]->node[ 3] = (Node) {0x1000000f,0x60000000};
  book->defs[0x00030cfb]->locs     = (u32*) malloc(4 * sizeof(u32));
  // not
  book->defs[0x00032cf8]           = (Term*) malloc(sizeof(Term));
  book->defs[0x00032cf8]->root     = 0xa0000000;
  book->defs[0x00032cf8]->alen     = 0;
  book->defs[0x00032cf8]->acts     = (Wire*) malloc(0 * sizeof(Wire));
  book->defs[0x00032cf8]->nlen     = 5;
  book->defs[0x00032cf8]->node     = (Node*) malloc(5 * sizeof(Node));
  book->defs[0x00032cf8]->node[ 0] = (Node) {0xa0000001,0xa0000003};
  book->defs[0x00032cf8]->node[ 1] = (Node) {0x50000004,0xa0000002};
  book->defs[0x00032cf8]->node[ 2] = (Node) {0x50000003,0x60000004};
  book->defs[0x00032cf8]->node[ 3] = (Node) {0x50000002,0xa0000004};
  book->defs[0x00032cf8]->node[ 4] = (Node) {0x50000001,0x60000002};
  book->defs[0x00032cf8]->locs     = (u32*) malloc(5 * sizeof(u32));
  // run
  book->defs[0x00036e72]           = (Term*) malloc(sizeof(Term));
  book->defs[0x00036e72]->root     = 0xa0000000;
  book->defs[0x00036e72]->alen     = 0;
  book->defs[0x00036e72]->acts     = (Wire*) malloc(0 * sizeof(Wire));
  book->defs[0x00036e72]->nlen     = 4;
  book->defs[0x00036e72]->node     = (Node*) malloc(4 * sizeof(Node));
  book->defs[0x00036e72]->node[ 0] = (Node) {0xa0000001,0x60000003};
  book->defs[0x00036e72]->node[ 1] = (Node) {0x10db9c99,0xa0000002};
  book->defs[0x00036e72]->node[ 2] = (Node) {0x10db9c93,0xa0000003};
  book->defs[0x00036e72]->node[ 3] = (Node) {0x1000000f,0x60000000};
  book->defs[0x00036e72]->locs     = (u32*) malloc(4 * sizeof(u32));
  // decI
  book->defs[0x00a299d3]           = (Term*) malloc(sizeof(Term));
  book->defs[0x00a299d3]->root     = 0xa0000000;
  book->defs[0x00a299d3]->alen     = 1;
  book->defs[0x00a299d3]->acts     = (Wire*) malloc(1 * sizeof(Wire));
  book->defs[0x00a299d3]->acts[ 0] = (Wire) {0x10030cfb,0xa0000001};
  book->defs[0x00a299d3]->nlen     = 2;
  book->defs[0x00a299d3]->node     = (Node*) malloc(2 * sizeof(Node));
  book->defs[0x00a299d3]->node[ 0] = (Node) {0x50000001,0x60000001};
  book->defs[0x00a299d3]->node[ 1] = (Node) {0x50000000,0x60000000};
  book->defs[0x00a299d3]->locs     = (u32*) malloc(2 * sizeof(u32));
  // decO
  book->defs[0x00a299d9]           = (Term*) malloc(sizeof(Term));
  book->defs[0x00a299d9]->root     = 0xa0000000;
  book->defs[0x00a299d9]->alen     = 2;
  book->defs[0x00a299d9]->acts     = (Wire*) malloc(2 * sizeof(Wire));
  book->defs[0x00a299d9]->acts[ 0] = (Wire) {0x10000013,0xa0000001};
  book->defs[0x00a299d9]->acts[ 1] = (Wire) {0x10028a67,0xa0000002};
  book->defs[0x00a299d9]->nlen     = 3;
  book->defs[0x00a299d9]->node     = (Node*) malloc(3 * sizeof(Node));
  book->defs[0x00a299d9]->node[ 0] = (Node) {0x50000002,0x60000001};
  book->defs[0x00a299d9]->node[ 1] = (Node) {0x60000002,0x60000000};
  book->defs[0x00a299d9]->node[ 2] = (Node) {0x50000000,0x50000001};
  book->defs[0x00a299d9]->locs     = (u32*) malloc(3 * sizeof(u32));
  // lowI
  book->defs[0x00c33ed3]           = (Term*) malloc(sizeof(Term));
  book->defs[0x00c33ed3]->root     = 0xa0000000;
  book->defs[0x00c33ed3]->alen     = 2;
  book->defs[0x00c33ed3]->acts     = (Wire*) malloc(2 * sizeof(Wire));
  book->defs[0x00c33ed3]->acts[ 0] = (Wire) {0x10000013,0xa0000001};
  book->defs[0x00c33ed3]->acts[ 1] = (Wire) {0x10000019,0xa0000002};
  book->defs[0x00c33ed3]->nlen     = 3;
  book->defs[0x00c33ed3]->node     = (Node*) malloc(3 * sizeof(Node));
  book->defs[0x00c33ed3]->node[ 0] = (Node) {0x50000001,0x60000002};
  book->defs[0x00c33ed3]->node[ 1] = (Node) {0x50000000,0x50000002};
  book->defs[0x00c33ed3]->node[ 2] = (Node) {0x60000001,0x60000000};
  book->defs[0x00c33ed3]->locs     = (u32*) malloc(3 * sizeof(u32));
  // lowO
  book->defs[0x00c33ed9]           = (Term*) malloc(sizeof(Term));
  book->defs[0x00c33ed9]->root     = 0xa0000000;
  book->defs[0x00c33ed9]->alen     = 2;
  book->defs[0x00c33ed9]->acts     = (Wire*) malloc(2 * sizeof(Wire));
  book->defs[0x00c33ed9]->acts[ 0] = (Wire) {0x10000019,0xa0000001};
  book->defs[0x00c33ed9]->acts[ 1] = (Wire) {0x10000019,0xa0000002};
  book->defs[0x00c33ed9]->nlen     = 3;
  book->defs[0x00c33ed9]->node     = (Node*) malloc(3 * sizeof(Node));
  book->defs[0x00c33ed9]->node[ 0] = (Node) {0x50000001,0x60000002};
  book->defs[0x00c33ed9]->node[ 1] = (Node) {0x50000000,0x50000002};
  book->defs[0x00c33ed9]->node[ 2] = (Node) {0x60000001,0x60000000};
  book->defs[0x00c33ed9]->locs     = (u32*) malloc(3 * sizeof(u32));
  // main
  book->defs[0x00c65b72]           = (Term*) malloc(sizeof(Term));
  book->defs[0x00c65b72]->root     = 0x60000000;
  book->defs[0x00c65b72]->alen     = 1;
  book->defs[0x00c65b72]->acts     = (Wire*) malloc(1 * sizeof(Wire));
  book->defs[0x00c65b72]->acts[ 0] = (Wire) {0x100009c4,0xa0000000};
  book->defs[0x00c65b72]->nlen     = 1;
  book->defs[0x00c65b72]->node     = (Node*) malloc(1 * sizeof(Node));
  book->defs[0x00c65b72]->node[ 0] = (Node) {0x10000bc4,0x40000000};
  book->defs[0x00c65b72]->locs     = (u32*) malloc(1 * sizeof(u32));
  // runI
  book->defs[0x00db9c93]           = (Term*) malloc(sizeof(Term));
  book->defs[0x00db9c93]->root     = 0xa0000000;
  book->defs[0x00db9c93]->alen     = 3;
  book->defs[0x00db9c93]->acts     = (Wire*) malloc(3 * sizeof(Wire));
  book->defs[0x00db9c93]->acts[ 0] = (Wire) {0x10036e72,0xa0000001};
  book->defs[0x00db9c93]->acts[ 1] = (Wire) {0x10028a67,0xa0000002};
  book->defs[0x00db9c93]->acts[ 2] = (Wire) {0x10000013,0xa0000003};
  book->defs[0x00db9c93]->nlen     = 4;
  book->defs[0x00db9c93]->node     = (Node*) malloc(4 * sizeof(Node));
  book->defs[0x00db9c93]->node[ 0] = (Node) {0x50000003,0x60000001};
  book->defs[0x00db9c93]->node[ 1] = (Node) {0x60000002,0x60000000};
  book->defs[0x00db9c93]->node[ 2] = (Node) {0x60000003,0x50000001};
  book->defs[0x00db9c93]->node[ 3] = (Node) {0x50000000,0x50000002};
  book->defs[0x00db9c93]->locs     = (u32*) malloc(4 * sizeof(u32));
  // runO
  book->defs[0x00db9c99]           = (Term*) malloc(sizeof(Term));
  book->defs[0x00db9c99]->root     = 0xa0000000;
  book->defs[0x00db9c99]->alen     = 3;
  book->defs[0x00db9c99]->acts     = (Wire*) malloc(3 * sizeof(Wire));
  book->defs[0x00db9c99]->acts[ 0] = (Wire) {0x10036e72,0xa0000001};
  book->defs[0x00db9c99]->acts[ 1] = (Wire) {0x10028a67,0xa0000002};
  book->defs[0x00db9c99]->acts[ 2] = (Wire) {0x10000019,0xa0000003};
  book->defs[0x00db9c99]->nlen     = 4;
  book->defs[0x00db9c99]->node     = (Node*) malloc(4 * sizeof(Node));
  book->defs[0x00db9c99]->node[ 0] = (Node) {0x50000003,0x60000001};
  book->defs[0x00db9c99]->node[ 1] = (Node) {0x60000002,0x60000000};
  book->defs[0x00db9c99]->node[ 2] = (Node) {0x60000003,0x50000001};
  book->defs[0x00db9c99]->node[ 3] = (Node) {0x50000000,0x50000002};
  book->defs[0x00db9c99]->locs     = (u32*) malloc(4 * sizeof(u32));
  // test
  book->defs[0x00e29df8]           = (Term*) malloc(sizeof(Term));
  book->defs[0x00e29df8]->root     = 0x6000001b;
  book->defs[0x00e29df8]->alen     = 1;
  book->defs[0x00e29df8]->acts     = (Wire*) malloc(1 * sizeof(Wire));
  book->defs[0x00e29df8]->acts[ 0] = (Wire) {0xa0000000,0xa0000011};
  book->defs[0x00e29df8]->nlen     = 29;
  book->defs[0x00e29df8]->node     = (Node*) malloc(29 * sizeof(Node));
  book->defs[0x00e29df8]->node[ 0] = (Node) {0xb0000001,0xa0000010};
  book->defs[0x00e29df8]->node[ 1] = (Node) {0xb0000002,0xa000000f};
  book->defs[0x00e29df8]->node[ 2] = (Node) {0xb0000003,0xa000000e};
  book->defs[0x00e29df8]->node[ 3] = (Node) {0xb0000004,0xa000000d};
  book->defs[0x00e29df8]->node[ 4] = (Node) {0xb0000005,0xa000000c};
  book->defs[0x00e29df8]->node[ 5] = (Node) {0xb0000006,0xa000000b};
  book->defs[0x00e29df8]->node[ 6] = (Node) {0xb0000007,0xa000000a};
  book->defs[0x00e29df8]->node[ 7] = (Node) {0xa0000008,0xa0000009};
  book->defs[0x00e29df8]->node[ 8] = (Node) {0x50000010,0x50000009};
  book->defs[0x00e29df8]->node[ 9] = (Node) {0x60000008,0x5000000a};
  book->defs[0x00e29df8]->node[10] = (Node) {0x60000009,0x5000000b};
  book->defs[0x00e29df8]->node[11] = (Node) {0x6000000a,0x5000000c};
  book->defs[0x00e29df8]->node[12] = (Node) {0x6000000b,0x5000000d};
  book->defs[0x00e29df8]->node[13] = (Node) {0x6000000c,0x5000000e};
  book->defs[0x00e29df8]->node[14] = (Node) {0x6000000d,0x5000000f};
  book->defs[0x00e29df8]->node[15] = (Node) {0x6000000e,0x60000010};
  book->defs[0x00e29df8]->node[16] = (Node) {0x50000008,0x6000000f};
  book->defs[0x00e29df8]->node[17] = (Node) {0xa0000012,0xa0000017};
  book->defs[0x00e29df8]->node[18] = (Node) {0xc0000013,0xa0000016};
  book->defs[0x00e29df8]->node[19] = (Node) {0xa0000014,0xa0000015};
  book->defs[0x00e29df8]->node[20] = (Node) {0x50000016,0x50000015};
  book->defs[0x00e29df8]->node[21] = (Node) {0x60000014,0x60000016};
  book->defs[0x00e29df8]->node[22] = (Node) {0x50000014,0x60000015};
  book->defs[0x00e29df8]->node[23] = (Node) {0xa0000018,0xa000001b};
  book->defs[0x00e29df8]->node[24] = (Node) {0xa0000019,0x60000019};
  book->defs[0x00e29df8]->node[25] = (Node) {0xa000001a,0x60000018};
  book->defs[0x00e29df8]->node[26] = (Node) {0x6000001a,0x5000001a};
  book->defs[0x00e29df8]->node[27] = (Node) {0xa000001c,0x40000000};
  book->defs[0x00e29df8]->node[28] = (Node) {0x6000001c,0x5000001c};
  book->defs[0x00e29df8]->locs     = (u32*) malloc(29 * sizeof(u32));
}

__host__ void boot(Net* net, Book* book, u32 id) {
  net->root = book->defs[id]->root;
  net->blen = book->defs[id]->alen;
  for (u32 i = 0; i < book->defs[id]->alen; ++i) {
    net->bags[i] = book->defs[id]->acts[i];
  }
  for (u32 i = 0; i < book->defs[id]->nlen; ++i) {
    net->node[i] = book->defs[id]->node[i];
  }
}

// term_a = (λx(x) λy(y))
__host__ void inject_term_a(Net* net) {
  net->root     = 0x60000001;
  net->blen     = 1;
  net->bags[ 0] = (Wire) {0xa0000000,0xa0000001};
  net->node[ 0] = (Node) {0x60000000,0x50000000};
  net->node[ 1] = (Node) {0xa0000002,0x40000000};
  net->node[ 2] = (Node) {0x60000002,0x50000002};
}

// term_b = (λfλx(f x) λy(y))
__host__ void inject_term_b(Net* net) {
  net->root     = 0x60000003;
  net->blen     = 1;
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
  net->blen     = 1;
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
  net->blen     = 1;
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
  net->blen     = 1;
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
  net->blen     = 1;
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
  net->blen     = 1;
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
  net->blen     = 1;
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
  net->blen     = 1;
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

// term_j = (c16 g_s g_z)
__host__ void inject_term_j(Net* net) {
  net->root     = 0x60000027;
  net->blen     = 1;
  net->bags[ 0] = (Wire) {0xa0000000,0xa0000021};
  net->node[ 0] = (Node) {0xb0000001,0xa0000020};
  net->node[ 1] = (Node) {0xb0000002,0xa000001f};
  net->node[ 2] = (Node) {0xb0000003,0xa000001e};
  net->node[ 3] = (Node) {0xb0000004,0xa000001d};
  net->node[ 4] = (Node) {0xb0000005,0xa000001c};
  net->node[ 5] = (Node) {0xb0000006,0xa000001b};
  net->node[ 6] = (Node) {0xb0000007,0xa000001a};
  net->node[ 7] = (Node) {0xb0000008,0xa0000019};
  net->node[ 8] = (Node) {0xb0000009,0xa0000018};
  net->node[ 9] = (Node) {0xb000000a,0xa0000017};
  net->node[10] = (Node) {0xb000000b,0xa0000016};
  net->node[11] = (Node) {0xb000000c,0xa0000015};
  net->node[12] = (Node) {0xb000000d,0xa0000014};
  net->node[13] = (Node) {0xb000000e,0xa0000013};
  net->node[14] = (Node) {0xb000000f,0xa0000012};
  net->node[15] = (Node) {0xa0000010,0xa0000011};
  net->node[16] = (Node) {0x50000020,0x50000011};
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
  net->node[27] = (Node) {0x6000001a,0x5000001c};
  net->node[28] = (Node) {0x6000001b,0x5000001d};
  net->node[29] = (Node) {0x6000001c,0x5000001e};
  net->node[30] = (Node) {0x6000001d,0x5000001f};
  net->node[31] = (Node) {0x6000001e,0x60000020};
  net->node[32] = (Node) {0x50000010,0x6000001f};
  net->node[33] = (Node) {0xa0000022,0xa0000027};
  net->node[34] = (Node) {0xc0000023,0xa0000024};
  net->node[35] = (Node) {0x50000025,0x50000026};
  net->node[36] = (Node) {0xa0000025,0x60000026};
  net->node[37] = (Node) {0x50000023,0xa0000026};
  net->node[38] = (Node) {0x60000023,0x60000024};
  net->node[39] = (Node) {0xa0000028,0x40000000};
  net->node[40] = (Node) {0x60000028,0x50000028};
}

// term_k = (c18 g_s g_z)
__host__ void inject_term_k(Net* net) {
  net->root     = 0x6000002b;
  net->blen     = 1;
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

// term_l = (c20 g_s g_z)
__host__ void inject_term_l(Net* net) {
  net->root     = 0x6000002f;
  net->blen     = 1;
  net->bags[ 0] = (Wire) {0xa0000000,0xa0000029};
  net->node[ 0] = (Node) {0xb0000001,0xa0000028};
  net->node[ 1] = (Node) {0xb0000002,0xa0000027};
  net->node[ 2] = (Node) {0xb0000003,0xa0000026};
  net->node[ 3] = (Node) {0xb0000004,0xa0000025};
  net->node[ 4] = (Node) {0xb0000005,0xa0000024};
  net->node[ 5] = (Node) {0xb0000006,0xa0000023};
  net->node[ 6] = (Node) {0xb0000007,0xa0000022};
  net->node[ 7] = (Node) {0xb0000008,0xa0000021};
  net->node[ 8] = (Node) {0xb0000009,0xa0000020};
  net->node[ 9] = (Node) {0xb000000a,0xa000001f};
  net->node[10] = (Node) {0xb000000b,0xa000001e};
  net->node[11] = (Node) {0xb000000c,0xa000001d};
  net->node[12] = (Node) {0xb000000d,0xa000001c};
  net->node[13] = (Node) {0xb000000e,0xa000001b};
  net->node[14] = (Node) {0xb000000f,0xa000001a};
  net->node[15] = (Node) {0xb0000010,0xa0000019};
  net->node[16] = (Node) {0xb0000011,0xa0000018};
  net->node[17] = (Node) {0xb0000012,0xa0000017};
  net->node[18] = (Node) {0xb0000013,0xa0000016};
  net->node[19] = (Node) {0xa0000014,0xa0000015};
  net->node[20] = (Node) {0x50000028,0x50000015};
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
  net->node[35] = (Node) {0x60000022,0x50000024};
  net->node[36] = (Node) {0x60000023,0x50000025};
  net->node[37] = (Node) {0x60000024,0x50000026};
  net->node[38] = (Node) {0x60000025,0x50000027};
  net->node[39] = (Node) {0x60000026,0x60000028};
  net->node[40] = (Node) {0x50000014,0x60000027};
  net->node[41] = (Node) {0xa000002a,0xa000002f};
  net->node[42] = (Node) {0xc000002b,0xa000002c};
  net->node[43] = (Node) {0x5000002d,0x5000002e};
  net->node[44] = (Node) {0xa000002d,0x6000002e};
  net->node[45] = (Node) {0x5000002b,0xa000002e};
  net->node[46] = (Node) {0x6000002b,0x6000002c};
  net->node[47] = (Node) {0xa0000030,0x40000000};
  net->node[48] = (Node) {0x60000030,0x50000030};
}

// term_m = (c23 g_s g_z)
__host__ void inject_term_m(Net* net) {
  net->root     = 0x60000035;
  net->blen     = 1;
  net->bags[ 0] = (Wire) {0xa0000000,0xa000002f};
  net->node[ 0] = (Node) {0xb0000001,0xa000002e};
  net->node[ 1] = (Node) {0xb0000002,0xa000002d};
  net->node[ 2] = (Node) {0xb0000003,0xa000002c};
  net->node[ 3] = (Node) {0xb0000004,0xa000002b};
  net->node[ 4] = (Node) {0xb0000005,0xa000002a};
  net->node[ 5] = (Node) {0xb0000006,0xa0000029};
  net->node[ 6] = (Node) {0xb0000007,0xa0000028};
  net->node[ 7] = (Node) {0xb0000008,0xa0000027};
  net->node[ 8] = (Node) {0xb0000009,0xa0000026};
  net->node[ 9] = (Node) {0xb000000a,0xa0000025};
  net->node[10] = (Node) {0xb000000b,0xa0000024};
  net->node[11] = (Node) {0xb000000c,0xa0000023};
  net->node[12] = (Node) {0xb000000d,0xa0000022};
  net->node[13] = (Node) {0xb000000e,0xa0000021};
  net->node[14] = (Node) {0xb000000f,0xa0000020};
  net->node[15] = (Node) {0xb0000010,0xa000001f};
  net->node[16] = (Node) {0xb0000011,0xa000001e};
  net->node[17] = (Node) {0xb0000012,0xa000001d};
  net->node[18] = (Node) {0xb0000013,0xa000001c};
  net->node[19] = (Node) {0xb0000014,0xa000001b};
  net->node[20] = (Node) {0xb0000015,0xa000001a};
  net->node[21] = (Node) {0xb0000016,0xa0000019};
  net->node[22] = (Node) {0xa0000017,0xa0000018};
  net->node[23] = (Node) {0x5000002e,0x50000018};
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
  net->node[35] = (Node) {0x60000022,0x50000024};
  net->node[36] = (Node) {0x60000023,0x50000025};
  net->node[37] = (Node) {0x60000024,0x50000026};
  net->node[38] = (Node) {0x60000025,0x50000027};
  net->node[39] = (Node) {0x60000026,0x50000028};
  net->node[40] = (Node) {0x60000027,0x50000029};
  net->node[41] = (Node) {0x60000028,0x5000002a};
  net->node[42] = (Node) {0x60000029,0x5000002b};
  net->node[43] = (Node) {0x6000002a,0x5000002c};
  net->node[44] = (Node) {0x6000002b,0x5000002d};
  net->node[45] = (Node) {0x6000002c,0x6000002e};
  net->node[46] = (Node) {0x50000017,0x6000002d};
  net->node[47] = (Node) {0xa0000030,0xa0000035};
  net->node[48] = (Node) {0xc0000031,0xa0000032};
  net->node[49] = (Node) {0x50000033,0x50000034};
  net->node[50] = (Node) {0xa0000033,0x60000034};
  net->node[51] = (Node) {0x50000031,0xa0000034};
  net->node[52] = (Node) {0x60000031,0x60000032};
  net->node[53] = (Node) {0xa0000036,0x40000000};
  net->node[54] = (Node) {0x60000036,0x50000036};
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
  //inject_term_m(h_net);

  // Allocates the initial book on device
  Book* h_book = mkbook();
  populate(h_book);
  printf("ptr at 9C6 is: %llu\n", h_book->defs[0x9C6]);

  // Boots the net with an initial term
  boot(h_net, h_book, 0x00c65b72); // main

  // Prints the initial net
  printf("\n");
  printf("INPUT\n");
  printf("=====\n\n");
  print_net(h_net);

  // Sends the net from host to device
  Net* d_net = net_to_device(h_net);

  // Sends the book from host to device
  Book* d_book = book_to_device(h_book);
  //Book* H_book = book_to_host(d_book);

  // Performs parallel reductions
  printf("\n");

  // Gets start time
  struct timespec start, end;
  u32 rwts;
  clock_gettime(CLOCK_MONOTONIC_RAW, &start);

  // Normalizes
  do_normalize(d_net, d_book);
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
  printf("\n");
  printf("NORMAL ~ rewrites=%d redexes=%d\n", norm->rwts, norm->blen);
  printf("======\n\n");
  //print_tree(norm, norm->root);
  print_net(norm);
  printf("\n");

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
