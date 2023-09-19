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

// This code is initially optimized for RTX 4090
const u32 BLOCK_LOG2    = 9;                         // log2 of block size
const u32 BLOCK_SIZE    = 1 << BLOCK_LOG2;           // threads per block
const u32 UNIT_LOG2     = 2;                         // log2 of unit size
const u32 UNIT_SIZE     = 1 << UNIT_LOG2;            // threads per unit
const u32 GROUP_LOG2    = BLOCK_LOG2 - UNIT_LOG2;    // log2 of group size
const u32 GROUP_SIZE    = 1 << GROUP_LOG2;           // units per group
const u32 NODE_LOG2     = 28;                        // log2 of node size
const u32 NODE_SIZE     = 1 << NODE_LOG2;            // max total nodes (2GB addressable)
const u32 HEAD_LOG2     = GROUP_LOG2 * 2;            // log2 of head size
const u32 HEAD_SIZE     = 1 << HEAD_LOG2;            // max head pointers
const u32 MAX_THREADS   = BLOCK_SIZE * BLOCK_SIZE;   // total number of active threads
const u32 MAX_UNITS     = GROUP_SIZE * GROUP_SIZE;   // total number of active units
const u32 MAX_NEW_REDEX = 4;                         // max new redexes per rewrite
const u32 SMEM_SIZE     = 4;                         // u32's shared by unit
const u32 RBAG_SIZE     = 32;                        // redexes per unit = 32
const u32 LHDS_SIZE     = 32;                        // max local heads
const u32 BAGS_SIZE     = MAX_UNITS * RBAG_SIZE;     // redexes per GPU

// Types
// -----

typedef u8  Tag; // pointer tag: 4-bit
typedef u32 Val; // pointer val: 28-bit

// Core terms
const Tag NIL = 0x0; // empty node
const Tag REF = 0x1; // reference to a definition (closed net)
const Tag ERA = 0x2; // unboxed eraser
const Tag VRR = 0x3; // variable pointing to root
const Tag VR1 = 0x4; // variable pointing to aux1 port of node
const Tag VR2 = 0x5; // variable pointing to aux2 port of node
const Tag RDR = 0x6; // redirection to root
const Tag RD1 = 0x7; // redirection to aux1 port of node
const Tag RD2 = 0x8; // redirection to aux2 port of node
const Tag NUM = 0x9; // unboxed number
const Tag CON = 0xA; // points to main port of con node
const Tag DUP = 0xB; // points to main port of dup node
const Tag TRI = 0xC; // points to main port of tri node
const Tag CTR = 0xF; // last constructor

// Special values
const u32 NEO = 0xFFFFFFFD; // recently allocated value
const u32 TMP = 0xFFFFFFFE; // node has been moved to redex bag
const u32 TKN = 0xFFFFFFFF; // value taken by another thread, will be replaced soon

// Worker types
const u32 A1 = 0; // focuses on the A node, P1 port
const u32 A2 = 1; // focuses on the A node, P2 port
const u32 B1 = 2; // focuses on the B node, P1 port
const u32 B2 = 3; // focuses on the B node, P2 port

// Ports (P1 or P2)
typedef u8 Port;
const u32 P1 = 0;
const u32 P2 = 1;

// Pointers = 4-bit tag + 28-bit val
typedef u32 Ptr;

// Nodes are pairs of pointers
typedef struct {
  Ptr ports[2];
} Node;

// Wires are pairs of pointers
typedef u64 Wire;

// Maximum number of defs in a book
const u32 MAX_DEFS = 1 << 24; // FIXME: make a proper HashMap

typedef struct {
  Ptr   root;
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
  Ptr   root; // root wire
  Wire* bags; // redex bags (active pairs)
  Node* node; // memory buffer with all nodes
  Wire* head; // head expansion buffer
  u32   done; // number of completed threads
  u64   rwts; // number of rewrites performed
} Net;

// A worker local data
typedef struct {
  u32   tid;  // thread id (local)
  u32   uid;  // unit id (global)
  u32   quad; // worker quad (A1|A2|B1|B2)
  u32   port; // worker port (P1|P2)
  u32   aloc; // where to alloc next node
  u64   rwts; // local rewrites performed
  u32*  sm32; // shared 32-bit buffer
  u64*  sm64; // shared 64-bit buffer
  u64*  rlen; // local redex bag length
  Wire* rbag; // local redex bag
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
  return ((u32)tag << 28) | (val & 0xFFFFFFF);
}

// Gets the tag of a pointer
__host__ __device__ inline Tag tag(Ptr ptr) {
  return (Tag)(ptr >> 28);
}

// Gets the value of a pointer
__host__ __device__ inline Val val(Ptr ptr) {
  return ptr & 0xFFFFFFF;
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
      || is_num(ptr)
      || is_ref(ptr);
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
__host__ __device__ Ptr enter(Net* net, Ptr ptr) {
  Ptr* ref = target(net, ptr);
  while (tag(*ref) >= RDR && tag(*ref) <= RD2) {
    ptr = *ref;
    ref = target(net, ptr);
  }
  return ptr;
}

// Transforms a variable into a redirection
__host__ __device__ inline Ptr redir(Ptr ptr) {
  return mkptr(tag(ptr) + (is_var(ptr) ? 3 : 0), val(ptr));
}

// Creates a new wire
__host__ __device__ inline Wire mkwire(Ptr p1, Ptr p2) {
  return (((u64)p1) << 32) | ((u64)p2);
}

// Gets the left element of a wire
__host__ __device__ inline Ptr wire_lft(Wire wire) {
  return wire >> 32;
}

// Gets the right element of a wire
__host__ __device__ inline Ptr wire_rgt(Wire wire) {
  return wire & 0xFFFFFFFF;
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

// Allocates one node in memory
__device__ inline u32 alloc(Worker *worker, Net *net) {
  while (true) {
    u64* ref = (u64*)&net->node[worker->aloc];
    u64  got = atomicCAS((u64*)ref, 0, ((u64)NEO << 32) | (u64)NEO);
    worker->aloc = (worker->aloc + 1) % NODE_SIZE;
    if (got == 0) {
      return (worker->aloc - 1) % NODE_SIZE;
    }
  }
}

// Allocates many nodes in memory
// TODO: use the entire squad to perform this
__device__ inline u32 alloc_many(Worker *worker, Net *net, u32 size) {
  u64 MKNEO = ((u64)NEO << 32) | (u64)NEO;
  u32 space = 0;
  while (true) {
    if (worker->aloc + size - space > NODE_SIZE) {
      worker->aloc = 0;
    }
    u64* ref = (u64*)&net->node[worker->aloc];
    u64  got = atomicCAS(ref, 0, MKNEO);
    if (got != 0) {
      for (u32 i = 0; i < space; ++i) {
        u32  index = (worker->aloc - space + i) % NODE_SIZE;
        Node clear = mknode(mkptr(NIL,0), mkptr(NIL,0));
        u64* ref = (u64*)&net->node[index];
        u64  got = atomicCAS(ref, MKNEO, 0);
      }
      space = 0;
    } else {
      space += 1;
    }
    worker->aloc = (worker->aloc + 1) % NODE_SIZE;
    if (space == size) {
      return (worker->aloc - space) % NODE_SIZE;
    }
  }
}

// Gets the value of a ref; waits if taken.
__device__ Ptr take(Ptr* ref) {
  Ptr got = atomicExch((u32*)ref, TKN);
  u32 K = 0;
  while (got == TKN) {
    //dbug(&K, "take");
    got = atomicExch((u32*)ref, TKN);
  }
  return got;
}

// Attempts to replace 'exp' by 'neo', until it succeeds
__device__ bool replace(Ptr* ref, Ptr exp, Ptr neo) {
  Ptr got = atomicCAS((u32*)ref, exp, neo);
  u32 K = 0;
  while (got != exp) {
    //dbug(&K, "replace");
    got = atomicCAS((u32*)ref, exp, neo);
  }
  return true;
}

// Splits elements of two arrays evenly between each-other
__device__ inline void split(u32 tid, u64* a_len, u64* a_arr, u64* b_len, u64* b_arr, u64 max_len) {
  __syncthreads();
  u64* A_len = *a_len < *b_len ? a_len : b_len;
  u64* B_len = *a_len < *b_len ? b_len : a_len;
  u64* A_arr = *a_len < *b_len ? a_arr : b_arr;
  u64* B_arr = *a_len < *b_len ? b_arr : a_arr;
  bool move  = *A_len + 1 < *B_len;
  u64  min   = *A_len;
  u64  max   = *B_len;
  __syncthreads();
  for (u64 t = 0; t < max_len / (UNIT_SIZE * 2); ++t) {
    u64 i = min + t * (UNIT_SIZE * 2) + tid;
    u64 value;
    if (move && i < max) {
      value = B_arr[i];
      B_arr[i] = 0;
    }
    __syncthreads();
    if (move && i < max) {
      if ((i - min) % 2 == 0) {
        A_arr[min + (t * (UNIT_SIZE * 2) + tid) / 2] = value;
      } else {
        B_arr[min + (t * (UNIT_SIZE * 2) + tid) / 2] = value;
      }
    }
  }
  __syncthreads();
  u64 old_A_len = *A_len;
  u64 old_B_len = *B_len;
  if (move && tid == 0) {
    u64 new_A_len = (*A_len + *B_len) / 2 + (*A_len + *B_len) % 2;
    u64 new_B_len = (*A_len + *B_len) / 2;
    *A_len = new_A_len;
    *B_len = new_B_len;
  }
  __syncthreads();
}

// Pops a redex
__device__ inline Wire pop_redex(Worker* worker) {
  if (worker->quad == A1) {
    Wire redex = mkwire(0,0);
    if (*worker->rlen > 0) {
      u64 index = *worker->rlen - 1;
      *worker->rlen -= 1;
      redex = worker->rbag[index];
      worker->rbag[index] = mkwire(0,0);
    }
    *worker->sm64 = redex;
  }
  __syncwarp();
  Wire got = *worker->sm64;
  __syncwarp();
  *worker->sm64 = 0;
  if (worker->quad <= A2) {
    return mkwire(wire_lft(got), wire_rgt(got));
  } else {
    return mkwire(wire_rgt(got), wire_lft(got));
  }
}

// Puts a redex
__device__ inline void put_redex(Worker* worker, Ptr a_ptr, Ptr b_ptr) {
  //printf("[%04x:%x] put_redex %08x %08x\n", worker->uid, worker->quad, a_ptr, b_ptr);

  // optimization: avoids pushing non-reactive redexes
  bool exclude
    =  is_era(a_ptr) && is_era(b_ptr)
    || is_ref(a_ptr) && is_era(b_ptr)
    || is_era(a_ptr) && is_ref(b_ptr)
    || is_ref(a_ptr) && is_ref(b_ptr);
  if (exclude) {
    worker->rwts += 1;
    return;
  }

  // pushes redex to end of bag
  u32 index = atomicAdd(worker->rlen, 1);
  if (index < RBAG_SIZE - 1) {
    worker->rbag[index] = mkwire(a_ptr, b_ptr);
  }
}

// Adjusts a dereferenced pointer
__device__ Ptr adjust(Worker* worker, Ptr ptr, u32 delta) {
  return mkptr(tag(ptr), has_loc(ptr) ? val(ptr) + delta : val(ptr));
}

// Dereferences a global definition
__device__ void deref(Worker* worker, Net* net, Book* book, Ptr* deref_ptr, Ptr parent) {
  // Loads definition
  Term* term = NULL;
  if (deref_ptr != NULL) {
    term = book->defs[val(*deref_ptr)];
  }

  // Allocates needed space
  if (term != NULL && worker->quad == A1) {
    worker->sm32[0] = alloc_many(worker, net, term->nlen);
  }
  __syncwarp();
  u32 delta = 0;
  if (term != NULL) {
    delta = worker->sm32[0];
  }

  // Loads dereferenced nodes, adjusted
  if (term != NULL) {
    for (u32 i = 0; i < div(term->nlen, UNIT_SIZE); ++i) {
      u32 loc = i * UNIT_SIZE + worker->quad;
      if (loc < term->nlen) {
        //printf("... node %u\n", loc);
        Node got = term->node[loc];
        Ptr  p1  = adjust(worker, got.ports[P1], delta);
        Ptr  p2  = adjust(worker, got.ports[P2], delta);
        replace(at(net, delta + loc, P1), NEO, p1);
        replace(at(net, delta + loc, P2), NEO, p2);
      }
    }
  }

  // Loads dereferenced redexes, adjusted
  if (term != NULL) {
    for (u32 i = 0; i < div(term->alen, UNIT_SIZE); ++i) {
      u32 loc = i * UNIT_SIZE + worker->quad;
      if (loc < term->alen) {
        Wire got = term->acts[loc];
        Ptr  p1  = adjust(worker, wire_lft(got), delta);
        Ptr  p2  = adjust(worker, wire_rgt(got), delta);
        put_redex(worker, p1, p2);
      }
    }
  }

  // Loads dereferenced root, adjusted
  if (term != NULL) {
    *deref_ptr = adjust(worker, term->root, delta);
  }
  __syncwarp();

  // Links root
  if (term != NULL && worker->quad == A1) {
    Ptr* trg = target(net, *deref_ptr);
    if (trg != NULL) {
      *trg = parent;
    }
  }
}

// Atomically links the node in 'src_ref' towards 'trg_ptr'.
__device__ void link(Worker* worker, Net* net, Book* book, Ptr* src_ref, Ptr src_ptr, Ptr dir_ptr) {

  // Create a new redex
  if (is_pri(src_ptr) && is_pri(dir_ptr)) {
    atomicCAS(src_ref, TKN, 0);
    put_redex(worker, *src_ref, dir_ptr);
    return;
  }

  // Move src towards either a VAR or another PRI
  if (is_pri(src_ptr) && is_red(dir_ptr)) {
    while (true) {
      //dbug(&K, "link");

      // Peek the target, which may not be owned by us.
      Ptr* trg_ref = target(net, dir_ptr);
      Ptr  trg_ptr = atomicAdd(trg_ref, 0);

      // If target is a redirection, clear and move forward.
      if (is_red(trg_ptr)) {
        // We own the redirection, so we can mutate it.
        *trg_ref = 0;
        dir_ptr = trg_ptr;
        continue;
      }

      // If target is a variable, try replacing it by the node.
      else if (is_var(trg_ptr)) {
        // Peeks the source node.
        Ptr src_ptr = *src_ref;

        // We don't own the var, so we must try replacing with a CAS.
        if (atomicCAS((u32*)trg_ref, trg_ptr, src_ptr) == trg_ptr) {
          // Collect the orphaned backward path.
          trg_ref = target(net, trg_ptr);
          trg_ptr = *trg_ref;
          while (is_red(trg_ptr)) {
            *trg_ref = 0;
            trg_ref = target(net, trg_ptr);
            trg_ptr = *trg_ref;
          }
          // Clear source location.
          *src_ref = 0;
          return;
        }

        // If the CAS failed, the var changed, so we try again.
        continue;
      }

      // If it is a node, two threads will reach this branch.
      else if (is_pri(trg_ptr) || is_ref(trg_ptr) || trg_ptr == TMP) {

        // Sort references, to avoid deadlocks.
        Ptr *fst_ref = src_ref < trg_ref ? src_ref : trg_ref;
        Ptr *snd_ref = src_ref < trg_ref ? trg_ref : src_ref;

        // Swap first reference by TMP placeholder.
        Ptr fst_ptr = atomicExch((u32*)fst_ref, TMP);

        // First to arrive creates a redex.
        if (fst_ptr != TMP) {
          Ptr snd_ptr = atomicExch((u32*)snd_ref, TMP);
          put_redex(worker, fst_ptr, snd_ptr);
          return;

        // Second to arrive clears up the memory.
        } else {
          *fst_ref = 0;
          replace((u32*)snd_ref, TMP, 0);
          return;
        }
      }

      // If it is taken, we wait.
      else if (trg_ptr == TKN) {
        continue;
      }

      // Shouldn't be reached.
      else {
        return;
      }
    }
  }

  // Optimization: safely shorten redirections
  if (is_red(src_ptr) && is_red(dir_ptr)) {
    while (true) {
      Ptr* ste_ref = target(net, src_ptr);
      Ptr  ste_ptr = *ste_ref;
      if (is_var(ste_ptr)) {
        Ptr* trg_ref = target(net, ste_ptr);
        Ptr  trg_ptr = atomicAdd(trg_ref, 0);
        if (is_red(trg_ptr)) {
          Ptr neo_ptr = mkptr(tag(trg_ptr) - 3, val(trg_ptr));
          Ptr updated = atomicCAS(ste_ref, ste_ptr, neo_ptr);
          if (updated == ste_ptr) {
            *trg_ref = 0;
            continue;
          }
        }
      }
      break;
    }
    return;
  }

}

// Rewrite
// -------

__device__ Worker init_worker(Net* net, bool flip) {
  __shared__ u32 SMEM[GROUP_SIZE * SMEM_SIZE];

  for (u32 i = 0; i < GROUP_SIZE * SMEM_SIZE / BLOCK_SIZE; ++i) {
    SMEM[i * BLOCK_SIZE + threadIdx.x] = 0;
  }
  __syncthreads();

  u32 tid = threadIdx.x;
  u32 gid = blockIdx.x * blockDim.x + tid;
  u32 uid = gid / UNIT_SIZE;
  u32 row = uid / GROUP_SIZE;
  u32 col = uid % GROUP_SIZE;

  Worker worker;
  worker.uid  = flip ? col * GROUP_SIZE + row : row * GROUP_SIZE + col;
  worker.tid  = threadIdx.x;
  worker.aloc = rng(clock() * (gid + 1)) % NODE_SIZE;
  worker.rwts = 0;
  worker.quad = worker.tid % 4;
  worker.port = worker.tid % 2;
  worker.sm32 = (u32*)(SMEM + worker.tid / UNIT_SIZE * SMEM_SIZE);
  worker.sm64 = (u64*)(SMEM + worker.tid / UNIT_SIZE * SMEM_SIZE);
  worker.rlen = net->bags + worker.uid * RBAG_SIZE;
  worker.rbag = worker.rlen + 1;

  return worker;
}

// An active wire is reduced by 4 parallel threads, each one performing "1/4" of
// the work. Each thread will be pointing to a node of the active pair, and an
// aux port of that node. So, when nodes A-B interact, we have 4 thread quads:
// - Thread A1: points to node A and its aux1
// - Thread A2: points to node A and its aux2
// - Thread B1: points to node B and its aux1
// - Thread B2: points to node B and its aux2
// This is organized so that local threads can perform the same instructions
// whenever possible. So, for example, in a commutation rule, all the 4 clones
// would be allocated at the same time.
__global__ void global_rewrite(Net* net, Book* book, u32 repeat, u32 tick, bool flip) {

  // Initializes local vars
  Worker worker = init_worker(net, flip);

  for (u32 turn = 0; turn < repeat; ++turn) {
    // Checks if we're full
    bool is_full = *worker.rlen > RBAG_SIZE - MAX_NEW_REDEX;

    // Pops a redex from local bag
    Wire redex;
    Ptr a_ptr, b_ptr;
    if (!is_full) {
      redex = pop_redex(&worker);
      a_ptr = wire_lft(redex);
      b_ptr = wire_rgt(redex);
    } else {
      //printf("[%04x:%x] full\n", worker.uid, worker.quad);
    }
    __syncwarp();

    // Dereferences
    Ptr* deref_ptr = NULL;
    if (is_ref(a_ptr) && is_ctr(b_ptr)) {
      deref_ptr = &a_ptr;
    }
    if (is_ref(b_ptr) && is_ctr(a_ptr)) {
      deref_ptr = &b_ptr;
    }
    deref(&worker, net, book, deref_ptr, mkptr(NIL,0));

    // Defines type of interaction
    bool rewrite = !is_full && a_ptr != 0 && b_ptr != 0;
    bool var_pri = rewrite && is_var(a_ptr) && is_pri(b_ptr) && worker.port == P1;
    bool era_ctr = rewrite && is_era(a_ptr) && is_ctr(b_ptr);
    bool ctr_era = rewrite && is_ctr(a_ptr) && is_era(b_ptr);
    bool con_con = rewrite && is_ctr(a_ptr) && is_ctr(b_ptr) && tag(a_ptr) == tag(b_ptr);
    bool con_dup = rewrite && is_ctr(a_ptr) && is_ctr(b_ptr) && tag(a_ptr) != tag(b_ptr);

    // Local rewrite variables
    Ptr  ak_dir; // dir to our aux port
    Ptr  bk_dir; // dir to other aux port
    Ptr *ak_ref; // ref to our aux port
    Ptr *bk_ref; // ref to other aux port
    Ptr  ak_ptr; // val of our aux port
    Ptr  mv_ptr; // val of ptr to send to other side
    u32  y0_idx; // idx of other clone idx

    // Inc rewrite count
    if (rewrite && worker.quad == A1) {
      worker.rwts += 1;
    }

    // Gets port here
    if (rewrite && (ctr_era || con_con || con_dup)) {
      ak_dir = mkptr(RD1 + worker.port, val(a_ptr));
      ak_ref = target(net, ak_dir);
      ak_ptr = take(ak_ref);
    }

    // Gets port there
    if (rewrite && (era_ctr || con_con || con_dup)) {
      bk_dir = mkptr(RD1 + worker.port, val(b_ptr));
      bk_ref = target(net, bk_dir);
    }

    // If era_ctr, send an erasure
    if (rewrite && era_ctr) {
      mv_ptr = mkptr(ERA, 0);
    }

    // If con_con, send a redirection
    if (rewrite && con_con) {
      mv_ptr = redir(ak_ptr);
    }

    // If con_dup, alloc clones base index
    if (rewrite && con_dup && worker.quad == A1) {
      worker.sm32[0] = alloc_many(&worker, net, 4);
    }
    __syncwarp();

    // If con_dup, create inner wires between clones
    if (rewrite && con_dup) {
      u32 al_loc = worker.sm32[0];
      u32 cx_loc = al_loc + worker.quad;
      u32 c1_loc = al_loc + (worker.quad <= A2 ? 2 : 0);
      u32 c2_loc = al_loc + (worker.quad <= A2 ? 3 : 1);
      replace(at(net, cx_loc, P1), NEO, mkptr(worker.port == P1 ? VR1 : VR2, c1_loc));
      replace(at(net, cx_loc, P2), NEO, mkptr(worker.port == P1 ? VR1 : VR2, c2_loc));
      mv_ptr = mkptr(tag(a_ptr), cx_loc);
    }
    __syncwarp();

    // Send ptr to other side
    if (rewrite && (era_ctr || con_con || con_dup)) {
      worker.sm32[worker.quad + (worker.quad <= A2 ? 2 : -2)] = mv_ptr;
    }
    __syncwarp();

    // Receive ptr from other side
    if (rewrite && (con_con || ctr_era || con_dup)) {
      *ak_ref = worker.sm32[worker.quad];
    }
    __syncwarp();

    // If var_pri, the var must be a deref root, so we just subst
    if (rewrite && var_pri && worker.port == P1) {
      atomicExch((u32*)target(net, a_ptr), b_ptr);
    }

    // Links the rewritten port
    Ptr* src_ref, src_ptr, dir_ptr;
    if (rewrite && (con_con || ctr_era || con_dup)) {
      src_ref = con_con ? ak_ref : ak_ref;
      src_ptr = *src_ref;
      dir_ptr = con_con ? redir(bk_dir) : redir(ak_ptr);
    }
    __syncwarp();
    if (rewrite && (con_con || ctr_era || con_dup)) {
      link(&worker, net, book, src_ref, src_ptr, dir_ptr);
    }
    __syncwarp();
  }

  // Splits redexes with neighbor
  u32  side  = ((worker.tid / UNIT_SIZE) >> (GROUP_LOG2 - 1 - (tick % GROUP_LOG2))) & 1;
  u32  lpad  = (1 << (GROUP_LOG2 - 1)) >> (tick % GROUP_LOG2);
  u32  gpad  = flip ? lpad * GROUP_SIZE : lpad;
  u32  a_uid = worker.uid;
  u32  b_uid = side ? worker.uid - gpad : worker.uid + gpad;
  u64* a_len = net->bags + a_uid * RBAG_SIZE;
  u64* b_len = net->bags + b_uid * RBAG_SIZE;
  //printf("[%04x:%x] tid=%x split! %04x ~ %04x | flip=%u tick=%u side=%u lpad=%u gpad=%u\n", worker.uid, worker.quad, worker.tid, a_uid, b_uid, flip, tick, side, lpad, gpad);
  split(worker.quad + side * UNIT_SIZE, a_len, a_len+1, b_len, b_len+1, RBAG_SIZE);
  __syncthreads();

  // When the work ends, sum stats
  if (worker.rwts > 0) {
    atomicAdd(&net->rwts, worker.rwts);
  }
}

void do_global_rewrite(Net* net, Book* book, u32 repeat, u32 tick, bool flip) {
  global_rewrite<<<GROUP_SIZE, BLOCK_SIZE>>>(net, book, repeat, tick, flip);
  // print any error launching this kernel:
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("CUDA error: %s\n", cudaGetErrorString(err));
  }
}

// Expand
// ------

// Collects local expansion heads recursively
__device__ void expand(Worker* worker, Net* net, Book* book, Ptr dir, u32* len, u32* lhds) {
  Ptr ptr = *target(net, dir);
  if (is_ctr(ptr)) {
    expand(worker, net, book, mkptr(VR1, val(ptr)), len, lhds);
    expand(worker, net, book, mkptr(VR2, val(ptr)), len, lhds);
  } else if (is_red(ptr)) {
    expand(worker, net, book, ptr, len, lhds);
  } else if (is_ref(ptr) && *len < LHDS_SIZE) {
    lhds[(*len)++] = dir;
  }
}

// Takes an initial head location for each unit
__global__ void global_expand_prepare(Net* net) {
  u32 uid = blockIdx.x * blockDim.x + threadIdx.x;

  // Traverses down
  u32 key = uid;
  Ptr dir = mkptr(VRR, 0);
  Ptr ptr, *ref;
  for (u32 depth = 0; depth < HEAD_LOG2; ++depth) {
    dir = enter(net, dir);
    ref = target(net, dir);
    if (ref != NULL) {
      ptr = *ref;
      if (is_ctr(ptr)) {
        dir = mkptr(key & 1 ? VR1 : VR2, val(ptr));
        key = key >> 1;
      }
    }
  }

  // Takes ptr
  dir = enter(net, dir);
  ref = target(net, dir);
  if (ref != NULL) {
    ptr = atomicExch(ref, TKN);
  }

  // Stores ptr
  if (ptr != TKN) {
    net->head[uid] = mkwire(dir, ptr);
  } else {
    net->head[uid] = mkwire(mkptr(NIL,0), mkptr(NIL,0));
  }

}

// Performs global expansion of heads
__global__ void global_expand(Net* net, Book* book) {
  __shared__ u32 HEAD[GROUP_SIZE * LHDS_SIZE];

  for (u32 i = 0; i < GROUP_SIZE * LHDS_SIZE / BLOCK_SIZE; ++i) {
    HEAD[i * BLOCK_SIZE + threadIdx.x] = 0;
  }
  __syncthreads();

  Worker worker = init_worker(net, 0);

  u32* head = HEAD + worker.tid / UNIT_SIZE * LHDS_SIZE;

  Wire got = net->head[worker.uid];
  Ptr  dir = wire_lft(got);
  Ptr* ref = target(net, dir);
  Ptr  ptr = wire_rgt(got);

  if (worker.quad == A1 && ptr != mkptr(NIL,0)) {
    *ref = ptr;
  }
  __syncthreads();

  u32 len = 0;
  if (worker.quad == A1 && ptr != mkptr(NIL,0)) {
    expand(&worker, net, book, dir, &len, head);
  }
  __syncthreads();

  for (u32 i = 0; i < LHDS_SIZE; ++i) {
    Ptr  dir = head[i];
    Ptr* ref = target(net, dir);
    if (ref != NULL && !is_ref(*ref)) {
      ref = NULL;
    }
    deref(&worker, net, book, ref, dir);
  }
  __syncthreads();
}

// Performs a global head expansion
void do_global_expand(Net* net, Book* book) {
  global_expand_prepare<<<GROUP_SIZE, GROUP_SIZE>>>(net);
  global_expand<<<GROUP_SIZE, BLOCK_SIZE>>>(net, book);
}

// Host<->Device
// -------------

__host__ Net* mknet() {
  Net* net  = (Net*)malloc(sizeof(Net));
  net->root = mkptr(NIL, 0);
  net->rwts = 0;
  net->done = 0;
  net->bags = (Wire*)malloc(BAGS_SIZE * sizeof(Wire));
  net->node = (Node*)malloc(NODE_SIZE * sizeof(Node));
  net->head = (Wire*)malloc(HEAD_SIZE * sizeof(Wire));
  memset(net->bags, 0, BAGS_SIZE * sizeof(Wire));
  memset(net->node, 0, NODE_SIZE * sizeof(Node));
  return net;
}

__host__ Net* net_to_gpu(Net* host_net) {
  // Allocate memory on the device for the Net object, and its data
  Net*  device_net;
  Wire* device_bags;
  Node* device_node;
  Wire* device_head;

  cudaMalloc((void**)&device_net, sizeof(Net));
  cudaMalloc((void**)&device_bags, BAGS_SIZE * sizeof(Wire));
  cudaMalloc((void**)&device_node, NODE_SIZE * sizeof(Node));
  cudaMalloc((void**)&device_head, HEAD_SIZE * sizeof(Wire));

  // Copy the host data to the device memory
  cudaMemcpy(device_bags, host_net->bags, BAGS_SIZE * sizeof(Wire), cudaMemcpyHostToDevice);
  cudaMemcpy(device_node, host_net->node, NODE_SIZE * sizeof(Node), cudaMemcpyHostToDevice);
  cudaMemcpy(device_head, host_net->head, HEAD_SIZE * sizeof(Wire), cudaMemcpyHostToDevice);

  // Create a temporary host Net object with device pointers
  Net temp_net  = *host_net;
  temp_net.bags = device_bags;
  temp_net.node = device_node;
  temp_net.head = device_head;

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
  host_net->node = (Node*)malloc(NODE_SIZE * sizeof(Node));
  host_net->head = (Wire*)malloc(HEAD_SIZE * sizeof(Wire));

  // Retrieve the device pointers for data
  Wire* device_bags;
  Node* device_node;
  Ptr*  device_head;
  cudaMemcpy(&device_bags, &(device_net->bags), sizeof(Wire*), cudaMemcpyDeviceToHost);
  cudaMemcpy(&device_node, &(device_net->node), sizeof(Node*), cudaMemcpyDeviceToHost);
  cudaMemcpy(&device_head, &(device_net->head), sizeof(Wire*), cudaMemcpyDeviceToHost);

  // Copy the device data to the host memory
  cudaMemcpy(host_net->bags, device_bags, BAGS_SIZE * sizeof(Wire), cudaMemcpyDeviceToHost);
  cudaMemcpy(host_net->node, device_node, NODE_SIZE * sizeof(Node), cudaMemcpyDeviceToHost);
  cudaMemcpy(host_net->head, device_head, HEAD_SIZE * sizeof(Wire), cudaMemcpyDeviceToHost);

  return host_net;
}

__host__ void net_free_on_gpu(Net* device_net) {
  // Retrieve the device pointers for data
  Wire* device_bags;
  Node* device_node;
  Wire* device_head;
  cudaMemcpy(&device_bags, &(device_net->bags), sizeof(Wire*), cudaMemcpyDeviceToHost);
  cudaMemcpy(&device_node, &(device_net->node), sizeof(Node*), cudaMemcpyDeviceToHost);
  cudaMemcpy(&device_head, &(device_net->head), sizeof(Wire*), cudaMemcpyDeviceToHost);

  // Free the device memory
  cudaFree(device_bags);
  cudaFree(device_node);
  cudaFree(device_head);
  cudaFree(device_net);
}

__host__ void net_free_on_cpu(Net* host_net) {
  free(host_net->bags);
  free(host_net->node);
  free(host_net->head);
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

  for (u32 i = 0; i < MAX_DEFS; ++i) {
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
  for (u32 i = 0; i < MAX_DEFS; ++i) {
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

__host__ const char* show_ptr(Ptr ptr, u32 slot) {
  static char buffer[8][20];
  if (ptr == 0) {
    strcpy(buffer[slot], "           ");
    return buffer[slot];
  } else if (ptr == TKN) {
    strcpy(buffer[slot], "[..........]");
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
      default:tag_str = tag(ptr) >= DUP ? "DUP" : "???"; break;
    }
    snprintf(buffer[slot], sizeof(buffer[slot]), "%s:%07X", tag_str, val(ptr));
    return buffer[slot];
  }
}

// Prints a net in hexadecimal, limited to a given size
void print_net(Net* net) {
  printf("Root:\n");
  printf("- %s\n", show_ptr(net->root,0));
  printf("Bags:\n");
  for (u32 i = 0; i < BAGS_SIZE; ++i) {
    if (i % RBAG_SIZE == 0 && net->bags[i] > 0) {
      printf("- [%07X] LEN=%llu\n", i, net->bags[i]);
    } else {
      //Ptr a = wire_lft(net->bags[i]);
      //Ptr b = wire_rgt(net->bags[i]);
      //if (a != 0 || b != 0) {
        //printf("- [%07X] %s %s\n", i, show_ptr(a,0), show_ptr(b,1));
      //}
    }
  }
  //printf("Node:\n");
  //for (u32 i = 0; i < NODE_SIZE; ++i) {
    //Ptr a = net->node[i].ports[P1];
    //Ptr b = net->node[i].ports[P2];
    //if (a != 0 || b != 0) {
      //printf("- [%07X] %s %s\n", i, show_ptr(a,0), show_ptr(b,1));
    //}
  //}
  printf("Rwts: %llu\n", net->rwts);
}
//void print_net(Net* net) {
  ////printf("Root:\n");
  ////printf("- %s\n", show_ptr(net->root,0));
  //printf("net.root = Ptr { data: 0x%08x };\n", net->root);
  ////printf("Bags:\n");
  //for (u32 i = 0; i < BAGS_SIZE; ++i) {
    //if (i % RBAG_SIZE == 0 && net->bags[i] > 0) {
      ////printf("- [%07X] LEN=%llu\n", i, net->bags[i]);
    //} else {
      //Ptr a = wire_lft(net->bags[i]);
      //Ptr b = wire_rgt(net->bags[i]);
      //if (a != 0 || b != 0) {
        ////printf("- [%07X] %s %s\n", i, show_ptr(a,0), show_ptr(b,1));
        //printf("net.acts.push((Ptr { data: 0x%08x }, Ptr { data: 0x%08x }));\n", a, b);
      //}
    //}
  //}
  ////printf("Node:\n");
  //for (u32 i = 0; i < NODE_SIZE; ++i) {
    //Ptr a = net->node[i].ports[P1];
    //Ptr b = net->node[i].ports[P2];
    //if (a != 0 || b != 0) {
      ////printf("- [%07X] %s %s\n", i, show_ptr(a,0), show_ptr(b,1));
      //printf("net.node[0x%07X] = Node::new(Ptr { data: 0x%08x }, Ptr { data: 0x%08x });\n", i, a, b);
    //}
  //}
  //printf("Rwts: %u\n", net->rwts);
  //printf("\n");
//}

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
  if (is_var(ptr)) {
    u32 got = map_lookup(var_ids, ptr);
    if (got == var_ids->size) {
      u32 name = var_ids->size;
      Ptr targ = *target(net, enter(net, ptr));
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
  book->defs[0x0000000f]->root     = 0xa0000000;
  book->defs[0x0000000f]->alen     = 0;
  book->defs[0x0000000f]->acts     = (Wire*) malloc(0 * sizeof(Wire));
  book->defs[0x0000000f]->nlen     = 3;
  book->defs[0x0000000f]->node     = (Node*) malloc(3 * sizeof(Node));
  book->defs[0x0000000f]->node[ 0] = (Node) {0x20000000,0xa0000001};
  book->defs[0x0000000f]->node[ 1] = (Node) {0x20000000,0xa0000002};
  book->defs[0x0000000f]->node[ 2] = (Node) {0x50000002,0x40000002};
  // F
  book->defs[0x00000010]           = (Term*) malloc(sizeof(Term));
  book->defs[0x00000010]->root     = 0xa0000000;
  book->defs[0x00000010]->alen     = 0;
  book->defs[0x00000010]->acts     = (Wire*) malloc(0 * sizeof(Wire));
  book->defs[0x00000010]->nlen     = 2;
  book->defs[0x00000010]->node     = (Node*) malloc(2 * sizeof(Node));
  book->defs[0x00000010]->node[ 0] = (Node) {0x20000000,0xa0000001};
  book->defs[0x00000010]->node[ 1] = (Node) {0x50000001,0x40000001};
  // I
  book->defs[0x00000013]           = (Term*) malloc(sizeof(Term));
  book->defs[0x00000013]->root     = 0xa0000000;
  book->defs[0x00000013]->alen     = 0;
  book->defs[0x00000013]->acts     = (Wire*) malloc(0 * sizeof(Wire));
  book->defs[0x00000013]->nlen     = 5;
  book->defs[0x00000013]->node     = (Node*) malloc(5 * sizeof(Node));
  book->defs[0x00000013]->node[ 0] = (Node) {0x40000003,0xa0000001};
  book->defs[0x00000013]->node[ 1] = (Node) {0x20000000,0xa0000002};
  book->defs[0x00000013]->node[ 2] = (Node) {0xa0000003,0xa0000004};
  book->defs[0x00000013]->node[ 3] = (Node) {0x40000000,0x50000004};
  book->defs[0x00000013]->node[ 4] = (Node) {0x20000000,0x50000003};
  // O
  book->defs[0x00000019]           = (Term*) malloc(sizeof(Term));
  book->defs[0x00000019]->root     = 0xa0000000;
  book->defs[0x00000019]->alen     = 0;
  book->defs[0x00000019]->acts     = (Wire*) malloc(0 * sizeof(Wire));
  book->defs[0x00000019]->nlen     = 5;
  book->defs[0x00000019]->node     = (Node*) malloc(5 * sizeof(Node));
  book->defs[0x00000019]->node[ 0] = (Node) {0x40000002,0xa0000001};
  book->defs[0x00000019]->node[ 1] = (Node) {0xa0000002,0xa0000003};
  book->defs[0x00000019]->node[ 2] = (Node) {0x40000000,0x50000004};
  book->defs[0x00000019]->node[ 3] = (Node) {0x20000000,0xa0000004};
  book->defs[0x00000019]->node[ 4] = (Node) {0x20000000,0x50000002};
  // S
  book->defs[0x0000001d]           = (Term*) malloc(sizeof(Term));
  book->defs[0x0000001d]->root     = 0xa0000000;
  book->defs[0x0000001d]->alen     = 0;
  book->defs[0x0000001d]->acts     = (Wire*) malloc(0 * sizeof(Wire));
  book->defs[0x0000001d]->nlen     = 4;
  book->defs[0x0000001d]->node     = (Node*) malloc(4 * sizeof(Node));
  book->defs[0x0000001d]->node[ 0] = (Node) {0x40000002,0xa0000001};
  book->defs[0x0000001d]->node[ 1] = (Node) {0xa0000002,0xa0000003};
  book->defs[0x0000001d]->node[ 2] = (Node) {0x40000000,0x50000003};
  book->defs[0x0000001d]->node[ 3] = (Node) {0x20000000,0x50000002};
  // T
  book->defs[0x0000001e]           = (Term*) malloc(sizeof(Term));
  book->defs[0x0000001e]->root     = 0xa0000000;
  book->defs[0x0000001e]->alen     = 0;
  book->defs[0x0000001e]->acts     = (Wire*) malloc(0 * sizeof(Wire));
  book->defs[0x0000001e]->nlen     = 2;
  book->defs[0x0000001e]->node     = (Node*) malloc(2 * sizeof(Node));
  book->defs[0x0000001e]->node[ 0] = (Node) {0x50000001,0xa0000001};
  book->defs[0x0000001e]->node[ 1] = (Node) {0x20000000,0x40000000};
  // Z
  book->defs[0x00000024]           = (Term*) malloc(sizeof(Term));
  book->defs[0x00000024]->root     = 0xa0000000;
  book->defs[0x00000024]->alen     = 0;
  book->defs[0x00000024]->acts     = (Wire*) malloc(0 * sizeof(Wire));
  book->defs[0x00000024]->nlen     = 2;
  book->defs[0x00000024]->node     = (Node*) malloc(2 * sizeof(Node));
  book->defs[0x00000024]->node[ 0] = (Node) {0x20000000,0xa0000001};
  book->defs[0x00000024]->node[ 1] = (Node) {0x50000001,0x40000001};
  // af
  book->defs[0x0000096a]           = (Term*) malloc(sizeof(Term));
  book->defs[0x0000096a]->root     = 0xa0000000;
  book->defs[0x0000096a]->alen     = 0;
  book->defs[0x0000096a]->acts     = (Wire*) malloc(0 * sizeof(Wire));
  book->defs[0x0000096a]->nlen     = 3;
  book->defs[0x0000096a]->node     = (Node*) malloc(3 * sizeof(Node));
  book->defs[0x0000096a]->node[ 0] = (Node) {0xa0000001,0x50000002};
  book->defs[0x0000096a]->node[ 1] = (Node) {0x10025a9d,0xa0000002};
  book->defs[0x0000096a]->node[ 2] = (Node) {0x10025aa4,0x50000000};
  // c0
  book->defs[0x000009c1]           = (Term*) malloc(sizeof(Term));
  book->defs[0x000009c1]->root     = 0xa0000000;
  book->defs[0x000009c1]->alen     = 0;
  book->defs[0x000009c1]->acts     = (Wire*) malloc(0 * sizeof(Wire));
  book->defs[0x000009c1]->nlen     = 2;
  book->defs[0x000009c1]->node     = (Node*) malloc(2 * sizeof(Node));
  book->defs[0x000009c1]->node[ 0] = (Node) {0x20000000,0xa0000001};
  book->defs[0x000009c1]->node[ 1] = (Node) {0x50000001,0x40000001};
  // c1
  book->defs[0x000009c2]           = (Term*) malloc(sizeof(Term));
  book->defs[0x000009c2]->root     = 0xa0000000;
  book->defs[0x000009c2]->alen     = 0;
  book->defs[0x000009c2]->acts     = (Wire*) malloc(0 * sizeof(Wire));
  book->defs[0x000009c2]->nlen     = 3;
  book->defs[0x000009c2]->node     = (Node*) malloc(3 * sizeof(Node));
  book->defs[0x000009c2]->node[ 0] = (Node) {0xa0000001,0xa0000002};
  book->defs[0x000009c2]->node[ 1] = (Node) {0x40000002,0x50000002};
  book->defs[0x000009c2]->node[ 2] = (Node) {0x40000001,0x50000001};
  // c2
  book->defs[0x000009c3]           = (Term*) malloc(sizeof(Term));
  book->defs[0x000009c3]->root     = 0xa0000000;
  book->defs[0x000009c3]->alen     = 0;
  book->defs[0x000009c3]->acts     = (Wire*) malloc(0 * sizeof(Wire));
  book->defs[0x000009c3]->nlen     = 5;
  book->defs[0x000009c3]->node     = (Node*) malloc(5 * sizeof(Node));
  book->defs[0x000009c3]->node[ 0] = (Node) {0xb0000001,0xa0000004};
  book->defs[0x000009c3]->node[ 1] = (Node) {0xa0000002,0xa0000003};
  book->defs[0x000009c3]->node[ 2] = (Node) {0x40000004,0x40000003};
  book->defs[0x000009c3]->node[ 3] = (Node) {0x50000002,0x50000004};
  book->defs[0x000009c3]->node[ 4] = (Node) {0x40000002,0x50000003};
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
  book->defs[0x000009c4]->node[ 3] = (Node) {0x40000006,0x40000004};
  book->defs[0x000009c4]->node[ 4] = (Node) {0x50000003,0x40000005};
  book->defs[0x000009c4]->node[ 5] = (Node) {0x50000004,0x50000006};
  book->defs[0x000009c4]->node[ 6] = (Node) {0x40000003,0x50000005};
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
  book->defs[0x000009c5]->node[ 4] = (Node) {0x40000008,0x40000005};
  book->defs[0x000009c5]->node[ 5] = (Node) {0x50000004,0x40000006};
  book->defs[0x000009c5]->node[ 6] = (Node) {0x50000005,0x40000007};
  book->defs[0x000009c5]->node[ 7] = (Node) {0x50000006,0x50000008};
  book->defs[0x000009c5]->node[ 8] = (Node) {0x40000004,0x50000007};
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
  book->defs[0x000009c6]->node[ 5] = (Node) {0x4000000a,0x40000006};
  book->defs[0x000009c6]->node[ 6] = (Node) {0x50000005,0x40000007};
  book->defs[0x000009c6]->node[ 7] = (Node) {0x50000006,0x40000008};
  book->defs[0x000009c6]->node[ 8] = (Node) {0x50000007,0x40000009};
  book->defs[0x000009c6]->node[ 9] = (Node) {0x50000008,0x5000000a};
  book->defs[0x000009c6]->node[10] = (Node) {0x40000005,0x50000009};
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
  book->defs[0x000009c7]->node[ 6] = (Node) {0x4000000c,0x40000007};
  book->defs[0x000009c7]->node[ 7] = (Node) {0x50000006,0x40000008};
  book->defs[0x000009c7]->node[ 8] = (Node) {0x50000007,0x40000009};
  book->defs[0x000009c7]->node[ 9] = (Node) {0x50000008,0x4000000a};
  book->defs[0x000009c7]->node[10] = (Node) {0x50000009,0x4000000b};
  book->defs[0x000009c7]->node[11] = (Node) {0x5000000a,0x5000000c};
  book->defs[0x000009c7]->node[12] = (Node) {0x40000006,0x5000000b};
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
  book->defs[0x000009c8]->node[ 7] = (Node) {0x4000000e,0x40000008};
  book->defs[0x000009c8]->node[ 8] = (Node) {0x50000007,0x40000009};
  book->defs[0x000009c8]->node[ 9] = (Node) {0x50000008,0x4000000a};
  book->defs[0x000009c8]->node[10] = (Node) {0x50000009,0x4000000b};
  book->defs[0x000009c8]->node[11] = (Node) {0x5000000a,0x4000000c};
  book->defs[0x000009c8]->node[12] = (Node) {0x5000000b,0x4000000d};
  book->defs[0x000009c8]->node[13] = (Node) {0x5000000c,0x5000000e};
  book->defs[0x000009c8]->node[14] = (Node) {0x40000007,0x5000000d};
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
  book->defs[0x000009c9]->node[ 8] = (Node) {0x40000010,0x40000009};
  book->defs[0x000009c9]->node[ 9] = (Node) {0x50000008,0x4000000a};
  book->defs[0x000009c9]->node[10] = (Node) {0x50000009,0x4000000b};
  book->defs[0x000009c9]->node[11] = (Node) {0x5000000a,0x4000000c};
  book->defs[0x000009c9]->node[12] = (Node) {0x5000000b,0x4000000d};
  book->defs[0x000009c9]->node[13] = (Node) {0x5000000c,0x4000000e};
  book->defs[0x000009c9]->node[14] = (Node) {0x5000000d,0x4000000f};
  book->defs[0x000009c9]->node[15] = (Node) {0x5000000e,0x50000010};
  book->defs[0x000009c9]->node[16] = (Node) {0x40000008,0x5000000f};
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
  book->defs[0x000009ca]->node[ 9] = (Node) {0x40000012,0x4000000a};
  book->defs[0x000009ca]->node[10] = (Node) {0x50000009,0x4000000b};
  book->defs[0x000009ca]->node[11] = (Node) {0x5000000a,0x4000000c};
  book->defs[0x000009ca]->node[12] = (Node) {0x5000000b,0x4000000d};
  book->defs[0x000009ca]->node[13] = (Node) {0x5000000c,0x4000000e};
  book->defs[0x000009ca]->node[14] = (Node) {0x5000000d,0x4000000f};
  book->defs[0x000009ca]->node[15] = (Node) {0x5000000e,0x40000010};
  book->defs[0x000009ca]->node[16] = (Node) {0x5000000f,0x40000011};
  book->defs[0x000009ca]->node[17] = (Node) {0x50000010,0x50000012};
  book->defs[0x000009ca]->node[18] = (Node) {0x40000009,0x50000011};
  // id
  book->defs[0x00000b68]           = (Term*) malloc(sizeof(Term));
  book->defs[0x00000b68]->root     = 0xa0000000;
  book->defs[0x00000b68]->alen     = 0;
  book->defs[0x00000b68]->acts     = (Wire*) malloc(0 * sizeof(Wire));
  book->defs[0x00000b68]->nlen     = 1;
  book->defs[0x00000b68]->node     = (Node*) malloc(1 * sizeof(Node));
  book->defs[0x00000b68]->node[ 0] = (Node) {0x50000000,0x40000000};
  // k0
  book->defs[0x00000bc1]           = (Term*) malloc(sizeof(Term));
  book->defs[0x00000bc1]->root     = 0xa0000000;
  book->defs[0x00000bc1]->alen     = 0;
  book->defs[0x00000bc1]->acts     = (Wire*) malloc(0 * sizeof(Wire));
  book->defs[0x00000bc1]->nlen     = 2;
  book->defs[0x00000bc1]->node     = (Node*) malloc(2 * sizeof(Node));
  book->defs[0x00000bc1]->node[ 0] = (Node) {0x20000000,0xa0000001};
  book->defs[0x00000bc1]->node[ 1] = (Node) {0x50000001,0x40000001};
  // k1
  book->defs[0x00000bc2]           = (Term*) malloc(sizeof(Term));
  book->defs[0x00000bc2]->root     = 0xa0000000;
  book->defs[0x00000bc2]->alen     = 0;
  book->defs[0x00000bc2]->acts     = (Wire*) malloc(0 * sizeof(Wire));
  book->defs[0x00000bc2]->nlen     = 3;
  book->defs[0x00000bc2]->node     = (Node*) malloc(3 * sizeof(Node));
  book->defs[0x00000bc2]->node[ 0] = (Node) {0xa0000001,0xa0000002};
  book->defs[0x00000bc2]->node[ 1] = (Node) {0x40000002,0x50000002};
  book->defs[0x00000bc2]->node[ 2] = (Node) {0x40000001,0x50000001};
  // k2
  book->defs[0x00000bc3]           = (Term*) malloc(sizeof(Term));
  book->defs[0x00000bc3]->root     = 0xa0000000;
  book->defs[0x00000bc3]->alen     = 0;
  book->defs[0x00000bc3]->acts     = (Wire*) malloc(0 * sizeof(Wire));
  book->defs[0x00000bc3]->nlen     = 5;
  book->defs[0x00000bc3]->node     = (Node*) malloc(5 * sizeof(Node));
  book->defs[0x00000bc3]->node[ 0] = (Node) {0xc0000001,0xa0000004};
  book->defs[0x00000bc3]->node[ 1] = (Node) {0xa0000002,0xa0000003};
  book->defs[0x00000bc3]->node[ 2] = (Node) {0x40000004,0x40000003};
  book->defs[0x00000bc3]->node[ 3] = (Node) {0x50000002,0x50000004};
  book->defs[0x00000bc3]->node[ 4] = (Node) {0x40000002,0x50000003};
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
  book->defs[0x00000bc4]->node[ 3] = (Node) {0x40000006,0x40000004};
  book->defs[0x00000bc4]->node[ 4] = (Node) {0x50000003,0x40000005};
  book->defs[0x00000bc4]->node[ 5] = (Node) {0x50000004,0x50000006};
  book->defs[0x00000bc4]->node[ 6] = (Node) {0x40000003,0x50000005};
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
  book->defs[0x00000bc5]->node[ 4] = (Node) {0x40000008,0x40000005};
  book->defs[0x00000bc5]->node[ 5] = (Node) {0x50000004,0x40000006};
  book->defs[0x00000bc5]->node[ 6] = (Node) {0x50000005,0x40000007};
  book->defs[0x00000bc5]->node[ 7] = (Node) {0x50000006,0x50000008};
  book->defs[0x00000bc5]->node[ 8] = (Node) {0x40000004,0x50000007};
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
  book->defs[0x00000bc6]->node[ 5] = (Node) {0x4000000a,0x40000006};
  book->defs[0x00000bc6]->node[ 6] = (Node) {0x50000005,0x40000007};
  book->defs[0x00000bc6]->node[ 7] = (Node) {0x50000006,0x40000008};
  book->defs[0x00000bc6]->node[ 8] = (Node) {0x50000007,0x40000009};
  book->defs[0x00000bc6]->node[ 9] = (Node) {0x50000008,0x5000000a};
  book->defs[0x00000bc6]->node[10] = (Node) {0x40000005,0x50000009};
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
  book->defs[0x00000bc7]->node[ 6] = (Node) {0x4000000c,0x40000007};
  book->defs[0x00000bc7]->node[ 7] = (Node) {0x50000006,0x40000008};
  book->defs[0x00000bc7]->node[ 8] = (Node) {0x50000007,0x40000009};
  book->defs[0x00000bc7]->node[ 9] = (Node) {0x50000008,0x4000000a};
  book->defs[0x00000bc7]->node[10] = (Node) {0x50000009,0x4000000b};
  book->defs[0x00000bc7]->node[11] = (Node) {0x5000000a,0x5000000c};
  book->defs[0x00000bc7]->node[12] = (Node) {0x40000006,0x5000000b};
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
  book->defs[0x00000bc8]->node[ 7] = (Node) {0x4000000e,0x40000008};
  book->defs[0x00000bc8]->node[ 8] = (Node) {0x50000007,0x40000009};
  book->defs[0x00000bc8]->node[ 9] = (Node) {0x50000008,0x4000000a};
  book->defs[0x00000bc8]->node[10] = (Node) {0x50000009,0x4000000b};
  book->defs[0x00000bc8]->node[11] = (Node) {0x5000000a,0x4000000c};
  book->defs[0x00000bc8]->node[12] = (Node) {0x5000000b,0x4000000d};
  book->defs[0x00000bc8]->node[13] = (Node) {0x5000000c,0x5000000e};
  book->defs[0x00000bc8]->node[14] = (Node) {0x40000007,0x5000000d};
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
  book->defs[0x00000bc9]->node[ 8] = (Node) {0x40000010,0x40000009};
  book->defs[0x00000bc9]->node[ 9] = (Node) {0x50000008,0x4000000a};
  book->defs[0x00000bc9]->node[10] = (Node) {0x50000009,0x4000000b};
  book->defs[0x00000bc9]->node[11] = (Node) {0x5000000a,0x4000000c};
  book->defs[0x00000bc9]->node[12] = (Node) {0x5000000b,0x4000000d};
  book->defs[0x00000bc9]->node[13] = (Node) {0x5000000c,0x4000000e};
  book->defs[0x00000bc9]->node[14] = (Node) {0x5000000d,0x4000000f};
  book->defs[0x00000bc9]->node[15] = (Node) {0x5000000e,0x50000010};
  book->defs[0x00000bc9]->node[16] = (Node) {0x40000008,0x5000000f};
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
  book->defs[0x00000bca]->node[ 9] = (Node) {0x40000012,0x4000000a};
  book->defs[0x00000bca]->node[10] = (Node) {0x50000009,0x4000000b};
  book->defs[0x00000bca]->node[11] = (Node) {0x5000000a,0x4000000c};
  book->defs[0x00000bca]->node[12] = (Node) {0x5000000b,0x4000000d};
  book->defs[0x00000bca]->node[13] = (Node) {0x5000000c,0x4000000e};
  book->defs[0x00000bca]->node[14] = (Node) {0x5000000d,0x4000000f};
  book->defs[0x00000bca]->node[15] = (Node) {0x5000000e,0x40000010};
  book->defs[0x00000bca]->node[16] = (Node) {0x5000000f,0x40000011};
  book->defs[0x00000bca]->node[17] = (Node) {0x50000010,0x50000012};
  book->defs[0x00000bca]->node[18] = (Node) {0x40000009,0x50000011};
  // afS
  book->defs[0x00025a9d]           = (Term*) malloc(sizeof(Term));
  book->defs[0x00025a9d]->root     = 0xa0000000;
  book->defs[0x00025a9d]->alen     = 3;
  book->defs[0x00025a9d]->acts     = (Wire*) malloc(3 * sizeof(Wire));
  book->defs[0x00025a9d]->acts[ 0] = mkwire(0xa0000002,0x1000096a);
  book->defs[0x00025a9d]->acts[ 1] = mkwire(0xa0000003,0x10025ca8);
  book->defs[0x00025a9d]->acts[ 2] = mkwire(0xa0000005,0x1000096a);
  book->defs[0x00025a9d]->nlen     = 6;
  book->defs[0x00025a9d]->node     = (Node*) malloc(6 * sizeof(Node));
  book->defs[0x00025a9d]->node[ 0] = (Node) {0xb0000001,0x50000004};
  book->defs[0x00025a9d]->node[ 1] = (Node) {0x40000005,0x40000002};
  book->defs[0x00025a9d]->node[ 2] = (Node) {0x50000001,0x40000004};
  book->defs[0x00025a9d]->node[ 3] = (Node) {0x50000005,0xa0000004};
  book->defs[0x00025a9d]->node[ 4] = (Node) {0x50000002,0x50000000};
  book->defs[0x00025a9d]->node[ 5] = (Node) {0x40000001,0x40000003};
  // afZ
  book->defs[0x00025aa4]           = (Term*) malloc(sizeof(Term));
  book->defs[0x00025aa4]->root     = 0x1000001e;
  book->defs[0x00025aa4]->alen     = 0;
  book->defs[0x00025aa4]->acts     = (Wire*) malloc(0 * sizeof(Wire));
  book->defs[0x00025aa4]->nlen     = 0;
  book->defs[0x00025aa4]->node     = (Node*) malloc(0 * sizeof(Node));
  // and
  book->defs[0x00025ca8]           = (Term*) malloc(sizeof(Term));
  book->defs[0x00025ca8]->root     = 0xa0000000;
  book->defs[0x00025ca8]->alen     = 0;
  book->defs[0x00025ca8]->acts     = (Wire*) malloc(0 * sizeof(Wire));
  book->defs[0x00025ca8]->nlen     = 9;
  book->defs[0x00025ca8]->node     = (Node*) malloc(9 * sizeof(Node));
  book->defs[0x00025ca8]->node[ 0] = (Node) {0xa0000001,0x50000005};
  book->defs[0x00025ca8]->node[ 1] = (Node) {0xa0000002,0xa0000005};
  book->defs[0x00025ca8]->node[ 2] = (Node) {0xa0000003,0x50000004};
  book->defs[0x00025ca8]->node[ 3] = (Node) {0x1000001e,0xa0000004};
  book->defs[0x00025ca8]->node[ 4] = (Node) {0x10000010,0x50000002};
  book->defs[0x00025ca8]->node[ 5] = (Node) {0xa0000006,0x50000000};
  book->defs[0x00025ca8]->node[ 6] = (Node) {0xa0000007,0x50000008};
  book->defs[0x00025ca8]->node[ 7] = (Node) {0x10000010,0xa0000008};
  book->defs[0x00025ca8]->node[ 8] = (Node) {0x10000010,0x50000006};
  // brn
  book->defs[0x00026db2]           = (Term*) malloc(sizeof(Term));
  book->defs[0x00026db2]->root     = 0xa0000000;
  book->defs[0x00026db2]->alen     = 0;
  book->defs[0x00026db2]->acts     = (Wire*) malloc(0 * sizeof(Wire));
  book->defs[0x00026db2]->nlen     = 3;
  book->defs[0x00026db2]->node     = (Node*) malloc(3 * sizeof(Node));
  book->defs[0x00026db2]->node[ 0] = (Node) {0xa0000001,0x50000002};
  book->defs[0x00026db2]->node[ 1] = (Node) {0x109b6c9d,0xa0000002};
  book->defs[0x00026db2]->node[ 2] = (Node) {0x109b6ca4,0x50000000};
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
  book->defs[0x00027081]->node[10] = (Node) {0x40000014,0x4000000b};
  book->defs[0x00027081]->node[11] = (Node) {0x5000000a,0x4000000c};
  book->defs[0x00027081]->node[12] = (Node) {0x5000000b,0x4000000d};
  book->defs[0x00027081]->node[13] = (Node) {0x5000000c,0x4000000e};
  book->defs[0x00027081]->node[14] = (Node) {0x5000000d,0x4000000f};
  book->defs[0x00027081]->node[15] = (Node) {0x5000000e,0x40000010};
  book->defs[0x00027081]->node[16] = (Node) {0x5000000f,0x40000011};
  book->defs[0x00027081]->node[17] = (Node) {0x50000010,0x40000012};
  book->defs[0x00027081]->node[18] = (Node) {0x50000011,0x40000013};
  book->defs[0x00027081]->node[19] = (Node) {0x50000012,0x50000014};
  book->defs[0x00027081]->node[20] = (Node) {0x4000000a,0x50000013};
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
  book->defs[0x00027082]->node[11] = (Node) {0x40000016,0x4000000c};
  book->defs[0x00027082]->node[12] = (Node) {0x5000000b,0x4000000d};
  book->defs[0x00027082]->node[13] = (Node) {0x5000000c,0x4000000e};
  book->defs[0x00027082]->node[14] = (Node) {0x5000000d,0x4000000f};
  book->defs[0x00027082]->node[15] = (Node) {0x5000000e,0x40000010};
  book->defs[0x00027082]->node[16] = (Node) {0x5000000f,0x40000011};
  book->defs[0x00027082]->node[17] = (Node) {0x50000010,0x40000012};
  book->defs[0x00027082]->node[18] = (Node) {0x50000011,0x40000013};
  book->defs[0x00027082]->node[19] = (Node) {0x50000012,0x40000014};
  book->defs[0x00027082]->node[20] = (Node) {0x50000013,0x40000015};
  book->defs[0x00027082]->node[21] = (Node) {0x50000014,0x50000016};
  book->defs[0x00027082]->node[22] = (Node) {0x4000000b,0x50000015};
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
  book->defs[0x00027083]->node[12] = (Node) {0x40000018,0x4000000d};
  book->defs[0x00027083]->node[13] = (Node) {0x5000000c,0x4000000e};
  book->defs[0x00027083]->node[14] = (Node) {0x5000000d,0x4000000f};
  book->defs[0x00027083]->node[15] = (Node) {0x5000000e,0x40000010};
  book->defs[0x00027083]->node[16] = (Node) {0x5000000f,0x40000011};
  book->defs[0x00027083]->node[17] = (Node) {0x50000010,0x40000012};
  book->defs[0x00027083]->node[18] = (Node) {0x50000011,0x40000013};
  book->defs[0x00027083]->node[19] = (Node) {0x50000012,0x40000014};
  book->defs[0x00027083]->node[20] = (Node) {0x50000013,0x40000015};
  book->defs[0x00027083]->node[21] = (Node) {0x50000014,0x40000016};
  book->defs[0x00027083]->node[22] = (Node) {0x50000015,0x40000017};
  book->defs[0x00027083]->node[23] = (Node) {0x50000016,0x50000018};
  book->defs[0x00027083]->node[24] = (Node) {0x4000000c,0x50000017};
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
  book->defs[0x00027084]->node[13] = (Node) {0x4000001a,0x4000000e};
  book->defs[0x00027084]->node[14] = (Node) {0x5000000d,0x4000000f};
  book->defs[0x00027084]->node[15] = (Node) {0x5000000e,0x40000010};
  book->defs[0x00027084]->node[16] = (Node) {0x5000000f,0x40000011};
  book->defs[0x00027084]->node[17] = (Node) {0x50000010,0x40000012};
  book->defs[0x00027084]->node[18] = (Node) {0x50000011,0x40000013};
  book->defs[0x00027084]->node[19] = (Node) {0x50000012,0x40000014};
  book->defs[0x00027084]->node[20] = (Node) {0x50000013,0x40000015};
  book->defs[0x00027084]->node[21] = (Node) {0x50000014,0x40000016};
  book->defs[0x00027084]->node[22] = (Node) {0x50000015,0x40000017};
  book->defs[0x00027084]->node[23] = (Node) {0x50000016,0x40000018};
  book->defs[0x00027084]->node[24] = (Node) {0x50000017,0x40000019};
  book->defs[0x00027084]->node[25] = (Node) {0x50000018,0x5000001a};
  book->defs[0x00027084]->node[26] = (Node) {0x4000000d,0x50000019};
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
  book->defs[0x00027085]->node[14] = (Node) {0x4000001c,0x4000000f};
  book->defs[0x00027085]->node[15] = (Node) {0x5000000e,0x40000010};
  book->defs[0x00027085]->node[16] = (Node) {0x5000000f,0x40000011};
  book->defs[0x00027085]->node[17] = (Node) {0x50000010,0x40000012};
  book->defs[0x00027085]->node[18] = (Node) {0x50000011,0x40000013};
  book->defs[0x00027085]->node[19] = (Node) {0x50000012,0x40000014};
  book->defs[0x00027085]->node[20] = (Node) {0x50000013,0x40000015};
  book->defs[0x00027085]->node[21] = (Node) {0x50000014,0x40000016};
  book->defs[0x00027085]->node[22] = (Node) {0x50000015,0x40000017};
  book->defs[0x00027085]->node[23] = (Node) {0x50000016,0x40000018};
  book->defs[0x00027085]->node[24] = (Node) {0x50000017,0x40000019};
  book->defs[0x00027085]->node[25] = (Node) {0x50000018,0x4000001a};
  book->defs[0x00027085]->node[26] = (Node) {0x50000019,0x4000001b};
  book->defs[0x00027085]->node[27] = (Node) {0x5000001a,0x5000001c};
  book->defs[0x00027085]->node[28] = (Node) {0x4000000e,0x5000001b};
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
  book->defs[0x00027086]->node[15] = (Node) {0x4000001e,0x40000010};
  book->defs[0x00027086]->node[16] = (Node) {0x5000000f,0x40000011};
  book->defs[0x00027086]->node[17] = (Node) {0x50000010,0x40000012};
  book->defs[0x00027086]->node[18] = (Node) {0x50000011,0x40000013};
  book->defs[0x00027086]->node[19] = (Node) {0x50000012,0x40000014};
  book->defs[0x00027086]->node[20] = (Node) {0x50000013,0x40000015};
  book->defs[0x00027086]->node[21] = (Node) {0x50000014,0x40000016};
  book->defs[0x00027086]->node[22] = (Node) {0x50000015,0x40000017};
  book->defs[0x00027086]->node[23] = (Node) {0x50000016,0x40000018};
  book->defs[0x00027086]->node[24] = (Node) {0x50000017,0x40000019};
  book->defs[0x00027086]->node[25] = (Node) {0x50000018,0x4000001a};
  book->defs[0x00027086]->node[26] = (Node) {0x50000019,0x4000001b};
  book->defs[0x00027086]->node[27] = (Node) {0x5000001a,0x4000001c};
  book->defs[0x00027086]->node[28] = (Node) {0x5000001b,0x4000001d};
  book->defs[0x00027086]->node[29] = (Node) {0x5000001c,0x5000001e};
  book->defs[0x00027086]->node[30] = (Node) {0x4000000f,0x5000001d};
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
  book->defs[0x00027087]->node[16] = (Node) {0x00000000,0x40000011};
  book->defs[0x00027087]->node[17] = (Node) {0x50000010,0x40000012};
  book->defs[0x00027087]->node[18] = (Node) {0x50000011,0x40000013};
  book->defs[0x00027087]->node[19] = (Node) {0x50000012,0x40000014};
  book->defs[0x00027087]->node[20] = (Node) {0x50000013,0x40000015};
  book->defs[0x00027087]->node[21] = (Node) {0x50000014,0x40000016};
  book->defs[0x00027087]->node[22] = (Node) {0x50000015,0x40000017};
  book->defs[0x00027087]->node[23] = (Node) {0x50000016,0x40000018};
  book->defs[0x00027087]->node[24] = (Node) {0x50000017,0x40000019};
  book->defs[0x00027087]->node[25] = (Node) {0x50000018,0x4000001a};
  book->defs[0x00027087]->node[26] = (Node) {0x50000019,0x4000001b};
  book->defs[0x00027087]->node[27] = (Node) {0x5000001a,0x4000001c};
  book->defs[0x00027087]->node[28] = (Node) {0x5000001b,0x4000001d};
  book->defs[0x00027087]->node[29] = (Node) {0x5000001c,0x4000001e};
  book->defs[0x00027087]->node[30] = (Node) {0x5000001d,0x4000001f};
  book->defs[0x00027087]->node[31] = (Node) {0x5000001e,0x50000020};
  book->defs[0x00027087]->node[32] = (Node) {0x00000000,0x5000001f};
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
  book->defs[0x00027088]->node[17] = (Node) {0x40000022,0x40000012};
  book->defs[0x00027088]->node[18] = (Node) {0x50000011,0x40000013};
  book->defs[0x00027088]->node[19] = (Node) {0x50000012,0x40000014};
  book->defs[0x00027088]->node[20] = (Node) {0x50000013,0x40000015};
  book->defs[0x00027088]->node[21] = (Node) {0x50000014,0x40000016};
  book->defs[0x00027088]->node[22] = (Node) {0x50000015,0x40000017};
  book->defs[0x00027088]->node[23] = (Node) {0x50000016,0x40000018};
  book->defs[0x00027088]->node[24] = (Node) {0x50000017,0x40000019};
  book->defs[0x00027088]->node[25] = (Node) {0x50000018,0x4000001a};
  book->defs[0x00027088]->node[26] = (Node) {0x50000019,0x4000001b};
  book->defs[0x00027088]->node[27] = (Node) {0x5000001a,0x4000001c};
  book->defs[0x00027088]->node[28] = (Node) {0x5000001b,0x4000001d};
  book->defs[0x00027088]->node[29] = (Node) {0x5000001c,0x4000001e};
  book->defs[0x00027088]->node[30] = (Node) {0x5000001d,0x4000001f};
  book->defs[0x00027088]->node[31] = (Node) {0x5000001e,0x40000020};
  book->defs[0x00027088]->node[32] = (Node) {0x5000001f,0x40000021};
  book->defs[0x00027088]->node[33] = (Node) {0x50000020,0x50000022};
  book->defs[0x00027088]->node[34] = (Node) {0x40000011,0x50000021};
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
  book->defs[0x00027089]->node[18] = (Node) {0x40000024,0x40000013};
  book->defs[0x00027089]->node[19] = (Node) {0x50000012,0x40000014};
  book->defs[0x00027089]->node[20] = (Node) {0x50000013,0x40000015};
  book->defs[0x00027089]->node[21] = (Node) {0x50000014,0x40000016};
  book->defs[0x00027089]->node[22] = (Node) {0x50000015,0x40000017};
  book->defs[0x00027089]->node[23] = (Node) {0x50000016,0x40000018};
  book->defs[0x00027089]->node[24] = (Node) {0x50000017,0x40000019};
  book->defs[0x00027089]->node[25] = (Node) {0x50000018,0x4000001a};
  book->defs[0x00027089]->node[26] = (Node) {0x50000019,0x4000001b};
  book->defs[0x00027089]->node[27] = (Node) {0x5000001a,0x4000001c};
  book->defs[0x00027089]->node[28] = (Node) {0x5000001b,0x4000001d};
  book->defs[0x00027089]->node[29] = (Node) {0x5000001c,0x4000001e};
  book->defs[0x00027089]->node[30] = (Node) {0x5000001d,0x4000001f};
  book->defs[0x00027089]->node[31] = (Node) {0x5000001e,0x40000020};
  book->defs[0x00027089]->node[32] = (Node) {0x5000001f,0x40000021};
  book->defs[0x00027089]->node[33] = (Node) {0x50000020,0x40000022};
  book->defs[0x00027089]->node[34] = (Node) {0x50000021,0x40000023};
  book->defs[0x00027089]->node[35] = (Node) {0x50000022,0x50000024};
  book->defs[0x00027089]->node[36] = (Node) {0x40000012,0x50000023};
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
  book->defs[0x0002708a]->node[19] = (Node) {0x40000026,0x40000014};
  book->defs[0x0002708a]->node[20] = (Node) {0x50000013,0x40000015};
  book->defs[0x0002708a]->node[21] = (Node) {0x50000014,0x40000016};
  book->defs[0x0002708a]->node[22] = (Node) {0x50000015,0x40000017};
  book->defs[0x0002708a]->node[23] = (Node) {0x50000016,0x40000018};
  book->defs[0x0002708a]->node[24] = (Node) {0x50000017,0x40000019};
  book->defs[0x0002708a]->node[25] = (Node) {0x50000018,0x4000001a};
  book->defs[0x0002708a]->node[26] = (Node) {0x50000019,0x4000001b};
  book->defs[0x0002708a]->node[27] = (Node) {0x5000001a,0x4000001c};
  book->defs[0x0002708a]->node[28] = (Node) {0x5000001b,0x4000001d};
  book->defs[0x0002708a]->node[29] = (Node) {0x5000001c,0x4000001e};
  book->defs[0x0002708a]->node[30] = (Node) {0x5000001d,0x4000001f};
  book->defs[0x0002708a]->node[31] = (Node) {0x5000001e,0x40000020};
  book->defs[0x0002708a]->node[32] = (Node) {0x5000001f,0x40000021};
  book->defs[0x0002708a]->node[33] = (Node) {0x50000020,0x40000022};
  book->defs[0x0002708a]->node[34] = (Node) {0x50000021,0x40000023};
  book->defs[0x0002708a]->node[35] = (Node) {0x50000022,0x40000024};
  book->defs[0x0002708a]->node[36] = (Node) {0x50000023,0x40000025};
  book->defs[0x0002708a]->node[37] = (Node) {0x50000024,0x50000026};
  book->defs[0x0002708a]->node[38] = (Node) {0x40000013,0x50000025};
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
  book->defs[0x000270c1]->node[20] = (Node) {0x40000028,0x40000015};
  book->defs[0x000270c1]->node[21] = (Node) {0x50000014,0x40000016};
  book->defs[0x000270c1]->node[22] = (Node) {0x50000015,0x40000017};
  book->defs[0x000270c1]->node[23] = (Node) {0x50000016,0x40000018};
  book->defs[0x000270c1]->node[24] = (Node) {0x50000017,0x40000019};
  book->defs[0x000270c1]->node[25] = (Node) {0x50000018,0x4000001a};
  book->defs[0x000270c1]->node[26] = (Node) {0x50000019,0x4000001b};
  book->defs[0x000270c1]->node[27] = (Node) {0x5000001a,0x4000001c};
  book->defs[0x000270c1]->node[28] = (Node) {0x5000001b,0x4000001d};
  book->defs[0x000270c1]->node[29] = (Node) {0x5000001c,0x4000001e};
  book->defs[0x000270c1]->node[30] = (Node) {0x5000001d,0x4000001f};
  book->defs[0x000270c1]->node[31] = (Node) {0x5000001e,0x40000020};
  book->defs[0x000270c1]->node[32] = (Node) {0x5000001f,0x40000021};
  book->defs[0x000270c1]->node[33] = (Node) {0x50000020,0x40000022};
  book->defs[0x000270c1]->node[34] = (Node) {0x50000021,0x40000023};
  book->defs[0x000270c1]->node[35] = (Node) {0x50000022,0x40000024};
  book->defs[0x000270c1]->node[36] = (Node) {0x50000023,0x40000025};
  book->defs[0x000270c1]->node[37] = (Node) {0x50000024,0x40000026};
  book->defs[0x000270c1]->node[38] = (Node) {0x50000025,0x40000027};
  book->defs[0x000270c1]->node[39] = (Node) {0x50000026,0x50000028};
  book->defs[0x000270c1]->node[40] = (Node) {0x40000014,0x50000027};
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
  book->defs[0x000270c2]->node[21] = (Node) {0x4000002a,0x40000016};
  book->defs[0x000270c2]->node[22] = (Node) {0x50000015,0x40000017};
  book->defs[0x000270c2]->node[23] = (Node) {0x50000016,0x40000018};
  book->defs[0x000270c2]->node[24] = (Node) {0x50000017,0x40000019};
  book->defs[0x000270c2]->node[25] = (Node) {0x50000018,0x4000001a};
  book->defs[0x000270c2]->node[26] = (Node) {0x50000019,0x4000001b};
  book->defs[0x000270c2]->node[27] = (Node) {0x5000001a,0x4000001c};
  book->defs[0x000270c2]->node[28] = (Node) {0x5000001b,0x4000001d};
  book->defs[0x000270c2]->node[29] = (Node) {0x5000001c,0x4000001e};
  book->defs[0x000270c2]->node[30] = (Node) {0x5000001d,0x4000001f};
  book->defs[0x000270c2]->node[31] = (Node) {0x5000001e,0x40000020};
  book->defs[0x000270c2]->node[32] = (Node) {0x5000001f,0x40000021};
  book->defs[0x000270c2]->node[33] = (Node) {0x50000020,0x40000022};
  book->defs[0x000270c2]->node[34] = (Node) {0x50000021,0x40000023};
  book->defs[0x000270c2]->node[35] = (Node) {0x50000022,0x40000024};
  book->defs[0x000270c2]->node[36] = (Node) {0x50000023,0x40000025};
  book->defs[0x000270c2]->node[37] = (Node) {0x50000024,0x40000026};
  book->defs[0x000270c2]->node[38] = (Node) {0x50000025,0x40000027};
  book->defs[0x000270c2]->node[39] = (Node) {0x50000026,0x40000028};
  book->defs[0x000270c2]->node[40] = (Node) {0x50000027,0x40000029};
  book->defs[0x000270c2]->node[41] = (Node) {0x50000028,0x5000002a};
  book->defs[0x000270c2]->node[42] = (Node) {0x40000015,0x50000029};
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
  book->defs[0x000270c3]->node[22] = (Node) {0x4000002c,0x40000017};
  book->defs[0x000270c3]->node[23] = (Node) {0x50000016,0x40000018};
  book->defs[0x000270c3]->node[24] = (Node) {0x50000017,0x40000019};
  book->defs[0x000270c3]->node[25] = (Node) {0x50000018,0x4000001a};
  book->defs[0x000270c3]->node[26] = (Node) {0x50000019,0x4000001b};
  book->defs[0x000270c3]->node[27] = (Node) {0x5000001a,0x4000001c};
  book->defs[0x000270c3]->node[28] = (Node) {0x5000001b,0x4000001d};
  book->defs[0x000270c3]->node[29] = (Node) {0x5000001c,0x4000001e};
  book->defs[0x000270c3]->node[30] = (Node) {0x5000001d,0x4000001f};
  book->defs[0x000270c3]->node[31] = (Node) {0x5000001e,0x40000020};
  book->defs[0x000270c3]->node[32] = (Node) {0x5000001f,0x40000021};
  book->defs[0x000270c3]->node[33] = (Node) {0x50000020,0x40000022};
  book->defs[0x000270c3]->node[34] = (Node) {0x50000021,0x40000023};
  book->defs[0x000270c3]->node[35] = (Node) {0x50000022,0x40000024};
  book->defs[0x000270c3]->node[36] = (Node) {0x50000023,0x40000025};
  book->defs[0x000270c3]->node[37] = (Node) {0x50000024,0x40000026};
  book->defs[0x000270c3]->node[38] = (Node) {0x50000025,0x40000027};
  book->defs[0x000270c3]->node[39] = (Node) {0x50000026,0x40000028};
  book->defs[0x000270c3]->node[40] = (Node) {0x50000027,0x40000029};
  book->defs[0x000270c3]->node[41] = (Node) {0x50000028,0x4000002a};
  book->defs[0x000270c3]->node[42] = (Node) {0x50000029,0x4000002b};
  book->defs[0x000270c3]->node[43] = (Node) {0x5000002a,0x5000002c};
  book->defs[0x000270c3]->node[44] = (Node) {0x40000016,0x5000002b};
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
  book->defs[0x000270c4]->node[23] = (Node) {0x4000002e,0x40000018};
  book->defs[0x000270c4]->node[24] = (Node) {0x50000017,0x40000019};
  book->defs[0x000270c4]->node[25] = (Node) {0x50000018,0x4000001a};
  book->defs[0x000270c4]->node[26] = (Node) {0x50000019,0x4000001b};
  book->defs[0x000270c4]->node[27] = (Node) {0x5000001a,0x4000001c};
  book->defs[0x000270c4]->node[28] = (Node) {0x5000001b,0x4000001d};
  book->defs[0x000270c4]->node[29] = (Node) {0x5000001c,0x4000001e};
  book->defs[0x000270c4]->node[30] = (Node) {0x5000001d,0x4000001f};
  book->defs[0x000270c4]->node[31] = (Node) {0x5000001e,0x40000020};
  book->defs[0x000270c4]->node[32] = (Node) {0x5000001f,0x40000021};
  book->defs[0x000270c4]->node[33] = (Node) {0x50000020,0x40000022};
  book->defs[0x000270c4]->node[34] = (Node) {0x50000021,0x40000023};
  book->defs[0x000270c4]->node[35] = (Node) {0x50000022,0x40000024};
  book->defs[0x000270c4]->node[36] = (Node) {0x50000023,0x40000025};
  book->defs[0x000270c4]->node[37] = (Node) {0x50000024,0x40000026};
  book->defs[0x000270c4]->node[38] = (Node) {0x50000025,0x40000027};
  book->defs[0x000270c4]->node[39] = (Node) {0x50000026,0x40000028};
  book->defs[0x000270c4]->node[40] = (Node) {0x50000027,0x40000029};
  book->defs[0x000270c4]->node[41] = (Node) {0x50000028,0x4000002a};
  book->defs[0x000270c4]->node[42] = (Node) {0x50000029,0x4000002b};
  book->defs[0x000270c4]->node[43] = (Node) {0x5000002a,0x4000002c};
  book->defs[0x000270c4]->node[44] = (Node) {0x5000002b,0x4000002d};
  book->defs[0x000270c4]->node[45] = (Node) {0x5000002c,0x5000002e};
  book->defs[0x000270c4]->node[46] = (Node) {0x40000017,0x5000002d};
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
  book->defs[0x000270c5]->node[24] = (Node) {0x40000030,0x40000019};
  book->defs[0x000270c5]->node[25] = (Node) {0x50000018,0x4000001a};
  book->defs[0x000270c5]->node[26] = (Node) {0x50000019,0x4000001b};
  book->defs[0x000270c5]->node[27] = (Node) {0x5000001a,0x4000001c};
  book->defs[0x000270c5]->node[28] = (Node) {0x5000001b,0x4000001d};
  book->defs[0x000270c5]->node[29] = (Node) {0x5000001c,0x4000001e};
  book->defs[0x000270c5]->node[30] = (Node) {0x5000001d,0x4000001f};
  book->defs[0x000270c5]->node[31] = (Node) {0x5000001e,0x40000020};
  book->defs[0x000270c5]->node[32] = (Node) {0x5000001f,0x40000021};
  book->defs[0x000270c5]->node[33] = (Node) {0x50000020,0x40000022};
  book->defs[0x000270c5]->node[34] = (Node) {0x50000021,0x40000023};
  book->defs[0x000270c5]->node[35] = (Node) {0x50000022,0x40000024};
  book->defs[0x000270c5]->node[36] = (Node) {0x50000023,0x40000025};
  book->defs[0x000270c5]->node[37] = (Node) {0x50000024,0x40000026};
  book->defs[0x000270c5]->node[38] = (Node) {0x50000025,0x40000027};
  book->defs[0x000270c5]->node[39] = (Node) {0x50000026,0x40000028};
  book->defs[0x000270c5]->node[40] = (Node) {0x50000027,0x40000029};
  book->defs[0x000270c5]->node[41] = (Node) {0x50000028,0x4000002a};
  book->defs[0x000270c5]->node[42] = (Node) {0x50000029,0x4000002b};
  book->defs[0x000270c5]->node[43] = (Node) {0x5000002a,0x4000002c};
  book->defs[0x000270c5]->node[44] = (Node) {0x5000002b,0x4000002d};
  book->defs[0x000270c5]->node[45] = (Node) {0x5000002c,0x4000002e};
  book->defs[0x000270c5]->node[46] = (Node) {0x5000002d,0x4000002f};
  book->defs[0x000270c5]->node[47] = (Node) {0x5000002e,0x50000030};
  book->defs[0x000270c5]->node[48] = (Node) {0x40000018,0x5000002f};
  // c25
  book->defs[0x000270c6]           = (Term*) malloc(sizeof(Term));
  book->defs[0x000270c6]->root     = 0xa0000000;
  book->defs[0x000270c6]->alen     = 0;
  book->defs[0x000270c6]->acts     = (Wire*) malloc(0 * sizeof(Wire));
  book->defs[0x000270c6]->nlen     = 51;
  book->defs[0x000270c6]->node     = (Node*) malloc(51 * sizeof(Node));
  book->defs[0x000270c6]->node[ 0] = (Node) {0xb0000001,0xa0000032};
  book->defs[0x000270c6]->node[ 1] = (Node) {0xb0000002,0xa0000031};
  book->defs[0x000270c6]->node[ 2] = (Node) {0xb0000003,0xa0000030};
  book->defs[0x000270c6]->node[ 3] = (Node) {0xb0000004,0xa000002f};
  book->defs[0x000270c6]->node[ 4] = (Node) {0xb0000005,0xa000002e};
  book->defs[0x000270c6]->node[ 5] = (Node) {0xb0000006,0xa000002d};
  book->defs[0x000270c6]->node[ 6] = (Node) {0xb0000007,0xa000002c};
  book->defs[0x000270c6]->node[ 7] = (Node) {0xb0000008,0xa000002b};
  book->defs[0x000270c6]->node[ 8] = (Node) {0xb0000009,0xa000002a};
  book->defs[0x000270c6]->node[ 9] = (Node) {0xb000000a,0xa0000029};
  book->defs[0x000270c6]->node[10] = (Node) {0xb000000b,0xa0000028};
  book->defs[0x000270c6]->node[11] = (Node) {0xb000000c,0xa0000027};
  book->defs[0x000270c6]->node[12] = (Node) {0xb000000d,0xa0000026};
  book->defs[0x000270c6]->node[13] = (Node) {0xb000000e,0xa0000025};
  book->defs[0x000270c6]->node[14] = (Node) {0xb000000f,0xa0000024};
  book->defs[0x000270c6]->node[15] = (Node) {0xb0000010,0xa0000023};
  book->defs[0x000270c6]->node[16] = (Node) {0xb0000011,0xa0000022};
  book->defs[0x000270c6]->node[17] = (Node) {0xb0000012,0xa0000021};
  book->defs[0x000270c6]->node[18] = (Node) {0xb0000013,0xa0000020};
  book->defs[0x000270c6]->node[19] = (Node) {0xb0000014,0xa000001f};
  book->defs[0x000270c6]->node[20] = (Node) {0xb0000015,0xa000001e};
  book->defs[0x000270c6]->node[21] = (Node) {0xb0000016,0xa000001d};
  book->defs[0x000270c6]->node[22] = (Node) {0xb0000017,0xa000001c};
  book->defs[0x000270c6]->node[23] = (Node) {0xb0000018,0xa000001b};
  book->defs[0x000270c6]->node[24] = (Node) {0xa0000019,0xa000001a};
  book->defs[0x000270c6]->node[25] = (Node) {0x00000000,0x40000032};
  book->defs[0x000270c6]->node[26] = (Node) {0x50000019,0x4000001b};
  book->defs[0x000270c6]->node[27] = (Node) {0x5000001a,0x4000001c};
  book->defs[0x000270c6]->node[28] = (Node) {0x5000001b,0x4000001d};
  book->defs[0x000270c6]->node[29] = (Node) {0x5000001c,0x4000001e};
  book->defs[0x000270c6]->node[30] = (Node) {0x5000001d,0x4000001f};
  book->defs[0x000270c6]->node[31] = (Node) {0x5000001e,0x40000020};
  book->defs[0x000270c6]->node[32] = (Node) {0x5000001f,0x40000021};
  book->defs[0x000270c6]->node[33] = (Node) {0x50000020,0x40000022};
  book->defs[0x000270c6]->node[34] = (Node) {0x50000021,0x40000023};
  book->defs[0x000270c6]->node[35] = (Node) {0x50000022,0x40000024};
  book->defs[0x000270c6]->node[36] = (Node) {0x50000023,0x40000025};
  book->defs[0x000270c6]->node[37] = (Node) {0x50000024,0x40000026};
  book->defs[0x000270c6]->node[38] = (Node) {0x50000025,0x40000027};
  book->defs[0x000270c6]->node[39] = (Node) {0x50000026,0x40000028};
  book->defs[0x000270c6]->node[40] = (Node) {0x50000027,0x40000029};
  book->defs[0x000270c6]->node[41] = (Node) {0x50000028,0x4000002a};
  book->defs[0x000270c6]->node[42] = (Node) {0x50000029,0x4000002b};
  book->defs[0x000270c6]->node[43] = (Node) {0x5000002a,0x4000002c};
  book->defs[0x000270c6]->node[44] = (Node) {0x5000002b,0x4000002d};
  book->defs[0x000270c6]->node[45] = (Node) {0x5000002c,0x4000002e};
  book->defs[0x000270c6]->node[46] = (Node) {0x5000002d,0x4000002f};
  book->defs[0x000270c6]->node[47] = (Node) {0x5000002e,0x40000030};
  book->defs[0x000270c6]->node[48] = (Node) {0x5000002f,0x40000031};
  book->defs[0x000270c6]->node[49] = (Node) {0x50000030,0x50000032};
  book->defs[0x000270c6]->node[50] = (Node) {0x50000019,0x50000031};
  // c26
  book->defs[0x000270c7]           = (Term*) malloc(sizeof(Term));
  book->defs[0x000270c7]->root     = 0xa0000000;
  book->defs[0x000270c7]->alen     = 0;
  book->defs[0x000270c7]->acts     = (Wire*) malloc(0 * sizeof(Wire));
  book->defs[0x000270c7]->nlen     = 53;
  book->defs[0x000270c7]->node     = (Node*) malloc(53 * sizeof(Node));
  book->defs[0x000270c7]->node[ 0] = (Node) {0xb0000001,0xa0000034};
  book->defs[0x000270c7]->node[ 1] = (Node) {0xb0000002,0xa0000033};
  book->defs[0x000270c7]->node[ 2] = (Node) {0xb0000003,0xa0000032};
  book->defs[0x000270c7]->node[ 3] = (Node) {0xb0000004,0xa0000031};
  book->defs[0x000270c7]->node[ 4] = (Node) {0xb0000005,0xa0000030};
  book->defs[0x000270c7]->node[ 5] = (Node) {0xb0000006,0xa000002f};
  book->defs[0x000270c7]->node[ 6] = (Node) {0xb0000007,0xa000002e};
  book->defs[0x000270c7]->node[ 7] = (Node) {0xb0000008,0xa000002d};
  book->defs[0x000270c7]->node[ 8] = (Node) {0xb0000009,0xa000002c};
  book->defs[0x000270c7]->node[ 9] = (Node) {0xb000000a,0xa000002b};
  book->defs[0x000270c7]->node[10] = (Node) {0xb000000b,0xa000002a};
  book->defs[0x000270c7]->node[11] = (Node) {0xb000000c,0xa0000029};
  book->defs[0x000270c7]->node[12] = (Node) {0xb000000d,0xa0000028};
  book->defs[0x000270c7]->node[13] = (Node) {0xb000000e,0xa0000027};
  book->defs[0x000270c7]->node[14] = (Node) {0xb000000f,0xa0000026};
  book->defs[0x000270c7]->node[15] = (Node) {0xb0000010,0xa0000025};
  book->defs[0x000270c7]->node[16] = (Node) {0xb0000011,0xa0000024};
  book->defs[0x000270c7]->node[17] = (Node) {0xb0000012,0xa0000023};
  book->defs[0x000270c7]->node[18] = (Node) {0xb0000013,0xa0000022};
  book->defs[0x000270c7]->node[19] = (Node) {0xb0000014,0xa0000021};
  book->defs[0x000270c7]->node[20] = (Node) {0xb0000015,0xa0000020};
  book->defs[0x000270c7]->node[21] = (Node) {0xb0000016,0xa000001f};
  book->defs[0x000270c7]->node[22] = (Node) {0xb0000017,0xa000001e};
  book->defs[0x000270c7]->node[23] = (Node) {0xb0000018,0xa000001d};
  book->defs[0x000270c7]->node[24] = (Node) {0xb0000019,0xa000001c};
  book->defs[0x000270c7]->node[25] = (Node) {0xa000001a,0xa000001b};
  book->defs[0x000270c7]->node[26] = (Node) {0x00000000,0x4000001b};
  book->defs[0x000270c7]->node[27] = (Node) {0x5000001a,0x40000034};
  book->defs[0x000270c7]->node[28] = (Node) {0x5000001b,0x4000001d};
  book->defs[0x000270c7]->node[29] = (Node) {0x5000001c,0x4000001e};
  book->defs[0x000270c7]->node[30] = (Node) {0x5000001d,0x4000001f};
  book->defs[0x000270c7]->node[31] = (Node) {0x5000001e,0x40000020};
  book->defs[0x000270c7]->node[32] = (Node) {0x5000001f,0x40000021};
  book->defs[0x000270c7]->node[33] = (Node) {0x50000020,0x40000022};
  book->defs[0x000270c7]->node[34] = (Node) {0x50000021,0x40000023};
  book->defs[0x000270c7]->node[35] = (Node) {0x50000022,0x40000024};
  book->defs[0x000270c7]->node[36] = (Node) {0x50000023,0x40000025};
  book->defs[0x000270c7]->node[37] = (Node) {0x50000024,0x40000026};
  book->defs[0x000270c7]->node[38] = (Node) {0x50000025,0x40000027};
  book->defs[0x000270c7]->node[39] = (Node) {0x50000026,0x40000028};
  book->defs[0x000270c7]->node[40] = (Node) {0x50000027,0x40000029};
  book->defs[0x000270c7]->node[41] = (Node) {0x50000028,0x4000002a};
  book->defs[0x000270c7]->node[42] = (Node) {0x50000029,0x4000002b};
  book->defs[0x000270c7]->node[43] = (Node) {0x5000002a,0x4000002c};
  book->defs[0x000270c7]->node[44] = (Node) {0x5000002b,0x4000002d};
  book->defs[0x000270c7]->node[45] = (Node) {0x5000002c,0x4000002e};
  book->defs[0x000270c7]->node[46] = (Node) {0x5000002d,0x4000002f};
  book->defs[0x000270c7]->node[47] = (Node) {0x5000002e,0x40000030};
  book->defs[0x000270c7]->node[48] = (Node) {0x5000002f,0x40000031};
  book->defs[0x000270c7]->node[49] = (Node) {0x50000030,0x40000032};
  book->defs[0x000270c7]->node[50] = (Node) {0x50000031,0x40000033};
  book->defs[0x000270c7]->node[51] = (Node) {0x50000032,0x50000034};
  book->defs[0x000270c7]->node[52] = (Node) {0x5000001b,0x50000033};
  // c_s
  book->defs[0x00027ff7]           = (Term*) malloc(sizeof(Term));
  book->defs[0x00027ff7]->root     = 0xa0000000;
  book->defs[0x00027ff7]->alen     = 0;
  book->defs[0x00027ff7]->acts     = (Wire*) malloc(0 * sizeof(Wire));
  book->defs[0x00027ff7]->nlen     = 7;
  book->defs[0x00027ff7]->node     = (Node*) malloc(7 * sizeof(Node));
  book->defs[0x00027ff7]->node[ 0] = (Node) {0xa0000001,0xa0000003};
  book->defs[0x00027ff7]->node[ 1] = (Node) {0x50000004,0xa0000002};
  book->defs[0x00027ff7]->node[ 2] = (Node) {0x40000006,0x40000005};
  book->defs[0x00027ff7]->node[ 3] = (Node) {0xb0000004,0xa0000006};
  book->defs[0x00027ff7]->node[ 4] = (Node) {0xa0000005,0x40000001};
  book->defs[0x00027ff7]->node[ 5] = (Node) {0x50000002,0x50000006};
  book->defs[0x00027ff7]->node[ 6] = (Node) {0x40000002,0x50000005};
  // c_z
  book->defs[0x00027ffe]           = (Term*) malloc(sizeof(Term));
  book->defs[0x00027ffe]->root     = 0xa0000000;
  book->defs[0x00027ffe]->alen     = 0;
  book->defs[0x00027ffe]->acts     = (Wire*) malloc(0 * sizeof(Wire));
  book->defs[0x00027ffe]->nlen     = 2;
  book->defs[0x00027ffe]->node     = (Node*) malloc(2 * sizeof(Node));
  book->defs[0x00027ffe]->node[ 0] = (Node) {0x20000000,0xa0000001};
  book->defs[0x00027ffe]->node[ 1] = (Node) {0x50000001,0x40000001};
  // dec
  book->defs[0x00028a67]           = (Term*) malloc(sizeof(Term));
  book->defs[0x00028a67]->root     = 0xa0000000;
  book->defs[0x00028a67]->alen     = 0;
  book->defs[0x00028a67]->acts     = (Wire*) malloc(0 * sizeof(Wire));
  book->defs[0x00028a67]->nlen     = 4;
  book->defs[0x00028a67]->node     = (Node*) malloc(4 * sizeof(Node));
  book->defs[0x00028a67]->node[ 0] = (Node) {0xa0000001,0x50000003};
  book->defs[0x00028a67]->node[ 1] = (Node) {0x10a299d9,0xa0000002};
  book->defs[0x00028a67]->node[ 2] = (Node) {0x10a299d3,0xa0000003};
  book->defs[0x00028a67]->node[ 3] = (Node) {0x1000000f,0x50000000};
  // ex0
  book->defs[0x00029f01]           = (Term*) malloc(sizeof(Term));
  book->defs[0x00029f01]->root     = 0x50000000;
  book->defs[0x00029f01]->alen     = 1;
  book->defs[0x00029f01]->acts     = (Wire*) malloc(1 * sizeof(Wire));
  book->defs[0x00029f01]->acts[ 0] = mkwire(0x100009c3,0xa0000000);
  book->defs[0x00029f01]->nlen     = 1;
  book->defs[0x00029f01]->node     = (Node*) malloc(1 * sizeof(Node));
  book->defs[0x00029f01]->node[ 0] = (Node) {0x10000bc3,0x30000000};
  // ex1
  book->defs[0x00029f02]           = (Term*) malloc(sizeof(Term));
  book->defs[0x00029f02]->root     = 0x50000001;
  book->defs[0x00029f02]->alen     = 1;
  book->defs[0x00029f02]->acts     = (Wire*) malloc(1 * sizeof(Wire));
  book->defs[0x00029f02]->acts[ 0] = mkwire(0x100270c5,0xa0000000);
  book->defs[0x00029f02]->nlen     = 2;
  book->defs[0x00029f02]->node     = (Node*) malloc(2 * sizeof(Node));
  book->defs[0x00029f02]->node[ 0] = (Node) {0x1002bff7,0xa0000001};
  book->defs[0x00029f02]->node[ 1] = (Node) {0x1002bffe,0x30000000};
  // ex2
  book->defs[0x00029f03]           = (Term*) malloc(sizeof(Term));
  book->defs[0x00029f03]->root     = 0x50000002;
  book->defs[0x00029f03]->alen     = 2;
  book->defs[0x00029f03]->acts     = (Wire*) malloc(2 * sizeof(Wire));
  book->defs[0x00029f03]->acts[ 0] = mkwire(0x100270c1,0xa0000000);
  book->defs[0x00029f03]->acts[ 1] = mkwire(0x10036e72,0xa0000002);
  book->defs[0x00029f03]->nlen     = 3;
  book->defs[0x00029f03]->node     = (Node*) malloc(3 * sizeof(Node));
  book->defs[0x00029f03]->node[ 0] = (Node) {0x10000013,0xa0000001};
  book->defs[0x00029f03]->node[ 1] = (Node) {0x1000000f,0x40000002};
  book->defs[0x00029f03]->node[ 2] = (Node) {0x50000001,0x30000000};
  // ex3
  book->defs[0x00029f04]           = (Term*) malloc(sizeof(Term));
  book->defs[0x00029f04]->root     = 0x50000002;
  book->defs[0x00029f04]->alen     = 2;
  book->defs[0x00029f04]->acts     = (Wire*) malloc(2 * sizeof(Wire));
  book->defs[0x00029f04]->acts[ 0] = mkwire(0x10027088,0xa0000000);
  book->defs[0x00029f04]->acts[ 1] = mkwire(0x10026db2,0xa0000002);
  book->defs[0x00029f04]->nlen     = 3;
  book->defs[0x00029f04]->node     = (Node*) malloc(3 * sizeof(Node));
  book->defs[0x00029f04]->node[ 0] = (Node) {0x1000001d,0xa0000001};
  book->defs[0x00029f04]->node[ 1] = (Node) {0x10000024,0x40000002};
  book->defs[0x00029f04]->node[ 2] = (Node) {0x50000001,0x30000000};
  // ex4
  book->defs[0x00029f05]           = (Term*) malloc(sizeof(Term));
  book->defs[0x00029f05]->root     = 0x50000000;
  book->defs[0x00029f05]->alen     = 2;
  book->defs[0x00029f05]->acts     = (Wire*) malloc(2 * sizeof(Wire));
  book->defs[0x00029f05]->acts[ 0] = mkwire(0xa0000000,0x1000096a);
  book->defs[0x00029f05]->acts[ 1] = mkwire(0xa0000001,0x100009c2);
  book->defs[0x00029f05]->nlen     = 3;
  book->defs[0x00029f05]->node     = (Node*) malloc(3 * sizeof(Node));
  book->defs[0x00029f05]->node[ 0] = (Node) {0x50000002,0x30000000};
  book->defs[0x00029f05]->node[ 1] = (Node) {0x1000001d,0xa0000002};
  book->defs[0x00029f05]->node[ 2] = (Node) {0x10000024,0x40000000};
  // g_s
  book->defs[0x0002bff7]           = (Term*) malloc(sizeof(Term));
  book->defs[0x0002bff7]->root     = 0xa0000000;
  book->defs[0x0002bff7]->alen     = 0;
  book->defs[0x0002bff7]->acts     = (Wire*) malloc(0 * sizeof(Wire));
  book->defs[0x0002bff7]->nlen     = 5;
  book->defs[0x0002bff7]->node     = (Node*) malloc(5 * sizeof(Node));
  book->defs[0x0002bff7]->node[ 0] = (Node) {0xc0000001,0xa0000002};
  book->defs[0x0002bff7]->node[ 1] = (Node) {0x40000003,0x40000004};
  book->defs[0x0002bff7]->node[ 2] = (Node) {0xa0000003,0x50000004};
  book->defs[0x0002bff7]->node[ 3] = (Node) {0x40000001,0xa0000004};
  book->defs[0x0002bff7]->node[ 4] = (Node) {0x50000001,0x50000002};
  // g_z
  book->defs[0x0002bffe]           = (Term*) malloc(sizeof(Term));
  book->defs[0x0002bffe]->root     = 0xa0000000;
  book->defs[0x0002bffe]->alen     = 0;
  book->defs[0x0002bffe]->acts     = (Wire*) malloc(0 * sizeof(Wire));
  book->defs[0x0002bffe]->nlen     = 1;
  book->defs[0x0002bffe]->node     = (Node*) malloc(1 * sizeof(Node));
  book->defs[0x0002bffe]->node[ 0] = (Node) {0x50000000,0x40000000};
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
  book->defs[0x0002f081]->node[10] = (Node) {0x40000014,0x4000000b};
  book->defs[0x0002f081]->node[11] = (Node) {0x5000000a,0x4000000c};
  book->defs[0x0002f081]->node[12] = (Node) {0x5000000b,0x4000000d};
  book->defs[0x0002f081]->node[13] = (Node) {0x5000000c,0x4000000e};
  book->defs[0x0002f081]->node[14] = (Node) {0x5000000d,0x4000000f};
  book->defs[0x0002f081]->node[15] = (Node) {0x5000000e,0x40000010};
  book->defs[0x0002f081]->node[16] = (Node) {0x5000000f,0x40000011};
  book->defs[0x0002f081]->node[17] = (Node) {0x50000010,0x40000012};
  book->defs[0x0002f081]->node[18] = (Node) {0x50000011,0x40000013};
  book->defs[0x0002f081]->node[19] = (Node) {0x50000012,0x50000014};
  book->defs[0x0002f081]->node[20] = (Node) {0x4000000a,0x50000013};
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
  book->defs[0x0002f082]->node[11] = (Node) {0x40000016,0x4000000c};
  book->defs[0x0002f082]->node[12] = (Node) {0x5000000b,0x4000000d};
  book->defs[0x0002f082]->node[13] = (Node) {0x5000000c,0x4000000e};
  book->defs[0x0002f082]->node[14] = (Node) {0x5000000d,0x4000000f};
  book->defs[0x0002f082]->node[15] = (Node) {0x5000000e,0x40000010};
  book->defs[0x0002f082]->node[16] = (Node) {0x5000000f,0x40000011};
  book->defs[0x0002f082]->node[17] = (Node) {0x50000010,0x40000012};
  book->defs[0x0002f082]->node[18] = (Node) {0x50000011,0x40000013};
  book->defs[0x0002f082]->node[19] = (Node) {0x50000012,0x40000014};
  book->defs[0x0002f082]->node[20] = (Node) {0x50000013,0x40000015};
  book->defs[0x0002f082]->node[21] = (Node) {0x50000014,0x50000016};
  book->defs[0x0002f082]->node[22] = (Node) {0x4000000b,0x50000015};
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
  book->defs[0x0002f083]->node[12] = (Node) {0x40000018,0x4000000d};
  book->defs[0x0002f083]->node[13] = (Node) {0x5000000c,0x4000000e};
  book->defs[0x0002f083]->node[14] = (Node) {0x5000000d,0x4000000f};
  book->defs[0x0002f083]->node[15] = (Node) {0x5000000e,0x40000010};
  book->defs[0x0002f083]->node[16] = (Node) {0x5000000f,0x40000011};
  book->defs[0x0002f083]->node[17] = (Node) {0x50000010,0x40000012};
  book->defs[0x0002f083]->node[18] = (Node) {0x50000011,0x40000013};
  book->defs[0x0002f083]->node[19] = (Node) {0x50000012,0x40000014};
  book->defs[0x0002f083]->node[20] = (Node) {0x50000013,0x40000015};
  book->defs[0x0002f083]->node[21] = (Node) {0x50000014,0x40000016};
  book->defs[0x0002f083]->node[22] = (Node) {0x50000015,0x40000017};
  book->defs[0x0002f083]->node[23] = (Node) {0x50000016,0x50000018};
  book->defs[0x0002f083]->node[24] = (Node) {0x4000000c,0x50000017};
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
  book->defs[0x0002f084]->node[13] = (Node) {0x4000001a,0x4000000e};
  book->defs[0x0002f084]->node[14] = (Node) {0x5000000d,0x4000000f};
  book->defs[0x0002f084]->node[15] = (Node) {0x5000000e,0x40000010};
  book->defs[0x0002f084]->node[16] = (Node) {0x5000000f,0x40000011};
  book->defs[0x0002f084]->node[17] = (Node) {0x50000010,0x40000012};
  book->defs[0x0002f084]->node[18] = (Node) {0x50000011,0x40000013};
  book->defs[0x0002f084]->node[19] = (Node) {0x50000012,0x40000014};
  book->defs[0x0002f084]->node[20] = (Node) {0x50000013,0x40000015};
  book->defs[0x0002f084]->node[21] = (Node) {0x50000014,0x40000016};
  book->defs[0x0002f084]->node[22] = (Node) {0x50000015,0x40000017};
  book->defs[0x0002f084]->node[23] = (Node) {0x50000016,0x40000018};
  book->defs[0x0002f084]->node[24] = (Node) {0x50000017,0x40000019};
  book->defs[0x0002f084]->node[25] = (Node) {0x50000018,0x5000001a};
  book->defs[0x0002f084]->node[26] = (Node) {0x4000000d,0x50000019};
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
  book->defs[0x0002f085]->node[14] = (Node) {0x4000001c,0x4000000f};
  book->defs[0x0002f085]->node[15] = (Node) {0x5000000e,0x40000010};
  book->defs[0x0002f085]->node[16] = (Node) {0x5000000f,0x40000011};
  book->defs[0x0002f085]->node[17] = (Node) {0x50000010,0x40000012};
  book->defs[0x0002f085]->node[18] = (Node) {0x50000011,0x40000013};
  book->defs[0x0002f085]->node[19] = (Node) {0x50000012,0x40000014};
  book->defs[0x0002f085]->node[20] = (Node) {0x50000013,0x40000015};
  book->defs[0x0002f085]->node[21] = (Node) {0x50000014,0x40000016};
  book->defs[0x0002f085]->node[22] = (Node) {0x50000015,0x40000017};
  book->defs[0x0002f085]->node[23] = (Node) {0x50000016,0x40000018};
  book->defs[0x0002f085]->node[24] = (Node) {0x50000017,0x40000019};
  book->defs[0x0002f085]->node[25] = (Node) {0x50000018,0x4000001a};
  book->defs[0x0002f085]->node[26] = (Node) {0x50000019,0x4000001b};
  book->defs[0x0002f085]->node[27] = (Node) {0x5000001a,0x5000001c};
  book->defs[0x0002f085]->node[28] = (Node) {0x4000000e,0x5000001b};
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
  book->defs[0x0002f086]->node[15] = (Node) {0x4000001e,0x40000010};
  book->defs[0x0002f086]->node[16] = (Node) {0x5000000f,0x40000011};
  book->defs[0x0002f086]->node[17] = (Node) {0x50000010,0x40000012};
  book->defs[0x0002f086]->node[18] = (Node) {0x50000011,0x40000013};
  book->defs[0x0002f086]->node[19] = (Node) {0x50000012,0x40000014};
  book->defs[0x0002f086]->node[20] = (Node) {0x50000013,0x40000015};
  book->defs[0x0002f086]->node[21] = (Node) {0x50000014,0x40000016};
  book->defs[0x0002f086]->node[22] = (Node) {0x50000015,0x40000017};
  book->defs[0x0002f086]->node[23] = (Node) {0x50000016,0x40000018};
  book->defs[0x0002f086]->node[24] = (Node) {0x50000017,0x40000019};
  book->defs[0x0002f086]->node[25] = (Node) {0x50000018,0x4000001a};
  book->defs[0x0002f086]->node[26] = (Node) {0x50000019,0x4000001b};
  book->defs[0x0002f086]->node[27] = (Node) {0x5000001a,0x4000001c};
  book->defs[0x0002f086]->node[28] = (Node) {0x5000001b,0x4000001d};
  book->defs[0x0002f086]->node[29] = (Node) {0x5000001c,0x5000001e};
  book->defs[0x0002f086]->node[30] = (Node) {0x4000000f,0x5000001d};
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
  book->defs[0x0002f087]->node[16] = (Node) {0x40000020,0x40000011};
  book->defs[0x0002f087]->node[17] = (Node) {0x50000010,0x40000012};
  book->defs[0x0002f087]->node[18] = (Node) {0x50000011,0x40000013};
  book->defs[0x0002f087]->node[19] = (Node) {0x50000012,0x40000014};
  book->defs[0x0002f087]->node[20] = (Node) {0x50000013,0x40000015};
  book->defs[0x0002f087]->node[21] = (Node) {0x50000014,0x40000016};
  book->defs[0x0002f087]->node[22] = (Node) {0x50000015,0x40000017};
  book->defs[0x0002f087]->node[23] = (Node) {0x50000016,0x40000018};
  book->defs[0x0002f087]->node[24] = (Node) {0x50000017,0x40000019};
  book->defs[0x0002f087]->node[25] = (Node) {0x50000018,0x4000001a};
  book->defs[0x0002f087]->node[26] = (Node) {0x50000019,0x4000001b};
  book->defs[0x0002f087]->node[27] = (Node) {0x5000001a,0x4000001c};
  book->defs[0x0002f087]->node[28] = (Node) {0x5000001b,0x4000001d};
  book->defs[0x0002f087]->node[29] = (Node) {0x5000001c,0x4000001e};
  book->defs[0x0002f087]->node[30] = (Node) {0x5000001d,0x4000001f};
  book->defs[0x0002f087]->node[31] = (Node) {0x5000001e,0x50000020};
  book->defs[0x0002f087]->node[32] = (Node) {0x40000010,0x5000001f};
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
  book->defs[0x0002f088]->node[17] = (Node) {0x40000022,0x40000012};
  book->defs[0x0002f088]->node[18] = (Node) {0x50000011,0x40000013};
  book->defs[0x0002f088]->node[19] = (Node) {0x50000012,0x40000014};
  book->defs[0x0002f088]->node[20] = (Node) {0x50000013,0x40000015};
  book->defs[0x0002f088]->node[21] = (Node) {0x50000014,0x40000016};
  book->defs[0x0002f088]->node[22] = (Node) {0x50000015,0x40000017};
  book->defs[0x0002f088]->node[23] = (Node) {0x50000016,0x40000018};
  book->defs[0x0002f088]->node[24] = (Node) {0x50000017,0x40000019};
  book->defs[0x0002f088]->node[25] = (Node) {0x50000018,0x4000001a};
  book->defs[0x0002f088]->node[26] = (Node) {0x50000019,0x4000001b};
  book->defs[0x0002f088]->node[27] = (Node) {0x5000001a,0x4000001c};
  book->defs[0x0002f088]->node[28] = (Node) {0x5000001b,0x4000001d};
  book->defs[0x0002f088]->node[29] = (Node) {0x5000001c,0x4000001e};
  book->defs[0x0002f088]->node[30] = (Node) {0x5000001d,0x4000001f};
  book->defs[0x0002f088]->node[31] = (Node) {0x5000001e,0x40000020};
  book->defs[0x0002f088]->node[32] = (Node) {0x5000001f,0x40000021};
  book->defs[0x0002f088]->node[33] = (Node) {0x50000020,0x50000022};
  book->defs[0x0002f088]->node[34] = (Node) {0x40000011,0x50000021};
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
  book->defs[0x0002f089]->node[18] = (Node) {0x40000024,0x40000013};
  book->defs[0x0002f089]->node[19] = (Node) {0x50000012,0x40000014};
  book->defs[0x0002f089]->node[20] = (Node) {0x50000013,0x40000015};
  book->defs[0x0002f089]->node[21] = (Node) {0x50000014,0x40000016};
  book->defs[0x0002f089]->node[22] = (Node) {0x50000015,0x40000017};
  book->defs[0x0002f089]->node[23] = (Node) {0x50000016,0x40000018};
  book->defs[0x0002f089]->node[24] = (Node) {0x50000017,0x40000019};
  book->defs[0x0002f089]->node[25] = (Node) {0x50000018,0x4000001a};
  book->defs[0x0002f089]->node[26] = (Node) {0x50000019,0x4000001b};
  book->defs[0x0002f089]->node[27] = (Node) {0x5000001a,0x4000001c};
  book->defs[0x0002f089]->node[28] = (Node) {0x5000001b,0x4000001d};
  book->defs[0x0002f089]->node[29] = (Node) {0x5000001c,0x4000001e};
  book->defs[0x0002f089]->node[30] = (Node) {0x5000001d,0x4000001f};
  book->defs[0x0002f089]->node[31] = (Node) {0x5000001e,0x40000020};
  book->defs[0x0002f089]->node[32] = (Node) {0x5000001f,0x40000021};
  book->defs[0x0002f089]->node[33] = (Node) {0x50000020,0x40000022};
  book->defs[0x0002f089]->node[34] = (Node) {0x50000021,0x40000023};
  book->defs[0x0002f089]->node[35] = (Node) {0x50000022,0x50000024};
  book->defs[0x0002f089]->node[36] = (Node) {0x40000012,0x50000023};
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
  book->defs[0x0002f08a]->node[19] = (Node) {0x40000026,0x40000014};
  book->defs[0x0002f08a]->node[20] = (Node) {0x50000013,0x40000015};
  book->defs[0x0002f08a]->node[21] = (Node) {0x50000014,0x40000016};
  book->defs[0x0002f08a]->node[22] = (Node) {0x50000015,0x40000017};
  book->defs[0x0002f08a]->node[23] = (Node) {0x50000016,0x40000018};
  book->defs[0x0002f08a]->node[24] = (Node) {0x50000017,0x40000019};
  book->defs[0x0002f08a]->node[25] = (Node) {0x50000018,0x4000001a};
  book->defs[0x0002f08a]->node[26] = (Node) {0x50000019,0x4000001b};
  book->defs[0x0002f08a]->node[27] = (Node) {0x5000001a,0x4000001c};
  book->defs[0x0002f08a]->node[28] = (Node) {0x5000001b,0x4000001d};
  book->defs[0x0002f08a]->node[29] = (Node) {0x5000001c,0x4000001e};
  book->defs[0x0002f08a]->node[30] = (Node) {0x5000001d,0x4000001f};
  book->defs[0x0002f08a]->node[31] = (Node) {0x5000001e,0x40000020};
  book->defs[0x0002f08a]->node[32] = (Node) {0x5000001f,0x40000021};
  book->defs[0x0002f08a]->node[33] = (Node) {0x50000020,0x40000022};
  book->defs[0x0002f08a]->node[34] = (Node) {0x50000021,0x40000023};
  book->defs[0x0002f08a]->node[35] = (Node) {0x50000022,0x40000024};
  book->defs[0x0002f08a]->node[36] = (Node) {0x50000023,0x40000025};
  book->defs[0x0002f08a]->node[37] = (Node) {0x50000024,0x50000026};
  book->defs[0x0002f08a]->node[38] = (Node) {0x40000013,0x50000025};
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
  book->defs[0x0002f0c1]->node[20] = (Node) {0x40000028,0x40000015};
  book->defs[0x0002f0c1]->node[21] = (Node) {0x50000014,0x40000016};
  book->defs[0x0002f0c1]->node[22] = (Node) {0x50000015,0x40000017};
  book->defs[0x0002f0c1]->node[23] = (Node) {0x50000016,0x40000018};
  book->defs[0x0002f0c1]->node[24] = (Node) {0x50000017,0x40000019};
  book->defs[0x0002f0c1]->node[25] = (Node) {0x50000018,0x4000001a};
  book->defs[0x0002f0c1]->node[26] = (Node) {0x50000019,0x4000001b};
  book->defs[0x0002f0c1]->node[27] = (Node) {0x5000001a,0x4000001c};
  book->defs[0x0002f0c1]->node[28] = (Node) {0x5000001b,0x4000001d};
  book->defs[0x0002f0c1]->node[29] = (Node) {0x5000001c,0x4000001e};
  book->defs[0x0002f0c1]->node[30] = (Node) {0x5000001d,0x4000001f};
  book->defs[0x0002f0c1]->node[31] = (Node) {0x5000001e,0x40000020};
  book->defs[0x0002f0c1]->node[32] = (Node) {0x5000001f,0x40000021};
  book->defs[0x0002f0c1]->node[33] = (Node) {0x50000020,0x40000022};
  book->defs[0x0002f0c1]->node[34] = (Node) {0x50000021,0x40000023};
  book->defs[0x0002f0c1]->node[35] = (Node) {0x50000022,0x40000024};
  book->defs[0x0002f0c1]->node[36] = (Node) {0x50000023,0x40000025};
  book->defs[0x0002f0c1]->node[37] = (Node) {0x50000024,0x40000026};
  book->defs[0x0002f0c1]->node[38] = (Node) {0x50000025,0x40000027};
  book->defs[0x0002f0c1]->node[39] = (Node) {0x50000026,0x50000028};
  book->defs[0x0002f0c1]->node[40] = (Node) {0x40000014,0x50000027};
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
  book->defs[0x0002f0c2]->node[21] = (Node) {0x4000002a,0x40000016};
  book->defs[0x0002f0c2]->node[22] = (Node) {0x50000015,0x40000017};
  book->defs[0x0002f0c2]->node[23] = (Node) {0x50000016,0x40000018};
  book->defs[0x0002f0c2]->node[24] = (Node) {0x50000017,0x40000019};
  book->defs[0x0002f0c2]->node[25] = (Node) {0x50000018,0x4000001a};
  book->defs[0x0002f0c2]->node[26] = (Node) {0x50000019,0x4000001b};
  book->defs[0x0002f0c2]->node[27] = (Node) {0x5000001a,0x4000001c};
  book->defs[0x0002f0c2]->node[28] = (Node) {0x5000001b,0x4000001d};
  book->defs[0x0002f0c2]->node[29] = (Node) {0x5000001c,0x4000001e};
  book->defs[0x0002f0c2]->node[30] = (Node) {0x5000001d,0x4000001f};
  book->defs[0x0002f0c2]->node[31] = (Node) {0x5000001e,0x40000020};
  book->defs[0x0002f0c2]->node[32] = (Node) {0x5000001f,0x40000021};
  book->defs[0x0002f0c2]->node[33] = (Node) {0x50000020,0x40000022};
  book->defs[0x0002f0c2]->node[34] = (Node) {0x50000021,0x40000023};
  book->defs[0x0002f0c2]->node[35] = (Node) {0x50000022,0x40000024};
  book->defs[0x0002f0c2]->node[36] = (Node) {0x50000023,0x40000025};
  book->defs[0x0002f0c2]->node[37] = (Node) {0x50000024,0x40000026};
  book->defs[0x0002f0c2]->node[38] = (Node) {0x50000025,0x40000027};
  book->defs[0x0002f0c2]->node[39] = (Node) {0x50000026,0x40000028};
  book->defs[0x0002f0c2]->node[40] = (Node) {0x50000027,0x40000029};
  book->defs[0x0002f0c2]->node[41] = (Node) {0x50000028,0x5000002a};
  book->defs[0x0002f0c2]->node[42] = (Node) {0x40000015,0x50000029};
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
  book->defs[0x0002f0c3]->node[22] = (Node) {0x4000002c,0x40000017};
  book->defs[0x0002f0c3]->node[23] = (Node) {0x50000016,0x40000018};
  book->defs[0x0002f0c3]->node[24] = (Node) {0x50000017,0x40000019};
  book->defs[0x0002f0c3]->node[25] = (Node) {0x50000018,0x4000001a};
  book->defs[0x0002f0c3]->node[26] = (Node) {0x50000019,0x4000001b};
  book->defs[0x0002f0c3]->node[27] = (Node) {0x5000001a,0x4000001c};
  book->defs[0x0002f0c3]->node[28] = (Node) {0x5000001b,0x4000001d};
  book->defs[0x0002f0c3]->node[29] = (Node) {0x5000001c,0x4000001e};
  book->defs[0x0002f0c3]->node[30] = (Node) {0x5000001d,0x4000001f};
  book->defs[0x0002f0c3]->node[31] = (Node) {0x5000001e,0x40000020};
  book->defs[0x0002f0c3]->node[32] = (Node) {0x5000001f,0x40000021};
  book->defs[0x0002f0c3]->node[33] = (Node) {0x50000020,0x40000022};
  book->defs[0x0002f0c3]->node[34] = (Node) {0x50000021,0x40000023};
  book->defs[0x0002f0c3]->node[35] = (Node) {0x50000022,0x40000024};
  book->defs[0x0002f0c3]->node[36] = (Node) {0x50000023,0x40000025};
  book->defs[0x0002f0c3]->node[37] = (Node) {0x50000024,0x40000026};
  book->defs[0x0002f0c3]->node[38] = (Node) {0x50000025,0x40000027};
  book->defs[0x0002f0c3]->node[39] = (Node) {0x50000026,0x40000028};
  book->defs[0x0002f0c3]->node[40] = (Node) {0x50000027,0x40000029};
  book->defs[0x0002f0c3]->node[41] = (Node) {0x50000028,0x4000002a};
  book->defs[0x0002f0c3]->node[42] = (Node) {0x50000029,0x4000002b};
  book->defs[0x0002f0c3]->node[43] = (Node) {0x5000002a,0x5000002c};
  book->defs[0x0002f0c3]->node[44] = (Node) {0x40000016,0x5000002b};
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
  book->defs[0x0002f0c4]->node[23] = (Node) {0x4000002e,0x40000018};
  book->defs[0x0002f0c4]->node[24] = (Node) {0x50000017,0x40000019};
  book->defs[0x0002f0c4]->node[25] = (Node) {0x50000018,0x4000001a};
  book->defs[0x0002f0c4]->node[26] = (Node) {0x50000019,0x4000001b};
  book->defs[0x0002f0c4]->node[27] = (Node) {0x5000001a,0x4000001c};
  book->defs[0x0002f0c4]->node[28] = (Node) {0x5000001b,0x4000001d};
  book->defs[0x0002f0c4]->node[29] = (Node) {0x5000001c,0x4000001e};
  book->defs[0x0002f0c4]->node[30] = (Node) {0x5000001d,0x4000001f};
  book->defs[0x0002f0c4]->node[31] = (Node) {0x5000001e,0x40000020};
  book->defs[0x0002f0c4]->node[32] = (Node) {0x5000001f,0x40000021};
  book->defs[0x0002f0c4]->node[33] = (Node) {0x50000020,0x40000022};
  book->defs[0x0002f0c4]->node[34] = (Node) {0x50000021,0x40000023};
  book->defs[0x0002f0c4]->node[35] = (Node) {0x50000022,0x40000024};
  book->defs[0x0002f0c4]->node[36] = (Node) {0x50000023,0x40000025};
  book->defs[0x0002f0c4]->node[37] = (Node) {0x50000024,0x40000026};
  book->defs[0x0002f0c4]->node[38] = (Node) {0x50000025,0x40000027};
  book->defs[0x0002f0c4]->node[39] = (Node) {0x50000026,0x40000028};
  book->defs[0x0002f0c4]->node[40] = (Node) {0x50000027,0x40000029};
  book->defs[0x0002f0c4]->node[41] = (Node) {0x50000028,0x4000002a};
  book->defs[0x0002f0c4]->node[42] = (Node) {0x50000029,0x4000002b};
  book->defs[0x0002f0c4]->node[43] = (Node) {0x5000002a,0x4000002c};
  book->defs[0x0002f0c4]->node[44] = (Node) {0x5000002b,0x4000002d};
  book->defs[0x0002f0c4]->node[45] = (Node) {0x5000002c,0x5000002e};
  book->defs[0x0002f0c4]->node[46] = (Node) {0x40000017,0x5000002d};
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
  book->defs[0x0002f0c5]->node[24] = (Node) {0x40000030,0x40000019};
  book->defs[0x0002f0c5]->node[25] = (Node) {0x50000018,0x4000001a};
  book->defs[0x0002f0c5]->node[26] = (Node) {0x50000019,0x4000001b};
  book->defs[0x0002f0c5]->node[27] = (Node) {0x5000001a,0x4000001c};
  book->defs[0x0002f0c5]->node[28] = (Node) {0x5000001b,0x4000001d};
  book->defs[0x0002f0c5]->node[29] = (Node) {0x5000001c,0x4000001e};
  book->defs[0x0002f0c5]->node[30] = (Node) {0x5000001d,0x4000001f};
  book->defs[0x0002f0c5]->node[31] = (Node) {0x5000001e,0x40000020};
  book->defs[0x0002f0c5]->node[32] = (Node) {0x5000001f,0x40000021};
  book->defs[0x0002f0c5]->node[33] = (Node) {0x50000020,0x40000022};
  book->defs[0x0002f0c5]->node[34] = (Node) {0x50000021,0x40000023};
  book->defs[0x0002f0c5]->node[35] = (Node) {0x50000022,0x40000024};
  book->defs[0x0002f0c5]->node[36] = (Node) {0x50000023,0x40000025};
  book->defs[0x0002f0c5]->node[37] = (Node) {0x50000024,0x40000026};
  book->defs[0x0002f0c5]->node[38] = (Node) {0x50000025,0x40000027};
  book->defs[0x0002f0c5]->node[39] = (Node) {0x50000026,0x40000028};
  book->defs[0x0002f0c5]->node[40] = (Node) {0x50000027,0x40000029};
  book->defs[0x0002f0c5]->node[41] = (Node) {0x50000028,0x4000002a};
  book->defs[0x0002f0c5]->node[42] = (Node) {0x50000029,0x4000002b};
  book->defs[0x0002f0c5]->node[43] = (Node) {0x5000002a,0x4000002c};
  book->defs[0x0002f0c5]->node[44] = (Node) {0x5000002b,0x4000002d};
  book->defs[0x0002f0c5]->node[45] = (Node) {0x5000002c,0x4000002e};
  book->defs[0x0002f0c5]->node[46] = (Node) {0x5000002d,0x4000002f};
  book->defs[0x0002f0c5]->node[47] = (Node) {0x5000002e,0x50000030};
  book->defs[0x0002f0c5]->node[48] = (Node) {0x40000018,0x5000002f};
  // low
  book->defs[0x00030cfb]           = (Term*) malloc(sizeof(Term));
  book->defs[0x00030cfb]->root     = 0xa0000000;
  book->defs[0x00030cfb]->alen     = 0;
  book->defs[0x00030cfb]->acts     = (Wire*) malloc(0 * sizeof(Wire));
  book->defs[0x00030cfb]->nlen     = 4;
  book->defs[0x00030cfb]->node     = (Node*) malloc(4 * sizeof(Node));
  book->defs[0x00030cfb]->node[ 0] = (Node) {0xa0000001,0x50000003};
  book->defs[0x00030cfb]->node[ 1] = (Node) {0x10c33ed9,0xa0000002};
  book->defs[0x00030cfb]->node[ 2] = (Node) {0x10c33ed3,0xa0000003};
  book->defs[0x00030cfb]->node[ 3] = (Node) {0x1000000f,0x50000000};
  // mul
  book->defs[0x00031e70]           = (Term*) malloc(sizeof(Term));
  book->defs[0x00031e70]->root     = 0xa0000000;
  book->defs[0x00031e70]->alen     = 0;
  book->defs[0x00031e70]->acts     = (Wire*) malloc(0 * sizeof(Wire));
  book->defs[0x00031e70]->nlen     = 5;
  book->defs[0x00031e70]->node     = (Node*) malloc(5 * sizeof(Node));
  book->defs[0x00031e70]->node[ 0] = (Node) {0xa0000001,0xa0000002};
  book->defs[0x00031e70]->node[ 1] = (Node) {0x50000003,0x50000004};
  book->defs[0x00031e70]->node[ 2] = (Node) {0xa0000003,0xa0000004};
  book->defs[0x00031e70]->node[ 3] = (Node) {0x40000004,0x40000001};
  book->defs[0x00031e70]->node[ 4] = (Node) {0x40000003,0x50000001};
  // nid
  book->defs[0x00032b68]           = (Term*) malloc(sizeof(Term));
  book->defs[0x00032b68]->root     = 0xa0000000;
  book->defs[0x00032b68]->alen     = 0;
  book->defs[0x00032b68]->acts     = (Wire*) malloc(0 * sizeof(Wire));
  book->defs[0x00032b68]->nlen     = 3;
  book->defs[0x00032b68]->node     = (Node*) malloc(3 * sizeof(Node));
  book->defs[0x00032b68]->node[ 0] = (Node) {0xa0000001,0x50000002};
  book->defs[0x00032b68]->node[ 1] = (Node) {0x10cada1d,0xa0000002};
  book->defs[0x00032b68]->node[ 2] = (Node) {0x10000024,0x50000000};
  // not
  book->defs[0x00032cf8]           = (Term*) malloc(sizeof(Term));
  book->defs[0x00032cf8]->root     = 0xa0000000;
  book->defs[0x00032cf8]->alen     = 0;
  book->defs[0x00032cf8]->acts     = (Wire*) malloc(0 * sizeof(Wire));
  book->defs[0x00032cf8]->nlen     = 5;
  book->defs[0x00032cf8]->node     = (Node*) malloc(5 * sizeof(Node));
  book->defs[0x00032cf8]->node[ 0] = (Node) {0xa0000001,0xa0000003};
  book->defs[0x00032cf8]->node[ 1] = (Node) {0x40000004,0xa0000002};
  book->defs[0x00032cf8]->node[ 2] = (Node) {0x40000003,0x50000004};
  book->defs[0x00032cf8]->node[ 3] = (Node) {0x40000002,0xa0000004};
  book->defs[0x00032cf8]->node[ 4] = (Node) {0x40000001,0x50000002};
  // run
  book->defs[0x00036e72]           = (Term*) malloc(sizeof(Term));
  book->defs[0x00036e72]->root     = 0xa0000000;
  book->defs[0x00036e72]->alen     = 0;
  book->defs[0x00036e72]->acts     = (Wire*) malloc(0 * sizeof(Wire));
  book->defs[0x00036e72]->nlen     = 4;
  book->defs[0x00036e72]->node     = (Node*) malloc(4 * sizeof(Node));
  book->defs[0x00036e72]->node[ 0] = (Node) {0xa0000001,0x50000003};
  book->defs[0x00036e72]->node[ 1] = (Node) {0x10db9c99,0xa0000002};
  book->defs[0x00036e72]->node[ 2] = (Node) {0x10db9c93,0xa0000003};
  book->defs[0x00036e72]->node[ 3] = (Node) {0x1000000f,0x50000000};
  // brnS
  book->defs[0x009b6c9d]           = (Term*) malloc(sizeof(Term));
  book->defs[0x009b6c9d]->root     = 0xa0000000;
  book->defs[0x009b6c9d]->alen     = 2;
  book->defs[0x009b6c9d]->acts     = (Wire*) malloc(2 * sizeof(Wire));
  book->defs[0x009b6c9d]->acts[ 0] = mkwire(0x10026db2,0xa0000003);
  book->defs[0x009b6c9d]->acts[ 1] = mkwire(0x10026db2,0xa0000004);
  book->defs[0x009b6c9d]->nlen     = 5;
  book->defs[0x009b6c9d]->node     = (Node*) malloc(5 * sizeof(Node));
  book->defs[0x009b6c9d]->node[ 0] = (Node) {0xb0000001,0xa0000002};
  book->defs[0x009b6c9d]->node[ 1] = (Node) {0x40000003,0x40000004};
  book->defs[0x009b6c9d]->node[ 2] = (Node) {0x50000003,0x50000004};
  book->defs[0x009b6c9d]->node[ 3] = (Node) {0x40000001,0x40000002};
  book->defs[0x009b6c9d]->node[ 4] = (Node) {0x50000001,0x50000002};
  // brnZ
  book->defs[0x009b6ca4]           = (Term*) malloc(sizeof(Term));
  book->defs[0x009b6ca4]->root     = 0x50000000;
  book->defs[0x009b6ca4]->alen     = 2;
  book->defs[0x009b6ca4]->acts     = (Wire*) malloc(2 * sizeof(Wire));
  book->defs[0x009b6ca4]->acts[ 0] = mkwire(0x10036e72,0xa0000000);
  book->defs[0x009b6ca4]->acts[ 1] = mkwire(0x10031e70,0xa0000001);
  book->defs[0x009b6ca4]->nlen     = 5;
  book->defs[0x009b6ca4]->node     = (Node*) malloc(5 * sizeof(Node));
  book->defs[0x009b6ca4]->node[ 0] = (Node) {0x50000004,0x30000000};
  book->defs[0x009b6ca4]->node[ 1] = (Node) {0x100009c3,0xa0000002};
  book->defs[0x009b6ca4]->node[ 2] = (Node) {0x100009c6,0xa0000003};
  book->defs[0x009b6ca4]->node[ 3] = (Node) {0x10000013,0xa0000004};
  book->defs[0x009b6ca4]->node[ 4] = (Node) {0x1000000f,0x40000000};
  // decI
  book->defs[0x00a299d3]           = (Term*) malloc(sizeof(Term));
  book->defs[0x00a299d3]->root     = 0xa0000000;
  book->defs[0x00a299d3]->alen     = 1;
  book->defs[0x00a299d3]->acts     = (Wire*) malloc(1 * sizeof(Wire));
  book->defs[0x00a299d3]->acts[ 0] = mkwire(0x10030cfb,0xa0000001);
  book->defs[0x00a299d3]->nlen     = 2;
  book->defs[0x00a299d3]->node     = (Node*) malloc(2 * sizeof(Node));
  book->defs[0x00a299d3]->node[ 0] = (Node) {0x40000001,0x50000001};
  book->defs[0x00a299d3]->node[ 1] = (Node) {0x40000000,0x50000000};
  // decO
  book->defs[0x00a299d9]           = (Term*) malloc(sizeof(Term));
  book->defs[0x00a299d9]->root     = 0xa0000000;
  book->defs[0x00a299d9]->alen     = 2;
  book->defs[0x00a299d9]->acts     = (Wire*) malloc(2 * sizeof(Wire));
  book->defs[0x00a299d9]->acts[ 0] = mkwire(0x10000013,0xa0000001);
  book->defs[0x00a299d9]->acts[ 1] = mkwire(0x10028a67,0xa0000002);
  book->defs[0x00a299d9]->nlen     = 3;
  book->defs[0x00a299d9]->node     = (Node*) malloc(3 * sizeof(Node));
  book->defs[0x00a299d9]->node[ 0] = (Node) {0x40000002,0x50000001};
  book->defs[0x00a299d9]->node[ 1] = (Node) {0x50000002,0x50000000};
  book->defs[0x00a299d9]->node[ 2] = (Node) {0x40000000,0x40000001};
  // lowI
  book->defs[0x00c33ed3]           = (Term*) malloc(sizeof(Term));
  book->defs[0x00c33ed3]->root     = 0xa0000000;
  book->defs[0x00c33ed3]->alen     = 2;
  book->defs[0x00c33ed3]->acts     = (Wire*) malloc(2 * sizeof(Wire));
  book->defs[0x00c33ed3]->acts[ 0] = mkwire(0x10000013,0xa0000001);
  book->defs[0x00c33ed3]->acts[ 1] = mkwire(0x10000019,0xa0000002);
  book->defs[0x00c33ed3]->nlen     = 3;
  book->defs[0x00c33ed3]->node     = (Node*) malloc(3 * sizeof(Node));
  book->defs[0x00c33ed3]->node[ 0] = (Node) {0x40000001,0x50000002};
  book->defs[0x00c33ed3]->node[ 1] = (Node) {0x40000000,0x40000002};
  book->defs[0x00c33ed3]->node[ 2] = (Node) {0x50000001,0x50000000};
  // lowO
  book->defs[0x00c33ed9]           = (Term*) malloc(sizeof(Term));
  book->defs[0x00c33ed9]->root     = 0xa0000000;
  book->defs[0x00c33ed9]->alen     = 2;
  book->defs[0x00c33ed9]->acts     = (Wire*) malloc(2 * sizeof(Wire));
  book->defs[0x00c33ed9]->acts[ 0] = mkwire(0x10000019,0xa0000001);
  book->defs[0x00c33ed9]->acts[ 1] = mkwire(0x10000019,0xa0000002);
  book->defs[0x00c33ed9]->nlen     = 3;
  book->defs[0x00c33ed9]->node     = (Node*) malloc(3 * sizeof(Node));
  book->defs[0x00c33ed9]->node[ 0] = (Node) {0x40000001,0x50000002};
  book->defs[0x00c33ed9]->node[ 1] = (Node) {0x40000000,0x40000002};
  book->defs[0x00c33ed9]->node[ 2] = (Node) {0x50000001,0x50000000};
  // nidS
  book->defs[0x00cada1d]           = (Term*) malloc(sizeof(Term));
  book->defs[0x00cada1d]->root     = 0xa0000000;
  book->defs[0x00cada1d]->alen     = 2;
  book->defs[0x00cada1d]->acts     = (Wire*) malloc(2 * sizeof(Wire));
  book->defs[0x00cada1d]->acts[ 0] = mkwire(0x1000001d,0xa0000001);
  book->defs[0x00cada1d]->acts[ 1] = mkwire(0x10032b68,0xa0000002);
  book->defs[0x00cada1d]->nlen     = 3;
  book->defs[0x00cada1d]->node     = (Node*) malloc(3 * sizeof(Node));
  book->defs[0x00cada1d]->node[ 0] = (Node) {0x40000002,0x50000001};
  book->defs[0x00cada1d]->node[ 1] = (Node) {0x50000002,0x50000000};
  book->defs[0x00cada1d]->node[ 2] = (Node) {0x40000000,0x40000001};
  // runI
  book->defs[0x00db9c93]           = (Term*) malloc(sizeof(Term));
  book->defs[0x00db9c93]->root     = 0xa0000000;
  book->defs[0x00db9c93]->alen     = 3;
  book->defs[0x00db9c93]->acts     = (Wire*) malloc(3 * sizeof(Wire));
  book->defs[0x00db9c93]->acts[ 0] = mkwire(0x10036e72,0xa0000001);
  book->defs[0x00db9c93]->acts[ 1] = mkwire(0x10028a67,0xa0000002);
  book->defs[0x00db9c93]->acts[ 2] = mkwire(0x10000013,0xa0000003);
  book->defs[0x00db9c93]->nlen     = 4;
  book->defs[0x00db9c93]->node     = (Node*) malloc(4 * sizeof(Node));
  book->defs[0x00db9c93]->node[ 0] = (Node) {0x40000003,0x50000001};
  book->defs[0x00db9c93]->node[ 1] = (Node) {0x50000002,0x50000000};
  book->defs[0x00db9c93]->node[ 2] = (Node) {0x50000003,0x40000001};
  book->defs[0x00db9c93]->node[ 3] = (Node) {0x40000000,0x40000002};
  // runO
  book->defs[0x00db9c99]           = (Term*) malloc(sizeof(Term));
  book->defs[0x00db9c99]->root     = 0xa0000000;
  book->defs[0x00db9c99]->alen     = 3;
  book->defs[0x00db9c99]->acts     = (Wire*) malloc(3 * sizeof(Wire));
  book->defs[0x00db9c99]->acts[ 0] = mkwire(0x10036e72,0xa0000001);
  book->defs[0x00db9c99]->acts[ 1] = mkwire(0x10028a67,0xa0000002);
  book->defs[0x00db9c99]->acts[ 2] = mkwire(0x10000019,0xa0000003);
  book->defs[0x00db9c99]->nlen     = 4;
  book->defs[0x00db9c99]->node     = (Node*) malloc(4 * sizeof(Node));
  book->defs[0x00db9c99]->node[ 0] = (Node) {0x40000003,0x50000001};
  book->defs[0x00db9c99]->node[ 1] = (Node) {0x50000002,0x50000000};
  book->defs[0x00db9c99]->node[ 2] = (Node) {0x50000003,0x40000001};
  book->defs[0x00db9c99]->node[ 3] = (Node) {0x40000000,0x40000002};
}

__host__ void boot(Net* net, u32 ref_id) {
  net->root = mkptr(REF, ref_id);
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
  //boot(cpu_net, 0x009b6ca4); // initial term

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
  do_global_expand(gpu_net, gpu_book);
  for (u32 tick = 0; tick < 128; ++tick) {
    do_global_rewrite(gpu_net, gpu_book, 16, tick, (tick / GROUP_LOG2) % 2);
  }
  do_global_expand(gpu_net, gpu_book);
  do_global_rewrite(gpu_net, gpu_book, 200000, 0, 0);
  cudaDeviceSynchronize();

  // Gets end time
  clock_gettime(CLOCK_MONOTONIC_RAW, &end);
  uint32_t delta_time = (end.tv_sec - start.tv_sec) * 1000 + (end.tv_nsec - start.tv_nsec) / 1000000;

  // Reads result back to cpu
  Net* norm = net_to_cpu(gpu_net);

  // Prints the output
  //print_tree(norm, norm->root);
  printf("\nNORMAL ~ rewrites=%llu\n======\n\n", norm->rwts);
  print_net(norm);
  printf("Time: %.3f s\n", ((double)delta_time) / 1000.0);
  printf("RPS : %.3f million\n", ((double)norm->rwts) / ((double)delta_time) / 1000.0);

  // Clears CPU memory
  net_free_on_gpu(gpu_net);
  book_free_on_gpu(gpu_book);

  // Clears GPU memory
  net_free_on_cpu(cpu_net);
  book_free_on_cpu(cpu_book);
  net_free_on_cpu(norm);

  return 0;
}
