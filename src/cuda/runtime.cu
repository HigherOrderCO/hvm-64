// NOTE: This file is adapted from hvm2.cu and must be kept in sync (for now)
// The only changes are that only kernel functions and related types are kept,
// host functions are removed, they are moved to the Rust host code.

//// Prelude

typedef unsigned char      uint8_t;
typedef unsigned short     uint16_t;
typedef unsigned int       uint32_t;
typedef unsigned long long uint64_t;

//// Runtime

typedef uint8_t u8;
typedef uint16_t u16;
typedef uint32_t u32;
typedef uint64_t u64;

// Configuration
// -------------

// This code is initially optimized for RTX 4090

// Bags dimensions (128x128 redex bags)
const u32 BAGS_WIDTH_L2  = 7;
const u32 BAGS_WIDTH     = 1 << BAGS_WIDTH_L2;
const u32 BAGS_HEIGHT_L2 = 7;
const u32 BAGS_HEIGHT    = 1 << BAGS_HEIGHT_L2;
const u32 BAGS_TOTAL_L2  = BAGS_WIDTH_L2 + BAGS_HEIGHT_L2;
const u32 BAGS_TOTAL     = 1 << BAGS_TOTAL_L2;

// Threads per Squad (4)
const u32 SQUAD_SIZE_L2 = 2;
const u32 SQUAD_SIZE    = 1 << SQUAD_SIZE_L2;

// Squads per Block (128)
const u32 GROUP_SIZE_L2 = BAGS_WIDTH_L2;
const u32 GROUP_SIZE    = 1 << GROUP_SIZE_L2;

// Threads per Block (512)
const u32 BLOCK_SIZE_L2 = GROUP_SIZE_L2 + SQUAD_SIZE_L2;
const u32 BLOCK_SIZE    = 1 << BLOCK_SIZE_L2;

// Heap Size (max total nodes = 256m = 2GB)
const u32 HEAP_SIZE_L2 = 28;
const u32 HEAP_SIZE    = 1 << HEAP_SIZE_L2;

// Jump Table (max book entries = 16m definitions)
const u32 JUMP_SIZE_L2 = 24;
const u32 JUMP_SIZE    = 1 << JUMP_SIZE_L2;

// Max Redexes per Interaction
const u32 MAX_NEW_REDEX = 16; // FIXME: use to check full rbags

// Local Attributes per Squad
const u32 SMEM_SIZE = 4; // local attributes

// Total Number of Squads
const u32 SQUAD_TOTAL_L2 = BAGS_TOTAL_L2;
const u32 SQUAD_TOTAL    = 1 << SQUAD_TOTAL_L2;

// Total Allocation Nodes per Squad
const u32 AREA_SIZE = HEAP_SIZE / SQUAD_TOTAL;

// Redexes per Redex Bag
const u32 RBAG_SIZE = 256;

// Total Redexes on All Bags
const u32 BAGS_SIZE = BAGS_TOTAL * RBAG_SIZE;

// Max Global Expansion Ptrs (1 per squad)
const u32 HEAD_SIZE_L2 = SQUAD_TOTAL_L2;
const u32 HEAD_SIZE    = 1 << HEAD_SIZE_L2;

// Max Local Expansion Ptrs per Squad
const u32 EXPANSIONS_PER_SQUAD = 16;

// Types
// -----

typedef u8  Tag; // pointer tag: 4-bit
typedef u32 Val; // pointer val: 28-bit

// Core terms
const Tag VR1 = 0x0; // variable to aux port 1
const Tag VR2 = 0x1; // variable to aux port 2
const Tag RD1 = 0x2; // redirect to aux port 1
const Tag RD2 = 0x3; // redirect to aux port 2
const Tag REF = 0x4; // lazy closed net
const Tag ERA = 0x5; // unboxed eraser
const Tag NUM = 0x6; // unboxed number
const Tag OP2 = 0x7; // numeric operation binary
const Tag OP1 = 0x8; // numeric operation unary
const Tag ITE = 0x9; // numeric if-then-else
const Tag CT0 = 0xA; // main port of con node 0
const Tag CT1 = 0xB; // main port of con node 1
const Tag CT2 = 0xC; // main port of con node 2
const Tag CT3 = 0xD; // main port of con node 3
const Tag CT4 = 0xE; // main port of con node 4
const Tag CT5 = 0xF; // main port of con node 5

// Special values
const u32 ROOT = 0x0 | VR2;  // pointer to root port
const u32 NONE = 0x00000000; // empty value, not allocated
const u32 GONE = 0xFFFFFFFE; // node has been moved to redex bag by paired thread
const u32 LOCK = 0xFFFFFFFF; // value taken by another thread, will be replaced soon
const u32 FAIL = 0xFFFFFFFF; // signals failure to allocate

// Unit types
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

// An interaction net
typedef struct {
  Wire* bags; // redex bags (active pairs)
  Node* heap; // memory buffer with all nodes
  Wire* head; // head expansion buffer
  u32*  jump; // book jump table
  u64   rwts; // number of rewrites performed
} Net;

// A unit local data
typedef struct {
  u32   tid;  // thread id (local)
  u32   gid;  // global id (global)
  u32   sid;  // squad id (local)
  u32   uid;  // squad id (global)
  u32   qid;  // quarter id (A1|A2|B1|B2)
  u32   port; // unit port (P1|P2)
  u64   rwts; // local rewrites performed
  u32   mask; // squad warp mask
  u32*  aloc; // where to alloc next node
  u32*  sm32; // shared 32-bit buffer
  u64*  sm64; // shared 64-bit buffer
  u64*  RBAG; // init of my redex bag
  u32*  rlen; // local redex bag length
  Wire* rbag; // local redex bag
} Unit;

// TermBook
// --------

__constant__ u32* BOOK;

typedef u32 Book; // stored in a flat buffer
/*
Book* init_book_on_gpu(u32* data, u32 size) {
  u32* gpu_book;
  cudaMalloc(&gpu_book, size * sizeof(u32));
  cudaMemcpy(gpu_book, data, size * sizeof(u32), cudaMemcpyHostToDevice);
  cudaMemcpyToSymbol(BOOK, &gpu_book, sizeof(u32*));
  return gpu_book;
}

void book_free_on_gpu(Book* gpu_book) {
  cudaFree(gpu_book);
}
 */
// Runtime
// -------

// Integer ceil division
__host__ __device__ inline u32 div(u32 a, u32 b) {
  return (a + b - 1) / b;
}

// Creates a new pointer
__host__ __device__ inline Ptr mkptr(Tag tag, Val val) {
  return (val << 4) | ((Val)tag);
}

// Gets the tag of a pointer
__host__ __device__ inline Tag tag(Ptr ptr) {
  return (Tag)(ptr & 0xF);
}

// Gets the value of a pointer
__host__ __device__ inline Val val(Ptr ptr) {
  return (Val)(ptr >> 4);
}

// Is this pointer a variable?
__host__ __device__ inline bool is_var(Ptr ptr) {
  return ptr != 0 && tag(ptr) >= VR1 && tag(ptr) <= VR2;
}

// Is this pointer a redirection?
__host__ __device__ inline bool is_red(Ptr ptr) {
  return tag(ptr) >= RD1 && tag(ptr) <= RD2;
}

// Is this pointer a constructor?
__host__ __device__ inline bool is_ctr(Ptr ptr) {
  return tag(ptr) >= CT0 && tag(ptr) < CT5; // FIXME: CT5 excluded
}

// Is this pointer an eraser?
__host__ __device__ inline bool is_era(Ptr ptr) {
  return tag(ptr) == ERA;
}

// Is this pointer a reference?
__host__ __device__ inline bool is_ref(Ptr ptr) {
  return tag(ptr) == REF;
}

// Is this pointer a main port?
__host__ __device__ inline bool is_pri(Ptr ptr) {
  return is_ctr(ptr) || is_era(ptr) || is_ref(ptr);
}

// Is this pointer carrying a location (that needs adjustment)?
__host__ __device__ inline bool has_loc(Ptr ptr) {
  return is_ctr(ptr) || is_var(ptr);
}

// Gets the target ref of a var or redirection pointer
__host__ __device__ inline Ptr* target(Net* net, Ptr ptr) {
  return &net->heap[val(ptr)].ports[ptr & 1];
}

// Traverses to the other side of a wire
__host__ __device__ inline Ptr enter(Net* net, Ptr ptr) {
  Ptr* ref = target(net, ptr);
  while (is_red(*ref)) {
    ptr = *ref;
    ref = target(net, ptr);
  }
  return ptr;
}

// Transforms a variable into a redirection
__host__ __device__ inline Ptr redir(Ptr ptr) {
  return mkptr(tag(ptr) + (is_var(ptr) ? 2 : 0), val(ptr));
}

// Transforms a redirection into a variable
__host__ __device__ inline Ptr undir(Ptr ptr) {
  return mkptr(tag(ptr) - (is_red(ptr) ? 2 : 0), val(ptr));
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
  return mknode(NONE, NONE);
}

// Checks if a node is nil
__host__ __device__ inline bool Node_is_nil(Node* node) {
  return node->ports[P1] == NONE && node->ports[P2] == NONE;
}

// Gets a reference to the index/port Ptr on the net
__device__ inline Ptr* at(Net* net, Val idx, Port port) {
  return &net->heap[idx].ports[port];
}

// Allocates one node in memory
__device__ u32 alloc(Unit *unit, Net *net, u32 size) {
  u32 size4 = div(size, (u32)4) * 4;
  u32 begin = unit->uid * AREA_SIZE;
  u32 space = 0;
  u32 index = *unit->aloc - (*unit->aloc % 4);
  for (u32 i = 0; i < 256; ++i) {
    Node node = net->heap[begin + index + unit->qid];
    bool null = Node_is_nil(&node);
    bool succ = __all_sync(unit->mask, null);
    index = (index + 4) % AREA_SIZE;
    space = succ && index > 0 ? space + 4 : 0;
    if (space == size4) {
      *unit->aloc = index;
      return (begin + index - space) % HEAP_SIZE;
    }
  }
  return FAIL;
}

// Gets the value of a ref; waits if taken.
__device__ inline Ptr take(Ptr* ref) {
  Ptr got = atomicExch((u32*)ref, LOCK);
  while (got == LOCK) {
    got = atomicExch((u32*)ref, LOCK);
  }
  return got;
}

// Attempts to replace 'exp' by 'neo', until it succeeds
__device__ inline bool replace(Ptr* ref, Ptr exp, Ptr neo) {
  Ptr got = atomicCAS((u32*)ref, exp, neo);
  while (got != exp) {
    got = atomicCAS((u32*)ref, exp, neo);
  }
  return true;
}

// Splits elements of two arrays evenly between each-other
// FIXME: it is desirable to split when size=1, to rotate out of starving squads
__device__ __noinline__ void split(u32 tid, u64* a_len, u64* a_arr, u64* b_len, u64* b_arr, u64 max_len) {
  __syncthreads();
  u64* A_len = *a_len < *b_len ? a_len : b_len;
  u64* B_len = *a_len < *b_len ? b_len : a_len;
  u64* A_arr = *a_len < *b_len ? a_arr : b_arr;
  u64* B_arr = *a_len < *b_len ? b_arr : a_arr;
  bool move  = *A_len + 1 < *B_len;
  u64  min   = *A_len;
  u64  max   = *B_len;
  __syncthreads();
  for (u64 t = 0; t < max_len / (SQUAD_SIZE * 2); ++t) {
    u64 i = min + t * (SQUAD_SIZE * 2) + tid;
    u64 value;
    if (move && i < max) {
      value = B_arr[i];
      B_arr[i] = 0;
    }
    __syncthreads();
    if (move && i < max) {
      if ((i - min) % 2 == 0) {
        A_arr[min + (t * (SQUAD_SIZE * 2) + tid) / 2] = value;
      } else {
        B_arr[min + (t * (SQUAD_SIZE * 2) + tid) / 2] = value;
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
__device__ Wire pop_redex(Unit* unit) {
  Wire redex = mkwire(0, 0);

  u32 rlen = *unit->rlen;
  if (rlen > 0 && rlen <= RBAG_SIZE - MAX_NEW_REDEX) {
    redex = unit->rbag[rlen-1];
  }
  __syncwarp(unit->mask);
  if (rlen > 0 && rlen <= RBAG_SIZE - MAX_NEW_REDEX) {
    unit->rbag[rlen-1] = mkwire(0, 0);
    *unit->rlen = rlen-1;
  }
  __syncwarp(unit->mask);

  if (unit->qid <= A2) {
    return mkwire(wire_lft(redex), wire_rgt(redex));
  } else {
    return mkwire(wire_rgt(redex), wire_lft(redex));
  }
}

// Puts a redex
__device__ void put_redex(Unit* unit, Ptr a_ptr, Ptr b_ptr) {
  // optimization: avoids pushing non-reactive redexes
  bool a_era = is_era(a_ptr);
  bool b_era = is_era(b_ptr);
  bool a_ref = is_ref(a_ptr);
  bool b_ref = is_ref(b_ptr);
  if ( a_era && b_era
    || a_ref && b_era
    || a_era && b_ref
    || a_ref && b_ref) {
    unit->rwts += 1;
    return;
  }

  // pushes redex to end of bag
  u32 index = atomicAdd(unit->rlen, 1);
  if (index < RBAG_SIZE - 1) {
    unit->rbag[index] = mkwire(a_ptr, b_ptr);
  } else {
    printf("ERROR: PUSHED TO FULL TBAG (NOT IMPLEMENTED YET)\n");
  }
}

// Adjusts a dereferenced pointer
__device__ inline Ptr adjust(Unit* unit, Ptr ptr, u32 delta) {
  return mkptr(tag(ptr), has_loc(ptr) ? val(ptr) + delta - 1 : val(ptr));
}

// Expands a reference
__device__ bool deref(Unit* unit, Net* net, Book* book, Ptr* ref, Ptr up) {
  // Assert ref is either a REF or NULL
  ref = ref != NULL && is_ref(*ref) ? ref : NULL;

  // Load definition
  const u32  jump = ref != NULL ? net->jump[val(*ref) & 0xFFFFFF] : 0;
  const u32  nlen = book[jump + 0];
  const u32  rlen = book[jump + 1];
  const u32* node = &book[jump + 2];
  const u32* acts = &book[jump + 2 + nlen * 2];

  // Allocate needed space
  u32 loc = FAIL;
  if (ref != NULL) {
    loc = alloc(unit, net, nlen - 1);
  }

  if (ref != NULL && loc != FAIL) {
    // Increment rewrite count.
    if (unit->qid == A1) {
      unit->rwts += 1;
    }

    // Load nodes, adjusted.
    for (u32 i = 0; i < div(nlen - 1, SQUAD_SIZE); ++i) {
      u32 idx = i * SQUAD_SIZE + unit->qid;
      if (idx < nlen - 1) {
        Ptr p1 = adjust(unit, node[2+idx*2+0], loc);
        Ptr p2 = adjust(unit, node[2+idx*2+1], loc);
        *at(net, loc + idx, P1) = p1;
        *at(net, loc + idx, P2) = p2;
      }
    }

    // Load redexes, adjusted.
    for (u32 i = 0; i < div(rlen, SQUAD_SIZE); ++i) {
      u32 idx = i * SQUAD_SIZE + unit->qid;
      if (idx < rlen) {
        Ptr p1 = adjust(unit, acts[idx*2+0], loc);
        Ptr p2 = adjust(unit, acts[idx*2+1], loc);
        put_redex(unit, p1, p2);
      }
    }

    // Load root, adjusted.
    *ref = adjust(unit, node[1], loc);

    // Link root.
    if (unit->qid == A1 && is_var(*ref)) {
      *target(net, *ref) = up;
    }
  }

  return ref == NULL || loc != FAIL;
}

// Rewrite
// -------

__device__ u32 interleave(u32 idx, u32 width, u32 height) {
  u32 old_row = idx / width;
  u32 old_col = idx % width;
  u32 new_row = old_col % height;
  u32 new_col = old_col / height + old_row * (width / height);
  return new_row * width + new_col;
}

// Local Squad Id (sid) to Global Squad Id (uid)
__device__ u32 sid_to_uid(u32 sid, bool flip) {
  return flip ? interleave(sid, BAGS_WIDTH, BAGS_HEIGHT) : sid;
}

__device__ Unit init_unit(Net* net, bool flip) {
  __shared__ u32 SMEM[GROUP_SIZE * SMEM_SIZE];
  __shared__ u32 ALOC[GROUP_SIZE];

  for (u32 i = 0; i < GROUP_SIZE * SMEM_SIZE / BLOCK_SIZE; ++i) {
    SMEM[i * BLOCK_SIZE + threadIdx.x] = 0;
  }
  __syncthreads();

  for (u32 i = 0; i < GROUP_SIZE / BLOCK_SIZE; ++i) {
    ALOC[i * BLOCK_SIZE + threadIdx.x] = 0;
  }
  __syncthreads();

  Unit unit;
  unit.tid  = threadIdx.x;
  unit.gid  = blockIdx.x * blockDim.x + unit.tid;
  unit.sid  = unit.gid / SQUAD_SIZE;
  unit.uid  = sid_to_uid(unit.sid, flip);
  unit.qid  = unit.tid % 4;
  unit.rwts = 0;
  unit.mask = ((1 << SQUAD_SIZE) - 1) << (unit.tid % 32 / SQUAD_SIZE * SQUAD_SIZE);
  unit.port = unit.tid % 2;
  unit.aloc = (u32*)(ALOC + unit.tid / SQUAD_SIZE); // locally cached
  unit.sm32 = (u32*)(SMEM + unit.tid / SQUAD_SIZE * SMEM_SIZE);
  unit.sm64 = (u64*)(SMEM + unit.tid / SQUAD_SIZE * SMEM_SIZE);
  unit.RBAG = net->bags + unit.uid * RBAG_SIZE;
  unit.rlen = (u32*)(unit.RBAG + 0); // TODO: cache locally
  unit.rbag = unit.RBAG + 1;
  *unit.aloc = 0; // TODO: randomize or persist

  return unit;
}

__device__ void save_unit(Unit* unit, Net* net) {
  if (unit->rwts > 0) {
    atomicAdd(&net->rwts, unit->rwts);
  }
}

__device__ void share_redexes(Unit* unit, Net* net, Book* book, u32 tick, bool flip) {
  u32  side  = ((unit->tid / SQUAD_SIZE) >> (BAGS_WIDTH_L2 - 1 - (tick % BAGS_WIDTH_L2))) & 1;
  u32  shift = (1 << (BAGS_WIDTH_L2 - 1)) >> (tick % BAGS_WIDTH_L2);
  u32  a_sid = unit->sid;
  u32  b_sid = side ? a_sid - shift : a_sid + shift;
  u32  a_uid = sid_to_uid(a_sid, flip);
  u32  b_uid = sid_to_uid(b_sid, flip);
  u64* a_len = net->bags + a_uid * RBAG_SIZE;
  u64* b_len = net->bags + b_uid * RBAG_SIZE;
  u32  sp_id = unit->tid % SQUAD_SIZE + side * SQUAD_SIZE;
  split(sp_id, a_len, a_len+1, b_len, b_len+1, RBAG_SIZE);
}

__device__ void atomic_join(Unit* unit, Net* net, Book* book, Ptr a_ptr, Ptr* a_ref, Ptr b_ptr) {
  while (true) {
    Ptr* ste_ref = target(net, b_ptr);
    Ptr  ste_ptr = *ste_ref;
    if (is_var(ste_ptr)) {
      Ptr* trg_ref = target(net, ste_ptr);
      Ptr  trg_ptr = atomicAdd(trg_ref, 0);
      if (is_red(trg_ptr)) {
        Ptr neo_ptr = undir(trg_ptr);
        Ptr updated = atomicCAS(ste_ref, ste_ptr, neo_ptr);
        if (updated == ste_ptr) {
          *trg_ref = 0;
          continue;
        }
      }
    }
    break;
  }
}

__device__ void atomic_link(Unit* unit, Net* net, Book* book, Ptr a_ptr, Ptr* a_ref, Ptr b_ptr) {
  while (true) {
    // Peek the target, which may not be owned by us.
    Ptr* t_ref = target(net, a_ptr);
    Ptr  t_ptr = atomicAdd(t_ref, 0);

    // If target is a redirection, clear and move forward.
    if (is_red(t_ptr)) {
      // We own the redirection, so we can mutate it.
      *t_ref = 0;
      a_ptr = t_ptr;
      continue;
    }

    // If target is a variable, try replacing it by the node.
    else if (is_var(t_ptr)) {
      // We don't own the var, so we must try replacing with a CAS.
      if (atomicCAS(t_ref, t_ptr, b_ptr) == t_ptr) {
        // Clear source location.
        *a_ref = 0;
        // Collect the orphaned backward path.
        t_ref = target(net, t_ptr);
        t_ptr = *t_ref;
        while (is_red(t_ptr)) {
          *t_ref = 0;
          t_ref = target(net, t_ptr);
          t_ptr = *t_ref;
        }
        return;
      }

      // If the CAS failed, the var changed, so we try again.
      continue;
    }

    // If it is a node, two threads will reach this branch.
    else if (is_pri(t_ptr) || is_ref(t_ptr) || t_ptr == GONE) {
      // Sort references, to avoid deadlocks.
      Ptr *x_ref = a_ref < t_ref ? a_ref : t_ref;
      Ptr *y_ref = a_ref < t_ref ? t_ref : a_ref;

      // Swap first reference by GONE placeholder.
      Ptr x_ptr = atomicExch(x_ref, GONE);

      // First to arrive creates a redex.
      if (x_ptr != GONE) {
        Ptr y_ptr = atomicExch(y_ref, GONE);
        put_redex(unit, x_ptr, y_ptr);
        return;

      // Second to arrive clears up the memory.
      } else {
        *x_ref = 0;
        replace(y_ref, GONE, 0);
        return;
      }
    }

    // If it is taken, we wait.
    else if (t_ptr == LOCK) {
      continue;
    }

    // Shouldn't be reached.
    else {
      return;
    }
  }
}

__device__ void atomic_subst(Unit* unit, Net* net, Book* book, Ptr a_ptr, Ptr a_dir, Ptr b_ptr, bool put) {
  Ptr* a_ref = target(net, a_dir);
  if (is_var(a_ptr)) {
    Ptr got = atomicCAS(target(net, a_ptr), a_dir, b_ptr);
    if (got == a_dir) {
      atomicExch(a_ref, NONE);
    } else if (is_var(b_ptr)) {
      atomicExch(a_ref, redir(b_ptr));
      atomic_join(unit, net, book, a_ptr, a_ref, redir(b_ptr));
    } else if (is_pri(b_ptr)) {
      atomicExch(a_ref, b_ptr);
      atomic_link(unit, net, book, a_ptr, a_ref, b_ptr);
    }
  } else if (is_pri(a_ptr) && is_pri(b_ptr)) {
    if (a_ptr < b_ptr || put) {
      put_redex(unit, b_ptr, a_ptr); // FIXME: swapping bloats rbag; why?
    }
    atomicExch(a_ref, NONE);
  } else {
    atomicExch(a_ref, NONE);
  }
}

__device__ void interact(Unit* unit, Net* net, Book* book) {
  // Pops a redex from local bag
  Wire redex = pop_redex(unit);
  Ptr  a_ptr = wire_lft(redex);
  Ptr  b_ptr = wire_rgt(redex);

  // Flag to abort in case of failure
  bool abort = false;

  // Dereferences
  Ptr* deref_ptr = NULL;
  if (is_ref(a_ptr) && is_ctr(b_ptr)) {
    deref_ptr = &a_ptr;
  }
  if (is_ref(b_ptr) && is_ctr(a_ptr)) {
    deref_ptr = &b_ptr;
  }
  if (!deref(unit, net, book, deref_ptr, NONE)) {
    abort = true;
  }

  // Defines type of interaction
  bool rewrite = a_ptr != 0 && b_ptr != 0;
  bool var_pri = rewrite && is_var(a_ptr) && is_pri(b_ptr) && unit->port == P1;
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
  Ptr  bk_ptr; // val to other aux port
  Ptr  mv_ptr; // val of ptr to send to other side
  u32  dp_loc; // duplication allocation index

  // If con_dup, alloc clones base index
  if (rewrite && con_dup) {
    dp_loc = alloc(unit, net, 4);
  }

  // Aborts if allocation failed
  if (rewrite && con_dup && dp_loc == FAIL) {
    abort = true;
  }

  // Reverts when abort=true
  if (rewrite && abort) {
    rewrite = false;
    put_redex(unit, a_ptr, b_ptr);
  }
  __syncwarp(unit->mask);

  // Inc rewrite count
  if (rewrite && unit->qid == A1) {
    unit->rwts += 1;
  }

  // Gets port here
  if (rewrite && (ctr_era || con_con || con_dup)) {
    ak_dir = mkptr(VR1 + unit->port, val(a_ptr));
    ak_ref = target(net, ak_dir);
    ak_ptr = take(ak_ref);
  }

  // Gets port there
  if (rewrite && (era_ctr || con_con || con_dup)) {
    bk_dir = mkptr(VR1 + unit->port, val(b_ptr));
    bk_ref = target(net, bk_dir);
  }

  // If era_ctr, send an erasure
  if (rewrite && era_ctr) {
    mv_ptr = mkptr(ERA, 0);
  }

  // If con_con, send a redirection
  if (rewrite && con_con) {
    mv_ptr = ak_ptr;
  }

  // If con_dup, create inner wires between clones
  if (rewrite && con_dup) {
    u32 cx_loc = dp_loc + unit->qid;
    u32 c1_loc = dp_loc + (unit->qid <= A2 ? 2 : 0);
    u32 c2_loc = dp_loc + (unit->qid <= A2 ? 3 : 1);
    atomicExch(target(net, mkptr(VR1, cx_loc)), mkptr(unit->port == P1 ? VR1 : VR2, c1_loc));
    atomicExch(target(net, mkptr(VR2, cx_loc)), mkptr(unit->port == P1 ? VR1 : VR2, c2_loc));
    mv_ptr = mkptr(tag(a_ptr), cx_loc);
  }
  __syncwarp(unit->mask);

  // Send ptr to other side
  if (rewrite && (era_ctr || con_con || con_dup)) {
    unit->sm32[unit->qid + (unit->qid <= A2 ? 2 : -2)] = mv_ptr;
  }
  __syncwarp(unit->mask);

  // Receive ptr from other side
  if (rewrite && (con_con || ctr_era || con_dup)) {
    bk_ptr = unit->sm32[unit->qid];
  }
  __syncwarp(unit->mask);

  // If var_pri, the var must be a deref root, so we just subst
  if (rewrite && var_pri && unit->port == P1) {
    atomicExch(target(net, a_ptr), b_ptr);
  }
  __syncwarp(unit->mask);

  // Substitutes
  if (rewrite && (con_con || ctr_era || con_dup)) {
    atomic_subst(unit, net, book, ak_ptr, ak_dir, bk_ptr, ctr_era || con_dup);
  }
  __syncwarp(unit->mask);
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
// __launch_bounds__(BLOCK_SIZE, 1)
extern "C" __global__ void global_rewrite(Net* net, Book* book, u32 repeat, u32 tick, bool flip) {
  // Initializes local vars
  Unit unit = init_unit(net, flip);

  // Performs interactions
  for (u32 turn = 0; turn < repeat; ++turn) {
    interact(&unit, net, book);
  }

  // Shares redexes with paired neighbor
  share_redexes(&unit, net, book, tick, flip);

  // When the work ends, sum stats
  save_unit(&unit, net);
}
/*
void do_global_rewrite(Net* net, Book* book, u32 repeat, u32 tick, bool flip) {
  global_rewrite<<<BAGS_HEIGHT, BLOCK_SIZE>>>(net, book, repeat, tick, flip);
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("CUDA error: %s\n", cudaGetErrorString(err));
  }
}
 */
// Expand
// ------

// Collects local expansion heads recursively
__device__ void expand(Unit* unit, Net* net, Book* book, Ptr dir, u32* len, u32* lhds) {
  Ptr ptr = *target(net, dir);
  if (is_ctr(ptr)) {
    expand(unit, net, book, mkptr(VR1, val(ptr)), len, lhds);
    expand(unit, net, book, mkptr(VR2, val(ptr)), len, lhds);
  } else if (is_red(ptr)) {
    expand(unit, net, book, ptr, len, lhds);
  } else if (is_ref(ptr) && *len < EXPANSIONS_PER_SQUAD) {
    lhds[(*len)++] = dir;
  }
}

// Takes an initial head location for each squad
extern "C" __global__ void global_expand_prepare(Net* net) {
  u32 uid = blockIdx.x * blockDim.x + threadIdx.x;

  // Traverses down
  u32 key = uid;
  Ptr dir = ROOT;
  Ptr ptr, *ref;
  for (u32 depth = 0; depth < BAGS_TOTAL_L2; ++depth) {
    dir = enter(net, dir);
    ref = target(net, dir);
    if (is_var(dir)) {
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
  if (is_var(dir)) {
    ptr = atomicExch(ref, LOCK);
  }

  // Stores ptr
  if (ptr != LOCK) {
    net->head[uid] = mkwire(dir, ptr);
  } else {
    net->head[uid] = mkwire(NONE, NONE);
  }

}

// Performs global expansion of heads
extern "C" __global__ void global_expand(Net* net, Book* book) {
  __shared__ u32 HEAD[GROUP_SIZE * EXPANSIONS_PER_SQUAD];

  for (u32 i = 0; i < GROUP_SIZE * EXPANSIONS_PER_SQUAD / BLOCK_SIZE; ++i) {
    HEAD[i * BLOCK_SIZE + threadIdx.x] = 0;
  }
  __syncthreads();

  Unit unit = init_unit(net, 0);

  u32* head = HEAD + unit.tid / SQUAD_SIZE * EXPANSIONS_PER_SQUAD;

  Wire got = net->head[unit.uid];
  Ptr  dir = wire_lft(got);
  Ptr* ref = target(net, dir);
  Ptr  ptr = wire_rgt(got);

  if (unit.qid == A1 && ptr != NONE) {
    *ref = ptr;
  }
  __syncthreads();

  u32 len = 0;
  if (unit.qid == A1 && ptr != NONE) {
    expand(&unit, net, book, dir, &len, head);
  }
  __syncthreads();

  for (u32 i = 0; i < EXPANSIONS_PER_SQUAD; ++i) {
    Ptr  dir = head[i];
    Ptr* ref = target(net, dir);
    if (!deref(&unit, net, book, ref, dir)) {
      printf("ERROR: DEREF FAILED ON EXPAND (NOT IMPLEMENTED YET)\n");
    }
  }
  __syncthreads();

  save_unit(&unit, net);
}
/*
// Performs a global head expansion (1 deref per bag)
void do_global_expand(Net* net, Book* book) {
  global_expand_prepare<<<BAGS_HEIGHT, GROUP_SIZE>>>(net);
  global_expand<<<BAGS_HEIGHT, BLOCK_SIZE>>>(net, book);
}

// Host<->Device
// -------------

__host__ Net* mknet(u32 root_fn, u32* jump_data, u32 jump_data_size) {
  Net* net  = (Net*)malloc(sizeof(Net));
  net->rwts = 0;
  net->bags = (Wire*)malloc(BAGS_SIZE * sizeof(Wire));
  net->heap = (Node*)malloc(HEAP_SIZE * sizeof(Node));
  net->head = (Wire*)malloc(HEAD_SIZE * sizeof(Wire));
  net->jump = (u32*) malloc(JUMP_SIZE * sizeof(u32));
  memset(net->bags, 0, BAGS_SIZE * sizeof(Wire));
  memset(net->heap, 0, HEAP_SIZE * sizeof(Node));
  memset(net->head, 0, HEAD_SIZE * sizeof(Wire));
  memset(net->jump, 0, JUMP_SIZE * sizeof(u32));
  *target(net, ROOT) = mkptr(REF, root_fn);
  for (u32 i = 0; i < jump_data_size / 2; ++i) {
    net->jump[jump_data[i*2+0]] = jump_data[i*2+1];
  }
  return net;
}

__host__ Net* net_to_gpu(Net* host_net) {
  // Allocate memory on the device for the Net object, and its data
  Net*  device_net;
  Wire* device_bags;
  Node* device_heap;
  Wire* device_head;
  u32*  device_jump;

  cudaMalloc((void**)&device_net, sizeof(Net));
  cudaMalloc((void**)&device_bags, BAGS_SIZE * sizeof(Wire));
  cudaMalloc((void**)&device_heap, HEAP_SIZE * sizeof(Node));
  cudaMalloc((void**)&device_head, HEAD_SIZE * sizeof(Wire));
  cudaMalloc((void**)&device_jump, JUMP_SIZE * sizeof(u32));

  // Copy the host data to the device memory
  cudaMemcpy(device_bags, host_net->bags, BAGS_SIZE * sizeof(Wire), cudaMemcpyHostToDevice);
  cudaMemcpy(device_heap, host_net->heap, HEAP_SIZE * sizeof(Node), cudaMemcpyHostToDevice);
  cudaMemcpy(device_head, host_net->head, HEAD_SIZE * sizeof(Wire), cudaMemcpyHostToDevice);
  cudaMemcpy(device_jump, host_net->jump, JUMP_SIZE * sizeof(u32), cudaMemcpyHostToDevice);

  // Create a temporary host Net object with device pointers
  Net temp_net  = *host_net;
  temp_net.bags = device_bags;
  temp_net.heap = device_heap;
  temp_net.head = device_head;
  temp_net.jump = device_jump;

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
  host_net->heap = (Node*)malloc(HEAP_SIZE * sizeof(Node));
  host_net->head = (Wire*)malloc(HEAD_SIZE * sizeof(Wire));
  host_net->jump = (u32*) malloc(JUMP_SIZE * sizeof(u32));

  // Retrieve the device pointers for data
  Wire* device_bags;
  Node* device_heap;
  Wire* device_head;
  u32*  device_jump;
  cudaMemcpy(&device_bags, &(device_net->bags), sizeof(Wire*), cudaMemcpyDeviceToHost);
  cudaMemcpy(&device_heap, &(device_net->heap), sizeof(Node*), cudaMemcpyDeviceToHost);
  cudaMemcpy(&device_head, &(device_net->head), sizeof(Wire*), cudaMemcpyDeviceToHost);
  cudaMemcpy(&device_jump, &(device_net->jump), sizeof(u32*), cudaMemcpyDeviceToHost);

  // Copy the device data to the host memory
  cudaMemcpy(host_net->bags, device_bags, BAGS_SIZE * sizeof(Wire), cudaMemcpyDeviceToHost);
  cudaMemcpy(host_net->heap, device_heap, HEAP_SIZE * sizeof(Node), cudaMemcpyDeviceToHost);
  cudaMemcpy(host_net->head, device_head, HEAD_SIZE * sizeof(Wire), cudaMemcpyDeviceToHost);
  cudaMemcpy(host_net->jump, device_jump, JUMP_SIZE * sizeof(u32),  cudaMemcpyDeviceToHost);

  return host_net;
}

__host__ void net_free_on_gpu(Net* device_net) {
  // Retrieve the device pointers for data
  Wire* device_bags;
  Node* device_heap;
  Wire* device_head;
  u32*  device_jump;
  cudaMemcpy(&device_bags, &(device_net->bags), sizeof(Wire*), cudaMemcpyDeviceToHost);
  cudaMemcpy(&device_heap, &(device_net->heap), sizeof(Node*), cudaMemcpyDeviceToHost);
  cudaMemcpy(&device_head, &(device_net->head), sizeof(Wire*), cudaMemcpyDeviceToHost);
  cudaMemcpy(&device_jump, &(device_net->jump), sizeof(u32*),  cudaMemcpyDeviceToHost);

  // Free the device memory
  cudaFree(device_bags);
  cudaFree(device_heap);
  cudaFree(device_head);
  cudaFree(device_jump);
  cudaFree(device_net);
}

__host__ void net_free_on_cpu(Net* host_net) {
  free(host_net->bags);
  free(host_net->heap);
  free(host_net->head);
  free(host_net->jump);
  free(host_net);
}

// Debugging
// ---------

__host__ const char* show_ptr(Ptr ptr, u32 slot) {
  static char buffer[8][20];
  if (ptr == NONE) {
    strcpy(buffer[slot], "           ");
    return buffer[slot];
  } else if (ptr == LOCK) {
    strcpy(buffer[slot], "[LOCK.....]");
    return buffer[slot];
  } else {
    const char* tag_str = NULL;
    switch (tag(ptr)) {
      case VR1: tag_str = "VR1"; break;
      case VR2: tag_str = "VR2"; break;
      case RD1: tag_str = "RD1"; break;
      case RD2: tag_str = "RD2"; break;
      case REF: tag_str = "REF"; break;
      case ERA: tag_str = "ERA"; break;
      case NUM: tag_str = "NUM"; break;
      case OP2: tag_str = "OP2"; break;
      case OP1: tag_str = "OP1"; break;
      case ITE: tag_str = "ITE"; break;
      case CT0: tag_str = "CT0"; break;
      case CT1: tag_str = "CT1"; break;
      case CT2: tag_str = "CT2"; break;
      case CT3: tag_str = "CT3"; break;
      case CT4: tag_str = "CT4"; break;
      case CT5: tag_str = "CT5"; break;
      default : tag_str = "???"; break;
    }
    snprintf(buffer[slot], sizeof(buffer[slot]), "%s:%07X", tag_str, val(ptr));
    return buffer[slot];
  }
}

// Prints a net in hexadecimal, limited to a given size
void print_net(Net* net) {
  printf("Bags:\n");
  for (u32 i = 0; i < BAGS_SIZE; ++i) {
    if (i % RBAG_SIZE == 0 && net->bags[i] > 0) {
      printf("- [%07X] LEN=%llu\n", i, net->bags[i]);
    } else if (i % RBAG_SIZE >= 1) {
      //Ptr a = wire_lft(net->bags[i]);
      //Ptr b = wire_rgt(net->bags[i]);
      //if (a != 0 || b != 0) {
        //printf("- [%07X] %s %s\n", i, show_ptr(a,0), show_ptr(b,1));
      //}
    }
  }
  //printf("Heap:\n");
  //for (u32 i = 0; i < HEAP_SIZE; ++i) {
    //Ptr a = net->heap[i].ports[P1];
    //Ptr b = net->heap[i].ports[P2];
    //if (a != 0 || b != 0) {
      //printf("- [%07X] %s %s\n", i, show_ptr(a,0), show_ptr(b,1));
    //}
  //}
  printf("Rwts: %llu\n", net->rwts);
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
      case RD1: case RD2:
        print_tree_go(net, *target(net, ptr), var_ids);
        break;
      default:
        printf("(%d ", tag(ptr) - CT0);
        print_tree_go(net, net->heap[val(ptr)].ports[P1], var_ids);
        printf(" ");
        print_tree_go(net, net->heap[val(ptr)].ports[P2], var_ids);
        printf(")");
    }
  }
}

__host__ void print_tree(Net* net, Ptr ptr) {
  var_ids.size = 0;
  print_tree_go(net, ptr, &var_ids);
  printf("\n");
}

// Book
// ----

const u32 F_E = 0xe;
const u32 F_F = 0xf;
const u32 F_I = 0x12;
const u32 F_O = 0x18;
const u32 F_S = 0x1c;
const u32 F_T = 0x1d;
const u32 F_Z = 0x23;
const u32 F_af = 0x929;
const u32 F_c0 = 0x980;
const u32 F_c1 = 0x981;
const u32 F_c2 = 0x982;
const u32 F_c3 = 0x983;
const u32 F_c4 = 0x984;
const u32 F_c5 = 0x985;
const u32 F_c6 = 0x986;
const u32 F_c7 = 0x987;
const u32 F_c8 = 0x988;
const u32 F_c9 = 0x989;
const u32 F_id = 0xb27;
const u32 F_k0 = 0xb80;
const u32 F_k1 = 0xb81;
const u32 F_k2 = 0xb82;
const u32 F_k3 = 0xb83;
const u32 F_k4 = 0xb84;
const u32 F_k5 = 0xb85;
const u32 F_k6 = 0xb86;
const u32 F_k7 = 0xb87;
const u32 F_k8 = 0xb88;
const u32 F_k9 = 0xb89;
const u32 F_afS = 0x24a5c;
const u32 F_afZ = 0x24a63;
const u32 F_and = 0x24c67;
const u32 F_brn = 0x25d71;
const u32 F_c10 = 0x26040;
const u32 F_c11 = 0x26041;
const u32 F_c12 = 0x26042;
const u32 F_c13 = 0x26043;
const u32 F_c14 = 0x26044;
const u32 F_c15 = 0x26045;
const u32 F_c16 = 0x26046;
const u32 F_c17 = 0x26047;
const u32 F_c18 = 0x26048;
const u32 F_c19 = 0x26049;
const u32 F_c20 = 0x26080;
const u32 F_c21 = 0x26081;
const u32 F_c22 = 0x26082;
const u32 F_c23 = 0x26083;
const u32 F_c24 = 0x26084;
const u32 F_c25 = 0x26085;
const u32 F_c26 = 0x26086;
const u32 F_c_s = 0x26fb6;
const u32 F_c_z = 0x26fbd;
const u32 F_dec = 0x27a26;
const u32 F_ex0 = 0x28ec0;
const u32 F_ex1 = 0x28ec1;
const u32 F_ex2 = 0x28ec2;
const u32 F_ex3 = 0x28ec3;
const u32 F_ex4 = 0x28ec4;
const u32 F_ex5 = 0x28ec5;
const u32 F_g_s = 0x2afb6;
const u32 F_g_z = 0x2afbd;
const u32 F_k10 = 0x2e040;
const u32 F_k11 = 0x2e041;
const u32 F_k12 = 0x2e042;
const u32 F_k13 = 0x2e043;
const u32 F_k14 = 0x2e044;
const u32 F_k15 = 0x2e045;
const u32 F_k16 = 0x2e046;
const u32 F_k17 = 0x2e047;
const u32 F_k18 = 0x2e048;
const u32 F_k19 = 0x2e049;
const u32 F_k20 = 0x2e080;
const u32 F_k21 = 0x2e081;
const u32 F_k22 = 0x2e082;
const u32 F_k23 = 0x2e083;
const u32 F_k24 = 0x2e084;
const u32 F_low = 0x2fcba;
const u32 F_mul = 0x30e2f;
const u32 F_nid = 0x31b27;
const u32 F_not = 0x31cb7;
const u32 F_run = 0x35e31;
const u32 F_brnS = 0x975c5c;
const u32 F_brnZ = 0x975c63;
const u32 F_decI = 0x9e8992;
const u32 F_decO = 0x9e8998;
const u32 F_lowI = 0xbf2e92;
const u32 F_lowO = 0xbf2e98;
const u32 F_nidS = 0xc6c9dc;
const u32 F_runI = 0xd78c52;
const u32 F_runO = 0xd78c58;

u32 BOOK_DATA[] = {
  // @E
  // .nlen
  0x00000004,
  // .rlen
  0x00000000,
  // .node
  0x00000000, 0x0000001A,  0x00000005, 0x0000002A,  0x00000005, 0x0000003A,  0x00000031, 0x00000030,
  // .rdex
  // @F
  // .nlen
  0x00000003,
  // .rlen
  0x00000000,
  // .node
  0x00000000, 0x0000001A,  0x00000005, 0x0000002A,  0x00000021, 0x00000020,
  // .rdex
  // @I
  // .nlen
  0x00000006,
  // .rlen
  0x00000000,
  // .node
  0x00000000, 0x0000001A,  0x00000040, 0x0000002A,  0x00000005, 0x0000003A,  0x0000004A, 0x0000005A,
  0x00000010, 0x00000051,  0x00000005, 0x00000041,
  // .rdex
  // @O
  // .nlen
  0x00000006,
  // .rlen
  0x00000000,
  // .node
  0x00000000, 0x0000001A,  0x00000030, 0x0000002A,  0x0000003A, 0x0000004A,  0x00000010, 0x00000051,
  0x00000005, 0x0000005A,  0x00000005, 0x00000031,
  // .rdex
  // @S
  // .nlen
  0x00000005,
  // .rlen
  0x00000000,
  // .node
  0x00000000, 0x0000001A,  0x00000030, 0x0000002A,  0x0000003A, 0x0000004A,  0x00000010, 0x00000041,
  0x00000005, 0x00000031,
  // .rdex
  // @T
  // .nlen
  0x00000003,
  // .rlen
  0x00000000,
  // .node
  0x00000000, 0x0000001A,  0x00000021, 0x0000002A,  0x00000005, 0x00000010,
  // .rdex
  // @Z
  // .nlen
  0x00000003,
  // .rlen
  0x00000000,
  // .node
  0x00000000, 0x0000001A,  0x00000005, 0x0000002A,  0x00000021, 0x00000020,
  // .rdex
  // @af
  // .nlen
  0x00000004,
  // .rlen
  0x00000000,
  // .node
  0x00000000, 0x0000001A,  0x0000002A, 0x00000031,  0x0024A5C4, 0x0000003A,  0x0024A634, 0x00000011,
  // .rdex
  // @c0
  // .nlen
  0x00000003,
  // .rlen
  0x00000000,
  // .node
  0x00000000, 0x0000001A,  0x00000005, 0x0000002A,  0x00000021, 0x00000020,
  // .rdex
  // @c1
  // .nlen
  0x00000004,
  // .rlen
  0x00000000,
  // .node
  0x00000000, 0x0000001A,  0x0000002A, 0x0000003A,  0x00000030, 0x00000031,  0x00000020, 0x00000021,
  // .rdex
  // @c2
  // .nlen
  0x00000006,
  // .rlen
  0x00000000,
  // .node
  0x00000000, 0x0000001A,  0x0000002B, 0x0000005A,  0x0000003A, 0x0000004A,  0x00000050, 0x00000040,
  0x00000031, 0x00000051,  0x00000030, 0x00000041,
  // .rdex
  // @c3
  // .nlen
  0x00000008,
  // .rlen
  0x00000000,
  // .node
  0x00000000, 0x0000001A,  0x0000002B, 0x0000007A,  0x0000003B, 0x0000006A,  0x0000004A, 0x0000005A,
  0x00000070, 0x00000050,  0x00000041, 0x00000060,  0x00000051, 0x00000071,  0x00000040, 0x00000061,
  // .rdex
  // @c4
  // .nlen
  0x0000000A,
  // .rlen
  0x00000000,
  // .node
  0x00000000, 0x0000001A,  0x0000002B, 0x0000009A,  0x0000003B, 0x0000008A,  0x0000004B, 0x0000007A,
  0x0000005A, 0x0000006A,  0x00000090, 0x00000060,  0x00000051, 0x00000070,  0x00000061, 0x00000080,
  0x00000071, 0x00000091,  0x00000050, 0x00000081,
  // .rdex
  // @c5
  // .nlen
  0x0000000C,
  // .rlen
  0x00000000,
  // .node
  0x00000000, 0x0000001A,  0x0000002B, 0x000000BA,  0x0000003B, 0x000000AA,  0x0000004B, 0x0000009A,
  0x0000005B, 0x0000008A,  0x0000006A, 0x0000007A,  0x000000B0, 0x00000070,  0x00000061, 0x00000080,
  0x00000071, 0x00000090,  0x00000081, 0x000000A0,  0x00000091, 0x000000B1,  0x00000060, 0x000000A1,
  // .rdex
  // @c6
  // .nlen
  0x0000000E,
  // .rlen
  0x00000000,
  // .node
  0x00000000, 0x0000001A,  0x0000002B, 0x000000DA,  0x0000003B, 0x000000CA,  0x0000004B, 0x000000BA,
  0x0000005B, 0x000000AA,  0x0000006B, 0x0000009A,  0x0000007A, 0x0000008A,  0x000000D0, 0x00000080,
  0x00000071, 0x00000090,  0x00000081, 0x000000A0,  0x00000091, 0x000000B0,  0x000000A1, 0x000000C0,
  0x000000B1, 0x000000D1,  0x00000070, 0x000000C1,
  // .rdex
  // @c7
  // .nlen
  0x00000010,
  // .rlen
  0x00000000,
  // .node
  0x00000000, 0x0000001A,  0x0000002B, 0x000000FA,  0x0000003B, 0x000000EA,  0x0000004B, 0x000000DA,
  0x0000005B, 0x000000CA,  0x0000006B, 0x000000BA,  0x0000007B, 0x000000AA,  0x0000008A, 0x0000009A,
  0x000000F0, 0x00000090,  0x00000081, 0x000000A0,  0x00000091, 0x000000B0,  0x000000A1, 0x000000C0,
  0x000000B1, 0x000000D0,  0x000000C1, 0x000000E0,  0x000000D1, 0x000000F1,  0x00000080, 0x000000E1,
  // .rdex
  // @c8
  // .nlen
  0x00000012,
  // .rlen
  0x00000000,
  // .node
  0x00000000, 0x0000001A,  0x0000002B, 0x0000011A,  0x0000003B, 0x0000010A,  0x0000004B, 0x000000FA,
  0x0000005B, 0x000000EA,  0x0000006B, 0x000000DA,  0x0000007B, 0x000000CA,  0x0000008B, 0x000000BA,
  0x0000009A, 0x000000AA,  0x00000110, 0x000000A0,  0x00000091, 0x000000B0,  0x000000A1, 0x000000C0,
  0x000000B1, 0x000000D0,  0x000000C1, 0x000000E0,  0x000000D1, 0x000000F0,  0x000000E1, 0x00000100,
  0x000000F1, 0x00000111,  0x00000090, 0x00000101,
  // .rdex
  // @c9
  // .nlen
  0x00000014,
  // .rlen
  0x00000000,
  // .node
  0x00000000, 0x0000001A,  0x0000002B, 0x0000013A,  0x0000003B, 0x0000012A,  0x0000004B, 0x0000011A,
  0x0000005B, 0x0000010A,  0x0000006B, 0x000000FA,  0x0000007B, 0x000000EA,  0x0000008B, 0x000000DA,
  0x0000009B, 0x000000CA,  0x000000AA, 0x000000BA,  0x00000130, 0x000000B0,  0x000000A1, 0x000000C0,
  0x000000B1, 0x000000D0,  0x000000C1, 0x000000E0,  0x000000D1, 0x000000F0,  0x000000E1, 0x00000100,
  0x000000F1, 0x00000110,  0x00000101, 0x00000120,  0x00000111, 0x00000131,  0x000000A0, 0x00000121,
  // .rdex
  // @id
  // .nlen
  0x00000002,
  // .rlen
  0x00000000,
  // .node
  0x00000000, 0x0000001A,  0x00000011, 0x00000010,
  // .rdex
  // @k0
  // .nlen
  0x00000003,
  // .rlen
  0x00000000,
  // .node
  0x00000000, 0x0000001A,  0x00000005, 0x0000002A,  0x00000021, 0x00000020,
  // .rdex
  // @k1
  // .nlen
  0x00000004,
  // .rlen
  0x00000000,
  // .node
  0x00000000, 0x0000001A,  0x0000002A, 0x0000003A,  0x00000030, 0x00000031,  0x00000020, 0x00000021,
  // .rdex
  // @k2
  // .nlen
  0x00000006,
  // .rlen
  0x00000000,
  // .node
  0x00000000, 0x0000001A,  0x0000002C, 0x0000005A,  0x0000003A, 0x0000004A,  0x00000050, 0x00000040,
  0x00000031, 0x00000051,  0x00000030, 0x00000041,
  // .rdex
  // @k3
  // .nlen
  0x00000008,
  // .rlen
  0x00000000,
  // .node
  0x00000000, 0x0000001A,  0x0000002C, 0x0000007A,  0x0000003C, 0x0000006A,  0x0000004A, 0x0000005A,
  0x00000070, 0x00000050,  0x00000041, 0x00000060,  0x00000051, 0x00000071,  0x00000040, 0x00000061,
  // .rdex
  // @k4
  // .nlen
  0x0000000A,
  // .rlen
  0x00000000,
  // .node
  0x00000000, 0x0000001A,  0x0000002C, 0x0000009A,  0x0000003C, 0x0000008A,  0x0000004C, 0x0000007A,
  0x0000005A, 0x0000006A,  0x00000090, 0x00000060,  0x00000051, 0x00000070,  0x00000061, 0x00000080,
  0x00000071, 0x00000091,  0x00000050, 0x00000081,
  // .rdex
  // @k5
  // .nlen
  0x0000000C,
  // .rlen
  0x00000000,
  // .node
  0x00000000, 0x0000001A,  0x0000002C, 0x000000BA,  0x0000003C, 0x000000AA,  0x0000004C, 0x0000009A,
  0x0000005C, 0x0000008A,  0x0000006A, 0x0000007A,  0x000000B0, 0x00000070,  0x00000061, 0x00000080,
  0x00000071, 0x00000090,  0x00000081, 0x000000A0,  0x00000091, 0x000000B1,  0x00000060, 0x000000A1,
  // .rdex
  // @k6
  // .nlen
  0x0000000E,
  // .rlen
  0x00000000,
  // .node
  0x00000000, 0x0000001A,  0x0000002C, 0x000000DA,  0x0000003C, 0x000000CA,  0x0000004C, 0x000000BA,
  0x0000005C, 0x000000AA,  0x0000006C, 0x0000009A,  0x0000007A, 0x0000008A,  0x000000D0, 0x00000080,
  0x00000071, 0x00000090,  0x00000081, 0x000000A0,  0x00000091, 0x000000B0,  0x000000A1, 0x000000C0,
  0x000000B1, 0x000000D1,  0x00000070, 0x000000C1,
  // .rdex
  // @k7
  // .nlen
  0x00000010,
  // .rlen
  0x00000000,
  // .node
  0x00000000, 0x0000001A,  0x0000002C, 0x000000FA,  0x0000003C, 0x000000EA,  0x0000004C, 0x000000DA,
  0x0000005C, 0x000000CA,  0x0000006C, 0x000000BA,  0x0000007C, 0x000000AA,  0x0000008A, 0x0000009A,
  0x000000F0, 0x00000090,  0x00000081, 0x000000A0,  0x00000091, 0x000000B0,  0x000000A1, 0x000000C0,
  0x000000B1, 0x000000D0,  0x000000C1, 0x000000E0,  0x000000D1, 0x000000F1,  0x00000080, 0x000000E1,
  // .rdex
  // @k8
  // .nlen
  0x00000012,
  // .rlen
  0x00000000,
  // .node
  0x00000000, 0x0000001A,  0x0000002C, 0x0000011A,  0x0000003C, 0x0000010A,  0x0000004C, 0x000000FA,
  0x0000005C, 0x000000EA,  0x0000006C, 0x000000DA,  0x0000007C, 0x000000CA,  0x0000008C, 0x000000BA,
  0x0000009A, 0x000000AA,  0x00000110, 0x000000A0,  0x00000091, 0x000000B0,  0x000000A1, 0x000000C0,
  0x000000B1, 0x000000D0,  0x000000C1, 0x000000E0,  0x000000D1, 0x000000F0,  0x000000E1, 0x00000100,
  0x000000F1, 0x00000111,  0x00000090, 0x00000101,
  // .rdex
  // @k9
  // .nlen
  0x00000014,
  // .rlen
  0x00000000,
  // .node
  0x00000000, 0x0000001A,  0x0000002C, 0x0000013A,  0x0000003C, 0x0000012A,  0x0000004C, 0x0000011A,
  0x0000005C, 0x0000010A,  0x0000006C, 0x000000FA,  0x0000007C, 0x000000EA,  0x0000008C, 0x000000DA,
  0x0000009C, 0x000000CA,  0x000000AA, 0x000000BA,  0x00000130, 0x000000B0,  0x000000A1, 0x000000C0,
  0x000000B1, 0x000000D0,  0x000000C1, 0x000000E0,  0x000000D1, 0x000000F0,  0x000000E1, 0x00000100,
  0x000000F1, 0x00000110,  0x00000101, 0x00000120,  0x00000111, 0x00000131,  0x000000A0, 0x00000121,
  // .rdex
  // @afS
  // .nlen
  0x00000007,
  // .rlen
  0x00000003,
  // .node
  0x00000000, 0x0000001A,  0x0000002B, 0x00000051,  0x00000060, 0x00000030,  0x00000021, 0x00000050,
  0x00000061, 0x0000005A,  0x00000031, 0x00000011,  0x00000020, 0x00000040,
  // .rdex
  0x0000003A, 0x00009294,  0x0000004A, 0x0024C674,  0x0000006A, 0x00009294,
  // @afZ
  // .nlen
  0x00000001,
  // .rlen
  0x00000000,
  // .node
  0x00000000, 0x000001D4,
  // .rdex
  // @and
  // .nlen
  0x0000000A,
  // .rlen
  0x00000000,
  // .node
  0x00000000, 0x0000001A,  0x0000002A, 0x00000061,  0x0000003A, 0x0000006A,  0x0000004A, 0x00000051,
  0x000001D4, 0x0000005A,  0x000000F4, 0x00000031,  0x0000007A, 0x00000011,  0x0000008A, 0x00000091,
  0x000000F4, 0x0000009A,  0x000000F4, 0x00000071,
  // .rdex
  // @brn
  // .nlen
  0x00000004,
  // .rlen
  0x00000000,
  // .node
  0x00000000, 0x0000001A,  0x0000002A, 0x00000031,  0x0975C5C4, 0x0000003A,  0x0975C634, 0x00000011,
  // .rdex
  // @c10
  // .nlen
  0x00000016,
  // .rlen
  0x00000000,
  // .node
  0x00000000, 0x0000001A,  0x0000002B, 0x0000015A,  0x0000003B, 0x0000014A,  0x0000004B, 0x0000013A,
  0x0000005B, 0x0000012A,  0x0000006B, 0x0000011A,  0x0000007B, 0x0000010A,  0x0000008B, 0x000000FA,
  0x0000009B, 0x000000EA,  0x000000AB, 0x000000DA,  0x000000BA, 0x000000CA,  0x00000150, 0x000000C0,
  0x000000B1, 0x000000D0,  0x000000C1, 0x000000E0,  0x000000D1, 0x000000F0,  0x000000E1, 0x00000100,
  0x000000F1, 0x00000110,  0x00000101, 0x00000120,  0x00000111, 0x00000130,  0x00000121, 0x00000140,
  0x00000131, 0x00000151,  0x000000B0, 0x00000141,
  // .rdex
  // @c11
  // .nlen
  0x00000018,
  // .rlen
  0x00000000,
  // .node
  0x00000000, 0x0000001A,  0x0000002B, 0x0000017A,  0x0000003B, 0x0000016A,  0x0000004B, 0x0000015A,
  0x0000005B, 0x0000014A,  0x0000006B, 0x0000013A,  0x0000007B, 0x0000012A,  0x0000008B, 0x0000011A,
  0x0000009B, 0x0000010A,  0x000000AB, 0x000000FA,  0x000000BB, 0x000000EA,  0x000000CA, 0x000000DA,
  0x00000170, 0x000000D0,  0x000000C1, 0x000000E0,  0x000000D1, 0x000000F0,  0x000000E1, 0x00000100,
  0x000000F1, 0x00000110,  0x00000101, 0x00000120,  0x00000111, 0x00000130,  0x00000121, 0x00000140,
  0x00000131, 0x00000150,  0x00000141, 0x00000160,  0x00000151, 0x00000171,  0x000000C0, 0x00000161,
  // .rdex
  // @c12
  // .nlen
  0x0000001A,
  // .rlen
  0x00000000,
  // .node
  0x00000000, 0x0000001A,  0x0000002B, 0x0000019A,  0x0000003B, 0x0000018A,  0x0000004B, 0x0000017A,
  0x0000005B, 0x0000016A,  0x0000006B, 0x0000015A,  0x0000007B, 0x0000014A,  0x0000008B, 0x0000013A,
  0x0000009B, 0x0000012A,  0x000000AB, 0x0000011A,  0x000000BB, 0x0000010A,  0x000000CB, 0x000000FA,
  0x000000DA, 0x000000EA,  0x00000190, 0x000000E0,  0x000000D1, 0x000000F0,  0x000000E1, 0x00000100,
  0x000000F1, 0x00000110,  0x00000101, 0x00000120,  0x00000111, 0x00000130,  0x00000121, 0x00000140,
  0x00000131, 0x00000150,  0x00000141, 0x00000160,  0x00000151, 0x00000170,  0x00000161, 0x00000180,
  0x00000171, 0x00000191,  0x000000D0, 0x00000181,
  // .rdex
  // @c13
  // .nlen
  0x0000001C,
  // .rlen
  0x00000000,
  // .node
  0x00000000, 0x0000001A,  0x0000002B, 0x000001BA,  0x0000003B, 0x000001AA,  0x0000004B, 0x0000019A,
  0x0000005B, 0x0000018A,  0x0000006B, 0x0000017A,  0x0000007B, 0x0000016A,  0x0000008B, 0x0000015A,
  0x0000009B, 0x0000014A,  0x000000AB, 0x0000013A,  0x000000BB, 0x0000012A,  0x000000CB, 0x0000011A,
  0x000000DB, 0x0000010A,  0x000000EA, 0x000000FA,  0x000001B0, 0x000000F0,  0x000000E1, 0x00000100,
  0x000000F1, 0x00000110,  0x00000101, 0x00000120,  0x00000111, 0x00000130,  0x00000121, 0x00000140,
  0x00000131, 0x00000150,  0x00000141, 0x00000160,  0x00000151, 0x00000170,  0x00000161, 0x00000180,
  0x00000171, 0x00000190,  0x00000181, 0x000001A0,  0x00000191, 0x000001B1,  0x000000E0, 0x000001A1,
  // .rdex
  // @c14
  // .nlen
  0x0000001E,
  // .rlen
  0x00000000,
  // .node
  0x00000000, 0x0000001A,  0x0000002B, 0x000001DA,  0x0000003B, 0x000001CA,  0x0000004B, 0x000001BA,
  0x0000005B, 0x000001AA,  0x0000006B, 0x0000019A,  0x0000007B, 0x0000018A,  0x0000008B, 0x0000017A,
  0x0000009B, 0x0000016A,  0x000000AB, 0x0000015A,  0x000000BB, 0x0000014A,  0x000000CB, 0x0000013A,
  0x000000DB, 0x0000012A,  0x000000EB, 0x0000011A,  0x000000FA, 0x0000010A,  0x000001D0, 0x00000100,
  0x000000F1, 0x00000110,  0x00000101, 0x00000120,  0x00000111, 0x00000130,  0x00000121, 0x00000140,
  0x00000131, 0x00000150,  0x00000141, 0x00000160,  0x00000151, 0x00000170,  0x00000161, 0x00000180,
  0x00000171, 0x00000190,  0x00000181, 0x000001A0,  0x00000191, 0x000001B0,  0x000001A1, 0x000001C0,
  0x000001B1, 0x000001D1,  0x000000F0, 0x000001C1,
  // .rdex
  // @c15
  // .nlen
  0x00000020,
  // .rlen
  0x00000000,
  // .node
  0x00000000, 0x0000001A,  0x0000002B, 0x000001FA,  0x0000003B, 0x000001EA,  0x0000004B, 0x000001DA,
  0x0000005B, 0x000001CA,  0x0000006B, 0x000001BA,  0x0000007B, 0x000001AA,  0x0000008B, 0x0000019A,
  0x0000009B, 0x0000018A,  0x000000AB, 0x0000017A,  0x000000BB, 0x0000016A,  0x000000CB, 0x0000015A,
  0x000000DB, 0x0000014A,  0x000000EB, 0x0000013A,  0x000000FB, 0x0000012A,  0x0000010A, 0x0000011A,
  0x000001F0, 0x00000110,  0x00000101, 0x00000120,  0x00000111, 0x00000130,  0x00000121, 0x00000140,
  0x00000131, 0x00000150,  0x00000141, 0x00000160,  0x00000151, 0x00000170,  0x00000161, 0x00000180,
  0x00000171, 0x00000190,  0x00000181, 0x000001A0,  0x00000191, 0x000001B0,  0x000001A1, 0x000001C0,
  0x000001B1, 0x000001D0,  0x000001C1, 0x000001E0,  0x000001D1, 0x000001F1,  0x00000100, 0x000001E1,
  // .rdex
  // @c16
  // .nlen
  0x00000022,
  // .rlen
  0x00000000,
  // .node
  0x00000000, 0x0000001A,  0x0000002B, 0x0000021A,  0x0000003B, 0x0000020A,  0x0000004B, 0x000001FA,
  0x0000005B, 0x000001EA,  0x0000006B, 0x000001DA,  0x0000007B, 0x000001CA,  0x0000008B, 0x000001BA,
  0x0000009B, 0x000001AA,  0x000000AB, 0x0000019A,  0x000000BB, 0x0000018A,  0x000000CB, 0x0000017A,
  0x000000DB, 0x0000016A,  0x000000EB, 0x0000015A,  0x000000FB, 0x0000014A,  0x0000010B, 0x0000013A,
  0x0000011A, 0x0000012A,  0x00000210, 0x00000120,  0x00000111, 0x00000130,  0x00000121, 0x00000140,
  0x00000131, 0x00000150,  0x00000141, 0x00000160,  0x00000151, 0x00000170,  0x00000161, 0x00000180,
  0x00000171, 0x00000190,  0x00000181, 0x000001A0,  0x00000191, 0x000001B0,  0x000001A1, 0x000001C0,
  0x000001B1, 0x000001D0,  0x000001C1, 0x000001E0,  0x000001D1, 0x000001F0,  0x000001E1, 0x00000200,
  0x000001F1, 0x00000211,  0x00000110, 0x00000201,
  // .rdex
  // @c17
  // .nlen
  0x00000024,
  // .rlen
  0x00000000,
  // .node
  0x00000000, 0x0000001A,  0x0000002B, 0x0000023A,  0x0000003B, 0x0000022A,  0x0000004B, 0x0000021A,
  0x0000005B, 0x0000020A,  0x0000006B, 0x000001FA,  0x0000007B, 0x000001EA,  0x0000008B, 0x000001DA,
  0x0000009B, 0x000001CA,  0x000000AB, 0x000001BA,  0x000000BB, 0x000001AA,  0x000000CB, 0x0000019A,
  0x000000DB, 0x0000018A,  0x000000EB, 0x0000017A,  0x000000FB, 0x0000016A,  0x0000010B, 0x0000015A,
  0x0000011B, 0x0000014A,  0x0000012A, 0x0000013A,  0x00000230, 0x00000130,  0x00000121, 0x00000140,
  0x00000131, 0x00000150,  0x00000141, 0x00000160,  0x00000151, 0x00000170,  0x00000161, 0x00000180,
  0x00000171, 0x00000190,  0x00000181, 0x000001A0,  0x00000191, 0x000001B0,  0x000001A1, 0x000001C0,
  0x000001B1, 0x000001D0,  0x000001C1, 0x000001E0,  0x000001D1, 0x000001F0,  0x000001E1, 0x00000200,
  0x000001F1, 0x00000210,  0x00000201, 0x00000220,  0x00000211, 0x00000231,  0x00000120, 0x00000221,
  // .rdex
  // @c18
  // .nlen
  0x00000026,
  // .rlen
  0x00000000,
  // .node
  0x00000000, 0x0000001A,  0x0000002B, 0x0000025A,  0x0000003B, 0x0000024A,  0x0000004B, 0x0000023A,
  0x0000005B, 0x0000022A,  0x0000006B, 0x0000021A,  0x0000007B, 0x0000020A,  0x0000008B, 0x000001FA,
  0x0000009B, 0x000001EA,  0x000000AB, 0x000001DA,  0x000000BB, 0x000001CA,  0x000000CB, 0x000001BA,
  0x000000DB, 0x000001AA,  0x000000EB, 0x0000019A,  0x000000FB, 0x0000018A,  0x0000010B, 0x0000017A,
  0x0000011B, 0x0000016A,  0x0000012B, 0x0000015A,  0x0000013A, 0x0000014A,  0x00000250, 0x00000140,
  0x00000131, 0x00000150,  0x00000141, 0x00000160,  0x00000151, 0x00000170,  0x00000161, 0x00000180,
  0x00000171, 0x00000190,  0x00000181, 0x000001A0,  0x00000191, 0x000001B0,  0x000001A1, 0x000001C0,
  0x000001B1, 0x000001D0,  0x000001C1, 0x000001E0,  0x000001D1, 0x000001F0,  0x000001E1, 0x00000200,
  0x000001F1, 0x00000210,  0x00000201, 0x00000220,  0x00000211, 0x00000230,  0x00000221, 0x00000240,
  0x00000231, 0x00000251,  0x00000130, 0x00000241,
  // .rdex
  // @c19
  // .nlen
  0x00000028,
  // .rlen
  0x00000000,
  // .node
  0x00000000, 0x0000001A,  0x0000002B, 0x0000027A,  0x0000003B, 0x0000026A,  0x0000004B, 0x0000025A,
  0x0000005B, 0x0000024A,  0x0000006B, 0x0000023A,  0x0000007B, 0x0000022A,  0x0000008B, 0x0000021A,
  0x0000009B, 0x0000020A,  0x000000AB, 0x000001FA,  0x000000BB, 0x000001EA,  0x000000CB, 0x000001DA,
  0x000000DB, 0x000001CA,  0x000000EB, 0x000001BA,  0x000000FB, 0x000001AA,  0x0000010B, 0x0000019A,
  0x0000011B, 0x0000018A,  0x0000012B, 0x0000017A,  0x0000013B, 0x0000016A,  0x0000014A, 0x0000015A,
  0x00000270, 0x00000150,  0x00000141, 0x00000160,  0x00000151, 0x00000170,  0x00000161, 0x00000180,
  0x00000171, 0x00000190,  0x00000181, 0x000001A0,  0x00000191, 0x000001B0,  0x000001A1, 0x000001C0,
  0x000001B1, 0x000001D0,  0x000001C1, 0x000001E0,  0x000001D1, 0x000001F0,  0x000001E1, 0x00000200,
  0x000001F1, 0x00000210,  0x00000201, 0x00000220,  0x00000211, 0x00000230,  0x00000221, 0x00000240,
  0x00000231, 0x00000250,  0x00000241, 0x00000260,  0x00000251, 0x00000271,  0x00000140, 0x00000261,
  // .rdex
  // @c20
  // .nlen
  0x0000002A,
  // .rlen
  0x00000000,
  // .node
  0x00000000, 0x0000001A,  0x0000002B, 0x0000029A,  0x0000003B, 0x0000028A,  0x0000004B, 0x0000027A,
  0x0000005B, 0x0000026A,  0x0000006B, 0x0000025A,  0x0000007B, 0x0000024A,  0x0000008B, 0x0000023A,
  0x0000009B, 0x0000022A,  0x000000AB, 0x0000021A,  0x000000BB, 0x0000020A,  0x000000CB, 0x000001FA,
  0x000000DB, 0x000001EA,  0x000000EB, 0x000001DA,  0x000000FB, 0x000001CA,  0x0000010B, 0x000001BA,
  0x0000011B, 0x000001AA,  0x0000012B, 0x0000019A,  0x0000013B, 0x0000018A,  0x0000014B, 0x0000017A,
  0x0000015A, 0x0000016A,  0x00000290, 0x00000160,  0x00000151, 0x00000170,  0x00000161, 0x00000180,
  0x00000171, 0x00000190,  0x00000181, 0x000001A0,  0x00000191, 0x000001B0,  0x000001A1, 0x000001C0,
  0x000001B1, 0x000001D0,  0x000001C1, 0x000001E0,  0x000001D1, 0x000001F0,  0x000001E1, 0x00000200,
  0x000001F1, 0x00000210,  0x00000201, 0x00000220,  0x00000211, 0x00000230,  0x00000221, 0x00000240,
  0x00000231, 0x00000250,  0x00000241, 0x00000260,  0x00000251, 0x00000270,  0x00000261, 0x00000280,
  0x00000271, 0x00000291,  0x00000150, 0x00000281,
  // .rdex
  // @c21
  // .nlen
  0x0000002C,
  // .rlen
  0x00000000,
  // .node
  0x00000000, 0x0000001A,  0x0000002B, 0x000002BA,  0x0000003B, 0x000002AA,  0x0000004B, 0x0000029A,
  0x0000005B, 0x0000028A,  0x0000006B, 0x0000027A,  0x0000007B, 0x0000026A,  0x0000008B, 0x0000025A,
  0x0000009B, 0x0000024A,  0x000000AB, 0x0000023A,  0x000000BB, 0x0000022A,  0x000000CB, 0x0000021A,
  0x000000DB, 0x0000020A,  0x000000EB, 0x000001FA,  0x000000FB, 0x000001EA,  0x0000010B, 0x000001DA,
  0x0000011B, 0x000001CA,  0x0000012B, 0x000001BA,  0x0000013B, 0x000001AA,  0x0000014B, 0x0000019A,
  0x0000015B, 0x0000018A,  0x0000016A, 0x0000017A,  0x000002B0, 0x00000170,  0x00000161, 0x00000180,
  0x00000171, 0x00000190,  0x00000181, 0x000001A0,  0x00000191, 0x000001B0,  0x000001A1, 0x000001C0,
  0x000001B1, 0x000001D0,  0x000001C1, 0x000001E0,  0x000001D1, 0x000001F0,  0x000001E1, 0x00000200,
  0x000001F1, 0x00000210,  0x00000201, 0x00000220,  0x00000211, 0x00000230,  0x00000221, 0x00000240,
  0x00000231, 0x00000250,  0x00000241, 0x00000260,  0x00000251, 0x00000270,  0x00000261, 0x00000280,
  0x00000271, 0x00000290,  0x00000281, 0x000002A0,  0x00000291, 0x000002B1,  0x00000160, 0x000002A1,
  // .rdex
  // @c22
  // .nlen
  0x0000002E,
  // .rlen
  0x00000000,
  // .node
  0x00000000, 0x0000001A,  0x0000002B, 0x000002DA,  0x0000003B, 0x000002CA,  0x0000004B, 0x000002BA,
  0x0000005B, 0x000002AA,  0x0000006B, 0x0000029A,  0x0000007B, 0x0000028A,  0x0000008B, 0x0000027A,
  0x0000009B, 0x0000026A,  0x000000AB, 0x0000025A,  0x000000BB, 0x0000024A,  0x000000CB, 0x0000023A,
  0x000000DB, 0x0000022A,  0x000000EB, 0x0000021A,  0x000000FB, 0x0000020A,  0x0000010B, 0x000001FA,
  0x0000011B, 0x000001EA,  0x0000012B, 0x000001DA,  0x0000013B, 0x000001CA,  0x0000014B, 0x000001BA,
  0x0000015B, 0x000001AA,  0x0000016B, 0x0000019A,  0x0000017A, 0x0000018A,  0x000002D0, 0x00000180,
  0x00000171, 0x00000190,  0x00000181, 0x000001A0,  0x00000191, 0x000001B0,  0x000001A1, 0x000001C0,
  0x000001B1, 0x000001D0,  0x000001C1, 0x000001E0,  0x000001D1, 0x000001F0,  0x000001E1, 0x00000200,
  0x000001F1, 0x00000210,  0x00000201, 0x00000220,  0x00000211, 0x00000230,  0x00000221, 0x00000240,
  0x00000231, 0x00000250,  0x00000241, 0x00000260,  0x00000251, 0x00000270,  0x00000261, 0x00000280,
  0x00000271, 0x00000290,  0x00000281, 0x000002A0,  0x00000291, 0x000002B0,  0x000002A1, 0x000002C0,
  0x000002B1, 0x000002D1,  0x00000170, 0x000002C1,
  // .rdex
  // @c23
  // .nlen
  0x00000030,
  // .rlen
  0x00000000,
  // .node
  0x00000000, 0x0000001A,  0x0000002B, 0x000002FA,  0x0000003B, 0x000002EA,  0x0000004B, 0x000002DA,
  0x0000005B, 0x000002CA,  0x0000006B, 0x000002BA,  0x0000007B, 0x000002AA,  0x0000008B, 0x0000029A,
  0x0000009B, 0x0000028A,  0x000000AB, 0x0000027A,  0x000000BB, 0x0000026A,  0x000000CB, 0x0000025A,
  0x000000DB, 0x0000024A,  0x000000EB, 0x0000023A,  0x000000FB, 0x0000022A,  0x0000010B, 0x0000021A,
  0x0000011B, 0x0000020A,  0x0000012B, 0x000001FA,  0x0000013B, 0x000001EA,  0x0000014B, 0x000001DA,
  0x0000015B, 0x000001CA,  0x0000016B, 0x000001BA,  0x0000017B, 0x000001AA,  0x0000018A, 0x0000019A,
  0x000002F0, 0x00000190,  0x00000181, 0x000001A0,  0x00000191, 0x000001B0,  0x000001A1, 0x000001C0,
  0x000001B1, 0x000001D0,  0x000001C1, 0x000001E0,  0x000001D1, 0x000001F0,  0x000001E1, 0x00000200,
  0x000001F1, 0x00000210,  0x00000201, 0x00000220,  0x00000211, 0x00000230,  0x00000221, 0x00000240,
  0x00000231, 0x00000250,  0x00000241, 0x00000260,  0x00000251, 0x00000270,  0x00000261, 0x00000280,
  0x00000271, 0x00000290,  0x00000281, 0x000002A0,  0x00000291, 0x000002B0,  0x000002A1, 0x000002C0,
  0x000002B1, 0x000002D0,  0x000002C1, 0x000002E0,  0x000002D1, 0x000002F1,  0x00000180, 0x000002E1,
  // .rdex
  // @c24
  // .nlen
  0x00000032,
  // .rlen
  0x00000000,
  // .node
  0x00000000, 0x0000001A,  0x0000002B, 0x0000031A,  0x0000003B, 0x0000030A,  0x0000004B, 0x000002FA,
  0x0000005B, 0x000002EA,  0x0000006B, 0x000002DA,  0x0000007B, 0x000002CA,  0x0000008B, 0x000002BA,
  0x0000009B, 0x000002AA,  0x000000AB, 0x0000029A,  0x000000BB, 0x0000028A,  0x000000CB, 0x0000027A,
  0x000000DB, 0x0000026A,  0x000000EB, 0x0000025A,  0x000000FB, 0x0000024A,  0x0000010B, 0x0000023A,
  0x0000011B, 0x0000022A,  0x0000012B, 0x0000021A,  0x0000013B, 0x0000020A,  0x0000014B, 0x000001FA,
  0x0000015B, 0x000001EA,  0x0000016B, 0x000001DA,  0x0000017B, 0x000001CA,  0x0000018B, 0x000001BA,
  0x0000019A, 0x000001AA,  0x00000310, 0x000001A0,  0x00000191, 0x000001B0,  0x000001A1, 0x000001C0,
  0x000001B1, 0x000001D0,  0x000001C1, 0x000001E0,  0x000001D1, 0x000001F0,  0x000001E1, 0x00000200,
  0x000001F1, 0x00000210,  0x00000201, 0x00000220,  0x00000211, 0x00000230,  0x00000221, 0x00000240,
  0x00000231, 0x00000250,  0x00000241, 0x00000260,  0x00000251, 0x00000270,  0x00000261, 0x00000280,
  0x00000271, 0x00000290,  0x00000281, 0x000002A0,  0x00000291, 0x000002B0,  0x000002A1, 0x000002C0,
  0x000002B1, 0x000002D0,  0x000002C1, 0x000002E0,  0x000002D1, 0x000002F0,  0x000002E1, 0x00000300,
  0x000002F1, 0x00000311,  0x00000190, 0x00000301,
  // .rdex
  // @c25
  // .nlen
  0x00000034,
  // .rlen
  0x00000000,
  // .node
  0x00000000, 0x0000001A,  0x0000002B, 0x0000033A,  0x0000003B, 0x0000032A,  0x0000004B, 0x0000031A,
  0x0000005B, 0x0000030A,  0x0000006B, 0x000002FA,  0x0000007B, 0x000002EA,  0x0000008B, 0x000002DA,
  0x0000009B, 0x000002CA,  0x000000AB, 0x000002BA,  0x000000BB, 0x000002AA,  0x000000CB, 0x0000029A,
  0x000000DB, 0x0000028A,  0x000000EB, 0x0000027A,  0x000000FB, 0x0000026A,  0x0000010B, 0x0000025A,
  0x0000011B, 0x0000024A,  0x0000012B, 0x0000023A,  0x0000013B, 0x0000022A,  0x0000014B, 0x0000021A,
  0x0000015B, 0x0000020A,  0x0000016B, 0x000001FA,  0x0000017B, 0x000001EA,  0x0000018B, 0x000001DA,
  0x0000019B, 0x000001CA,  0x000001AA, 0x000001BA,  0x00000000, 0x00000330,  0x000001A1, 0x000001C0,
  0x000001B1, 0x000001D0,  0x000001C1, 0x000001E0,  0x000001D1, 0x000001F0,  0x000001E1, 0x00000200,
  0x000001F1, 0x00000210,  0x00000201, 0x00000220,  0x00000211, 0x00000230,  0x00000221, 0x00000240,
  0x00000231, 0x00000250,  0x00000241, 0x00000260,  0x00000251, 0x00000270,  0x00000261, 0x00000280,
  0x00000271, 0x00000290,  0x00000281, 0x000002A0,  0x00000291, 0x000002B0,  0x000002A1, 0x000002C0,
  0x000002B1, 0x000002D0,  0x000002C1, 0x000002E0,  0x000002D1, 0x000002F0,  0x000002E1, 0x00000300,
  0x000002F1, 0x00000310,  0x00000301, 0x00000320,  0x00000311, 0x00000331,  0x000001A1, 0x00000321,
  // .rdex
  // @c26
  // .nlen
  0x00000036,
  // .rlen
  0x00000000,
  // .node
  0x00000000, 0x0000001A,  0x0000002B, 0x0000035A,  0x0000003B, 0x0000034A,  0x0000004B, 0x0000033A,
  0x0000005B, 0x0000032A,  0x0000006B, 0x0000031A,  0x0000007B, 0x0000030A,  0x0000008B, 0x000002FA,
  0x0000009B, 0x000002EA,  0x000000AB, 0x000002DA,  0x000000BB, 0x000002CA,  0x000000CB, 0x000002BA,
  0x000000DB, 0x000002AA,  0x000000EB, 0x0000029A,  0x000000FB, 0x0000028A,  0x0000010B, 0x0000027A,
  0x0000011B, 0x0000026A,  0x0000012B, 0x0000025A,  0x0000013B, 0x0000024A,  0x0000014B, 0x0000023A,
  0x0000015B, 0x0000022A,  0x0000016B, 0x0000021A,  0x0000017B, 0x0000020A,  0x0000018B, 0x000001FA,
  0x0000019B, 0x000001EA,  0x000001AB, 0x000001DA,  0x000001BA, 0x000001CA,  0x00000000, 0x000001C0,
  0x000001B1, 0x00000350,  0x000001C1, 0x000001E0,  0x000001D1, 0x000001F0,  0x000001E1, 0x00000200,
  0x000001F1, 0x00000210,  0x00000201, 0x00000220,  0x00000211, 0x00000230,  0x00000221, 0x00000240,
  0x00000231, 0x00000250,  0x00000241, 0x00000260,  0x00000251, 0x00000270,  0x00000261, 0x00000280,
  0x00000271, 0x00000290,  0x00000281, 0x000002A0,  0x00000291, 0x000002B0,  0x000002A1, 0x000002C0,
  0x000002B1, 0x000002D0,  0x000002C1, 0x000002E0,  0x000002D1, 0x000002F0,  0x000002E1, 0x00000300,
  0x000002F1, 0x00000310,  0x00000301, 0x00000320,  0x00000311, 0x00000330,  0x00000321, 0x00000340,
  0x00000331, 0x00000351,  0x000001C1, 0x00000341,
  // .rdex
  // @c_s
  // .nlen
  0x00000008,
  // .rlen
  0x00000000,
  // .node
  0x00000000, 0x0000001A,  0x0000002A, 0x0000004A,  0x00000051, 0x0000003A,  0x00000070, 0x00000060,
  0x0000005B, 0x0000007A,  0x0000006A, 0x00000020,  0x00000031, 0x00000071,  0x00000030, 0x00000061,
  // .rdex
  // @c_z
  // .nlen
  0x00000003,
  // .rlen
  0x00000000,
  // .node
  0x00000000, 0x0000001A,  0x00000005, 0x0000002A,  0x00000021, 0x00000020,
  // .rdex
  // @dec
  // .nlen
  0x00000005,
  // .rlen
  0x00000000,
  // .node
  0x00000000, 0x0000001A,  0x0000002A, 0x00000041,  0x09E89984, 0x0000003A,  0x09E89924, 0x0000004A,
  0x000000E4, 0x00000011,
  // .rdex
  // @ex0
  // .nlen
  0x00000002,
  // .rlen
  0x00000001,
  // .node
  0x00000000, 0x00000011,  0x0000B824, 0x00000001,
  // .rdex
  0x00009824, 0x0000001A,
  // @ex1
  // .nlen
  0x00000003,
  // .rlen
  0x00000001,
  // .node
  0x00000000, 0x00000021,  0x002AFB64, 0x0000002A,  0x002AFBD4, 0x00000001,
  // .rdex
  0x00260844, 0x0000001A,
  // @ex2
  // .nlen
  0x00000004,
  // .rlen
  0x00000002,
  // .node
  0x00000000, 0x00000031,  0x00000124, 0x0000002A,  0x000000E4, 0x00000030,  0x00000021, 0x00000001,
  // .rdex
  0x00260864, 0x0000001A,  0x0035E314, 0x0000003A,
  // @ex3
  // .nlen
  0x00000004,
  // .rlen
  0x00000002,
  // .node
  0x00000000, 0x00000031,  0x000001C4, 0x0000002A,  0x00000234, 0x00000030,  0x00000021, 0x00000001,
  // .rdex
  0x00260464, 0x0000001A,  0x0025D714, 0x0000003A,
  // @ex4
  // .nlen
  0x00000003,
  // .rlen
  0x00000001,
  // .node
  0x00000000, 0x00000021,  0x000007B6, 0x00000027,  0x00001416, 0x00000001,
  // .rdex
  0x00000036, 0x00000017,
  // @ex5
  // .nlen
  0x00000003,
  // .rlen
  0x00000001,
  // .node
  0x00000000, 0x00000011,  0x0000002A, 0x00000001,  0x000007B6, 0x00001416,
  // .rdex
  0x00000016, 0x00000019,
  // @g_s
  // .nlen
  0x00000006,
  // .rlen
  0x00000000,
  // .node
  0x00000000, 0x0000001A,  0x0000002C, 0x0000003A,  0x00000040, 0x00000050,  0x0000004A, 0x00000051,
  0x00000020, 0x0000005A,  0x00000021, 0x00000031,
  // .rdex
  // @g_z
  // .nlen
  0x00000002,
  // .rlen
  0x00000000,
  // .node
  0x00000000, 0x0000001A,  0x00000011, 0x00000010,
  // .rdex
  // @k10
  // .nlen
  0x00000016,
  // .rlen
  0x00000000,
  // .node
  0x00000000, 0x0000001A,  0x0000002C, 0x0000015A,  0x0000003C, 0x0000014A,  0x0000004C, 0x0000013A,
  0x0000005C, 0x0000012A,  0x0000006C, 0x0000011A,  0x0000007C, 0x0000010A,  0x0000008C, 0x000000FA,
  0x0000009C, 0x000000EA,  0x000000AC, 0x000000DA,  0x000000BA, 0x000000CA,  0x00000150, 0x000000C0,
  0x000000B1, 0x000000D0,  0x000000C1, 0x000000E0,  0x000000D1, 0x000000F0,  0x000000E1, 0x00000100,
  0x000000F1, 0x00000110,  0x00000101, 0x00000120,  0x00000111, 0x00000130,  0x00000121, 0x00000140,
  0x00000131, 0x00000151,  0x000000B0, 0x00000141,
  // .rdex
  // @k11
  // .nlen
  0x00000018,
  // .rlen
  0x00000000,
  // .node
  0x00000000, 0x0000001A,  0x0000002C, 0x0000017A,  0x0000003C, 0x0000016A,  0x0000004C, 0x0000015A,
  0x0000005C, 0x0000014A,  0x0000006C, 0x0000013A,  0x0000007C, 0x0000012A,  0x0000008C, 0x0000011A,
  0x0000009C, 0x0000010A,  0x000000AC, 0x000000FA,  0x000000BC, 0x000000EA,  0x000000CA, 0x000000DA,
  0x00000170, 0x000000D0,  0x000000C1, 0x000000E0,  0x000000D1, 0x000000F0,  0x000000E1, 0x00000100,
  0x000000F1, 0x00000110,  0x00000101, 0x00000120,  0x00000111, 0x00000130,  0x00000121, 0x00000140,
  0x00000131, 0x00000150,  0x00000141, 0x00000160,  0x00000151, 0x00000171,  0x000000C0, 0x00000161,
  // .rdex
  // @k12
  // .nlen
  0x0000001A,
  // .rlen
  0x00000000,
  // .node
  0x00000000, 0x0000001A,  0x0000002C, 0x0000019A,  0x0000003C, 0x0000018A,  0x0000004C, 0x0000017A,
  0x0000005C, 0x0000016A,  0x0000006C, 0x0000015A,  0x0000007C, 0x0000014A,  0x0000008C, 0x0000013A,
  0x0000009C, 0x0000012A,  0x000000AC, 0x0000011A,  0x000000BC, 0x0000010A,  0x000000CC, 0x000000FA,
  0x000000DA, 0x000000EA,  0x00000190, 0x000000E0,  0x000000D1, 0x000000F0,  0x000000E1, 0x00000100,
  0x000000F1, 0x00000110,  0x00000101, 0x00000120,  0x00000111, 0x00000130,  0x00000121, 0x00000140,
  0x00000131, 0x00000150,  0x00000141, 0x00000160,  0x00000151, 0x00000170,  0x00000161, 0x00000180,
  0x00000171, 0x00000191,  0x000000D0, 0x00000181,
  // .rdex
  // @k13
  // .nlen
  0x0000001C,
  // .rlen
  0x00000000,
  // .node
  0x00000000, 0x0000001A,  0x0000002C, 0x000001BA,  0x0000003C, 0x000001AA,  0x0000004C, 0x0000019A,
  0x0000005C, 0x0000018A,  0x0000006C, 0x0000017A,  0x0000007C, 0x0000016A,  0x0000008C, 0x0000015A,
  0x0000009C, 0x0000014A,  0x000000AC, 0x0000013A,  0x000000BC, 0x0000012A,  0x000000CC, 0x0000011A,
  0x000000DC, 0x0000010A,  0x000000EA, 0x000000FA,  0x000001B0, 0x000000F0,  0x000000E1, 0x00000100,
  0x000000F1, 0x00000110,  0x00000101, 0x00000120,  0x00000111, 0x00000130,  0x00000121, 0x00000140,
  0x00000131, 0x00000150,  0x00000141, 0x00000160,  0x00000151, 0x00000170,  0x00000161, 0x00000180,
  0x00000171, 0x00000190,  0x00000181, 0x000001A0,  0x00000191, 0x000001B1,  0x000000E0, 0x000001A1,
  // .rdex
  // @k14
  // .nlen
  0x0000001E,
  // .rlen
  0x00000000,
  // .node
  0x00000000, 0x0000001A,  0x0000002C, 0x000001DA,  0x0000003C, 0x000001CA,  0x0000004C, 0x000001BA,
  0x0000005C, 0x000001AA,  0x0000006C, 0x0000019A,  0x0000007C, 0x0000018A,  0x0000008C, 0x0000017A,
  0x0000009C, 0x0000016A,  0x000000AC, 0x0000015A,  0x000000BC, 0x0000014A,  0x000000CC, 0x0000013A,
  0x000000DC, 0x0000012A,  0x000000EC, 0x0000011A,  0x000000FA, 0x0000010A,  0x000001D0, 0x00000100,
  0x000000F1, 0x00000110,  0x00000101, 0x00000120,  0x00000111, 0x00000130,  0x00000121, 0x00000140,
  0x00000131, 0x00000150,  0x00000141, 0x00000160,  0x00000151, 0x00000170,  0x00000161, 0x00000180,
  0x00000171, 0x00000190,  0x00000181, 0x000001A0,  0x00000191, 0x000001B0,  0x000001A1, 0x000001C0,
  0x000001B1, 0x000001D1,  0x000000F0, 0x000001C1,
  // .rdex
  // @k15
  // .nlen
  0x00000020,
  // .rlen
  0x00000000,
  // .node
  0x00000000, 0x0000001A,  0x0000002C, 0x000001FA,  0x0000003C, 0x000001EA,  0x0000004C, 0x000001DA,
  0x0000005C, 0x000001CA,  0x0000006C, 0x000001BA,  0x0000007C, 0x000001AA,  0x0000008C, 0x0000019A,
  0x0000009C, 0x0000018A,  0x000000AC, 0x0000017A,  0x000000BC, 0x0000016A,  0x000000CC, 0x0000015A,
  0x000000DC, 0x0000014A,  0x000000EC, 0x0000013A,  0x000000FC, 0x0000012A,  0x0000010A, 0x0000011A,
  0x000001F0, 0x00000110,  0x00000101, 0x00000120,  0x00000111, 0x00000130,  0x00000121, 0x00000140,
  0x00000131, 0x00000150,  0x00000141, 0x00000160,  0x00000151, 0x00000170,  0x00000161, 0x00000180,
  0x00000171, 0x00000190,  0x00000181, 0x000001A0,  0x00000191, 0x000001B0,  0x000001A1, 0x000001C0,
  0x000001B1, 0x000001D0,  0x000001C1, 0x000001E0,  0x000001D1, 0x000001F1,  0x00000100, 0x000001E1,
  // .rdex
  // @k16
  // .nlen
  0x00000022,
  // .rlen
  0x00000000,
  // .node
  0x00000000, 0x0000001A,  0x0000002C, 0x0000021A,  0x0000003C, 0x0000020A,  0x0000004C, 0x000001FA,
  0x0000005C, 0x000001EA,  0x0000006C, 0x000001DA,  0x0000007C, 0x000001CA,  0x0000008C, 0x000001BA,
  0x0000009C, 0x000001AA,  0x000000AC, 0x0000019A,  0x000000BC, 0x0000018A,  0x000000CC, 0x0000017A,
  0x000000DC, 0x0000016A,  0x000000EC, 0x0000015A,  0x000000FC, 0x0000014A,  0x0000010C, 0x0000013A,
  0x0000011A, 0x0000012A,  0x00000210, 0x00000120,  0x00000111, 0x00000130,  0x00000121, 0x00000140,
  0x00000131, 0x00000150,  0x00000141, 0x00000160,  0x00000151, 0x00000170,  0x00000161, 0x00000180,
  0x00000171, 0x00000190,  0x00000181, 0x000001A0,  0x00000191, 0x000001B0,  0x000001A1, 0x000001C0,
  0x000001B1, 0x000001D0,  0x000001C1, 0x000001E0,  0x000001D1, 0x000001F0,  0x000001E1, 0x00000200,
  0x000001F1, 0x00000211,  0x00000110, 0x00000201,
  // .rdex
  // @k17
  // .nlen
  0x00000024,
  // .rlen
  0x00000000,
  // .node
  0x00000000, 0x0000001A,  0x0000002C, 0x0000023A,  0x0000003C, 0x0000022A,  0x0000004C, 0x0000021A,
  0x0000005C, 0x0000020A,  0x0000006C, 0x000001FA,  0x0000007C, 0x000001EA,  0x0000008C, 0x000001DA,
  0x0000009C, 0x000001CA,  0x000000AC, 0x000001BA,  0x000000BC, 0x000001AA,  0x000000CC, 0x0000019A,
  0x000000DC, 0x0000018A,  0x000000EC, 0x0000017A,  0x000000FC, 0x0000016A,  0x0000010C, 0x0000015A,
  0x0000011C, 0x0000014A,  0x0000012A, 0x0000013A,  0x00000230, 0x00000130,  0x00000121, 0x00000140,
  0x00000131, 0x00000150,  0x00000141, 0x00000160,  0x00000151, 0x00000170,  0x00000161, 0x00000180,
  0x00000171, 0x00000190,  0x00000181, 0x000001A0,  0x00000191, 0x000001B0,  0x000001A1, 0x000001C0,
  0x000001B1, 0x000001D0,  0x000001C1, 0x000001E0,  0x000001D1, 0x000001F0,  0x000001E1, 0x00000200,
  0x000001F1, 0x00000210,  0x00000201, 0x00000220,  0x00000211, 0x00000231,  0x00000120, 0x00000221,
  // .rdex
  // @k18
  // .nlen
  0x00000026,
  // .rlen
  0x00000000,
  // .node
  0x00000000, 0x0000001A,  0x0000002C, 0x0000025A,  0x0000003C, 0x0000024A,  0x0000004C, 0x0000023A,
  0x0000005C, 0x0000022A,  0x0000006C, 0x0000021A,  0x0000007C, 0x0000020A,  0x0000008C, 0x000001FA,
  0x0000009C, 0x000001EA,  0x000000AC, 0x000001DA,  0x000000BC, 0x000001CA,  0x000000CC, 0x000001BA,
  0x000000DC, 0x000001AA,  0x000000EC, 0x0000019A,  0x000000FC, 0x0000018A,  0x0000010C, 0x0000017A,
  0x0000011C, 0x0000016A,  0x0000012C, 0x0000015A,  0x0000013A, 0x0000014A,  0x00000250, 0x00000140,
  0x00000131, 0x00000150,  0x00000141, 0x00000160,  0x00000151, 0x00000170,  0x00000161, 0x00000180,
  0x00000171, 0x00000190,  0x00000181, 0x000001A0,  0x00000191, 0x000001B0,  0x000001A1, 0x000001C0,
  0x000001B1, 0x000001D0,  0x000001C1, 0x000001E0,  0x000001D1, 0x000001F0,  0x000001E1, 0x00000200,
  0x000001F1, 0x00000210,  0x00000201, 0x00000220,  0x00000211, 0x00000230,  0x00000221, 0x00000240,
  0x00000231, 0x00000251,  0x00000130, 0x00000241,
  // .rdex
  // @k19
  // .nlen
  0x00000028,
  // .rlen
  0x00000000,
  // .node
  0x00000000, 0x0000001A,  0x0000002C, 0x0000027A,  0x0000003C, 0x0000026A,  0x0000004C, 0x0000025A,
  0x0000005C, 0x0000024A,  0x0000006C, 0x0000023A,  0x0000007C, 0x0000022A,  0x0000008C, 0x0000021A,
  0x0000009C, 0x0000020A,  0x000000AC, 0x000001FA,  0x000000BC, 0x000001EA,  0x000000CC, 0x000001DA,
  0x000000DC, 0x000001CA,  0x000000EC, 0x000001BA,  0x000000FC, 0x000001AA,  0x0000010C, 0x0000019A,
  0x0000011C, 0x0000018A,  0x0000012C, 0x0000017A,  0x0000013C, 0x0000016A,  0x0000014A, 0x0000015A,
  0x00000270, 0x00000150,  0x00000141, 0x00000160,  0x00000151, 0x00000170,  0x00000161, 0x00000180,
  0x00000171, 0x00000190,  0x00000181, 0x000001A0,  0x00000191, 0x000001B0,  0x000001A1, 0x000001C0,
  0x000001B1, 0x000001D0,  0x000001C1, 0x000001E0,  0x000001D1, 0x000001F0,  0x000001E1, 0x00000200,
  0x000001F1, 0x00000210,  0x00000201, 0x00000220,  0x00000211, 0x00000230,  0x00000221, 0x00000240,
  0x00000231, 0x00000250,  0x00000241, 0x00000260,  0x00000251, 0x00000271,  0x00000140, 0x00000261,
  // .rdex
  // @k20
  // .nlen
  0x0000002A,
  // .rlen
  0x00000000,
  // .node
  0x00000000, 0x0000001A,  0x0000002C, 0x0000029A,  0x0000003C, 0x0000028A,  0x0000004C, 0x0000027A,
  0x0000005C, 0x0000026A,  0x0000006C, 0x0000025A,  0x0000007C, 0x0000024A,  0x0000008C, 0x0000023A,
  0x0000009C, 0x0000022A,  0x000000AC, 0x0000021A,  0x000000BC, 0x0000020A,  0x000000CC, 0x000001FA,
  0x000000DC, 0x000001EA,  0x000000EC, 0x000001DA,  0x000000FC, 0x000001CA,  0x0000010C, 0x000001BA,
  0x0000011C, 0x000001AA,  0x0000012C, 0x0000019A,  0x0000013C, 0x0000018A,  0x0000014C, 0x0000017A,
  0x0000015A, 0x0000016A,  0x00000290, 0x00000160,  0x00000151, 0x00000170,  0x00000161, 0x00000180,
  0x00000171, 0x00000190,  0x00000181, 0x000001A0,  0x00000191, 0x000001B0,  0x000001A1, 0x000001C0,
  0x000001B1, 0x000001D0,  0x000001C1, 0x000001E0,  0x000001D1, 0x000001F0,  0x000001E1, 0x00000200,
  0x000001F1, 0x00000210,  0x00000201, 0x00000220,  0x00000211, 0x00000230,  0x00000221, 0x00000240,
  0x00000231, 0x00000250,  0x00000241, 0x00000260,  0x00000251, 0x00000270,  0x00000261, 0x00000280,
  0x00000271, 0x00000291,  0x00000150, 0x00000281,
  // .rdex
  // @k21
  // .nlen
  0x0000002C,
  // .rlen
  0x00000000,
  // .node
  0x00000000, 0x0000001A,  0x0000002C, 0x000002BA,  0x0000003C, 0x000002AA,  0x0000004C, 0x0000029A,
  0x0000005C, 0x0000028A,  0x0000006C, 0x0000027A,  0x0000007C, 0x0000026A,  0x0000008C, 0x0000025A,
  0x0000009C, 0x0000024A,  0x000000AC, 0x0000023A,  0x000000BC, 0x0000022A,  0x000000CC, 0x0000021A,
  0x000000DC, 0x0000020A,  0x000000EC, 0x000001FA,  0x000000FC, 0x000001EA,  0x0000010C, 0x000001DA,
  0x0000011C, 0x000001CA,  0x0000012C, 0x000001BA,  0x0000013C, 0x000001AA,  0x0000014C, 0x0000019A,
  0x0000015C, 0x0000018A,  0x0000016A, 0x0000017A,  0x000002B0, 0x00000170,  0x00000161, 0x00000180,
  0x00000171, 0x00000190,  0x00000181, 0x000001A0,  0x00000191, 0x000001B0,  0x000001A1, 0x000001C0,
  0x000001B1, 0x000001D0,  0x000001C1, 0x000001E0,  0x000001D1, 0x000001F0,  0x000001E1, 0x00000200,
  0x000001F1, 0x00000210,  0x00000201, 0x00000220,  0x00000211, 0x00000230,  0x00000221, 0x00000240,
  0x00000231, 0x00000250,  0x00000241, 0x00000260,  0x00000251, 0x00000270,  0x00000261, 0x00000280,
  0x00000271, 0x00000290,  0x00000281, 0x000002A0,  0x00000291, 0x000002B1,  0x00000160, 0x000002A1,
  // .rdex
  // @k22
  // .nlen
  0x0000002E,
  // .rlen
  0x00000000,
  // .node
  0x00000000, 0x0000001A,  0x0000002C, 0x000002DA,  0x0000003C, 0x000002CA,  0x0000004C, 0x000002BA,
  0x0000005C, 0x000002AA,  0x0000006C, 0x0000029A,  0x0000007C, 0x0000028A,  0x0000008C, 0x0000027A,
  0x0000009C, 0x0000026A,  0x000000AC, 0x0000025A,  0x000000BC, 0x0000024A,  0x000000CC, 0x0000023A,
  0x000000DC, 0x0000022A,  0x000000EC, 0x0000021A,  0x000000FC, 0x0000020A,  0x0000010C, 0x000001FA,
  0x0000011C, 0x000001EA,  0x0000012C, 0x000001DA,  0x0000013C, 0x000001CA,  0x0000014C, 0x000001BA,
  0x0000015C, 0x000001AA,  0x0000016C, 0x0000019A,  0x0000017A, 0x0000018A,  0x000002D0, 0x00000180,
  0x00000171, 0x00000190,  0x00000181, 0x000001A0,  0x00000191, 0x000001B0,  0x000001A1, 0x000001C0,
  0x000001B1, 0x000001D0,  0x000001C1, 0x000001E0,  0x000001D1, 0x000001F0,  0x000001E1, 0x00000200,
  0x000001F1, 0x00000210,  0x00000201, 0x00000220,  0x00000211, 0x00000230,  0x00000221, 0x00000240,
  0x00000231, 0x00000250,  0x00000241, 0x00000260,  0x00000251, 0x00000270,  0x00000261, 0x00000280,
  0x00000271, 0x00000290,  0x00000281, 0x000002A0,  0x00000291, 0x000002B0,  0x000002A1, 0x000002C0,
  0x000002B1, 0x000002D1,  0x00000170, 0x000002C1,
  // .rdex
  // @k23
  // .nlen
  0x00000030,
  // .rlen
  0x00000000,
  // .node
  0x00000000, 0x0000001A,  0x0000002C, 0x000002FA,  0x0000003C, 0x000002EA,  0x0000004C, 0x000002DA,
  0x0000005C, 0x000002CA,  0x0000006C, 0x000002BA,  0x0000007C, 0x000002AA,  0x0000008C, 0x0000029A,
  0x0000009C, 0x0000028A,  0x000000AC, 0x0000027A,  0x000000BC, 0x0000026A,  0x000000CC, 0x0000025A,
  0x000000DC, 0x0000024A,  0x000000EC, 0x0000023A,  0x000000FC, 0x0000022A,  0x0000010C, 0x0000021A,
  0x0000011C, 0x0000020A,  0x0000012C, 0x000001FA,  0x0000013C, 0x000001EA,  0x0000014C, 0x000001DA,
  0x0000015C, 0x000001CA,  0x0000016C, 0x000001BA,  0x0000017C, 0x000001AA,  0x0000018A, 0x0000019A,
  0x000002F0, 0x00000190,  0x00000181, 0x000001A0,  0x00000191, 0x000001B0,  0x000001A1, 0x000001C0,
  0x000001B1, 0x000001D0,  0x000001C1, 0x000001E0,  0x000001D1, 0x000001F0,  0x000001E1, 0x00000200,
  0x000001F1, 0x00000210,  0x00000201, 0x00000220,  0x00000211, 0x00000230,  0x00000221, 0x00000240,
  0x00000231, 0x00000250,  0x00000241, 0x00000260,  0x00000251, 0x00000270,  0x00000261, 0x00000280,
  0x00000271, 0x00000290,  0x00000281, 0x000002A0,  0x00000291, 0x000002B0,  0x000002A1, 0x000002C0,
  0x000002B1, 0x000002D0,  0x000002C1, 0x000002E0,  0x000002D1, 0x000002F1,  0x00000180, 0x000002E1,
  // .rdex
  // @k24
  // .nlen
  0x00000032,
  // .rlen
  0x00000000,
  // .node
  0x00000000, 0x0000001A,  0x0000002C, 0x0000031A,  0x0000003C, 0x0000030A,  0x0000004C, 0x000002FA,
  0x0000005C, 0x000002EA,  0x0000006C, 0x000002DA,  0x0000007C, 0x000002CA,  0x0000008C, 0x000002BA,
  0x0000009C, 0x000002AA,  0x000000AC, 0x0000029A,  0x000000BC, 0x0000028A,  0x000000CC, 0x0000027A,
  0x000000DC, 0x0000026A,  0x000000EC, 0x0000025A,  0x000000FC, 0x0000024A,  0x0000010C, 0x0000023A,
  0x0000011C, 0x0000022A,  0x0000012C, 0x0000021A,  0x0000013C, 0x0000020A,  0x0000014C, 0x000001FA,
  0x0000015C, 0x000001EA,  0x0000016C, 0x000001DA,  0x0000017C, 0x000001CA,  0x0000018C, 0x000001BA,
  0x0000019A, 0x000001AA,  0x00000310, 0x000001A0,  0x00000191, 0x000001B0,  0x000001A1, 0x000001C0,
  0x000001B1, 0x000001D0,  0x000001C1, 0x000001E0,  0x000001D1, 0x000001F0,  0x000001E1, 0x00000200,
  0x000001F1, 0x00000210,  0x00000201, 0x00000220,  0x00000211, 0x00000230,  0x00000221, 0x00000240,
  0x00000231, 0x00000250,  0x00000241, 0x00000260,  0x00000251, 0x00000270,  0x00000261, 0x00000280,
  0x00000271, 0x00000290,  0x00000281, 0x000002A0,  0x00000291, 0x000002B0,  0x000002A1, 0x000002C0,
  0x000002B1, 0x000002D0,  0x000002C1, 0x000002E0,  0x000002D1, 0x000002F0,  0x000002E1, 0x00000300,
  0x000002F1, 0x00000311,  0x00000190, 0x00000301,
  // .rdex
  // @low
  // .nlen
  0x00000005,
  // .rlen
  0x00000000,
  // .node
  0x00000000, 0x0000001A,  0x0000002A, 0x00000041,  0x0BF2E984, 0x0000003A,  0x0BF2E924, 0x0000004A,
  0x000000E4, 0x00000011,
  // .rdex
  // @mul
  // .nlen
  0x00000006,
  // .rlen
  0x00000000,
  // .node
  0x00000000, 0x0000001A,  0x0000002A, 0x0000003A,  0x00000041, 0x00000051,  0x0000004A, 0x0000005A,
  0x00000050, 0x00000020,  0x00000040, 0x00000021,
  // .rdex
  // @nid
  // .nlen
  0x00000004,
  // .rlen
  0x00000000,
  // .node
  0x00000000, 0x0000001A,  0x0000002A, 0x00000031,  0x0C6C9DC4, 0x0000003A,  0x00000234, 0x00000011,
  // .rdex
  // @not
  // .nlen
  0x00000006,
  // .rlen
  0x00000000,
  // .node
  0x00000000, 0x0000001A,  0x0000002A, 0x0000004A,  0x00000050, 0x0000003A,  0x00000040, 0x00000051,
  0x00000030, 0x0000005A,  0x00000020, 0x00000031,
  // .rdex
  // @run
  // .nlen
  0x00000005,
  // .rlen
  0x00000000,
  // .node
  0x00000000, 0x0000001A,  0x0000002A, 0x00000041,  0x0D78C584, 0x0000003A,  0x0D78C524, 0x0000004A,
  0x000000E4, 0x00000011,
  // .rdex
  // @brnS
  // .nlen
  0x00000006,
  // .rlen
  0x00000002,
  // .node
  0x00000000, 0x0000001A,  0x0000002B, 0x0000003A,  0x00000040, 0x00000050,  0x00000041, 0x00000051,
  0x00000020, 0x00000030,  0x00000021, 0x00000031,
  // .rdex
  0x0025D714, 0x0000004A,  0x0025D714, 0x0000005A,
  // @brnZ
  // .nlen
  0x00000004,
  // .rlen
  0x00000002,
  // .node
  0x00000000, 0x00000011,  0x00000031, 0x00000001,  0x00000124, 0x0000003A,  0x000000E4, 0x00000010,
  // .rdex
  0x0035E314, 0x0000001A,  0x00260414, 0x0000002A,
  // @decI
  // .nlen
  0x00000003,
  // .rlen
  0x00000001,
  // .node
  0x00000000, 0x0000001A,  0x00000020, 0x00000021,  0x00000010, 0x00000011,
  // .rdex
  0x002FCBA4, 0x0000002A,
  // @decO
  // .nlen
  0x00000004,
  // .rlen
  0x00000002,
  // .node
  0x00000000, 0x0000001A,  0x00000030, 0x00000021,  0x00000031, 0x00000011,  0x00000010, 0x00000020,
  // .rdex
  0x00000124, 0x0000002A,  0x0027A264, 0x0000003A,
  // @lowI
  // .nlen
  0x00000004,
  // .rlen
  0x00000002,
  // .node
  0x00000000, 0x0000001A,  0x00000020, 0x00000031,  0x00000010, 0x00000030,  0x00000021, 0x00000011,
  // .rdex
  0x00000124, 0x0000002A,  0x00000184, 0x0000003A,
  // @lowO
  // .nlen
  0x00000004,
  // .rlen
  0x00000002,
  // .node
  0x00000000, 0x0000001A,  0x00000020, 0x00000031,  0x00000010, 0x00000030,  0x00000021, 0x00000011,
  // .rdex
  0x00000184, 0x0000002A,  0x00000184, 0x0000003A,
  // @nidS
  // .nlen
  0x00000004,
  // .rlen
  0x00000002,
  // .node
  0x00000000, 0x0000001A,  0x00000030, 0x00000021,  0x00000031, 0x00000011,  0x00000010, 0x00000020,
  // .rdex
  0x000001C4, 0x0000002A,  0x0031B274, 0x0000003A,
  // @runI
  // .nlen
  0x00000005,
  // .rlen
  0x00000003,
  // .node
  0x00000000, 0x0000001A,  0x00000040, 0x00000021,  0x00000031, 0x00000011,  0x00000041, 0x00000020,
  0x00000010, 0x00000030,
  // .rdex
  0x0035E314, 0x0000002A,  0x0027A264, 0x0000003A,  0x00000124, 0x0000004A,
  // @runO
  // .nlen
  0x00000005,
  // .rlen
  0x00000003,
  // .node
  0x00000000, 0x0000001A,  0x00000040, 0x00000021,  0x00000031, 0x00000011,  0x00000041, 0x00000020,
  0x00000010, 0x00000030,
  // .rdex
  0x0035E314, 0x0000002A,  0x0027A264, 0x0000003A,  0x00000184, 0x0000004A,
};
u32 JUMP_DATA[] = {
  0x0000000E, 0x00000000, // @E
  0x0000000F, 0x0000000A, // @F
  0x00000012, 0x00000012, // @I
  0x00000018, 0x00000020, // @O
  0x0000001C, 0x0000002E, // @S
  0x0000001D, 0x0000003A, // @T
  0x00000023, 0x00000042, // @Z
  0x00000929, 0x0000004A, // @af
  0x00000980, 0x00000054, // @c0
  0x00000981, 0x0000005C, // @c1
  0x00000982, 0x00000066, // @c2
  0x00000983, 0x00000074, // @c3
  0x00000984, 0x00000086, // @c4
  0x00000985, 0x0000009C, // @c5
  0x00000986, 0x000000B6, // @c6
  0x00000987, 0x000000D4, // @c7
  0x00000988, 0x000000F6, // @c8
  0x00000989, 0x0000011C, // @c9
  0x00000B27, 0x00000146, // @id
  0x00000B80, 0x0000014C, // @k0
  0x00000B81, 0x00000154, // @k1
  0x00000B82, 0x0000015E, // @k2
  0x00000B83, 0x0000016C, // @k3
  0x00000B84, 0x0000017E, // @k4
  0x00000B85, 0x00000194, // @k5
  0x00000B86, 0x000001AE, // @k6
  0x00000B87, 0x000001CC, // @k7
  0x00000B88, 0x000001EE, // @k8
  0x00000B89, 0x00000214, // @k9
  0x00024A5C, 0x0000023E, // @afS
  0x00024A63, 0x00000254, // @afZ
  0x00024C67, 0x00000258, // @and
  0x00025D71, 0x0000026E, // @brn
  0x00026040, 0x00000278, // @c10
  0x00026041, 0x000002A6, // @c11
  0x00026042, 0x000002D8, // @c12
  0x00026043, 0x0000030E, // @c13
  0x00026044, 0x00000348, // @c14
  0x00026045, 0x00000386, // @c15
  0x00026046, 0x000003C8, // @c16
  0x00026047, 0x0000040E, // @c17
  0x00026048, 0x00000458, // @c18
  0x00026049, 0x000004A6, // @c19
  0x00026080, 0x000004F8, // @c20
  0x00026081, 0x0000054E, // @c21
  0x00026082, 0x000005A8, // @c22
  0x00026083, 0x00000606, // @c23
  0x00026084, 0x00000668, // @c24
  0x00026085, 0x000006CE, // @c25
  0x00026086, 0x00000738, // @c26
  0x00026FB6, 0x000007A6, // @c_s
  0x00026FBD, 0x000007B8, // @c_z
  0x00027A26, 0x000007C0, // @dec
  0x00028EC0, 0x000007CC, // @ex0
  0x00028EC1, 0x000007D4, // @ex1
  0x00028EC2, 0x000007DE, // @ex2
  0x00028EC3, 0x000007EC, // @ex3
  0x00028EC4, 0x000007FA, // @ex4
  0x00028EC5, 0x00000804, // @ex5
  0x0002AFB6, 0x0000080E, // @g_s
  0x0002AFBD, 0x0000081C, // @g_z
  0x0002E040, 0x00000822, // @k10
  0x0002E041, 0x00000850, // @k11
  0x0002E042, 0x00000882, // @k12
  0x0002E043, 0x000008B8, // @k13
  0x0002E044, 0x000008F2, // @k14
  0x0002E045, 0x00000930, // @k15
  0x0002E046, 0x00000972, // @k16
  0x0002E047, 0x000009B8, // @k17
  0x0002E048, 0x00000A02, // @k18
  0x0002E049, 0x00000A50, // @k19
  0x0002E080, 0x00000AA2, // @k20
  0x0002E081, 0x00000AF8, // @k21
  0x0002E082, 0x00000B52, // @k22
  0x0002E083, 0x00000BB0, // @k23
  0x0002E084, 0x00000C12, // @k24
  0x0002FCBA, 0x00000C78, // @low
  0x00030E2F, 0x00000C84, // @mul
  0x00031B27, 0x00000C92, // @nid
  0x00031CB7, 0x00000C9C, // @not
  0x00035E31, 0x00000CAA, // @run
  0x00975C5C, 0x00000CB6, // @brnS
  0x00975C63, 0x00000CC8, // @brnZ
  0x009E8992, 0x00000CD6, // @decI
  0x009E8998, 0x00000CE0, // @decO
  0x00BF2E92, 0x00000CEE, // @lowI
  0x00BF2E98, 0x00000CFC, // @lowO
  0x00C6C9DC, 0x00000D0A, // @nidS
  0x00D78C52, 0x00000D18, // @runI
  0x00D78C58, 0x00000D2A, // @runO
};

const size_t BOOK_DATA_SIZE = sizeof(BOOK_DATA) / sizeof(u32);
const size_t JUMP_DATA_SIZE = sizeof(JUMP_DATA) / sizeof(u32);

// Main
// ----

int main() {
  // Prints device info
  int device;
  cudaDeviceProp prop;
  cudaGetDevice(&device);
  cudaGetDeviceProperties(&prop, device);
  printf("CUDA Device: %s, Compute Capability: %d.%d\n\n", prop.name, prop.major, prop.minor);
  printf("Total global memory: %zu bytes\n", prop.totalGlobalMem);
  printf("Shared memory per block: %zu bytes\n", prop.sharedMemPerBlock);
  printf("Registers per block: %d\n", prop.regsPerBlock);
  printf("Warp size: %d\n", prop.warpSize);
  printf("Maximum threads per block: %d\n", prop.maxThreadsPerBlock);
  printf("Maximum thread dimensions: (%d, %d, %d)\n", prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
  printf("Maximum grid dimensions: (%d, %d, %d)\n", prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
  printf("Clock rate: %d kHz\n", prop.clockRate);
  printf("Total constant memory: %zu bytes\n", prop.totalConstMem);
  printf("Compute capability: %d.%d\n", prop.major, prop.minor);
  printf("Number of multiprocessors: %d\n", prop.multiProcessorCount);
  printf("Concurrent copy and execution: %s\n", (prop.deviceOverlap ? "Yes" : "No"));
  printf("Kernel execution timeout: %s\n", (prop.kernelExecTimeoutEnabled ? "Yes" : "No"));

  // Prints info about the do_global_rewrite kernel
  cudaFuncAttributes attr;
  cudaError_t err = cudaFuncGetAttributes(&attr, global_rewrite);
  if (err != cudaSuccess) {
    printf("CUDA error: %s\n", cudaGetErrorString(err));
  } else {
    printf("\n");
    printf("Number of registers used: %d\n", attr.numRegs);
    printf("Shared memory used: %zu bytes\n", attr.sharedSizeBytes);
    printf("Constant memory used: %zu bytes\n", attr.constSizeBytes);
    printf("Size of local memory frame: %zu bytes\n", attr.localSizeBytes);
    printf("Maximum number of threads per block: %d\n", attr.maxThreadsPerBlock);
    printf("Number of PTX versions supported: %d\n", attr.ptxVersion);
    printf("Number of Binary versions supported: %d\n", attr.binaryVersion);
  }

  // Allocates net on CPU
  Net* cpu_net = mknet(F_ex3, JUMP_DATA, JUMP_DATA_SIZE);

  // Prints the input net
  printf("\nINPUT\n=====\n\n");
  print_net(cpu_net);

  // Uploads net and book to GPU
  Net* gpu_net = net_to_gpu(cpu_net);
  Book* gpu_book = init_book_on_gpu(BOOK_DATA, BOOK_DATA_SIZE);

  // Marks init time
  struct timespec start, end;
  // clock_gettime(CLOCK_MONOTONIC_RAW, &start);

  // Normalizes
  do_global_expand(gpu_net, gpu_book);
  for (u32 tick = 0; tick < 128; ++tick) {
    do_global_rewrite(gpu_net, gpu_book, 16, tick, (tick / BAGS_WIDTH_L2) % 2);
  }
  do_global_expand(gpu_net, gpu_book);
  do_global_rewrite(gpu_net, gpu_book, 200000, 0, 0);
  cudaDeviceSynchronize();

  // Gets end time
  // clock_gettime(CLOCK_MONOTONIC_RAW, &end);
  uint32_t delta_time = (end.tv_sec - start.tv_sec) * 1000 + (end.tv_nsec - start.tv_nsec) / 1000000;

  // Reads result back to cpu
  Net* norm = net_to_cpu(gpu_net);

  // Prints the output
  printf("\nNORMAL ~ rewrites=%llu\n======\n\n", norm->rwts);
  //print_tree(norm, norm->root);
  print_net(norm);
  printf("Time: %.3f s\n", ((double)delta_time) / 1000.0);
  printf("RPS : %.3f million\n", ((double)norm->rwts) / ((double)delta_time) / 1000.0);

  // Clears CPU memory
  net_free_on_gpu(gpu_net);
  book_free_on_gpu(gpu_book);

  // Clears GPU memory
  net_free_on_cpu(cpu_net);
  net_free_on_cpu(norm);

  return 0;
}
 */