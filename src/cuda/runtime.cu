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
