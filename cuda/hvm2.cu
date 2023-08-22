#include <stdint.h>
#include <stdbool.h>
#include <string.h>
#include <stdio.h>

typedef uint8_t u8;
typedef uint32_t u32;
typedef uint64_t u64;
typedef unsigned long long int a64;

// Configuration
// -------------

// This code is initially optimized for nVidia RTX 4090

// Total number of SMs in a chip (fixed per GPU)
const u32 SM_COUNT = 128;
//const u32 SM_COUNT = 4;

// Clock rate, in hertz
const u32 CLOCK_RATE = 2520000000;

// Shared memory size, in bytes (fixed per GPU)
const u32 SM_SIZE = 128 * 1024; // 128 KB

// Number of threads per warp (fixed per GPU)
const u32 WARP_SIZE = 32;

// Warps per SM (adjustable: aim to max occupancy)
const u32 WARPS_PER_SM = 4;

// Number of threads per block (derived)
const u32 BLOCK_SIZE = WARP_SIZE * WARPS_PER_SM;
//const u32 BLOCK_SIZE = 16;

// Total number of active parallel threads (derived)
const u32 TOTAL_THREADS = SM_COUNT * BLOCK_SIZE;

// Threads used per rewrite unit
const u32 UNIT_SIZE = 4;

// Total number of units
const u32 TOTAL_UNITS = TOTAL_THREADS / UNIT_SIZE;

// Total number of redexes per thread
const u32 RBAG_AREA = 256;

// Total number of bag redex entries 
const u32 RBAG_SIZE = RBAG_AREA * TOTAL_UNITS;

// Padding to avoid bank conflicts
const u32 BANK_PAD = 32;

// Total size of RPUT object
const u32 RPUT_SIZE = BANK_PAD * TOTAL_UNITS;

// Total number of nodes (fixed due to 2^28 addressable space)
const u32 NODE_SIZE = 1 << 28;

// Spacing between units, in number of nodes
const u32 UNIT_ALLOC_PAD = NODE_SIZE / TOTAL_UNITS;

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
const Tag GOR = 0x7; // redirection to root
const Tag GO1 = 0x8; // redirection to aux1 port of node
const Tag GO2 = 0x9; // redirection to aux2 port of node
const Tag CON = 0xA; // points to main port of con node
const Tag DUP = 0xB; // points to main port of dup node
const Tag TRI = 0xC; // points to main port of tri node
const Tag QUA = 0xD; // points to main port of qua node
const Tag QUI = 0xE; // points to main port of qui node
const Tag SEX = 0xF; // points to main port of sex node
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
typedef struct {
  Ptr ports[2];
} Node;

// Wires are pairs of pointers
typedef struct {
  Ptr lft;
  Ptr rgt;
} Wire;

// An interaction net 
typedef struct {
  Ptr   root; // root wire
  Wire* rbag; // global redex bag (active pairs)
  u32*  rput; // where given thread will alloc the next redex
  Node* node; // global memory buffer with all nodes
  u32   actv; // number of active threads
  u32   rwts; // number of rewrites performed
} Net;

// A worker local data
typedef struct {
  u32 tid;   // local thread id (on the block)
  u32 gid;   // global thread id (on the kernel)
  u32 unit;  // unit id (index on redex array)
  u32 frac;  // worker frac (A1|A2|B1|B2)
  u32 port;  // worker port (P1|P2)
  Ptr a_ptr; // left pointer of active wire
  Ptr b_ptr; // right pointer of active wire
  u32 tick;  // random number generator
  u32 seed;  // random number generator
  u32 aloc;  // where to alloc next node
  u32 rpop;  // where to pop next redex
  u32 rwts;  // total rewrites this performed
} Worker;

// Runtime
// -------

__device__ inline u32 rng(u32 seed) {
  return seed * 16843009 + 3014898611;
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
  if (tag(ptr) == VRR || tag(ptr) == GOR) {
    return &net->root;
  } else if (tag(ptr) == VR1 || tag(ptr) == GO1) {
    return &net->node[val(ptr)].ports[P1];
  } else if (tag(ptr) == VR2 || tag(ptr) == GO2) {
    return &net->node[val(ptr)].ports[P2];
  } else {
    return NULL;
  }
}

// Traverses to the other side of a wire
__host__ __device__ Ptr enter(Net* net, Ptr ptr) {
  ptr = *target(net, ptr);
  while (tag(ptr) >= GOR && tag(ptr) <= GO2) {
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
  while (true) {
    u32  idx = (worker->unit * UNIT_ALLOC_PAD + worker->aloc * 4 + worker->frac) % NODE_SIZE;
    a64* ref = &((a64*)net->node)[idx];
    u64  exp = 0;
    u64  neo = *((u64*)&mknode(BSY, BSY));
    u64  got = atomicCAS(ref, exp, neo);
    if (got == 0) {
      //printf("[%d] alloc at %d\n", worker->gid, idx);
      return idx;
    } else {
      worker->aloc = (worker->aloc + 1) % NODE_SIZE;
    }
  }
}

// Creates a new active pair
__device__ inline void put_redex(Worker* worker, Net* net, Ptr a_ptr, Ptr b_ptr) {
  //bool rand = worker->seed % 4 == 0;
  //u32  unit = (rand ? worker->seed : worker->unit) % TOTAL_UNITS;
  worker->seed = rng(worker->seed);
  bool rand    = (worker->seed % 7) == 0;
  //printf("[%d|%d] ... seed=%u\n", worker->gid, worker->unit, worker->seed);
  u32  unit    = (rand ? worker->seed : worker->unit) % TOTAL_UNITS;
  //printf("[%d] seed=%d | pick=%d | %d\n", worker->gid, worker->seed, worker->seed % TOTAL_UNITS, rand);
  //printf("[%d] unit: %d | %u\n", worker->gid, unit, worker->seed);
  while (true) {
    u32  idx = atomicAdd(&net->rput[unit * BANK_PAD], 1);
    a64* ref = (a64*)&net->rbag[unit * RBAG_AREA + (idx % RBAG_AREA)];
    u64  exp = 0;
    u64  neo = *((u64*)&(Wire){a_ptr, b_ptr});
    u64  got = atomicCAS(ref, exp, neo);
    if (got == 0) {
      //if (worker->gid == 1) {
        //printf("[%d|%d] pushed at %d of %d | rand=%d seed=%u\n", worker->gid, worker->unit, idx, unit, rand, worker->seed);
      //}
      return;
    } else {
      unit = rng(unit) % TOTAL_UNITS;
    }
  }
}

// Attempts to get an active pair
__device__ inline Wire pop_redex(Worker* worker, Net* net) {
  a64* ref = (a64*)&net->rbag[worker->unit * RBAG_AREA + (worker->rpop % RBAG_AREA)];
  u64  neo = 0;
  u64  got = atomicExch(ref, neo);
  if (got != neo) {
    //printf("[%d] popped at %d\n", worker->gid, worker->rpop);
    worker->rpop++;
    return *((Wire*)&got);
  }
  return (Wire) { lft: 0, rgt: 0 };
}

// Empties a slot in memory
__device__ Ptr clear(Ptr* ref) {
  atomicCAS((u32*)ref, BSY, 0);
}

// Gets the value of a ref; waits if busy
__device__ Ptr deref(Ptr* ref) {
  Ptr got = atomicExch((u32*)ref, BSY);
  while (got == BSY) {
    got = atomicExch((u32*)ref, BSY);
  }
  return got;
}

// Attempts to replace 'exp' by 'neo', until it succeeds
__device__ Ptr replace(Ptr* ref, Ptr exp, Ptr neo) {
  Ptr got = atomicCAS((u32*)ref, exp, neo);
  while (got != exp) {
    got = atomicCAS((u32*)ref, exp, neo);
  }
  return got;
}

// Result of a march
typedef struct {
  Ptr* ref;
  Ptr  ptr;
} Target;

// Marches towards a target, clearing redirections
__device__ Target march(Net* net, Ptr ptr) {
  // Gets the immediate target
  Target targ;
  targ.ref = target(net, ptr);
  targ.ptr = deref(targ.ref);
  // While it is a redirection, clear and get the next target
  while (tag(targ.ptr) >= GOR && tag(targ.ptr) <= GO2) {
    clear(targ.ref);
    targ.ref = target(net, targ.ptr);
    targ.ptr = deref(targ.ref);
  }
  return targ;
}

// Substs the node in 'self_ref' into the destination of 'target_ptr'
// - If the destination is a VAR => move the node into it
// - If the destination is a NOD => create an active pair
__device__ void subst(Worker* worker, Net* net, Ptr* self_ref, Ptr target_ptr) {
  // Marches towards target_ptr
  Target targ = march(net, target_ptr);
  // If the final target is a var, move this node into it
  if (tag(targ.ptr) >= VRR && tag(targ.ptr) <= VR2) { 
    Ptr* last_ref = targ.ref; // save the var's ref
    targ = march(net, targ.ptr); // clear the backwards path
    clear(self_ref); // clear our ref (here, targ.ref == self_ref)
    replace(last_ref, BSY, targ.ptr); // move our node to the last ref
  // Otherwise, create a new active pair. Since two marches reach this branch,
  // we priorize the one with largest ref, in order to avoid race conditions.
  } else if (self_ref < targ.ref) {
    atomicExch((u32*)targ.ref, targ.ptr); // puts the other node back
  } else {
    replace(targ.ref, BSY, 0); // clear the other node
    put_redex(worker, net, deref(self_ref), targ.ptr); // create redex
    clear(self_ref); // clear our node
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
__global__ void work(Net* net) {
  // Shared buffer for warp communication
  __shared__ u32  XLOC[BLOCK_SIZE];
  __shared__ Wire WIRE[BLOCK_SIZE];

  // Activity status
  bool curr_active, last_active;

  // Initializes local vars
  Worker worker;
  worker.tid  = threadIdx.x;
  worker.gid  = blockIdx.x * blockDim.x + threadIdx.x;
  worker.unit = worker.gid / UNIT_SIZE;
  worker.tick = 0;
  worker.seed = (42 * (worker.gid + 1)) % UINT_MAX;
  worker.aloc = 0;
  worker.rpop = 0;
  worker.rwts = 0;
  worker.frac = worker.tid % 4;
  worker.port = worker.tid % 2;

  // Broadcasts initial activity status
  if (worker.frac == A1) {
    curr_active = true;
    last_active = true;
    atomicAdd(&net->actv, 1);
  }

  while (true) {
    for (u32 tick = 0; tick < 256; ++tick) {
      // If group leader, attempts to pop an active wire
      if (worker.frac == A1) {
        WIRE[worker.tid/4] = pop_redex(&worker, net);
      }
      __syncwarp();

      // Reads redex ptrs
      Wire wire    = WIRE[worker.tid/4];
      worker.a_ptr = worker.frac <= A2 ? wire.lft : wire.rgt;
      worker.b_ptr = worker.frac <= A2 ? wire.rgt : wire.lft;

      // Checks if we got redex, and what type
      bool rdex = worker.a_ptr != 0;
      bool anni = rdex && tag(worker.a_ptr) == tag(worker.b_ptr);
      bool comm = rdex && tag(worker.a_ptr) != tag(worker.b_ptr);

      // Updates activity status
      curr_active = rdex;

      // Prints message
      if (rdex) {
        //printf("[%d] %u | rdex: %8X <-> %8X | %d | \n", worker.gid, tick, worker.a_ptr, worker.b_ptr, comm ? 1 : 0);
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
        ak_ptr = deref(ak_ref);
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
        replace(at(net, xk_loc, P1), BSY, mkptr(VRK, XLOC[worker.tid + ADD[worker.frac] + 0]));
        replace(at(net, xk_loc, P2), BSY, mkptr(VRK, XLOC[worker.tid + ADD[worker.frac] + 1]));
      }

      // Send ptr to other side
      if (rdex) {
        replace(bk_ref, BSY, xk_ptr);
      }

      // If anni and we sent a NOD, subst the node there, towards our port
      // If comm and we have a VAR, subst the clone here, towards that var
      if (anni && !var(ak_ptr) || comm && var(ak_ptr)) {
        u32  GOK  = worker.port == P1 ? GO1 : GO2;
        Ptr *self = comm ? ak_ref        : bk_ref;
        Ptr  targ = comm ? redir(ak_ptr) : mkptr(GOK, val(worker.a_ptr)); 
        subst(&worker, net, self, targ);
      }

      // If comm and we have a NOD, form an active pair with the clone we got
      if (comm && !var(ak_ptr)) {
        put_redex(&worker, net, ak_ptr, deref(ak_ref));
        clear(ak_ref);
      }
    }

    // Broadcasts updated activity status
    if (worker.frac == A1) {
      if (curr_active && !last_active) {
        //printf("[%d] active\n", worker.gid);
        atomicAdd(&net->actv, 1);
      }
      if (!curr_active && last_active) {
        //printf("[%d] inactive\n", worker.gid);
        atomicSub(&net->actv, 1);
      }
      last_active = curr_active;
    }

    // Stops when all units are inactive
    if (!curr_active && atomicAdd(&net->actv, 0) == 0) {
      break;
    }

  }

  // When the work ends, sum stats
  if (worker.frac == A1) {
    //printf("- %08X rwts [%d]\n", worker.rwts, worker.gid);
    atomicAdd(&net->rwts, worker.rwts);
  }
}

// Host<->Device
// -------------

__host__ Net* mknet() {
  Net* net  = (Net*)malloc(sizeof(Net));
  net->root = mkptr(NIL, 0);
  net->rwts = 0;
  net->actv = 0;
  net->rbag = (Wire*)malloc(RBAG_SIZE * sizeof(Wire));
  net->rput = (u32*) malloc(RPUT_SIZE * sizeof(u32));
  net->node = (Node*)malloc(NODE_SIZE * sizeof(Node));
  memset(net->rbag, 0, RBAG_SIZE * sizeof(Wire));
  memset(net->rput, 0, RPUT_SIZE * sizeof(u32));
  memset(net->node, 0, NODE_SIZE * sizeof(Node));
  return net;
}

__host__ Net* net_to_device(Net* host_net) {
  // Allocate memory on the device for the Net object, and its data
  Net*  device_net;
  Wire* device_rbag;
  u32*  device_rput;
  Node* device_node;

  cudaMalloc((void**)&device_net, sizeof(Net));
  cudaMalloc((void**)&device_rbag, RBAG_SIZE * sizeof(Wire));
  cudaMalloc((void**)&device_rput, RPUT_SIZE * sizeof(u32));
  cudaMalloc((void**)&device_node, NODE_SIZE * sizeof(Node));

  // Copy the host data to the device memory
  cudaMemcpy(device_rbag, host_net->rbag, RBAG_SIZE * sizeof(Wire), cudaMemcpyHostToDevice);
  cudaMemcpy(device_rput, host_net->rput, RPUT_SIZE * sizeof(u32),  cudaMemcpyHostToDevice);
  cudaMemcpy(device_node, host_net->node, NODE_SIZE * sizeof(Node), cudaMemcpyHostToDevice);

  // Create a temporary host Net object with device pointers
  Net temp_net  = *host_net;
  temp_net.rbag = device_rbag;
  temp_net.rput = device_rput;
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
  host_net->rbag = (Wire*)malloc(RBAG_SIZE * sizeof(Wire));
  host_net->rput = (u32*) malloc(RPUT_SIZE * sizeof(u32));
  host_net->node = (Node*)malloc(NODE_SIZE * sizeof(Node));

  // Retrieve the device pointers for data
  Wire* device_rbag;
  u32*  device_rput;
  Node* device_node;
  cudaMemcpy(&device_rbag, &(device_net->rbag), sizeof(Wire*), cudaMemcpyDeviceToHost);
  cudaMemcpy(&device_rput, &(device_net->rput), sizeof(u32*),  cudaMemcpyDeviceToHost);
  cudaMemcpy(&device_node, &(device_net->node), sizeof(Node*), cudaMemcpyDeviceToHost);

  // Copy the device data to the host memory
  cudaMemcpy(host_net->rbag, device_rbag, RBAG_SIZE * sizeof(Wire), cudaMemcpyDeviceToHost);
  cudaMemcpy(host_net->rput, device_rput, RPUT_SIZE * sizeof(u32),  cudaMemcpyDeviceToHost);
  cudaMemcpy(host_net->node, device_node, NODE_SIZE * sizeof(Node), cudaMemcpyDeviceToHost);

  return host_net;
}

__host__ void net_free_on_device(Net* device_net) {
  // Retrieve the device pointers for data
  Wire* device_rbag;
  u32*  device_rput;
  Node* device_node;
  cudaMemcpy(&device_rbag, &(device_net->rbag), sizeof(Wire*), cudaMemcpyDeviceToHost);
  cudaMemcpy(&device_rput, &(device_net->rput), sizeof(u32*),  cudaMemcpyDeviceToHost);
  cudaMemcpy(&device_node, &(device_net->node), sizeof(Node*), cudaMemcpyDeviceToHost);

  // Free the device memory
  cudaFree(device_rbag);
  cudaFree(device_rput);
  cudaFree(device_node);
  cudaFree(device_net);
}

__host__ void net_free_on_host(Net* host_net) {
  free(host_net->rbag);
  free(host_net->rput);
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
      case GOR: tag_str = "GOR"; break;
      case GO1: tag_str = "GO1"; break;
      case GO2: tag_str = "GO2"; break;
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
  printf("RBag:\n");
  for (u32 i = 0; i < RBAG_SIZE; ++i) {
    Ptr a = net->rbag[i].lft;
    Ptr b = net->rbag[i].rgt;
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
  //printf("Used: %u\n", net->used_mem);
  printf("Rwts: %u\n", net->rwts);
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
      case GOR: case GO1: case GO2:
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
  net->rput[ 0] = 1;
  net->rbag[ 0] = (Wire) {0xa0000000,0xa0000001};
  net->node[ 0] = (Node) {0x60000000,0x50000000};
  net->node[ 1] = (Node) {0xa0000002,0x40000000};
  net->node[ 2] = (Node) {0x60000002,0x50000002};
}

// term_b = (λfλx(f x) λy(y))
__host__ void inject_term_b(Net* net) {
  net->root     = 0x60000003;
  net->rput[ 0] = 1;
  net->rbag[ 0] = (Wire) {0xa0000000,0xa0000003};
  net->node[ 0] = (Node) {0xa0000001,0xa0000002};
  net->node[ 1] = (Node) {0x50000002,0x60000002};
  net->node[ 2] = (Node) {0x50000001,0x60000001};
  net->node[ 3] = (Node) {0xa0000004,0x40000000};
  net->node[ 4] = (Node) {0x60000004,0x50000004};
}

// term_c = (λfλx(f (f x)) λx(x))
__host__ void inject_term_c(Net* net) {
  net->root     = 0x60000005;
  net->rput[ 0] = 1;
  net->rbag[ 0] = (Wire) {0xa0000000,0xa0000005};
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
  net->rput[ 0] = 1;
  net->rbag[ 0] = (Wire) {0xa0000000,0xa0000005};
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
  net->rput[ 0] = 1;
  net->rbag[ 0] = (Wire) {0xa0000000,0xa0000005};
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
  net->rput[ 0] = 1;
  net->rbag[ 0] = (Wire) {0xa0000000,0xa0000007};
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
  net->rput[ 0] = 1;
  net->rbag[ 0] = (Wire) {0xa0000000,0xa0000011};
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
  net->rput[ 0] = 1;
  net->rbag[ 0] = (Wire) {0xa0000000,0xa0000019};
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

// term_h = (c16 g_s g_z)
__host__ void inject_term_i(Net* net) {
  net->root     = 0x60000027;
  net->rput[ 0] = 1;
  net->rbag[ 0] = (Wire) {0xa0000000,0xa0000021};
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

// term_h = (c18 g_s g_z)
__host__ void inject_term_j(Net* net) {
  net->root    = 0x6000002b;
  net->rput[ 0] = 1;
  net->rbag[ 0] = (Wire) {0xa0000000,0xa0000025};
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
  inject_term_g(h_net); // works up to j

  // Prints the initial net
  print_net(h_net);

  // Sends the net from host to device
  Net* d_net = net_to_device(h_net);

  // Performs parallel reductions
  work<<<SM_COUNT, BLOCK_SIZE>>>(d_net);

  // Add synchronization and error checking
  cudaDeviceSynchronize();
  cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess) {
    printf("CUDA error: %s\n", cudaGetErrorString(error));
  }

  // Reads the normalized net from device to host
  Net* norm = net_to_host(d_net);

  // Prints the normal form (raw data)
  //print_net(norm);

  // Prints the normal form (as a tree)
  printf("Norm:\n");
  //print_tree(norm, norm->root);
  printf("\n");

  // Prints just the rwts
  printf("Rwts: %d\n", norm->rwts);

  // Free device memory
  net_free_on_device(d_net);

  // Free host memory
  net_free_on_host(h_net);
  net_free_on_host(norm);

  return 0;
}
