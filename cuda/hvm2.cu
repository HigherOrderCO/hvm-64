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

// Clock rate, in hertz
const u32 CLOCK_RATE = 2520000000;

// Active cycles before a worker halts
const u32 CYCLES_TO_HALT = 1000000; // about 1/256 seconds on RTX 4090

// Shared memory size, in bytes (fixed per GPU)
const u32 SM_SIZE = 128 * 1024; // 128 KB

// Number of threads per warp (fixed per GPU)
const u32 WARP_SIZE = 32;

// Warps per SM (adjustable: aim to max occupancy)
const u32 WARPS_PER_SM = 4;

// Number of threads per block (derived)
const u32 BLOCK_SIZE = WARP_SIZE * WARPS_PER_SM;

// Total number of active parallel threads (derived)
const u32 TOTAL_THREADS = SM_COUNT * BLOCK_SIZE;

// Total number of thread blocks
const u32 TOTAL_BLOCKS = TOTAL_THREADS / BLOCK_SIZE;

// Threads used per rewrite unit
const u32 UNIT_SIZE = 4;

// Total number of units
const u32 TOTAL_UNITS = TOTAL_THREADS / UNIT_SIZE;

// Total number of rewrite units (adjustable ratios)
const u32 ANNI_SIZE = TOTAL_UNITS / 2;
const u32 COMM_SIZE = TOTAL_UNITS / 2;

// Total number of nodes (fixed due to 2^28 addressable space)
const u32 NODE_SIZE = 1 << 28;

// Spacing between units, in number of nodes
const u32 UNIT_NODE_SPACE = NODE_SIZE / TOTAL_UNITS;

// Unit types
const u32 ANNI = 0;
const u32 COMM = 1;

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

// Thread shards
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
  Ptr   root;
  Wire* anni_data;
  Wire* comm_data;
  Node* node_data;
  u32   rewrites;
} Net;

// A worker local data
typedef struct {
  u32   tid;       // local thread id (on the block)
  u32   gid;       // global thread id (on the kernel)
  u32   unit;      // unit id (index on acts array)
  u32   type;      // unit type (ANNI|COMM)
  u32   alloc_at;  // where to alloc next node
  Wire* acts_data; // redex buffer we pull from
  u32   acts_size; // max redexes in our buffer
  u32   rewrites;  // total rewrites this performed
  u32   shard;     // aimed shard (A1|A2|B1|B2)
  u32   port;      // aimed port (P1|P2)
  Ptr   a_ptr;     // left pointer of active wire
  Ptr   b_ptr;     // right pointer of active wire
  u32*  shared;    // shared memory for communication
} Worker;

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
  if (tag(ptr) == VRR || tag(ptr) == GOR) {
    return &net->root;
  } else if (tag(ptr) == VR1 || tag(ptr) == GO1) {
    return &net->node_data[val(ptr)].ports[P1];
  } else if (tag(ptr) == VR2 || tag(ptr) == GO2) {
    return &net->node_data[val(ptr)].ports[P2];
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
  return &net->node_data[idx].ports[port];
}

// Allocates a new node in memory
__device__ inline u32 alloc(Worker *worker, Net *net) {
  while (true) {
    if (atomicCAS(&((a64*)net->node_data)[worker->alloc_at + worker->shard], 0, *((u64*)&mknode(BSY, BSY))) == 0) {
      return worker->alloc_at + worker->shard;
    }
    worker->alloc_at = (worker->alloc_at + 4) % NODE_SIZE;
  }
}

// Creates a new active pair
__device__ inline void activ(Worker* worker, Net* net, Ptr a_ptr, Ptr b_ptr) {
  bool anni = tag(a_ptr) == tag(b_ptr);
  u32  aloc = worker->unit;
  a64* data = anni ? (a64*)net->anni_data : (a64*)net->comm_data;
  u32  size = anni ? ANNI_SIZE : COMM_SIZE;
  Wire wire = {a_ptr, b_ptr};
  while (true) {
    if (atomicCAS(&data[aloc], 0, *((u64*)&wire)) == 0) {
      break;
    }
    aloc = (aloc + 1) % size; 
  }
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

// Attempts to replace 'from' by 'to', until it succeeds
__device__ Ptr replace(Ptr* ref, Ptr from, Ptr to) {
  Ptr got = atomicCAS((u32*)ref, from, to);
  while (got != from) {
    got = atomicCAS((u32*)ref, from, to);
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
    Ptr* last_ref = targ.ref;         // save the var's ref
    targ = march(net, targ.ptr);      // clear the backwards path
    clear(self_ref);                  // clear our ref (here, targ.ref == self_ref)
    replace(last_ref, BSY, targ.ptr); // move our node to the last ref
  // Otherwise, create a new active pair. Since two marches reach this branch,
  // we priorize the one with largest ref, in order to avoid race conditions.
  } else if (self_ref < targ.ref) {
    atomicExch((u32*)targ.ref, targ.ptr); // puts the other node back
  } else {
    replace(targ.ref, BSY, 0);                     // clear the other node
    activ(worker, net, deref(self_ref), targ.ptr); // create active pair
    clear(self_ref);                               // clear our node
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

// Annihilation Interaction
__device__ void anni(Worker* self, Net* net) {
  //printf("[%d] anni: tid=%d unit=%d type=%d alloc_at=%d shard=%d port=%d | %llu | %8X <-> %8X\n", self->gid, self->tid, self->unit, self->type, self->alloc_at, self->shard, self->port, clock64(), self->a_ptr, self->b_ptr);

  // Takes my aux port
  Ptr* ak_ref = at(net, val(self->a_ptr), self->port);
  Ptr  ak_ptr = deref(ak_ref);

  // Gets a reference to the other aux port
  Ptr* bk_ref = at(net, val(self->b_ptr), self->port);

  // Redirection to be sent to other side
  u32 xk_ptr = redir(ak_ptr);

  // Send redirection to other side
  replace(bk_ref, BSY, xk_ptr);

  // If we sent a NOD, subst the node in their port, towards our port
  if (!var(ak_ptr)) {
    subst(self, net, bk_ref, mkptr(self->port == P1 ? GO1 : GO2, val(self->a_ptr)));
  }
}

// Commutation Interaction
__device__ void comm(Worker* self, Net* net) {
  //printf("[%d] comm: tid=%d unit=%d type=%d alloc_at=%d shard=%d port=%d | %llu | %8X <-> %8X\n", self->gid, self->tid, self->unit, self->type, self->alloc_at, self->shard, self->port, clock64(), self->a_ptr, self->b_ptr);

  // Takes my aux port
  Ptr* ak_ref = at(net, val(self->a_ptr), self->port);
  Ptr  ak_ptr = deref(ak_ref);

  // Gets a reference to the other aux port
  Ptr* bk_ref = at(net, val(self->b_ptr), self->port);

  // Alloc the clone to send later
  u32 xk_loc = alloc(self, net);
  u32 xk_ptr = mkptr(tag(self->a_ptr), xk_loc);

  // Communicate alloc index to local threads
  self->shared[self->tid] = xk_loc;
  u32 yi;
  switch (self->shard) {
    case A1: yi = self->tid + 2; break;
    case A2: yi = self->tid + 1; break;
    case B1: yi = self->tid - 2; break;
    case B2: yi = self->tid - 3; break;
  }

  // Receive alloc indices from local threads
  __syncthreads();

  //printf("[%d] alloc %d | %d %d\n", self->gid, xk_loc, self->shared[yi+0], self->shared[yi+1]);

  // Fill the clone
  Ptr clone_p1 = mkptr(self->port == P1 ? VR1 : VR2, self->shared[yi+0]);
  Ptr clone_p2 = mkptr(self->port == P1 ? VR1 : VR2, self->shared[yi+1]);
  replace(at(net, xk_loc, P1), BSY, clone_p1);
  replace(at(net, xk_loc, P2), BSY, clone_p2);

  // Send clone to other side
  replace(bk_ref, BSY, xk_ptr);

  // If we have a VAR, subst the clone we got towards that var
  if (var(ak_ptr)) {
    subst(self, net, ak_ref, redir(ak_ptr));
  // If we have a NOD, form an active pair with the clone we got
  } else {
    Ptr got = deref(ak_ref);
    clear(ak_ref);
    activ(self, net, ak_ptr, got);
  }

  //printf("[%d] DONE: tid=%d unit=%d type=%d alloc_at=%d shard=%d port=%d | %llu\n", self->gid, self->tid, self->unit, self->type, self->alloc_at, self->shard, self->port, clock64());
}

__global__ void work(Net* net) {
  // Shared buffer for warp communication
  __shared__ u32 shared[BLOCK_SIZE];

  // Initializes local vars
  Worker worker;
  worker.tid        = threadIdx.x;
  worker.gid        = blockIdx.x * blockDim.x + threadIdx.x;
  worker.unit       = (worker.gid / UNIT_SIZE) % ANNI_SIZE;
  worker.type       = worker.gid < TOTAL_THREADS/2 ? ANNI : COMM;
  worker.acts_data  = worker.type == ANNI ? net->anni_data : net->comm_data;
  worker.acts_size  = worker.type == ANNI ? ANNI_SIZE : COMM_SIZE;
  worker.alloc_at   = UNIT_NODE_SPACE * (worker.gid / UNIT_SIZE);
  worker.rewrites   = 0;
  worker.shard      = worker.tid % 4;
  worker.port       = worker.shard % 2 == 0 ? P1 : P2;
  worker.shared     = shared;

  //printf("[%d] ON: tid=%d unit=%d type=%d alloc_at=%d shard=%d port=%d\n", worker.gid, worker.tid, worker.unit, worker.type, worker.alloc_at, worker.shard, worker.port);

  // Clears shared buffer
  shared[worker.tid] = 0;
  __syncthreads();

  // Pops an active wire
  Wire active_wire;
  if (worker.shard == A1) {
    *((u64*)&active_wire) = atomicExch((a64*)&worker.acts_data[worker.unit], 0);
    if (active_wire.lft != 0) {
      worker.rewrites += 1;
      shared[worker.tid + 0] = active_wire.lft;
      shared[worker.tid + 1] = active_wire.rgt;
      shared[worker.tid + 2] = active_wire.rgt;
      shared[worker.tid + 3] = active_wire.lft;
    }
  }
  __syncthreads();

  // Sets main ports
  worker.a_ptr = shared[worker.port == P1 ? worker.tid + 0 : worker.tid - 1];
  worker.b_ptr = shared[worker.port == P1 ? worker.tid + 1 : worker.tid + 0];

  // Performs a reduction
  if (worker.a_ptr != 0) {
    if (worker.type == ANNI) {
      anni(&worker, net);
    } else {
      comm(&worker, net);
    }
  }

  // When the work ends, sum stats
  if (worker.shard == A1) {
    atomicAdd(&net->rewrites, worker.rewrites);
  }
}

// Host<->Device
// -------------

__host__ Net* mknet() {
  Net* net       = (Net*)malloc(sizeof(Net));
  net->root      = mkptr(NIL, 0);
  net->rewrites  = 0;
  net->anni_data = (Wire*)malloc(ANNI_SIZE * sizeof(Wire));
  net->comm_data = (Wire*)malloc(COMM_SIZE * sizeof(Wire));
  net->node_data = (Node*)malloc(NODE_SIZE * sizeof(Node));
  memset(net->anni_data, 0, ANNI_SIZE * sizeof(Wire));
  memset(net->comm_data, 0, COMM_SIZE * sizeof(Wire));
  memset(net->node_data, 0, NODE_SIZE * sizeof(Node));
  return net;
}

__host__ Net* net_to_device(Net* host_net) {
  // Allocate memory on the device for the Net object, and its data
  Net* device_net;
  Wire* device_anni_data;
  Wire* device_comm_data;
  Node* device_node_data;

  cudaMalloc((void**)&device_net, sizeof(Net));
  cudaMalloc((void**)&device_anni_data, ANNI_SIZE * sizeof(Wire));
  cudaMalloc((void**)&device_comm_data, COMM_SIZE * sizeof(Wire));
  cudaMalloc((void**)&device_node_data, NODE_SIZE * sizeof(Node));

  // Copy the host data to the device memory
  cudaMemcpy(device_anni_data, host_net->anni_data, ANNI_SIZE * sizeof(Wire), cudaMemcpyHostToDevice);
  cudaMemcpy(device_comm_data, host_net->comm_data, COMM_SIZE * sizeof(Wire), cudaMemcpyHostToDevice);
  cudaMemcpy(device_node_data, host_net->node_data, NODE_SIZE * sizeof(Node), cudaMemcpyHostToDevice);

  // Create a temporary host Net object with device pointers
  Net temp_net = *host_net;
  temp_net.anni_data = device_anni_data;
  temp_net.comm_data = device_comm_data;
  temp_net.node_data = device_node_data;

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
  host_net->anni_data = (Wire*)malloc(ANNI_SIZE * sizeof(Wire));
  host_net->comm_data = (Wire*)malloc(COMM_SIZE * sizeof(Wire));
  host_net->node_data = (Node*)malloc(NODE_SIZE * sizeof(Node));

  // Retrieve the device pointers for data
  Wire* device_anni_data;
  Wire* device_comm_data;
  Node* device_node_data;
  cudaMemcpy(&device_anni_data, &(device_net->anni_data), sizeof(Wire*), cudaMemcpyDeviceToHost);
  cudaMemcpy(&device_comm_data, &(device_net->comm_data), sizeof(Wire*), cudaMemcpyDeviceToHost);
  cudaMemcpy(&device_node_data, &(device_net->node_data), sizeof(Node*), cudaMemcpyDeviceToHost);

  // Copy the device data to the host memory
  cudaMemcpy(host_net->anni_data, device_anni_data, ANNI_SIZE * sizeof(Wire), cudaMemcpyDeviceToHost);
  cudaMemcpy(host_net->comm_data, device_comm_data, COMM_SIZE * sizeof(Wire), cudaMemcpyDeviceToHost);
  cudaMemcpy(host_net->node_data, device_node_data, NODE_SIZE * sizeof(Node), cudaMemcpyDeviceToHost);

  return host_net;
}

__host__ void net_free_on_device(Net* device_net) {
  // Retrieve the device pointers for data
  Wire* device_anni_data;
  Wire* device_comm_data;
  Node* device_node_data;
  cudaMemcpy(&device_anni_data, &(device_net->anni_data), sizeof(Wire*), cudaMemcpyDeviceToHost);
  cudaMemcpy(&device_comm_data, &(device_net->comm_data), sizeof(Wire*), cudaMemcpyDeviceToHost);
  cudaMemcpy(&device_node_data, &(device_net->node_data), sizeof(Node*), cudaMemcpyDeviceToHost);

  // Free the device memory
  cudaFree(device_anni_data);
  cudaFree(device_comm_data);
  cudaFree(device_node_data);
  cudaFree(device_net);
}

__host__ void net_free_on_host(Net* host_net) {
  free(host_net->anni_data);
  free(host_net->comm_data);
  free(host_net->node_data);
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
  printf("Anni:\n");
  for (u32 i = 0; i < ANNI_SIZE; ++i) {
    Ptr a = net->anni_data[i].lft;
    Ptr b = net->anni_data[i].rgt;
    if (a != 0 || b != 0) {
      printf("- [%07X] %s %s\n", i, Ptr_show(a,0), Ptr_show(b,1));
    }
  }
  printf("Comm:\n");
  for (u32 i = 0; i < COMM_SIZE; ++i) {
    Ptr a = net->comm_data[i].lft;
    Ptr b = net->comm_data[i].rgt;
    if (a != 0 || b != 0) {
      printf("- [%07X] %s %s\n", i, Ptr_show(a,0), Ptr_show(b,1));
    }
  }
  printf("Node:\n");
  for (u32 i = 0; i < NODE_SIZE; ++i) {
    Ptr a = net->node_data[i].ports[P1];
    Ptr b = net->node_data[i].ports[P2];
    if (a != 0 || b != 0) {
      printf("- [%07X] %s %s\n", i, Ptr_show(a,0), Ptr_show(b,1));
    }
  }
  //printf("Used: %zu\n", net->used_mem);
  printf("Rwts: %zu\n", net->rewrites);
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
        print_tree_go(net, net->node_data[val(ptr)].ports[P1], var_ids);
        printf(" ");
        print_tree_go(net, net->node_data[val(ptr)].ports[P2], var_ids);
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
  net->root          = 0x60000001;
  net->anni_data[ 0] = (Wire) {0xa0000000,0xa0000001};
  net->node_data[ 0] = (Node) {0x60000000,0x50000000};
  net->node_data[ 1] = (Node) {0xa0000002,0x40000000};
  net->node_data[ 2] = (Node) {0x60000002,0x50000002};
}

// term_b = (λfλx(f x) λy(y))
__host__ void inject_term_b(Net* net) {
  net->root          = 0x60000003;
  net->anni_data[ 0] = (Wire) {0xa0000000,0xa0000003};
  net->node_data[ 0] = (Node) {0xa0000001,0xa0000002};
  net->node_data[ 1] = (Node) {0x50000002,0x60000002};
  net->node_data[ 2] = (Node) {0x50000001,0x60000001};
  net->node_data[ 3] = (Node) {0xa0000004,0x40000000};
  net->node_data[ 4] = (Node) {0x60000004,0x50000004};
}

// term_c = (λfλx(f (f x)) λx(x))
__host__ void inject_term_c(Net* net) {
  net->root          = 0x60000005;
  net->anni_data[ 0] = (Wire) {0xa0000000,0xa0000005};
  net->anni_data[ 0] = (Wire) {0xa0000005,0xa0000000};
  net->node_data[ 0] = (Node) {0xb0000001,0xa0000004};
  net->node_data[ 1] = (Node) {0xa0000002,0xa0000003};
  net->node_data[ 2] = (Node) {0x50000004,0x50000003};
  net->node_data[ 3] = (Node) {0x60000002,0x60000004};
  net->node_data[ 4] = (Node) {0x50000002,0x60000003};
  net->node_data[ 5] = (Node) {0xa0000006,0x40000000};
  net->node_data[ 6] = (Node) {0x60000006,0x50000006};
}

// term_d = (λfλx(f (f x)) λgλy(g (g y)))
__host__ void inject_term_d(Net* net) {
  net->root          = 0x60000005;
  net->anni_data[ 0] = (Wire) {0xa0000000,0xa0000005};
  net->node_data[ 0] = (Node) {0xb0000001,0xa0000004};
  net->node_data[ 1] = (Node) {0xa0000002,0xa0000003};
  net->node_data[ 2] = (Node) {0x50000004,0x50000003};
  net->node_data[ 3] = (Node) {0x60000002,0x60000004};
  net->node_data[ 4] = (Node) {0x50000002,0x60000003};
  net->node_data[ 5] = (Node) {0xa0000006,0x40000000};
  net->node_data[ 6] = (Node) {0xc0000007,0xa000000a};
  net->node_data[ 7] = (Node) {0xa0000008,0xa0000009};
  net->node_data[ 8] = (Node) {0x5000000a,0x50000009};
  net->node_data[ 9] = (Node) {0x60000008,0x6000000a};
  net->node_data[10] = (Node) {0x50000008,0x60000009};
}

// term_e = (c2 g_s g_z)
__host__ void inject_term_e(Net* net) {
  net->root          = 0x6000000b;
  net->anni_data[ 0] = (Wire) {0xa0000000,0xa0000005};
  net->node_data[ 0] = (Node) {0xb0000001,0xa0000004};
  net->node_data[ 1] = (Node) {0xa0000002,0xa0000003};
  net->node_data[ 2] = (Node) {0x50000004,0x50000003};
  net->node_data[ 3] = (Node) {0x60000002,0x60000004};
  net->node_data[ 4] = (Node) {0x50000002,0x60000003};
  net->node_data[ 5] = (Node) {0xa0000006,0xa000000b};
  net->node_data[ 6] = (Node) {0xc0000007,0xa0000008};
  net->node_data[ 7] = (Node) {0x50000009,0x5000000a};
  net->node_data[ 8] = (Node) {0xa0000009,0x6000000a};
  net->node_data[ 9] = (Node) {0x50000007,0xa000000a};
  net->node_data[10] = (Node) {0x60000007,0x60000008};
  net->node_data[11] = (Node) {0xa000000c,0x40000000};
  net->node_data[12] = (Node) {0x6000000c,0x5000000c};
}

// term_f = (c3 g_s g_z)
__host__ void inject_term_f(Net* net) {
  net->root          = 0x6000000d;
  net->anni_data[ 0] = (Wire) {0xa0000000,0xa0000007};
  net->node_data[ 0] = (Node) {0xb0000001,0xa0000006};
  net->node_data[ 1] = (Node) {0xb0000002,0xa0000005};
  net->node_data[ 2] = (Node) {0xa0000003,0xa0000004};
  net->node_data[ 3] = (Node) {0x50000006,0x50000004};
  net->node_data[ 4] = (Node) {0x60000003,0x50000005};
  net->node_data[ 5] = (Node) {0x60000004,0x60000006};
  net->node_data[ 6] = (Node) {0x50000003,0x60000005};
  net->node_data[ 7] = (Node) {0xa0000008,0xa000000d};
  net->node_data[ 8] = (Node) {0xc0000009,0xa000000a};
  net->node_data[ 9] = (Node) {0x5000000b,0x5000000c};
  net->node_data[10] = (Node) {0xa000000b,0x6000000c};
  net->node_data[11] = (Node) {0x50000009,0xa000000c};
  net->node_data[12] = (Node) {0x60000009,0x6000000a};
  net->node_data[13] = (Node) {0xa000000e,0x40000000};
  net->node_data[14] = (Node) {0x6000000e,0x5000000e};
}

// term_g = (c5 g_s g_z)
__host__ void inject_term_g(Net* net) {
  net->root          = 0x60000011;
  net->anni_data[ 0] = (Wire) {0xa0000000,0xa000000b};
  net->node_data[ 0] = (Node) {0xb0000001,0xa000000a};
  net->node_data[ 1] = (Node) {0xb0000002,0xa0000009};
  net->node_data[ 2] = (Node) {0xb0000003,0xa0000008};
  net->node_data[ 3] = (Node) {0xb0000004,0xa0000007};
  net->node_data[ 4] = (Node) {0xa0000005,0xa0000006};
  net->node_data[ 5] = (Node) {0x5000000a,0x50000006};
  net->node_data[ 6] = (Node) {0x60000005,0x50000007};
  net->node_data[ 7] = (Node) {0x60000006,0x50000008};
  net->node_data[ 8] = (Node) {0x60000007,0x50000009};
  net->node_data[ 9] = (Node) {0x60000008,0x6000000a};
  net->node_data[10] = (Node) {0x50000005,0x60000009};
  net->node_data[11] = (Node) {0xa000000c,0xa0000011};
  net->node_data[12] = (Node) {0xc000000d,0xa000000e};
  net->node_data[13] = (Node) {0x5000000f,0x50000010};
  net->node_data[14] = (Node) {0xa000000f,0x60000010};
  net->node_data[15] = (Node) {0x5000000d,0xa0000010};
  net->node_data[16] = (Node) {0x6000000d,0x6000000e};
  net->node_data[17] = (Node) {0xa0000012,0x40000000};
  net->node_data[18] = (Node) {0x60000012,0x50000012};
}

// term_h = (c12 g_s g_z)
__host__ void inject_term_h(Net* net) {
  net->root          = 0x6000001f;
  net->anni_data[ 0] = (Wire) {0xa0000000,0xa0000019};
  net->node_data[ 0] = (Node) {0xb0000001,0xa0000018};
  net->node_data[ 1] = (Node) {0xb0000002,0xa0000017};
  net->node_data[ 2] = (Node) {0xb0000003,0xa0000016};
  net->node_data[ 3] = (Node) {0xb0000004,0xa0000015};
  net->node_data[ 4] = (Node) {0xb0000005,0xa0000014};
  net->node_data[ 5] = (Node) {0xb0000006,0xa0000013};
  net->node_data[ 6] = (Node) {0xb0000007,0xa0000012};
  net->node_data[ 7] = (Node) {0xb0000008,0xa0000011};
  net->node_data[ 8] = (Node) {0xb0000009,0xa0000010};
  net->node_data[ 9] = (Node) {0xb000000a,0xa000000f};
  net->node_data[10] = (Node) {0xb000000b,0xa000000e};
  net->node_data[11] = (Node) {0xa000000c,0xa000000d};
  net->node_data[12] = (Node) {0x50000018,0x5000000d};
  net->node_data[13] = (Node) {0x6000000c,0x5000000e};
  net->node_data[14] = (Node) {0x6000000d,0x5000000f};
  net->node_data[15] = (Node) {0x6000000e,0x50000010};
  net->node_data[16] = (Node) {0x6000000f,0x50000011};
  net->node_data[17] = (Node) {0x60000010,0x50000012};
  net->node_data[18] = (Node) {0x60000011,0x50000013};
  net->node_data[19] = (Node) {0x60000012,0x50000014};
  net->node_data[20] = (Node) {0x60000013,0x50000015};
  net->node_data[21] = (Node) {0x60000014,0x50000016};
  net->node_data[22] = (Node) {0x60000015,0x50000017};
  net->node_data[23] = (Node) {0x60000016,0x60000018};
  net->node_data[24] = (Node) {0x5000000c,0x60000017};
  net->node_data[25] = (Node) {0xa000001a,0xa000001f};
  net->node_data[26] = (Node) {0xc000001b,0xa000001c};
  net->node_data[27] = (Node) {0x5000001d,0x5000001e};
  net->node_data[28] = (Node) {0xa000001d,0x6000001e};
  net->node_data[29] = (Node) {0x5000001b,0xa000001e};
  net->node_data[30] = (Node) {0x6000001b,0x6000001c};
  net->node_data[31] = (Node) {0xa0000020,0x40000000};
  net->node_data[32] = (Node) {0x60000020,0x50000020};
}

// term_h = (c16 g_s g_z)
__host__ void inject_term_i(Net* net) {
  net->root          = 0x60000027;
  net->anni_data[ 0] = (Wire) {0xa0000000,0xa0000021};
  net->node_data[ 0] = (Node) {0xb0000001,0xa0000020};
  net->node_data[ 1] = (Node) {0xb0000002,0xa000001f};
  net->node_data[ 2] = (Node) {0xb0000003,0xa000001e};
  net->node_data[ 3] = (Node) {0xb0000004,0xa000001d};
  net->node_data[ 4] = (Node) {0xb0000005,0xa000001c};
  net->node_data[ 5] = (Node) {0xb0000006,0xa000001b};
  net->node_data[ 6] = (Node) {0xb0000007,0xa000001a};
  net->node_data[ 7] = (Node) {0xb0000008,0xa0000019};
  net->node_data[ 8] = (Node) {0xb0000009,0xa0000018};
  net->node_data[ 9] = (Node) {0xb000000a,0xa0000017};
  net->node_data[10] = (Node) {0xb000000b,0xa0000016};
  net->node_data[11] = (Node) {0xb000000c,0xa0000015};
  net->node_data[12] = (Node) {0xb000000d,0xa0000014};
  net->node_data[13] = (Node) {0xb000000e,0xa0000013};
  net->node_data[14] = (Node) {0xb000000f,0xa0000012};
  net->node_data[15] = (Node) {0xa0000010,0xa0000011};
  net->node_data[16] = (Node) {0x50000020,0x50000011};
  net->node_data[17] = (Node) {0x60000010,0x50000012};
  net->node_data[18] = (Node) {0x60000011,0x50000013};
  net->node_data[19] = (Node) {0x60000012,0x50000014};
  net->node_data[20] = (Node) {0x60000013,0x50000015};
  net->node_data[21] = (Node) {0x60000014,0x50000016};
  net->node_data[22] = (Node) {0x60000015,0x50000017};
  net->node_data[23] = (Node) {0x60000016,0x50000018};
  net->node_data[24] = (Node) {0x60000017,0x50000019};
  net->node_data[25] = (Node) {0x60000018,0x5000001a};
  net->node_data[26] = (Node) {0x60000019,0x5000001b};
  net->node_data[27] = (Node) {0x6000001a,0x5000001c};
  net->node_data[28] = (Node) {0x6000001b,0x5000001d};
  net->node_data[29] = (Node) {0x6000001c,0x5000001e};
  net->node_data[30] = (Node) {0x6000001d,0x5000001f};
  net->node_data[31] = (Node) {0x6000001e,0x60000020};
  net->node_data[32] = (Node) {0x50000010,0x6000001f};
  net->node_data[33] = (Node) {0xa0000022,0xa0000027};
  net->node_data[34] = (Node) {0xc0000023,0xa0000024};
  net->node_data[35] = (Node) {0x50000025,0x50000026};
  net->node_data[36] = (Node) {0xa0000025,0x60000026};
  net->node_data[37] = (Node) {0x50000023,0xa0000026};
  net->node_data[38] = (Node) {0x60000023,0x60000024};
  net->node_data[39] = (Node) {0xa0000028,0x40000000};
  net->node_data[40] = (Node) {0x60000028,0x50000028};
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

  // Allocates the initial net on device
  Net* h_net = mknet();
  inject_term_h(h_net);

  // Prints the initial net
  print_net(h_net);

  // Sends the net from host to device
  Net* d_net = net_to_device(h_net);

  // Performs parallel reductions
  for (u32 i = 0; i < 128; ++i) {
    work<<<TOTAL_BLOCKS, BLOCK_SIZE>>>(d_net);
  }

  // Reads the normalized net from device to host
  Net* norm = net_to_host(d_net);

  // Prints the normal form (raw data)
  //print_net(norm);

  // Prints the nromal form (as a tree)
  printf("Normal:\n");
  print_tree(norm, norm->root);
  printf("\n");

  // Prints just the rewrites
  printf("Rwts: %d\n", norm->rewrites);

  //// Free device memory
  net_free_on_device(d_net);

  //// Free host memory
  net_free_on_host(h_net);
  net_free_on_host(norm);

  return 0;
}
