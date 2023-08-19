#include <stdint.h>
#include <stdbool.h>
#include <string.h>
#include <stdio.h>

typedef uint8_t u8;
typedef uint32_t u32;
typedef uint64_t u64;
typedef unsigned long long int a64;

// Shared Memory Size = 128 KB
const u32 SM_SIZE = 128 * 1024;

// Threads per Block = 128
const u32 BLOCK_SIZE = 128;

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
const Tag CON = 0xa; // points to main port of con node
const Tag DUP = 0xb; // points to main port of dup node; higher labels also dups
const u32 BSY = 0xFFFFFFFF; // value taken by another thread, will be replaced soon

// Thread types
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
  Ptr    root;
  Wire*  anni_data;
  u32    anni_size;
  Wire*  comm_data;
  u32    comm_size;
  Node*  node_data;
  u32    node_size;
  u32    used_mem;
  u32    rewrites;
} Net;

// A worker local data
typedef struct {
  u32  tid;
  u32  gid;
  u32  alloc_at;
  u32  new_anni;
  u32  new_comm;
  u32  rewrites;
  u32  used_mem;
  u32  type;
  u32  port;
  Ptr  a_ptr;
  Ptr  b_ptr;
  u32* shared;
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

// Inserts a value in the first empty slot
__device__ inline u32 insert(a64* arr, u32 len, u32* idx, a64 val) {
  while (true) {
    if (atomicCAS(&arr[*idx], 0, val) == 0) {
      return *idx;
    }
    *idx = (*idx + 1) % len; 
  }
}

// Allocates a new node in memory
__device__ inline u32 alloc(Worker *worker, Net *net) {
  return insert((a64*)net->node_data, net->node_size, &worker->alloc_at, *((u64*)&mknode(BSY, BSY)));
}

// Creates a new active pair
__device__ inline void activ(Worker* worker, Net* net, Ptr a_ptr, Ptr b_ptr) {
  bool anni = tag(a_ptr) == tag(b_ptr);
  a64* data = anni ? (a64*)net->anni_data : (a64*)net->comm_data;
  u32  size = anni ? net->anni_size       : net->comm_size;
  u32* newx = anni ? &worker->new_anni    : &worker->new_comm;
  Wire wire = {a_ptr, b_ptr};
  insert(data, size, newx, *((u64*)&wire));
}

// Empties a slot in memory
__device__ Ptr clear(Ptr* ref) {
  atomicExch((u32*)ref, 0);
}

// Gets the value of a ref; waits if busy
__device__ Ptr deref(Ptr* ref, Ptr new_value) {
  Ptr got = atomicExch((u32*)ref, new_value);
  while (got == BSY) {
    got = atomicExch((u32*)ref, new_value);
  }
  return got;
}

// Attempts to replace 'from' by 'to' until it succeeds
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
__device__ Target march(Worker* worker, Net* net, Ptr ptr) {
  // Gets the immediate target
  Target targ;
  targ.ref = target(net, ptr);
  targ.ptr = deref(targ.ref, BSY);
  // While it is a redirection, clear and get the next target
  while (tag(targ.ptr) >= GOR && tag(targ.ptr) <= GO2) {
    clear(targ.ref);
    targ.ref = target(net, targ.ptr);
    targ.ptr = deref(targ.ref, BSY);
  }
  return targ;
}

// Substs the node in 'self_ref' into the destination of 'target_ptr'
// - If the destination is a VAR => move the node into it
// - If the destination is a NOD => create an active pair
__device__ void subst(Worker* worker, Net* net, Ptr* self_ref, Ptr target_ptr) {
  // Marches towards target_ptr
  Target targ = march(worker, net, target_ptr);
  // If the final target is a var, move this node into it
  if (tag(targ.ptr) >= VRR && tag(targ.ptr) <= VR2) { 
    Ptr* last_ref = targ.ref;            // Saves the var's ref
    targ = march(worker, net, targ.ptr); // Clears the backwards path
    clear(targ.ref);                     // Clears our ref (targ_ref == self_ref)
    replace(last_ref, BSY, targ.ptr);    // Moves our node to the last ref
  // Otherwise, create a new active pair. Since two marches reach this branch,
  // we priorize the one with largest ref, in order to avoid race conditions.
  } else if (self_ref < targ.ref) {
    atomicExch((u32*)targ.ref, targ.ptr); // puts targ_ptr back
  } else {
    replace(targ.ref, BSY, 0);
    activ(worker, net, deref(self_ref, 0), targ.ptr); // create active pair
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

__device__ Worker new_worker(Net* net) {
  // Shared buffer for warp communication
  __shared__ u32 shared[BLOCK_SIZE];

  // Initializes local vars
  Worker worker;
  worker.tid      = threadIdx.x;
  worker.gid      = blockIdx.x * blockDim.x + threadIdx.x;
  worker.alloc_at = 0;
  worker.new_anni = 0;
  worker.new_comm = 0;
  worker.rewrites = 0;
  worker.used_mem = 0; 
  worker.type     = worker.tid % 4;
  worker.port     = worker.type % 2 == 0 ? P1 : P2;
  worker.shared   = shared;

  // Gets the active wire
  Wire active_wire;
  if (worker.type == 0) {
    *((u64*)&active_wire)  = atomicExch((a64*)&net->anni_data[worker.gid / 4], 0);
    shared[worker.tid + 0] = active_wire.lft;
    shared[worker.tid + 1] = active_wire.rgt;
    shared[worker.tid + 2] = active_wire.rgt;
    shared[worker.tid + 3] = active_wire.lft;
  }

  // Synchronizes writes
  __syncthreads();

  // Gets main ports
  worker.a_ptr = shared[worker.port == P1 ? worker.tid + 0 : worker.tid - 1];
  worker.b_ptr = shared[worker.port == P1 ? worker.tid + 1 : worker.tid + 0];

  return worker;
}

// Annihilation Interaction
__global__ void anni(Net* net) {
  // Creates the local worker object
  Worker* self = &new_worker(net);

  // If there is nothing to do, return
  if (self->a_ptr == 0 || self->b_ptr == 0) { return; }

  // Takes my aux port
  Ptr* ak_ref = at(net, val(self->a_ptr), self->port);
  Ptr  ak_ptr = deref(ak_ref, BSY);

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
__global__ void comm(Net* net) {
  // Creates the local worker object
  Worker* self = &new_worker(net);

  // If there is nothing to do, return
  if (self->a_ptr == 0 || self->b_ptr == 0) { return; }

  // Takes my aux port
  Ptr* ak_ref = at(net, val(self->a_ptr), self->port);
  Ptr  ak_ptr = deref(ak_ref, BSY);

  // Gets a reference to the other aux port
  Ptr* bk_ref = at(net, val(self->b_ptr), self->port);

  // Alloc the clone to send later
  u32 xk_loc = alloc(self, net);
  u32 xk_ptr = mkptr(tag(self->a_ptr), xk_loc);

  // Communicate alloc index to local threads
  self->shared[self->tid] = xk_loc;
  u32 yi;
  switch (self->type) {
    case A1: yi = self->tid + 2; break;
    case A2: yi = self->tid + 1; break;
    case B1: yi = self->tid - 2; break;
    case B2: yi = self->tid - 3; break;
  }

  // Receive alloc indices from local threads
  __syncthreads();

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
    activ(self, net, ak_ptr, deref(ak_ref,0));
  }
}

// Host<->Device
// -------------

__host__ Net* Net_new(Ptr root, Wire* anni_data, u32 anni_size, Wire* comm_data, u32 comm_size, Node* node_data, u32 node_size) {
  // Allocate and initialize a new Net object on the host
  Net* host_net       = (Net*)malloc(sizeof(Net));
  host_net->root      = root;
  host_net->anni_size = anni_size;
  host_net->comm_size = comm_size;
  host_net->node_size = node_size;
  host_net->used_mem  = 0;
  host_net->rewrites  = 0;
  host_net->anni_data = anni_data;
  host_net->comm_data = comm_data;
  host_net->node_data = node_data;
  return host_net;
}

__host__ Net* Net_to_device(Net* host_net) {
  // Allocate memory on the device for the Net object, and its data
  Net* device_net;
  Wire* device_anni_data;
  Wire* device_comm_data;
  Node* device_node_data;

  cudaMalloc((void**)&device_net, sizeof(Net));
  cudaMalloc((void**)&device_anni_data, host_net->anni_size * sizeof(Wire));
  cudaMalloc((void**)&device_comm_data, host_net->comm_size * sizeof(Wire));
  cudaMalloc((void**)&device_node_data, host_net->node_size * sizeof(Node));

  // Copy the host data to the device memory
  cudaMemcpy(device_anni_data, host_net->anni_data, host_net->anni_size * sizeof(Wire), cudaMemcpyHostToDevice);
  cudaMemcpy(device_comm_data, host_net->comm_data, host_net->comm_size * sizeof(Wire), cudaMemcpyHostToDevice);
  cudaMemcpy(device_node_data, host_net->node_data, host_net->node_size * sizeof(Node), cudaMemcpyHostToDevice);

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

__host__ Net* Net_to_host(Net* device_net) {
  // Create a new host Net object
  Net* host_net = (Net*)malloc(sizeof(Net));

  // Copy the device Net object to the host memory
  cudaMemcpy(host_net, device_net, sizeof(Net), cudaMemcpyDeviceToHost);

  // Allocate host memory for data
  host_net->anni_data = (Wire*)malloc(host_net->anni_size * sizeof(Wire));
  host_net->comm_data = (Wire*)malloc(host_net->comm_size * sizeof(Wire));
  host_net->node_data = (Node*)malloc(host_net->node_size * sizeof(Node));

  // Retrieve the device pointers for data
  Wire* device_anni_data;
  Wire* device_comm_data;
  Node* device_node_data;
  cudaMemcpy(&device_anni_data, &(device_net->anni_data), sizeof(Wire*), cudaMemcpyDeviceToHost);
  cudaMemcpy(&device_comm_data, &(device_net->comm_data), sizeof(Wire*), cudaMemcpyDeviceToHost);
  cudaMemcpy(&device_node_data, &(device_net->node_data), sizeof(Node*), cudaMemcpyDeviceToHost);

  // Copy the device data to the host memory
  cudaMemcpy(host_net->anni_data, device_anni_data, host_net->anni_size * sizeof(Wire), cudaMemcpyDeviceToHost);
  cudaMemcpy(host_net->comm_data, device_comm_data, host_net->comm_size * sizeof(Wire), cudaMemcpyDeviceToHost);
  cudaMemcpy(host_net->node_data, device_node_data, host_net->node_size * sizeof(Node), cudaMemcpyDeviceToHost);

  return host_net;
}

// Debugging
// ---------

__host__ const char* Ptr_show(Ptr ptr, u32 slot) {
  static char buffer[8][12];
  if (ptr == 0) {
    strcpy(buffer[slot], "           ");
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
      case 0xF: tag_str = "BSY"; break;
      default : tag_str = "UNK"; break;
    }
    snprintf(buffer[slot], sizeof(buffer[slot]), "%s:%07X", tag_str, val(ptr));
    return buffer[slot];
  }
}

// Prints a net in hexadecimal, limited to a given size
void Net_print(Net* net, u32 lim_size) {
  printf("Root:\n");
  printf("- %s\n", Ptr_show(net->root,0));
  printf("Anni:\n");
  for (u32 i = 0; i < lim_size && i < net->anni_size; ++i) {
    Ptr a = net->anni_data[i].lft;
    Ptr b = net->anni_data[i].rgt;
    if (a != 0 || b != 0) {
      printf("- [%07X] %s %s\n", i, Ptr_show(a,0), Ptr_show(b,1));
    }
  }
  printf("Comm:\n");
  for (u32 i = 0; i < lim_size && i < net->comm_size; ++i) {
    Ptr a = net->comm_data[i].lft;
    Ptr b = net->comm_data[i].rgt;
    if (a != 0 || b != 0) {
      printf("- [%07X] %s %s\n", i, Ptr_show(a,0), Ptr_show(b,1));
    }
  }
  printf("Node:\n");
  for (u32 i = 0; i < lim_size && i < net->node_size; ++i) {
    Ptr a = net->node_data[i].ports[P1];
    Ptr b = net->node_data[i].ports[P2];
    if (a != 0 || b != 0) {
      printf("- [%07X] %s %s\n", i, Ptr_show(a,0), Ptr_show(b,1));
    }
  }
  printf("Used: %zu\n", net->used_mem);
  printf("Rwts: %zu\n", net->rewrites);
}

// ~
// ~
// ~

// Main
// ----

int main() {
  printf("Starting...\n");
  
  // Prints device info
  int device;
  cudaDeviceProp prop;
  cudaGetDevice(&device);
  cudaGetDeviceProperties(&prop, device);
  printf("CUDA Device: %s, Compute Capability: %d.%d\n", prop.name, prop.major, prop.minor);

  // Host data
  u32 anni_size   = 32;
  u32 comm_size   = 32;
  u32 node_size   = 256;
  Wire* anni_data = (Wire*)malloc(anni_size * sizeof(Wire));
  Wire* comm_data = (Wire*)malloc(comm_size * sizeof(Wire));
  Node* node_data = (Node*)malloc(node_size * sizeof(Node));
  memset(anni_data, 0, anni_size * sizeof(Wire));
  memset(comm_data, 0, comm_size * sizeof(Wire));
  memset(node_data, 0, node_size * sizeof(Node));

  // (λx(x) λy(y))
  //u32 root      = 0x60000001;
  //anni_data[ 0] = (Wire) {0xa0000000,0xa0000001};
  //node_data[ 0] = (Node) {0x60000000,0x50000000};
  //node_data[ 1] = (Node) {0xa0000002,0x40000000};
  //node_data[ 2] = (Node) {0x60000002,0x50000002};

  // (λfλx(f x) λy(y))
  //u32 root      = 0x60000003;
  //anni_data[ 0] = (Wire) {0xa0000000,0xa0000003};
  //node_data[ 0] = (Node) {0xa0000001,0xa0000002};
  //node_data[ 1] = (Node) {0x50000002,0x60000002};
  //node_data[ 2] = (Node) {0x50000001,0x60000001};
  //node_data[ 3] = (Node) {0xa0000004,0x40000000};
  //node_data[ 4] = (Node) {0x60000004,0x50000004};

  // (λfλx(f (f x)) λx(x))
  u32 root      = 0x60000005;
  anni_data[ 0] = (Wire) {0xa0000000,0xa0000005};
  anni_data[ 0] = (Wire) {0xa0000005,0xa0000000};
  node_data[ 0] = (Node) {0xb0000001,0xa0000004};
  node_data[ 1] = (Node) {0xa0000002,0xa0000003};
  node_data[ 2] = (Node) {0x50000004,0x50000003};
  node_data[ 3] = (Node) {0x60000002,0x60000004};
  node_data[ 4] = (Node) {0x50000002,0x60000003};
  node_data[ 5] = (Node) {0xa0000006,0x40000000};
  node_data[ 6] = (Node) {0x60000006,0x50000006};

  Net* h_net = Net_new(root, anni_data, anni_size, comm_data, comm_size, node_data, node_size);
  Net_print(h_net, 32);
  printf("--------------\n");

  Net* d_net = Net_to_device(h_net);

  anni<<<1,4>>>(d_net);
  //work<<<1,4>>>(d_net);
  //work<<<1,3>>>(d_net);

  Net* h_net2 = Net_to_host(d_net);
  Net_print(h_net2, 32);
  printf("--------------\n");

  //// Free device memory
  //cudaFree(device_net->anni_data);
  //cudaFree(device_net->comm_data);
  //cudaFree(device_net->node_data);
  //cudaFree(device_net);

  //// Free host memory
  //free(updated_host_net->anni_data);
  //free(updated_host_net->comm_data);
  //free(updated_host_net->node_data);
  //free(updated_host_net);

  return 0;
}


