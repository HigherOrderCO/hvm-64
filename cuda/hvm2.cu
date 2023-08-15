#include <stdint.h>
#include <stdbool.h>
#include <string.h>
#include <stdio.h>

typedef uint8_t u8;
typedef uint32_t u32;
typedef uint64_t u64;
typedef unsigned long long int a64;

typedef u32 Val;

typedef u8 Tag;
const Tag NIL = 0; // empty node
const Tag REF = 1; // reference to a definition (closed net)
const Tag NUM = 2; // unboxed number
const Tag ERA = 3; // unboxed eraser
const Tag VRT = 4; // variable pointing to root
const Tag VR1 = 5; // variable pointing to aux1 port of node
const Tag VR2 = 6; // variable pointing to aux2 port of node
const Tag CON = 7; // points to main port of con node
const Tag DUP = 8; // points to main port of dup node; higher labels also dups

// Shared Memory Size = 128 KB
const size_t SM_SIZE = 128 * 1024;

// just prints a hello, world
int main() {
  printf("Hello, world!\n");
  // show the compute capability of this cuda version
  int device;
  cudaDeviceProp prop;
  cudaGetDevice(&device);
  cudaGetDeviceProperties(&prop, device);
  printf("CUDA Device: %s, Compute Capability: %d.%d\n", prop.name, prop.major, prop.minor);
  return 0;
}

typedef u8 Port;
const size_t P1 = 0;
const size_t P2 = 1;

typedef u32 Ptr;

typedef struct {
  u32 ports[2];
} Node;

typedef struct {
  Ptr lft;
  Ptr rgt;
} Wire;

typedef struct {
  Ptr    root;
  Wire*  acts_data;
  size_t acts_size;
  Node*  node_data;
  size_t node_size;
  size_t used;
  size_t rwts;
  size_t next;
} Net;

// Ptr
// ---

__host__ __device__ inline Ptr Ptr_new(Tag tag, Val val) {
  return ((u32)tag << 28) | (val & 0x0FFFFFFF);
}

__host__ __device__ inline Tag Ptr_tag(Ptr* ptr) {
  return (Tag)(*ptr >> 28);
}

__host__ __device__ inline Val Ptr_val(Ptr* ptr) {
  return *ptr & 0x0FFFFFFF;
}

__host__ __device__ inline Ptr* Ptr_target(Net* net, Ptr* ptr) {
  if (Ptr_tag(ptr) == VRT) {
    return &net->root;
  } else if (Ptr_tag(ptr) == VR1) {
    return &net->node_data[Ptr_val(ptr)].ports[P1];
  } else if (Ptr_tag(ptr) == VR2) {
    return &net->node_data[Ptr_val(ptr)].ports[P2];
  } else {
    return NULL;
  }
}

__device__ inline void Ptr_mov(Ptr* ptr, u32 add) {
  Tag tag = Ptr_tag(ptr);
  if (tag >= VR1) {
    *ptr = Ptr_new(tag, Ptr_val(ptr) + add);
  }
}

// Node
// ----

__host__ __device__ inline Node Node_new(Ptr p1, Ptr p2) {
  Node node;
  node.ports[P1] = p1;
  node.ports[P2] = p2;
  return node;
}

__host__ __device__ inline Node Node_nil() {
  return Node_new(Ptr_new(NIL, 0), Ptr_new(NIL, 0));
}

__host__ __device__ inline void Node_set(Node* node, Port port, Ptr ptr) {
  node->ports[port] = ptr;
}

__host__ __device__ inline Ptr Node_get(Node* node, Port port) {
  return node->ports[port];
}

__device__ inline Ptr Node_exch(Node* node, Port port, Ptr ptr) {
  return atomicExch((a64*)&node->ports[port], ptr);
}

__device__ inline Ptr Node_CAS(Node* node, Port port, Ptr expected, Ptr desired) {
  return atomicCAS((a64*)&node->ports[port], expected, desired);
}

// Net
// ---

__host__ inline Net Net_new(size_t acts_size, size_t node_size) {
  Net net;
  net.root      = Ptr_new(NIL, 0);
  net.acts_data = (Wire*)malloc(acts_size * sizeof(Wire));
  net.acts_size = acts_size;
  net.node_data = (Node*)malloc(node_size * sizeof(Node));
  net.node_size = node_size;
  net.used      = 0;
  net.rwts      = 0;
  net.next      = 0;
  return net;
}

// Allocator
// ---------

__device__ inline u32 Net_alloc(Net *net, Node *term_data, u32 term_size, u32 *cursor) {
  int length = 0;
  while (true) {
    u64 expected = 0;
    u64 detected = atomicCAS((a64*)&net->node_data[*cursor], expected, *((u64*)&Node_new(Ptr_new(NIL, 0), Ptr_new(NIL, term_size - length))));
    length = length + 1;
    if (detected == expected) {
      if (length == term_size) {
        for (int i = 0; i < length; ++i) {
          u64  ini = *cursor - length;
          Node val = term_data[i];
          Ptr_mov(&val.ports[0], ini);
          Ptr_mov(&val.ports[1], ini);
          atomicExch((a64*)&net->node_data[ini + i], *((u64*)&val));
        }
        return *cursor - length;
      }
      *cursor = (*cursor + 1) % net->node_size;
      length = *cursor == 0 ? 0 : length;
    } else {
      for (int i = 0; i < length; ++i) {
        u64 ini = *cursor - length;
        atomicExch((a64*)&net->node_data[ini + i], 0);
      }
      Ptr p2 = Node_get((Node*)&detected, P2);
      u64 inc = Ptr_tag(&p2) == NIL ? Ptr_val(&p2) : 1;
      *cursor = (*cursor + inc) % net->node_size;
      length = 0;
    }
  }
}

__device__ inline void Net_free(Net *net, u32 index) {
  atomicExch((a64*)&net->node_data[index], 0);
}

__device__ inline Ptr Net_set(Net* net, Val idx, Port port, Ptr ptr) {
  Node_set(&net->node_data[idx], port, ptr);
}

__device__ inline Ptr Net_get(Net* net, Val idx, Port port) {
  return Node_get(&net->node_data[idx], port);
}

__device__ inline Ptr Net_exch(Net* net, Val idx, Port port, Ptr ptr) {
  return Node_exch(&net->node_data[idx], port, ptr);
}

__device__ inline Ptr Net_CAS(Net* net, Val idx, Port port, Ptr expected, Ptr desired) {
  return Node_CAS(&net->node_data[idx], port, expected, desired);
}

__device__ inline Ptr* Net_ref(Net* net, Val idx, Port port) {
  return &net->node_data[idx].ports[port];
}

// Reference algorithm for Link:
// fn link(x, y):
//   loop:
//     a = take(net[x])
//     b = take(net[y])
//     r = CAS(net[a], x, b)
//     if r:
//       s = CAS(net[b], y, a)
//       if s:
//         break
//       store(net[a], x)
//     store(net[x], a)
//     store(net[y], b)
//     backoff

// Performs an atomic link
__device__ inline void Net_link(Net* net, Val a0_idx, Port a0_port, Val b0_idx, Port b0_port) {
  unsigned int ns = 4;
  while (true) {
    Ptr* a0_ref = Net_ref(net, a0_idx, a0_port);
    Ptr* b0_ref = Net_ref(net, b0_idx, b0_port);
    Ptr  a0_ptr = atomicExch((a64*)a0_ref, 0);
    Ptr  b0_ptr = atomicExch((a64*)b0_ref, 0);
    Ptr* a1_ref = Ptr_target(net, &a0_ptr);
    Ptr* b1_ref = Ptr_target(net, &b0_ptr);
    Ptr  a1_exp = Ptr_new(a0_port == P1 ? VR1 : VR2, a0_idx);
    Ptr  b1_exp = Ptr_new(b0_port == P1 ? VR1 : VR2, b0_idx);
    Ptr  a1_got = a1_ref == NULL ? a1_exp : atomicCAS(a1_ref, a1_exp, b0_ptr);
    if (a1_got == a1_exp) {
      Ptr b1_got = b1_ref == NULL ? b1_exp : atomicCAS(b1_ref, b1_exp, a0_ptr);
      if (b1_got == b1_exp) {
        break;
      }
      atomicExch((a64*)a1_ref, a1_got);
    }
    atomicExch((a64*)a0_ref, a0_ptr);
    atomicExch((a64*)b0_ref, b0_ptr);
    __nanosleep(ns);
    ns = ns < 256 ? ns * 2 : ns;
  }
}

// Same as Net_link, but when we know 'b' can't be a var
__device__ inline void Net_move(Net* net, Val a0_idx, Port a0_port, Ptr b0_ptr) {
  unsigned int ns = 4;
  while (true) {
    Ptr* a0_ref = Net_ref(net, a0_idx, a0_port);
    Ptr  a0_ptr = atomicExch((a64*)a0_ref, 0);
    Ptr* a1_ref = Ptr_target(net, &a0_ptr);
    Ptr  a1_exp = Ptr_new(a0_port == P1 ? VR1 : VR2, a0_idx);
    Ptr  a1_got = a1_ref == NULL ? a1_exp : atomicCAS(a1_ref, a1_exp, b0_ptr);
    if (a1_got == a1_exp) {
      break;
    }
    atomicExch((a64*)a0_ref, a0_ptr);
    __nanosleep(ns);
    ns = ns < 256 ? ns * 2 : ns;
  }
}

__device__ inline void Net_interact(Net* net, Ptr* a, Ptr* b, u32* cursor) {
  Tag a_tag = Ptr_tag(a);
  Tag b_tag = Ptr_tag(b);
  // Collect (for closed nets)
  if (a_tag == REF && b_tag == ERA) return;
  if (a_tag == ERA && b_tag == REF) return;
  // Dereference
  //Net_load_ref(net, book, a);
  //Net_load_ref(net, book, b);
  // Annihilation
  if (a_tag >= CON && b_tag >= CON && a_tag == b_tag) {
    Net_link(net, Ptr_val(a), P1, Ptr_val(b), P1);
    Net_link(net, Ptr_val(a), P2, Ptr_val(b), P2);
    Net_free(net, Ptr_val(a));
    Net_free(net, Ptr_val(b));
    //atomicAdd(&net->rwts, 1);
  // Commutation
  } else if (a_tag >= CON && b_tag >= CON && a_tag != b_tag) {
    Node term_data[4] = {
      Node_new(Ptr_new(VR1, 2), Ptr_new(VR1, 3)), // x1
      Node_new(Ptr_new(VR2, 2), Ptr_new(VR2, 3)), // x2
      Node_new(Ptr_new(VR1, 0), Ptr_new(VR1, 1)), // y1
      Node_new(Ptr_new(VR2, 0), Ptr_new(VR2, 1))  // y2
    };
    u32 ix = Net_alloc(net, term_data, 4, cursor);
    Net_move(net, Ptr_val(a), P1, Ptr_new(b_tag, ix+0));
    Net_move(net, Ptr_val(a), P2, Ptr_new(b_tag, ix+1));
    Net_move(net, Ptr_val(b), P1, Ptr_new(a_tag, ix+2));
    Net_move(net, Ptr_val(b), P2, Ptr_new(a_tag, ix+3));
    Net_free(net, Ptr_val(a));
    Net_free(net, Ptr_val(b));
    //atomicAdd(&net->rwts, 1);
  // Erasure
  } else if (a_tag >= CON && b_tag == ERA) {
    Net_move(net, Ptr_val(a), P1, Ptr_new(ERA, 0));
    Net_move(net, Ptr_val(a), P2, Ptr_new(ERA, 0));
    Net_free(net, Ptr_val(a));
    //atomicAdd(&net->rwts, 1);
  // Erasure
  } else if (a_tag == ERA && b_tag >= CON) {
    Net_move(net, Ptr_val(b), P1, Ptr_new(ERA, 0));
    Net_move(net, Ptr_val(b), P2, Ptr_new(ERA, 0));
    Net_free(net, Ptr_val(b));
    //atomicAdd(&net->rwts, 1);
  // Stuck
  } else {
    // Omitted: self.acts_data.push((*a,*b));
  }
}

//__host__ inline void Net_boot(Net* net, Book* book, uint32_t ref_id) {
  //Ptr root = Ptr_new(REF, ref_id);
  //Net_load_ref(net, book, &root);
  //net->root = root;
//}
