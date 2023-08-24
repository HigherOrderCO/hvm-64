#ifndef RUNTIME_CUH
#define RUNTIME_CUH

#include "types.cuh"

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
__host__ __device__ inline Ptr enter(Net* net, Ptr ptr) {
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

__device__ inline u32 alloc(Worker *worker, Net *net);
__device__ inline void put_redex(Worker* worker, Ptr a_ptr, Ptr b_ptr);
__device__ Ptr take(Ptr* ref);
__device__ void replace(u32 id, Ptr* ref, Ptr exp, Ptr neo);
__device__ void link(Worker* worker, Net* net, Ptr* nod_ref, Ptr dir_ptr);

__device__ int scansum(u32* arr);
__device__ void local_scatter(Net* net);
__global__ void global_scatter_prepare_0(Net* net);
__global__ void global_scatter_prepare_1(Net* net);
__global__ void global_scatter(Net* net);
__global__ void global_rewrite(Net* net);
void do_global_scatter(Net* net);
void do_global_rewrite(Net* net);

#endif
