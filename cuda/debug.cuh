#ifndef DEBUG_CUH
#define DEBUG_CUH

#include "types.cuh"
#include "runtime.cuh"

__device__ __host__ void stop(const char* tag);
__device__ __host__ bool dbug(u32* K, const char* tag);

__host__ const char* show_ptr(Ptr ptr, u32 slot);
__host__ void print_net(Net* net);
__host__ void print_tree_go(Net* net, Ptr ptr, Map* var_ids);
__host__ void print_tree(Net* net, Ptr ptr);

__host__ void map_insert(Map* map, u32 key, u32 val);
__host__ u32 map_lookup(Map* map, u32 key);

#endif
