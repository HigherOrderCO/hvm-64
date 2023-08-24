#ifndef MEMORY_CUH
#define MEMORY_CUH

#include "types.cuh"
#include "runtime.cuh"

__host__ Net* mknet();
__host__ Net* net_to_device(Net* host_net);
__host__ Net* net_to_host(Net* device_net);
__host__ void net_free_on_device(Net* device_net);
__host__ void net_free_on_host(Net* host_net);

#endif
