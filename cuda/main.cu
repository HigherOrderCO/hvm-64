#include "types.cuh"
#include "runtime.cuh"
#include "debug.cuh"
#include "memory.cuh"
#include "map.cuh"
#include "tests.cuh"

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
  inject_term_i(h_net);

  // Prints the initial net
  printf("\n");
  printf("INPUT\n");
  printf("=====\n\n");
  print_net(h_net);

  // Sends the net from host to device
  Net* d_net = net_to_device(h_net);

  // Performs parallel reductions
  printf("\n");
  for (u64 i = 0; i < 128; ++i) {
    do_global_rewrite(d_net);
    do_global_scatter(d_net);
  }

  // Add synchronization and error checking
  cudaDeviceSynchronize();
  cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess) {
    printf("CUDA error: %s\n", cudaGetErrorString(error));
  }

  // Reads the normalized net from device to host
  Net* norm = net_to_host(d_net);

  // Prints the normal form (raw data)
  printf("\n");
  printf("NORMAL (%d | %d)\n", norm->rwts, norm->blen);
  printf("======\n\n");
  //print_tree(norm, norm->root);
  print_net(norm);
  printf("\n");

  // Free device memory
  net_free_on_device(d_net);

  // Free host memory
  net_free_on_host(h_net);
  net_free_on_host(norm);

  return 0;
}

