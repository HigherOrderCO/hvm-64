#include <stdint.h>
#include <stdio.h>
#if defined(__unix__) || defined(__unix) || defined(__APPLE__)
  #include <unistd.h>
#endif

// Import runtime
#include "runtime.cu"

// Import program to run
#include "programs/default_benchmark.cu"
// #include "programs/dec_bits.cu"
// #include "programs/dec_bits_tree.cu"

// This file contains the host functions for the standalone C++ version

const size_t BOOK_DATA_SIZE = sizeof(BOOK_DATA) / sizeof(u32);
const size_t JUMP_DATA_SIZE = sizeof(JUMP_DATA) / sizeof(u32);

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

void do_global_rewrite(Net* net, Book* book, u32 repeat, u32 tick, bool flip) {
  global_rewrite<<<BAGS_HEIGHT, BLOCK_SIZE>>>(net, book, repeat, tick, flip);
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("CUDA error: %s\n", cudaGetErrorString(err));
  }
}

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
  Map var_ids;
  var_ids.size = 0;
  print_tree_go(net, ptr, &var_ids);
  printf("\n");
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
  Net* cpu_net = mknet(F_main, JUMP_DATA, JUMP_DATA_SIZE);

  // Prints the input net
  printf("\nINPUT\n=====\n\n");
  print_net(cpu_net);

  // Uploads net and book to GPU
  Net* gpu_net = net_to_gpu(cpu_net);
  Book* gpu_book = init_book_on_gpu(BOOK_DATA, BOOK_DATA_SIZE);

  // Marks init time
  struct timespec start, end;

#if defined(__unix__) || defined(__unix) || defined(__APPLE__)
  clock_gettime(CLOCK_MONOTONIC_RAW, &start);
#endif

  // Normalizes
  do_global_expand(gpu_net, gpu_book);
  for (u32 tick = 0; tick < 128; ++tick) {
    do_global_rewrite(gpu_net, gpu_book, 16, tick, (tick / BAGS_WIDTH_L2) % 2);
  }
  do_global_expand(gpu_net, gpu_book);
  do_global_rewrite(gpu_net, gpu_book, 200000, 0, 0);
  cudaDeviceSynchronize();

#if defined(__unix__) || defined(__unix) || defined(__APPLE__)
  // Gets end time
  clock_gettime(CLOCK_MONOTONIC_RAW, &end);
#endif

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
