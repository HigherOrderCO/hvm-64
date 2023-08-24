#include "memory.cuh"

__host__ Net* mknet() {
  Net* net  = (Net*)malloc(sizeof(Net));
  net->root = mkptr(NIL, 0);
  net->rwts = 0;
  net->done = 0;
  net->blen = 0;
  net->bags = (Wire*)malloc(BAGS_SIZE * sizeof(Wire));
  net->gidx = (u32*) malloc(GIDX_SIZE * sizeof(u32));
  net->gtmp = (u32*) malloc(GTMP_SIZE * sizeof(u32));
  net->node = (Node*)malloc(NODE_SIZE * sizeof(Node));
  memset(net->bags, 0, BAGS_SIZE * sizeof(Wire));
  memset(net->gidx, 0, GIDX_SIZE * sizeof(u32));
  memset(net->gtmp, 0, GTMP_SIZE * sizeof(u32));
  memset(net->node, 0, NODE_SIZE * sizeof(Node));
  return net;
}

__host__ Net* net_to_device(Net* host_net) {
  // Allocate memory on the device for the Net object, and its data
  Net*  device_net;
  Wire* device_bags;
  u32*  device_gidx;
  u32*  device_gtmp;
  Node* device_node;

  cudaMalloc((void**)&device_net, sizeof(Net));
  cudaMalloc((void**)&device_bags, BAGS_SIZE * sizeof(Wire));
  cudaMalloc((void**)&device_gidx, GIDX_SIZE * sizeof(u32));
  cudaMalloc((void**)&device_gtmp, GTMP_SIZE * sizeof(u32));
  cudaMalloc((void**)&device_node, NODE_SIZE * sizeof(Node));

  // Copy the host data to the device memory
  cudaMemcpy(device_bags, host_net->bags, BAGS_SIZE * sizeof(Wire), cudaMemcpyHostToDevice);
  cudaMemcpy(device_gidx, host_net->gidx, GIDX_SIZE * sizeof(u32),  cudaMemcpyHostToDevice);
  cudaMemcpy(device_gtmp, host_net->gtmp, GTMP_SIZE * sizeof(u32),  cudaMemcpyHostToDevice);
  cudaMemcpy(device_node, host_net->node, NODE_SIZE * sizeof(Node), cudaMemcpyHostToDevice);

  // Create a temporary host Net object with device pointers
  Net temp_net  = *host_net;
  temp_net.bags = device_bags;
  temp_net.gidx = device_gidx;
  temp_net.gtmp = device_gtmp;
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
  host_net->bags = (Wire*)malloc(BAGS_SIZE * sizeof(Wire));
  host_net->gidx = (u32*) malloc(GIDX_SIZE * sizeof(u32));
  host_net->gtmp = (u32*) malloc(GTMP_SIZE * sizeof(u32));
  host_net->node = (Node*)malloc(NODE_SIZE * sizeof(Node));

  // Retrieve the device pointers for data
  Wire* device_bags;
  u32*  device_gidx;
  u32*  device_gtmp;
  Node* device_node;
  cudaMemcpy(&device_bags, &(device_net->bags), sizeof(Wire*), cudaMemcpyDeviceToHost);
  cudaMemcpy(&device_gidx, &(device_net->gidx), sizeof(u32*),  cudaMemcpyDeviceToHost);
  cudaMemcpy(&device_gtmp, &(device_net->gtmp), sizeof(u32*),  cudaMemcpyDeviceToHost);
  cudaMemcpy(&device_node, &(device_net->node), sizeof(Node*), cudaMemcpyDeviceToHost);

  // Copy the device data to the host memory
  cudaMemcpy(host_net->bags, device_bags, BAGS_SIZE * sizeof(Wire), cudaMemcpyDeviceToHost);
  cudaMemcpy(host_net->gidx, device_gidx, GIDX_SIZE * sizeof(u32),  cudaMemcpyDeviceToHost);
  cudaMemcpy(host_net->gtmp, device_gtmp, GTMP_SIZE * sizeof(u32),  cudaMemcpyDeviceToHost);
  cudaMemcpy(host_net->node, device_node, NODE_SIZE * sizeof(Node), cudaMemcpyDeviceToHost);

  return host_net;
}

__host__ void net_free_on_device(Net* device_net) {
  // Retrieve the device pointers for data
  Wire* device_bags;
  u32*  device_gidx;
  u32*  device_gtmp;
  Node* device_node;
  cudaMemcpy(&device_bags, &(device_net->bags), sizeof(Wire*), cudaMemcpyDeviceToHost);
  cudaMemcpy(&device_gidx, &(device_net->gidx), sizeof(u32*),  cudaMemcpyDeviceToHost);
  cudaMemcpy(&device_gtmp, &(device_net->gtmp), sizeof(u32*),  cudaMemcpyDeviceToHost);
  cudaMemcpy(&device_node, &(device_net->node), sizeof(Node*), cudaMemcpyDeviceToHost);

  // Free the device memory
  cudaFree(device_bags);
  cudaFree(device_gidx);
  cudaFree(device_gtmp);
  cudaFree(device_node);
  cudaFree(device_net);
}

__host__ void net_free_on_host(Net* host_net) {
  free(host_net->bags);
  free(host_net->gidx);
  free(host_net->gtmp);
  free(host_net->node);
  free(host_net);
}

