#include <cuda_runtime.h>
#include <stdio.h>

#define CHECK(call) {\
  const cudaError_t err = call;\
  if(err != cudaSuccess) {\
    printf("Error: %s: %d ", __FILE__, __LINE__);\
    printf("code: %d reason: %s\n", err, cudaGetErrorString(err));\
    exit (-10*err);\
  }\
}

void initial_int(int *ip, int size) {
  for (int i=0; i<size; i++) {
    ip[i] = i;
  }
}

void get_matrix(int *C, const int nx, const int ny) {
  int *ic = C;
  int n = 0;
  printf("[+] Matrix: (%d.%d) =============================================================\n[>]", nx, ny);
  for (int iy=0; iy<ny; iy++) {
    for (int ix=0; ix<nx; ix++) 
      printf("%3d", ic[ix]);
      if (n==2) printf("\n[>]");
      n++;
  }
  ic += nx;
  printf("\n");
  printf("[+] ===========================================================================\n");
}

__global__ void get_thread_index(int *A, const int nx, const int ny) {

  int ix = threadIdx.x + blockIdx.x * blockDim.x;
  int iy = threadIdx.y + blockIdx.y * blockDim.y;

  unsigned int idx = iy*nx +ix;
  printf(
    "[>] thread_id (%d, %d) block_id (%d, %d) coordinates(%d, %d) global index %2d ival %2d\n",
    threadIdx.x, threadIdx.y, blockIdx.x, blockIdx.y, ix, iy, idx, A[idx]
  );
}

int main(int argc, char **argv) {
  printf("[x] ===========================================================================\n");
  printf("[-] Fetching Device Info..\n");

  int id = 0;
  cudaDeviceProp prop;
  CHECK(cudaGetDeviceProperties(&prop, id));
	
  printf("[+] Working With Device %d: %s\n", id, prop.name);
  CHECK(cudaSetDevice(id));

  printf("[-] Declaring Matrix Dimensions..\n");
  // set matric dimensions
  int nx = 8;
  int ny = 6;
  int nxy = nx * ny;
  int bytes = nxy * sizeof(float);

  printf("[-] Allocating Host Memory..\n");
  // allocate host mem
  int *ha;
  ha = (int *)malloc(bytes);

  printf("[-] Initializing Host Matrix..\n");
  // initialize host matrix
  initial_int(ha, nxy);
  get_matrix(ha, nx, ny);

  printf("[-] Allocating Device Memory..\n");
  // allocate device mem
  int *dmata;
  cudaMalloc((void **) &dmata, bytes);

  printf("[-] Transferring Payload From Host to Device..\n");
  // transfer data from host to device
  cudaMemcpy(ha, dmata, bytes, cudaMemcpyHostToDevice);

  printf("[-] Declaring Block Dimensions..\n");
  // prep kernel
  dim3 block (4, 2);
  dim3 grid ((nx+block.x -1) / block.x, (ny+block.y -1) /block.y);

  printf("[+] Invoking the Kernel.\n");
  // invoke the kernel
  get_thread_index <<<grid,block>>> (dmata, nx, ny);
  printf("[+] == Results ================================================================\n");
  // check for errors
  CHECK(cudaDeviceSynchronize());

  printf("[+] ---------------------------------------------------------------------------\n");
  printf("[-] Freeing Memory.. Resetting Device..\n");
  // free mem
  cudaFree(dmata);
  free(ha);
  // device reset
  printf("[+] Device Reset.\n");
  cudaDeviceReset();
  printf("[x] ===========================================================================\n");
  return 0;
}

/* Output
[x] ===========================================================================
[-] Fetching Device Info..
[+] Working With Device 0: NVIDIA GeForce RTX 3060 Laptop GPU
[-] Declaring Matrix Dimensions..
[-] Allocating Host Memory..
[-] Initializing Host Matrix..
[+] Matrix: (8.6) =============================================================
[>]  0  1  2  3  4  5  6  7  0  1  2  3  4  5  6  7  0  1  2  3  4  5  6  7
[>]  0  1  2  3  4  5  6  7  0  1  2  3  4  5  6  7  0  1  2  3  4  5  6  7
[+] ===========================================================================
[-] Allocating Device Memory..
[-] Transferring Payload From Host to Device..
[-] Declaring Block Dimensions..
[+] Invoking the Kernel.
[+] == Results ================================================================
[>] thread_id (0, 0) block_id (1, 2) coordinates(4, 4) global index 36 ival  0
[>] thread_id (1, 0) block_id (1, 2) coordinates(5, 4) global index 37 ival  0
[>] thread_id (2, 0) block_id (1, 2) coordinates(6, 4) global index 38 ival  0
[>] thread_id (3, 0) block_id (1, 2) coordinates(7, 4) global index 39 ival  0
[>] thread_id (0, 1) block_id (1, 2) coordinates(4, 5) global index 44 ival  0
[>] thread_id (1, 1) block_id (1, 2) coordinates(5, 5) global index 45 ival  0
[>] thread_id (2, 1) block_id (1, 2) coordinates(6, 5) global index 46 ival  0
[>] thread_id (3, 1) block_id (1, 2) coordinates(7, 5) global index 47 ival  0
[>] thread_id (0, 0) block_id (0, 1) coordinates(0, 2) global index 16 ival  0
[>] thread_id (1, 0) block_id (0, 1) coordinates(1, 2) global index 17 ival  0
[>] thread_id (2, 0) block_id (0, 1) coordinates(2, 2) global index 18 ival  0
[>] thread_id (3, 0) block_id (0, 1) coordinates(3, 2) global index 19 ival  0
[>] thread_id (0, 1) block_id (0, 1) coordinates(0, 3) global index 24 ival  0
[>] thread_id (1, 1) block_id (0, 1) coordinates(1, 3) global index 25 ival  0
[>] thread_id (2, 1) block_id (0, 1) coordinates(2, 3) global index 26 ival  0
[>] thread_id (3, 1) block_id (0, 1) coordinates(3, 3) global index 27 ival  0
[>] thread_id (0, 0) block_id (1, 0) coordinates(4, 0) global index  4 ival  0
[>] thread_id (1, 0) block_id (1, 0) coordinates(5, 0) global index  5 ival  0
[>] thread_id (2, 0) block_id (1, 0) coordinates(6, 0) global index  6 ival  0
[>] thread_id (3, 0) block_id (1, 0) coordinates(7, 0) global index  7 ival  0
[>] thread_id (0, 1) block_id (1, 0) coordinates(4, 1) global index 12 ival  0
[>] thread_id (1, 1) block_id (1, 0) coordinates(5, 1) global index 13 ival  0
[>] thread_id (2, 1) block_id (1, 0) coordinates(6, 1) global index 14 ival  0
[>] thread_id (3, 1) block_id (1, 0) coordinates(7, 1) global index 15 ival  0
[>] thread_id (0, 0) block_id (0, 2) coordinates(0, 4) global index 32 ival  0
[>] thread_id (1, 0) block_id (0, 2) coordinates(1, 4) global index 33 ival  0
[>] thread_id (2, 0) block_id (0, 2) coordinates(2, 4) global index 34 ival  0
[>] thread_id (3, 0) block_id (0, 2) coordinates(3, 4) global index 35 ival  0
[>] thread_id (0, 1) block_id (0, 2) coordinates(0, 5) global index 40 ival  0
[>] thread_id (1, 1) block_id (0, 2) coordinates(1, 5) global index 41 ival  0
[>] thread_id (2, 1) block_id (0, 2) coordinates(2, 5) global index 42 ival  0
[>] thread_id (3, 1) block_id (0, 2) coordinates(3, 5) global index 43 ival  0
[>] thread_id (0, 0) block_id (0, 0) coordinates(0, 0) global index  0 ival  0
[>] thread_id (1, 0) block_id (0, 0) coordinates(1, 0) global index  1 ival  0
[>] thread_id (2, 0) block_id (0, 0) coordinates(2, 0) global index  2 ival  0
[>] thread_id (3, 0) block_id (0, 0) coordinates(3, 0) global index  3 ival  0
[>] thread_id (0, 1) block_id (0, 0) coordinates(0, 1) global index  8 ival  0
[>] thread_id (1, 1) block_id (0, 0) coordinates(1, 1) global index  9 ival  0
[>] thread_id (2, 1) block_id (0, 0) coordinates(2, 1) global index 10 ival  0
[>] thread_id (3, 1) block_id (0, 0) coordinates(3, 1) global index 11 ival  0
[>] thread_id (0, 0) block_id (1, 1) coordinates(4, 2) global index 20 ival  0
[>] thread_id (1, 0) block_id (1, 1) coordinates(5, 2) global index 21 ival  0
[>] thread_id (2, 0) block_id (1, 1) coordinates(6, 2) global index 22 ival  0
[>] thread_id (3, 0) block_id (1, 1) coordinates(7, 2) global index 23 ival  0
[>] thread_id (0, 1) block_id (1, 1) coordinates(4, 3) global index 28 ival  0
[>] thread_id (1, 1) block_id (1, 1) coordinates(5, 3) global index 29 ival  0
[>] thread_id (2, 1) block_id (1, 1) coordinates(6, 3) global index 30 ival  0
[>] thread_id (3, 1) block_id (1, 1) coordinates(7, 3) global index 31 ival  0
[+] ---------------------------------------------------------------------------
[-] Freeing Memory.. Resetting Device..
[+] Device Reset.
[x] ===========================================================================
*/
