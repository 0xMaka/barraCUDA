#include <stdio.h>
#include <cuda_runtime.h>
#include <time.h>

__global__ void warmup(float *c) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  float a, b;
  a = b = 0.0f;
  if ((tid / warpSize) % 2 == 0) {
    a = 100.0f;
  } else {
    b = 200.0f;
  }
  c[tid] = a + b;
}

__global__ void math_kernel0(float *c) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  float a, b;
  a = b = 0.0f;
  if (tid % 2 == 0) {
    a = 100.0f;
  } else {
    b = 200.0f;
  }
  c[tid] = a + b;
}

__global__ void math_kernel1(float *c) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  float a, b;
  a = b = 0.0f;
  if ((tid / warpSize) % 2 == 0) {
    a = 100.0f;
  } else {
    b = 200.0f;
  }
  c[tid] = a + b;
}

__global__ void math_kernel2(float *c) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  float a, b;
  a = b = 0.0f;
  bool ipred = (tid % 2 == 0);
  if (ipred) {
    a = 100.0f;
  } else {
    b = 200.0f;
  }
  c[tid] = a + b;
}

__global__ void math_kernel3(float *c) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  float a, b;
  a = b = 0.0f;

  int itid = tid >> 5;
  if (itid & 0x01 == 0) {
    a = 100.0f;
  } else {
    b = 200.0f;
  }
  c[tid] = a + b;
}
int main(int argc, char **argv) {
  int dev = 0;
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, dev);
  printf("[x] ----------------------------------------------------------\n");
  printf("[>] Working with Device %d: %s\n", dev, prop.name);

  int size = 64;
  int blocksize = 64;

  if (argc > 1) blocksize = atoi(argv[1]);
  if (argc > 2) size = atoi(argv[2]);

  printf("[-] Data size: %d\n", size);

  dim3 block (blocksize, 1);
  dim3 grid ((size + block.x - 1) / block.x, 1);

  printf("[-] Execution Configure (block %d grid %d)\n", block.x, grid.x);

  float *dc;
  size_t bytes = size *sizeof(float);
  cudaMalloc((float**)&dc, bytes);

  clock_t t0, t1;
  cudaDeviceSynchronize();
  t0 = clock();
  // run warm up to remove overhead
  warmup <<< grid, block >>> (dc);
  cudaDeviceSynchronize();
  t1 = clock();
  printf(
    "[+] Warmup       <<< %4d, %4d >>> elapsed in: %f\n", 
    block.x, grid.x, (double)(t1-t0)/CLOCKS_PER_SEC
  );

  cudaDeviceSynchronize();
  t0 = clock();
  math_kernel0 <<< grid, block >>> (dc);
  cudaDeviceSynchronize();
  t1 = clock();
  printf(
    "[+] Math Kernel0 <<< %4d, %4d >>> elapsed in: %f\n", 
    block.x, grid.x, (double)(t1-t0)/CLOCKS_PER_SEC
  );

  cudaDeviceSynchronize();
  t0 = clock();
  math_kernel1 <<< grid, block >>> (dc);
  cudaDeviceSynchronize();
  t1 = clock();
  printf(
    "[+] Math Kernel1 <<< %4d, %4d >>> elapsed in: %f\n", 
    block.x, grid.x, (double)(t1-t0)/CLOCKS_PER_SEC
  );

  cudaDeviceSynchronize();
  t0 = clock();
  math_kernel2 <<< grid, block >>> (dc);
  cudaDeviceSynchronize();
  t1 = clock();
  printf(
    "[+] Math Kernel2 <<< %4d, %4d >>> elapsed in: %f\n", 
    block.x, grid.x, (double)(t1-t0)/CLOCKS_PER_SEC
  );

  cudaDeviceSynchronize();
  t0 = clock();
  math_kernel3 <<< grid, block >>> (dc);
  cudaDeviceSynchronize();
  t1 = clock();
  printf(
    "[+] Math Kernel3 <<< %4d, %4d >>> elapsed in: %f\n", 
    block.x, grid.x, (double)(t1-t0)/CLOCKS_PER_SEC
  );

  printf("[x] ---------------------------------------------------------\n");

  cudaFree(dc);
  cudaDeviceReset();
  return EXIT_SUCCESS;
}


