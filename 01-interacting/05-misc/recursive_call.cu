#include <stdio.h>
#include <cuda_runtime.h>

__global__ void nestedhw(int const iSize, int iDepth) {
  int tid = threadIdx.x;
  printf("[+] Rescursion = %d: we flaggin' from thread %d block %d\n", iDepth, tid, blockIdx.x);

  if (iSize == 1) return;

  int nThreads = iSize >> 1;

  // launch child grid recursively
  if (tid == 0 && nThreads > 0) {
    nestedhw <<< 1, nThreads >>> (nThreads, ++iDepth);
    printf("[-] ------------> nested execution depth: %d\n", iDepth);
  }
}

int main (int argc, char **argv) {
  int size = 8;
  int blocksize  = 8;
  int igrid = 1;

  if (argc > 1) {
    igrid = atoi(argv[1]);
    size = igrid * blocksize;
  }

  dim3 block (blocksize, 1);
  dim3 grid ((size + block.x - 1) / block.x, 1);
  printf("[x] =================================================\n");
  printf("[>] Execution Configuration ( block %d grid %d )\n", block.x, grid.x);
  nestedhw <<< grid, block >>> (block.x, 0);

  printf("[x] -------------------------------------------------\n");

  cudaGetLastError();
  cudaDeviceReset();
  printf("[x] =================================================\n");
}




  
