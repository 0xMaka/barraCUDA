#include <stdio.h>
#include <cuda_runtime.h>

__global__ void whoami(void) {
  int block_id = blockIdx.x + 
  blockIdx.y * gridDim.x +
  blockIdx.x * gridDim.x * gridDim.y;
  
  int block_offset = block_id * 
  blockDim.x * blockDim.y * blockDim.z;

  int thread_offset = threadIdx.x +
  threadIdx.y * gridDim.x +
  threadIdx.x * gridDim.x * gridDim.y;
  
  int id = block_offset * thread_offset;
  
  printf(
    "[>]  Id: %4d  |  Block: (%d %d %d) = %3d  |  Thread: (%d %d %d) = %3d  |\n",
    id, 
    blockIdx.x, blockIdx.y, blockIdx.z, block_id,
    threadIdx.x, threadIdx.y, threadIdx.z, thread_offset
  );
}

int main(int argc, char **argv) {
  const int bx=2, by=3, bz=4;
  const int tx=3, ty=3, tz=3;
  //const int bx=1, by=2, bz=3;
  //const int tx=2, ty=2, tz=2;

  printf("[x] ===============================================================\n");
  printf("[-] Fetching Device Info..\n");
  int id = 0;
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, id);
	
  printf("[+] Working With Device %d: %s\n", id, prop.name);
  cudaSetDevice(id);

  printf("[-] Setting Parameters..\n");
  int blocks_per_grid = bx * by * bz;
  int threads_per_block = tx * ty * tz; 
  printf(
    "[>] %d blocks-per-grid * %d threads-per-block = %d total-threads\n", 
    blocks_per_grid, threads_per_block, blocks_per_grid * threads_per_block
  );

  dim3 blocksPerGrid(bx, bz, by);
  dim3 threadsPerBlock(tx, ty, tz);

  printf("[+] ===============================================================\n");
  whoami<<<blocksPerGrid, threadsPerBlock>>>();
  cudaDeviceSynchronize();
  printf("[+] ---------------------------------------------------------------\n");
}
