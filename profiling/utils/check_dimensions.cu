#include <cuda_runtime.h>
#include <stdio.h>

__global__ void check_index(void) {
  // pre defined uint3 data types
  printf(
    "[+] threadIdx: (%d, %d, %d) blockIdx: (%d, %d, %d) blockDim: (%d, %d, %d) gridDim: (%d, %d, %d)\n", 
    threadIdx.x, threadIdx.y, threadIdx.z, blockIdx.x, blockIdx.y, blockIdx.z, blockDim.x, blockDim.y, blockDim.z, gridDim.x, gridDim.y, gridDim.z
  );
}

int main(int argc, char **argv) {
  // define total data elements
  int elements = 6;
  // define grid block structure: choose a block size then calculate grid size from data and block sizes
	// here we define a 1D block with 3 threads and a 1D grid
    dim3 block(3);  // manually defined dim3 data types
	  dim3 grid ((elements + block.x -1) / block.x); // grid size rounded up to multiple of block size

  printf("[x] Checking grid/block indices and dimensions.. \n");
  // check grid and block dimensions from host side
  printf("[+] grid.x: %d, grid.y: %d, grid.z: %d\n", grid.x, grid.y, grid.z);
  printf("[+] block.x: %d, block.y: %d, block.z: %d\n", block.x, block.y, block.z);
  
	// check grid and block dimensions from device side 
  check_index <<<grid, block>>> ();
	//reset device before leave leaving
  cudaDeviceReset();
	return 0;
}
