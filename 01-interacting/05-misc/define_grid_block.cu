#include <cuda_runtime.h>
#include <stdio.h>

// - For a given data size the general steps to determine block and grid dimensions are:
// - - Select a block size
// - - Calculate the grid size based on data size and block size
// - When selecting block dimensions we usually need to consider:
// - - Performance characteristics of the kernel
// - - limitations of GPU resources

// example uses a 1D grid/1D blocks to illustrate that a change in block size alters grid size accordingly

int main(int argc, char **argv) {
  // define total data elements
  int elements = 1024;

  // define grid and block structure
	dim3 block = (1024);
	dim3 grid ((elements + block.x -1) / block.x);
	printf("[+] grid.x %d block.x %d \n", grid.x, block.x);

  // reset block
  block.x = 512;
	grid.x = (elements + block.x -1) / block.x;
	printf("[+] grid.x %d block.x %d \n", grid.x, block.x);

  // reset block
  block.x = 256;
	grid.x = (elements + block.x -1) / block.x;
	printf("[+] grid.x %d block.x %d \n", grid.x, block.x);

  // reset block
  block.x = 128;
	grid.x = (elements + block.x -1) / block.x;
	printf("[+] grid.x %d block.x %d \n", grid.x, block.x);

  // reset device
  cudaDeviceReset();
  return 0;
}

/* ex Output:
	 [+] grid.x 1 block.x 1024 
	 [+] grid.x 2 block.x 512 
	 [+] grid.x 4 block.x 256 
	 [+] grid.x 8 block.x 128
 */


