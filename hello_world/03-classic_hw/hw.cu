#include <stdio.h>

__global__ void hello(void) {
  printf("Hello world, from GPU!\n");
}

int main(void) {
  hello<<<1,5>>>();
	cudaError_t errSync = cudaGetLastError();
  cudaError_t errAsync = cudaDeviceSynchronize();

	if (errSync != cudaSuccess)
    printf("Sync kernel error!\n%s\n", cudaGetErrorString(errSync));
  if (errAsync != cudaSuccess)
    printf("Async kernel error!\n%s\n", cudaGetErrorString(errAsync));
  
	return 0;
}


