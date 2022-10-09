#include <stdio.h>

__global__ void hello(void) {
  printf("Hello world, from GPU!\n");
  printf("I am kernel number %d\n", threadIdx.x);
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

/* ex. Output
Hello world, from GPU!
Hello world, from GPU!
Hello world, from GPU!
Hello world, from GPU!
Hello world, from GPU!
I am kernel number 0
I am kernel number 1
I am kernel number 2
I am kernel number 3
I am kernel number 4
*/
