#include <iostream>
#include <math.h>

__global__ void add(int n, float *x, float *y) {
  // cuda c++ provides keywords to return indices of running threads
  // threadIdx.x contains index of current thread within its block
  // blockDim.x contains number of threads in the block
  int index = threadIdx.x;
  int stride = blockDim.x;
  for (int i=index; i<n; i+=stride)
	  y[i] = x[i] + y[i];
}

int main(void) {
  int N = 1<<20; //20mill elements
  float *x, *y;	
  // allocate unified memory accessible from gpu or cpu
  cudaMallocManaged(&x, N*sizeof(float));
  cudaMallocManaged(&y, N*sizeof(float));

  // initialize x and y arrays on the host (cpu)
  for (int i=0; i<N; i++) {
    x[i] = 1.0f;
	y[i] = 2.0f;
  }

  // run kernel on 1mill elements, 256 threads of the gpu
  add<<<1,256>>>(N,x,y);
  
  // wait for gpu to finish before accessing results
  cudaDeviceSynchronize();

  // check for errors (all values should be 3.0f)
  float maxError = 0.0f;
    for (int i=0; i<N; i++) 
      maxError = fmax(maxError, fabs(y[i]-3.0f));
  std::cout << "Max error: " << maxError << std::endl;

  // free memory
  cudaFree(x);
  cudaFree(y);
  return 0;
}
