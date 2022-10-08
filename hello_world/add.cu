#include <iostream>
#include <math.h>

// # ex. using a single thread

// ## __global__ dunder is a cuda specifier for a kernal function
// - adds the elements of two arrays
__global__ void add(int n, float *x, float *y) {
  for (int i=0; i<n; i++)
		y[i] = x[i] + y[i];
}

int main(void) {
	int N = 1<<20; //20mill elements
	float *x, *y;	
	// allocate unified memory accessible from gpu or cpu
	cudaMallocManaged(&x, N*sizeof(float)); // new's replaced by cudaMallocManaged
	cudaMallocManaged(&y, N*sizeof(float)); // returns pointer accessible by host, or device code

	// initialize x and y arrays on the host (cpu)
	for (int i=0; i<N; i++) {
    x[i] = 1.0f;
		y[i] = 2.0f;
	}

	// run kernel on 1mill elements of the cpu
	add<<<1,1>>>(N,x,y); // launch kernel on single thread

	// launching kernel does not block calling cpu thread..
	// so wait for gpu to finish before accessing results
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
