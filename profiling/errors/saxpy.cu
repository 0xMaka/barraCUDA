#include <stdio.h> 
__global__ void saxpy(int n, float a, float *x, float *y) {
  int i = (blockIdx.x * blockDim.x) + threadIdx.x;
	if(i < n) // if in bounds
    y[i] = (a * x[i]) + y[i]; // single-precision ax+y
}
int main(void) {
  int N = 1<<20; 
  float *x, *y, *devx, *devy; // host and device pointers
  x = (float*)malloc(N*sizeof(float)); // points to host array
  y = (float*)malloc(N*sizeof(float));

  cudaMalloc(&devx, N*sizeof(float));  // points to device array
  cudaMalloc(&devy, N*sizeof(float));

  for (int i=0; i<N; i++) {
    x[i] = 1.0f; // initialize host arrays
    y[i] = 2.0f;
  }
  // source: host pointer
  // dest: device pointer
  cudaMemcpy(devx, x, N*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(devy, y, N*sizeof(float), cudaMemcpyHostToDevice);
  // launch kernel 
  saxpy<<<(N+255)/256,256>>>(N, 2.0f, devx, devy);

	// check for kernel errors
	// checks for invalid config params
	// sync errors, such as too many threads per block
	cudaError_t errSync = cudaGetLastError();
	// async errors, such as out of bound memory access
	// needs a device sync, so is expensive..
	// use with care. ie not in production
	cudaError_t errAsync = cudaDeviceSynchronize();
  if (errSync != cudaSuccess)
    printf("Sync kernel error: %s\n", cudaGetErrorString(errSync));
	if (errAsync != cudaSuccess)
    printf("Async kernel error: %s\n", cudaGetErrorString(cudaGetLastError()));

	// cp back results after running kernel function
  cudaMemcpy(y, devy, N*sizeof(float), cudaMemcpyDeviceToHost);

  float maxError = 0.0f;
  for(int i=0; i<N; i++)
    maxError = max(maxError, abs(y[i]-4.0f));
  printf("Max error %.*f\n", 6, maxError);			

  free(x);  // free host mem
  free(y);
  cudaFree(devx);  // free device mem
  cudaFree(devy);
}
