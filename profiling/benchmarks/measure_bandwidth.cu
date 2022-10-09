#include <stdio.h>

// # Memory Bandwidth
// - Memory Bandwidth is the theoretical maximum amount of data the bus can handle at a given time 
// - - it is a key factor in how quickly a GPU can access and utilize its framebuffer

// ## Theoretical vs Effective Memory Bandwidth
// - Theoretical bandwidth can be calculated using hardware specifications
// - - Look for clock rate and memory interface width

// RTX-3060-Mobile - Spec lists a Memory Bandwidth of 336 GB/s
// BUS_WIDTH = 192
// CLOCK_RATE = 1750 
// DDR_MULTI = 8 // DDR6
// BWTheoretical = (((CLOCK_RATE * (10 ** 6)) * (BUS_WIDTH/8)) * DDR_MULI) / (10 ** 9) 
// BWTheoretical == 336 GB/s

// ## Effective Bandwidth
// - Effective bandwidth needs to be timed measuring specfic activities
// - - We can use the equation: BWEffective = ((RB) + (WB)) / (t * (10 ** 9))
// Where BWEffective = bandwidth in GB's units
// RB = number of bytes per kernel
// WB = number of bytes written per kernel
// t = elapsed time in seconds

// Device/Kernel function
// a, i, n are stored in device thread registers
// *x, y* are pointers to device memory
__global__ void saxpy(int n, float a, float *x, float *y) {
  int i = (blockIdx.x * blockDim.x) + threadIdx.x;
  if(i < n) // if in bounds
    y[i] = (a * x[i]) + y[i]; // single-precision ax+y
}
// Host function
int main(void) {
  int N = 20 * (1<<20);
  float *x, *y, *devx, *devy; // host and device pointers
  x = (float*)malloc(N*sizeof(float)); // points to host array
  y = (float*)malloc(N*sizeof(float));

  cudaMalloc(&devx, N*sizeof(float));  // points to device array
  cudaMalloc(&devy, N*sizeof(float));

  for (int i=0; i<N; i++) {
    x[i] = 1.0f; // initialize host arrays
    y[i] = 2.0f;
  }

  // timer using cuda event api
  cudaEvent_t t0, t1;
  cudaEventCreate(&t0);
  cudaEventCreate(&t1);

  // init dev arrays via cp, std::memcpy + directional arg
  // source: host pointer
  // dest: device pointer
  cudaMemcpy(devx, x, N*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(devy, y, N*sizeof(float), cudaMemcpyHostToDevice);
	
  // start timer
  cudaEventRecord(t0);
  // launch kernel 
  // Between trip chevs is the `execution config`
  saxpy<<<(N+512)/512,512>>>(N, 2.0f, devx, devy);

  cudaEventRecord(t1);
  
  // cp back results after running kernel function
  cudaMemcpy(y, devy, N*sizeof(float), cudaMemcpyDeviceToHost);

  cudaEventSynchronize(t1);
  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, t0, t1);

  float maxError = 0.0f;
  for(int i=0; i<N; i++)
    maxError = max(maxError, abs(y[i]-4.0f));
  printf("Max error %.*f\n", 6, maxError);			
  printf("Effective Bandwidth (GB/s): %.*f\n", 6, N*4*3/milliseconds/1e6);
  // about 316 gb/s

  free(x);  // free host mem
  free(y);
  cudaFree(devx);  // free device mem
  cudaFree(devy);
}
