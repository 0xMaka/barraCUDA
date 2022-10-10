#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

// device function - sum array
__global__ 
void d_sum_array(float *A, float *B, float *C, const int N) {
  int i = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (i < N)
    C[i] = A[i] + B[i];
}

// host function - sum array
void h_sum_array(float *A, float *B, float *C, const int N) {
  for(int idx=0; idx<N; idx++)
    C[idx] = A[idx] + B[idx];
}
// random data generator
void initial_data(float *ip, int size) {
  time_t t;
  srand((unsigned int) time(&t));
  for (int i=0; i<size; i++) {
    ip[i] = (float) ( rand() & 0xFF )/10.0f;
  }
}

// result checker
void verify(float *href, float *dref, const int N) {
  bool result = 1;
  double threshold = 1.0E-8;
  for (int i=0; i<N; i++) {
    if (abs(href[i] - dref[i]) > threshold) {
      result = 0;
      printf("[!!] No Match\n");
      printf("[!!] href: %5.2f dref: %5.2f @ current: %d\n", href[i], dref[i], i);
      break;
    }
  if (result)
    printf("[+] Results are matching!\n");
  return;
  }
}
  
int main(int argc, char **argv) {
	printf("[+] Setting up device...\n");
  
	// list some device info
	int id = 0;
	cudaDeviceProp prop;
	cudaGetDeviceProperties(&prop, id);
	printf("[=] %s with id %d\n", prop.name, id);

  // set vector size
  int elements = 1<<20;
  printf("[+] Setting vector size: %d\n", elements);

  // allocate mem on the cpu
  size_t bytes = elements * sizeof(float);
  // - - host pointers a and b
  // - - host reference pointer, device reference pointer
  float *ha, *hb, *href, *dref;
  ha = (float *)malloc(bytes);
  hb = (float *)malloc(bytes);
  href = (float *)malloc(bytes);
  dref = (float *)malloc(bytes);
  
	printf("[+] Generating data with %d elements\n", elements);
  // generate data
  initial_data(ha, elements);
  initial_data(hb, elements);
  
  // pad blocks
  memset(dref, 0, bytes);
  memset(href, 0, bytes);

  // allocate global mem on the gpu
  float *da, *db, *dc;
  cudaMalloc((float **)&da, bytes);
  cudaMalloc((float **)&db, bytes);
  cudaMalloc((float **)&dc, bytes);

  printf("[+] Copying data to device: %d\n", id);
  // transfer data to device
  cudaMemcpy(da, ha, bytes, cudaMemcpyHostToDevice);
  cudaMemcpy(db, hb, bytes, cudaMemcpyHostToDevice);
  cudaMemcpy(dc, dref, bytes, cudaMemcpyHostToDevice);

	// set up timer
	clock_t t0, t1;

	printf("[+] Running host side..\n");
  t0 = clock();
  // run host side
  h_sum_array(ha, hb, href, elements);
  t1 = clock();
	printf("[+] CPU side array finished in: %f\n", (double)(t1-t0)/CLOCKS_PER_SEC);

	// kernel prep
  int len = 512;
  dim3 block (len);
  dim3 grid ((elements + block.x -1) / block.x);
  
	printf("[+] Running Device side..\n");
	t0 = clock();
	// run kernel
  d_sum_array<<<grid,block>>>(da, db, dc, elements);
  
	// check for errors
	cudaError_t errSync = cudaGetLastError();
  cudaError_t errAsync = cudaDeviceSynchronize();
	t1 = clock();
	printf("[+] GPU side array finished in: %f\n", (double)(t1-t0)/CLOCKS_PER_SEC);
  
  printf("[+] Checking for errors..\n");
	if (errSync != cudaSuccess)
    printf("[!!] Sync kernel error: %s\n", cudaGetErrorString(errSync));
  if (errAsync != cudaSuccess)
    printf("[!!] Async kernel error: %s\n", cudaGetErrorString(errAsync));

  printf("[+] Copying data from device: %d\n", id);
  // transfer data back from device
  cudaMemcpy(dref, dc, bytes, cudaMemcpyDeviceToHost);

  printf("[+] Comparing results..\n");
  // display result
  verify(href, dref, elements);

  printf("[=] Cleaning up..\n");
  // free memory
  cudaFree(da);
  cudaFree(db);
  cudaFree(dc);
  free(ha);
  free(hb);
  free(href);
  free(dref);

  printf("[x] Closing.\n");
  return 0;
}

/* ex. Output
[+] Setting up device...
[=] NVIDIA GeForce RTX 3060 Laptop GPU with id 0
[+] Setting vector size: 1048576
[+] Generating data with 1048576 elements
[+] Copying data to device: 0
[+] Running host side..
[+] CPU side array finished in: 0.001096
[+] Running Device side..
[+] GPU side array finished in: 0.000061
[+] Checking for errors..
[+] Copying data from device: 0
[+] Comparing results..
[+] Results are matching!
[=] Cleaning up..
[x] Closing.
 */
