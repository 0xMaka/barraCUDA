#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

// device function - sum array
__global__ 
void d_sum_array(float *A, float *B, float *C, const int N) {
  // grid-stride-loop 
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
	  printf("[+] Matching\n");
	return;
  }
}
  
int main(int argc, char **argv) {
  // set vector size
  int elements = 1024;
  printf("[+] Vector size: %d\n", elements);

  // allocate mem on the cpu
  size_t bytes = elements * sizeof(float);
  // host pointers a and b
  // host reference pointer, gpu reference pointer
  float *ha, *hb, *href, *dref;
  ha = (float *)malloc(bytes);
  hb = (float *)malloc(bytes);
  href = (float *)malloc(bytes);
  dref = (float *)malloc(bytes);

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

  // transfer data to device
  cudaMemcpy(da, ha, bytes, cudaMemcpyHostToDevice);
  cudaMemcpy(db, hb, bytes, cudaMemcpyHostToDevice);
  cudaMemcpy(dc, dref, bytes, cudaMemcpyHostToDevice);

  // kernel prep
  int len = 512;
  dim3 block (len);
  dim3 grid ((elements + block.x -1) / block.x);
  // run kernel
  d_sum_array<<<grid,block>>>(da, db, dc, elements);
  
  // sync up
  cudaDeviceSynchronize();
  // transfer data back from device
  cudaMemcpy(dref, dc, bytes, cudaMemcpyDeviceToHost);

  // redo host side to check against
  h_sum_array(ha, hb, href, elements);
  // display result
  verify(href, dref, elements);

  // free memory
  cudaFree(da);
  cudaFree(db);
  cudaFree(dc);
  free(ha);
  free(hb);
  free(href);
  free(dref);

  return 0;
}
