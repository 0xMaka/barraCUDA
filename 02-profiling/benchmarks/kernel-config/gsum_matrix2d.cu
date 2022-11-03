#include <cuda_runtime.h>
#include <stdio.h>
#include <time.h>

#define CHECK(call) { \
  const cudaError_t err = call; \
    if (err != cudaSuccess) { \
      printf("[!] Error: %s:%d", __FILE__, __LINE__); \
      printf("code: %d, reason: %s\n", err, cudaGetErrorString(err)); \
      exit(-10*err); \
    }\
}\

void initial_data(float *ip, const int size) {
  for (int i=0; i<size; i++)
    ip[i] = (float)(rand() &0xFF) / 10.0f;
  return;
}

void hsum_matrix(float *A, float *B, float *C, const int nx, const int ny) {
  float *ia = A;
  float *ib = B;
  float *ic = C;

  for (int iy=0; iy<ny; iy++) {
    for (int ix=0; ix<nx; ix++) {
      ic[ix] = ia[ix] + ib[ix];
    }
    ia += nx; ib += nx; ic += nx;
  }
  return;
}

__global__ void gsum_matrix(float *matA, float *matB, float *matC, int nx, int ny) {
  unsigned int ix = threadIdx.x + blockIdx.x * blockDim.x;
  unsigned int iy = threadIdx.y + blockIdx.y * blockDim.y;

  unsigned int idx = iy * nx + ix;
  if (ix < nx && iy < ny)
    matC[idx] = matA[idx] + matB[idx];
}

void verify(float *href, float *dref, const int N) {
  bool result = 1;
  double threshold = 1.0E-8;
  for (int i=0; i<N; i++) {
    if (abs(href[i] - dref[i]) > threshold) {
      result = 0;
      printf("[!] No Match\n");
      printf("[!] href: %5.2f dref: %5.2f @ current: %d\n", href[i], dref[i], i);
      break;
    }
  if (result)
    printf("[+] Arrays Match.\n");
  return;
  }
}

int main (int argc, char **argv) {
  printf("[x] ==========================================================\n");
  printf("[-] Loading Device Info..\n");

  int id = 0;
  cudaDeviceProp prop;
  CHECK(cudaGetDeviceProperties(&prop, id));
  
  printf("[+] Working With Device %d: %s\n", id, prop.name);
  CHECK(cudaSetDevice(id));

  int nx = 1 << 14;
  int ny = 1 << 14;
  int nxy = nx*ny;
  int bytes = nxy * sizeof(float);
  printf("[+] Matrix Size -> nx: %d ny: %d\n", nx, ny);

  float *ha, *hb, *href, *dref;
  
  ha = (float *) malloc(bytes);
  hb = (float *) malloc(bytes);
  href = (float *) malloc(bytes);
  dref = (float *) malloc(bytes);
  
  printf("[-] Initializing Data..\n");
  time_t t0, t1;
  t0 = clock();
  initial_data(ha, nxy);
  initial_data(hb, nxy);
  t1 = clock();
  printf("[+] Initialized in: %f\n", (double) (t1-t0) / CLOCKS_PER_SEC);

  memset(href, 0, bytes);
  memset(dref, 0, bytes);
  
  printf("[>] Running Host Side.\n"); 
  double times[2];

  t0 = clock();
  hsum_matrix(ha, hb, href, nx, ny);
  t1 = clock();
  times[0] = (double) (t1-t0) / CLOCKS_PER_SEC;

  float *dmatA, *dmatB, *dmatC;
  
  cudaMalloc((void **) &dmatA, bytes);
  cudaMalloc((void **) &dmatB, bytes);
  cudaMalloc((void **) &dmatC, bytes);

  cudaMemcpy(dmatA, ha, bytes, cudaMemcpyHostToDevice);
  cudaMemcpy(dmatB, hb, bytes, cudaMemcpyHostToDevice);

  int dimx = 32;
  //int dimx = 16;
  //int dimy = 32;
  int dimy = 16;

  dim3 block(dimx, dimy);
  dim3 grid((nx + block.x - 1) / block.x, (ny + block.y - 1) / block.y);

  printf("[>] Running Device Side.\n"); 
  t0 = clock();
  gsum_matrix <<< grid, block >>> (dmatA, dmatB, dmatC, nx, ny);

  cudaDeviceSynchronize();
  t1 = clock();
  times[1] = (double) (t1-t0) / CLOCKS_PER_SEC;

  cudaMemcpy(dref, dmatC, bytes, cudaMemcpyDeviceToHost);

  verify(href, dref, nxy);
  printf("[+] ----------------------------------------------------------\n");
  printf(
    "[>] CPU: %f | GPU: %f | <<<(%d,%d), (%d,%d)>>>\n", 
    times[0], times[1], grid.x, grid.y, block.x, block.y
  );
  printf("[x] ==========================================================\n");

  cudaFree(dmatA); cudaFree(dmatB); cudaFree(dmatC);
  free(ha); free(hb); free(href); free(dref);

  return 0;
}

/* ex Output
[x] ==========================================================
[-] Loading Device Info..
[+] Working With Device 0: NVIDIA GeForce RTX 3060 Laptop GPU
[+] Matrix Size -> nx: 16384 ny: 16384
[-] Initializing Data..
[+] Initialized in: 6.138594
[>] Running Host Side.
[>] Running Device Side.
[+] Arrays Match.
[+] ----------------------------------------------------------
[>] CPU: 0.241003 | GPU: 0.010223 | <<<(1024,1024), (16,16)>>>
[x] ==========================================================
*/
