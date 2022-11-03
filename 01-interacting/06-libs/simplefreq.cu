#include <iostream>

#include <thrust/device_ptr.h>
#include <thrust/count.h>
#include <thrust/execution_policy.h>

#include "range.hpp" // python like for loop by github.com/harrism
using namespace util::lang;

template<typename T>
using step_range = typename range_proxy<T>::step_range_proxy;

template <typename T>
__device__ step_range<T> grid_stride_range(T begin, T end) {
  begin += blockDim.x * blockIdx.x + threadIdx.x;
  return range(begin, end).step(gridDim.x * blockDim.x);
}

template <typename T, typename Predicate>

__device__ void count_if(int *count, T *data, int n, Predicate p) {
  for (auto i : grid_stride_range(0, n)) {
    if (p(data[i])) 
      atomicAdd(count, 1);
  }
}

__global__ void xyz_freq(int *count, char *text, int n) {
  const char letters[] {'f','o','x'};
  count_if(count, text, n, [&](char c) {
    for (const auto x : letters) {
      if (c==x)
        return true;
    return false;
    }
  });
}

int main(void) {
  const char *file = "toadoftoadhall.txt";

  int nbytes = 16*100000;
  char *htxt = (char *)malloc(nbytes);

  char *dtxt;
  cudaMalloc((void **)&dtxt, nbytes);

  FILE *fp = fopen(file, "r");
  int len = fread(htxt, sizeof(char), nbytes, fp);
  fclose(fp);
  std::cout << "[+] Read " << len << " bytes from " << file << std::endl;
  
  cudaMemcpy(dtxt, htxt, len, cudaMemcpyHostToDevice);

  int href = 0;
  int *dcount;
  cudaMalloc(&dcount, sizeof(int));
  cudaMemset(dcount, 0, sizeof(int));

  xyz_freq<<<8,256>>>(dcount, dtxt, len);
  cudaMemcpy(&href, dcount, sizeof(int), cudaMemcpyDeviceToHost);

  std::cout << "[>] Found " << href << " instances of f, o, x." << std::endl;
  cudaFree(dcount); cudaFree(dtxt);
  free(htxt);

  return 0;
}

