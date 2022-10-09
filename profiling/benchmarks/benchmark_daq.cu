#include <iostream>
#include <chrono>

/* we typically don't need all device properties, just a few.
  couldaDeviceGetAttribute() is slow as it returns everything
  if wanting to be selective and save time, we can use cudaDeviceGetAttribute()
  To see just how much time we can save let's benchmark queries..
*/
int devId = 0;
void get_all(void) {
  cudaDeviceProp prop;
  auto t0 = std::chrono::high_resolution_clock::now();
	for(int i=0; i<25; i++)
    cudaGetDeviceProperties(&prop, devId);
  auto t1 = std::chrono::high_resolution_clock::now();
  std::cout << "cudaGetDeviceProperties -> "
  <<std::chrono::duration_cast<std::chrono::microseconds>(t1-t0).count()/25.0
	<< "us" << std::endl;
}

void get_selected(void) {
  int smemSize, numProcs;
  auto t0 = std::chrono::high_resolution_clock::now();
  for (int i=0; i<25; i++) {
    cudaDeviceGetAttribute(
      &smemSize,
      cudaDevAttrMaxSharedMemoryPerBlock, devId
    );
    cudaDeviceGetAttribute(
      &numProcs,
      cudaDevAttrMultiProcessorCount, devId
    );
	}
  auto t1 = std::chrono::high_resolution_clock::now();
  std::cout << "cudaGetDeviceAttribute -> "
  <<std::chrono::duration_cast<std::chrono::microseconds>(t1-t0).count()/25.0
  << "us" << std::endl;
}

int main(void) {
  get_all();
	get_selected();
  return 0;
}
/* ex. Output
cudaGetDeviceProperties -> 3664.08us
cudaGetDeviceAttribute -> 0.04us
*/
