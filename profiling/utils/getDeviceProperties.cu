#include <stdio.h>

int main() {
  int nDevices;
  // returns number of cuda capable devices
  cudaError_t err = cudaGetDeviceCount(&nDevices);
  if (err != cudaSuccess)
    printf("%s\n", cudaGetErrorString(err));
  for (int i=0; i<nDevices; i++) {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, i);
    printf("Device Number: %d\n", i);
    printf("Device Name: %s\n", prop.name);
    printf("Memory Clock Rate (KHz); %d\n", prop.memoryClockRate);
    printf("Memory Bus Width (bits): %d\n", prop.memoryBusWidth);
    printf("Peak Memory Bandwidth (GB/s): %f\n\n", 2.0*prop.memoryClockRate*(prop.memoryBusWidth/8)/1.0e6);
  }
}
/* ex. Output
Device Number: 0
Device Name: NVIDIA GeForce RTX 3060 Laptop GPU
Memory Clock Rate (KHz); 7001000
Memory Bus Width (bits): 192
Peak Memory Bandwidth (GB/s): 336.048000
*/
