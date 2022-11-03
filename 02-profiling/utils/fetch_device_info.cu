#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

int main (int argc, char **argv) {
  printf("[x] ===================================================================\n");   
  printf("[-] Fetching Device Info..\n");    
  int deviceCount = 0;
  cudaError_t err_id = cudaGetDeviceCount(&deviceCount);

  if (err_id != cudaSuccess) {
    printf("[!] cudaDeviceCount: %d\n[>] %s\n", (int) err_id, cudaGetErrorString(err_id));
    printf("[>] Result = FAIL\n");
    exit(EXIT_FAILURE);
  }

  if (deviceCount == 0) {
    printf("[!] No Devices Found.");
  } else {
    printf("[+] Detected %d CUDA capable devices(s)\n", deviceCount);
  }

  int dev, driverVersion = 0, runtimeVersion = 0;
  dev = 0;
  cudaSetDevice(dev);
  cudaDeviceProp prop;
  
  cudaGetDeviceProperties(&prop, dev);
  printf("[+] Device %d: %s\n", dev, prop.name);
  cudaDriverGetVersion(&driverVersion);
  cudaRuntimeGetVersion(&runtimeVersion);

  printf(
    "[>] CUDA Driver / Runtime Versions: %d.%d / %d.%d\n", 
    driverVersion / 1000, (driverVersion % 100) / 10,
    runtimeVersion / 1000, (runtimeVersion % 100) / 10
  );
  printf("[>] CUDA Capability Major / Minor Version Numbers: %d.%d\n", prop.major, prop.minor);
  printf(
    "[>] Total Amount Of Global memory: %.2f MBytes (%llu bytes)\n", 
    (float) prop.totalGlobalMem / (pow(1024.0,3)), (unsigned long long) prop.totalGlobalMem
  );

  printf("[>] GPU Clock Rate: %.0f mhZ (%.2f GHz)\n", prop.clockRate * 1e-3f, prop.clockRate * 1e-6f);
  printf("[>] Memory Clock Rate: %.0f MHz\n", prop.memoryClockRate * 1e-3f);
  printf("[>] Memory Bus Width: %d-bit\n", prop.memoryBusWidth);
  
  if (prop.l2CacheSize)
    printf("[>] L2 Cache Size: %d bytes\n", prop.l2CacheSize);

  printf(
    "[+] Max Texture Dimension Size (x,y,z): \n[>] 1D=(%d), 2D=(%d,%d), 3D=(%d,%d,%d)\n",
    prop.maxTexture1D, 
    prop.maxTexture2D[0], prop.maxTexture2D[1], 
    prop.maxTexture3D[0], prop.maxTexture3D[1], prop.maxTexture3D[2]
  );

  printf(
    "[+] Max Layered Texture Size (dim) * layers: \n[>] 1D=(%d) * %d, 2D=(%d,%d) * %d\n",
    prop.maxTexture1DLayered[0], prop.maxTexture1DLayered[1], 
    prop.maxTexture2DLayered[0], prop.maxTexture2DLayered[1], prop.maxTexture2DLayered[2]
  );

  printf("[>] Total Amount of Constant Memory: %lu bytes\n", prop.totalConstMem);
  printf("[>] Total Amount Of Shared memory per block: %lu bytes\n", prop.sharedMemPerBlock);
  printf("[>] Total Number of Registers Per Block: %d\n", prop.regsPerBlock);
  printf("[>] Warp Size: %d\n", prop.warpSize);

  printf("[>] Total Number Thread Per MultiProcessor: %d\n", prop.maxThreadsPerMultiProcessor);

  printf(
    "[>] Maximum Number Of Threads Per Block: %d\n", prop.maxThreadsPerBlock
  );

  printf(
    "[>] Max Size Of Each Dimension Per Block: %d * %d * %d\n",
    prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]
  );

  printf(
    "[>] Max Size Of Each Dimension Per Grid: %d * %d * %d\n",
    prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]
  );

  printf("[>] Maximum Memory Pitch: %lu byte\n", prop.memPitch);

  printf("[x] ===================================================================\n");   
  exit(EXIT_SUCCESS);
}
/* ex. Output
[x] ===================================================================
[-] Fetching Device Info..
[+] Detected 1 CUDA capable devices(s)
[+] Device 0: "NVIDIA GeForce RTX 3060 Laptop GPU"
[>] CUDA Driver / Runtime Versions: 11.8 / 11.8
[>] CUDA Capability Major / Minor Version Numbers: 8.6
[>] Total Amount Of Global memory: 5.66 MBytes (6076956672 bytes)
[>] GPU Clock Rate: 1425 mhZ (1.42 GHz)
[>] Memory Clock Rate: 7001 MHz
[>] Memory Bus Width: 192-bit
[>] L2 Cache Size: 3145728 bytes
[+] Max Texture Dimension Size (x,y,z):
[>] 1D=(131072), 2D=(131072,65536), 3D=(16384,16384,16384)
[+] Max Layered Texture Size (dim) * layers:
[>] 1D=(32768) * 2048, 2D=(32768,32768) * 2048
[>] Total Amount of Constant Memory: 65536 bytes
[>] Total Amount Of Shared memory per block: 49152 bytes
[>] Total Number of Registers Per Block: 65536
[>] Warp Size: 32
[>] Total Number Thread Per MultiProcessor: 1536
[>] Maximum Number Of Threads Per Block: 1024
[>] Max Size Of Each Dimension Per Block: 1024 * 1024 * 64
[>] Max Size Of Each Dimension Per Grid: 2147483647 * 65535 * 65535
[>] Maximum Memory Pitch: 2147483647 byte
[x] ===================================================================
*/

