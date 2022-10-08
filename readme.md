# CUDA - Notes and experimentation 🐟

## Basics:
### \_\_global\_\_
```
__global__ void my_func(void);
```
- Indicates a function that:
- - Runs on the GPU (device)
- - Is called from the CPU (host), or other device code.

### nvcc - nvidia's c complier
- Parses which functions to process:
- - Device functions such as  those decorated by the \_\_global\_\_ dundar, are processed by the nvidia compiler.
- - Host functions, such a standard main() are processed by the systems compiler, such as gcc.

### Unified memory

- Provides a single memory space, accessible by all hosts and devices on the system.
- - cudaMallocManaged() - returns a pointer that can be accessessed from host, or device code.
- - Once done with the data, pointer should be passed to cudaFree().

### grid-strid-loop
```
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  for (int i = index; i < n; i += stride)
```

- ```(blockIdx.x * blockDim.x) + threadIdx.x``` is idiomatic CUDA.
- - Each thread obtains index via computing offset relative to start of block.
- - (block index * block size) + thread index

- - ```(blockDim.x * gridDim.x)``` sets *stride* to the total number of threads in grid

### Host-Device Synchronization
- Data transfer between host and device using cudaMemcpy() is synchronous
- - Transfers are blocking
- - Synchronous data transfers will not begin until all previously issued cuda calls have completed
- - Additional calls cannot be made until transfer is complete
- - Once kernels are launched control returns to host

**NOTE** Kernal launches are asynchronous

---

### Tasks:
- [ ] Create a character frequency analyizer and adapt it to complete the 3rd matisano challenge.

- [ ] Explore Monte Carlo options pricing, before analyzing sim utility from a crypto native perspective.
