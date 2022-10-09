# CUDA - Personal notes and experimentation 🐟
---

## Basics:

Cuda programs involve running code on two different platforms at the same time.
- A host system with one or more CPUs 
- One or more CUDA-enabled GPU devices

Host (CPU) controls flow of program, offloading explicitly defined kernel functions to the device (GPU).

An ex. sequence would be to:
- - Declare and allocate host and device memory
- - Initialize host data
- - Transfer data from the host to device
- - Execute one or more kernels to operate on data
- - Transfer results from device to host

GPU's provide much higher instruction throughput and memory bandwidth than CPU's.

- CPU's are designed to execute a sequence of operations (a thread), as fast as possible
- - Can execute a few tens in parallel 
- GPU's amortize slower single-thread performance for greater throughput
- - Excel at executing thousands of threads in parallel, with transistors devoted to data processing over data caching and flow control

### \_\_global\_\_ (keyword)
```
__global__ void my_func(void);
```
- Indicates a function that:
- - Runs on the GPU (device)
- - Is called from the CPU (host), or other device code

### nvcc - nvidia's c complier
- Parses which functions to process:
- - Device functions such as  those decorated by the \_\_global\_\_ dundar, are processed by the nvidia compiler
- - Host functions, such a standard main() are processed by the systems compiler, such as gcc

### Unified Memory
Offers a *single-pointer-to-data* model. 
Similar to cuda zero-copy memory, except here seperates memory and execution areas to increase access speed, as opposed to location being pinned in system memory.
- Provides a single memory space, accessible by all hosts and devices on the system.
- - cudaMallocManaged() - returns a pointer that can be accessessed from host, or device code
- - Once done with the data, pointer should be passed to cudaFree()

### grid-stride-loop
```
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  for (int i = index; i < n; i += stride)
    y[i] = x[i] + y[i];
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

**NOTE** :bulb: Kernel launches are asynchronous

### Memory Bandwidth
- Memory Bandwidth is the theoretical maximum amount of data the bus can handle at a given time 
- - it is a key factor in how quickly a GPU can access and utilize its framebuffer

#### Theoretical vs Effective Memory Bandwidth
- Theoretical bandwidth can be calculated using hardware specifications
- - Look for clock rate and memory interface width

- RTX-3060-Mobile - Spec lists a Memory Bandwidth of 336 GB/s
- - BUSWIDTH = 192
- - CLOCKRATE = 1750 
- - DDRMULTI = 8 // DDR6
$$^B_WTheoretical = (((CLOCKRATE * (10 ** 6)) * (BUSWIDTH/8)) * DDRMULI) / (10 ** 9)$$ 

1750 * 10** 6 * (192/8) * 8 / 10** 9 = 336 GB/s

#### Effective Bandwidth
- Effective bandwidth needs to be timed by measuring specfic activities
- - We can use the equation: 
$$^B_WEffective = ((^RB) + (^WB)) / (t * (10 ** 9))$$

- Where: 
- - BWEffective = bandwidth in GB's units
- - RB = number of bytes per kernel
- - WB = number of bytes written per kernel
- - t = elapsed time in seconds

Running on the device above results averaged 316 GB/s
```
Effective Bandwidth (GB/s): 316.280736
```

---

### Tasks:
- [ ] Create a character frequency analyizer and adapt it to complete the 3rd matisano challenge.

- [ ] Explore Monte Carlo options pricing, before analyzing sim utility from a crypto native perspective.


---

#### List of external resources:

- Device specs and datasheets (3060ti desktop/3060 mobile)
- - https://www.nvidia.com/en-gb/geforce/graphics-cards/30-series/rtx-3060-3060ti/
- - https://www.notebookcheck.net/NVIDIA-GeForce-RTX-3060-Mobile-GPU-Benchmarks-and-Specs.497453.0.html

- Efficient GPU Programming with Modern C++ 
- - https://www.youtube.com/watch?v=8pJS3n4MITM
- Unified Memory for CUDA Beginners
- - https://developer.nvidia.com/blog/unified-memory-cuda-beginners/
- An Easy Introduction to CUDA C and C++
- - https://developer.nvidia.com/blog/easy-introduction-cuda-c-and-c/
- Intro to CUDA (part 2): Programming Model
- - https://www.youtube.com/watch?v=cKI20rITSvo
- How to Implement Performance Metrics in CUDA C/C++
- - https://developer.nvidia.com/blog/how-implement-performance-metrics-cuda-cc/
- How to Query Device Properties and Handle Errors in CUDA C/C++
- - https://developer.nvidia.com/blog/how-query-device-properties-and-handle-errors-cuda-cc/
- CUDA-Accelerated Monte-Carlo for HPC
- - https://www.nvidia.com/docs/IO/116711/sc11-montecarlo.pdf
- CUDA C++ Best Practices Guide
- - https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html
- CUDA C++ Programming Guide
- - https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html
