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

---