# CUDA - Notes and experimentation 🐟

## Basics:
### KEYWORD - \_\_global\_\_: 
```
__global__ void my_func(void);
```

- Indicates a function that:
- - Runs on the GPU (host)
- - Is called from the CPU (host), or other device code.

### nvcc - nvidia's c complier
- Parses which functions to process:
- - Device functions, those decorated by the \_\_global\_\_ dundar, are processed by the nvidia compiler.
- - Host functions, such a standard main() are processed host systems compiler such as gcc.
