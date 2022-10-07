
# CUDA - Notes and expirementation

## Basics:
### __\global\__
#### NOTE: 
__global__ void my_func(void) {
...
}

#### KEYWORD: \__global\__ dunder, indicates a function that:
- Runs on the GPU (host)
- Is called from the CPU (host), or other device code.

### nvcc - nvidia's c complier
- Device fucntions, those decorated by the __global__ dundar are compuiled by nvidia compiler
