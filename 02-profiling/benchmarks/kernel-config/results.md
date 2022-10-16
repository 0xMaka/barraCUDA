## Results

- A summary of performance for different execution parameters.

| Kernel Config | Elapsed Timed | Block Number|
|:--------------|:-------------:|:------------|
|   (32, 32)    |   0.010837    |   512x512   |
|   (32, 16)    |   0.010182    |   512x1024  |
|   (16, 16)    |   0.010224    |  1024x1024  |

**NOTE** Increasing the number of blocks does not guarantee an increase in performance.

| Kernel function |     Configuration   | Elapsed Time |
|:----------------|:-------------------:|:-------------|
|  gsum\_matrix1d |    (128,1),(128,1)  |   0.011637   |
|  gsum\_matrix2d |  (512,1024),(32,16) |   0.010154   |
| gsum\_matrixmix |  (64,16384),(256,1) |  0.0026614   |

- From the examples we can see
- - Changing configs effects performance
- - Is worth spending time on the kernel as naive implimentations are unlikley to yield the best results
- - Is worth experimenting with different block, grid combos, on sets of data.
