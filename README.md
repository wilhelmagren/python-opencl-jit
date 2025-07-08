<div align="center">

# python-opencl-jit
##### Benchmark of LRU cached JIT compiler for opencl kernels.

</div>


This is just a small benchmark script for comparing numpy timings vs OpenCL + JIT compiler
and LRU-cached JIT compiled kernels.

Hardware:
- NVIDIA GeForce RTX 4070
    - Frequency: 1920 MHz (turbo: 2505 MHz)
    - Memory: GDDR6X 12 GB 21000 MHz 192 bits
- AMD Ryzen 7 7700X
    - Frequency: 4.5 GHz (turbo: 5.4 GHz)
    - Cores: 8
    - Threads: 16
    - Cache: L2 8 MB, L3 32 MB

Results (matmul and add ops):

```
---------------------- SUMMARY -------------------
[NumPy]         average: 2.058 ms       (runs=100)
[JIT]           average: 73.421 ms      (runs=100)
[JIT-CACHED]    average: 12.067 ms      (runs=100)
--------------------------------------------------

```


