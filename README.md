# GPU

Hands-On GPU Learning

## Miscellaneous

Some information may be useful for theoretical calculations.

### Bandwidth vs Throughput

| Aspect              | Bandwidth                            | Throughput                              |
|---------------------|--------------------------------------|------------------------------------------|
| What it measures     | **Data transfer rate**               | **Compute or instruction rate**          |
| Unit                | GB/s                                 | FLOPS, operations/sec                    |
| Related to          | **Memory system**                    | **ALUs / cores**                         |
| Bottleneck when?    | Moving lots of data                  | Performing lots of calculations          |
| Optimized by        | Coalesced access, memory hierarchy   | Occupancy, instruction-level parallelism |

### C++ / CUDA Data Size Table

| Data Type             | Size (Bytes) | Bit Width | Description / GPU Notes                          |
|-----------------------|--------------|-----------|--------------------------------------------------|
| `bool`                | 1            | 8 bits    | Stored as 1 byte, often packed in structures     |
| `char`                | 1            | 8 bits    | Signed/unsigned 8-bit integer                    |
| `unsigned char`       | 1            | 8 bits    |                                                  |
| `short`               | 2            | 16 bits   |                                                  |
| `unsigned short`      | 2            | 16 bits   |                                                  |
| `int`                 | 4            | 32 bits   | Most common integer type                         |
| `unsigned int`        | 4            | 32 bits   |                                                  |
| `long` (Windows)      | 4            | 32 bits   |                                                  |
| `long` (Linux 64-bit) | 8            | 64 bits   | Platform-dependent                               |
| `long long`           | 8            | 64 bits   | Use for portable 64-bit integer                  |
| `unsigned long long`  | 8            | 64 bits   |                                                  |
| `size_t`              | 4/8          | 32/64 bits| Platform-dependent size type                     |
| `float`               | 4            | 32 bits   | FP32 (Single-precision float)                    |
| `double`              | 8            | 64 bits   | FP64 (Double-precision float)                    |
| `__half` (CUDA)       | 2            | 16 bits   | FP16 (half-precision float, GPU-accelerated)     |
| `__nv_bfloat16` (CUDA)| 2            | 16 bits   | Brain floating-point (used in AI)                |
| `int2`                | 8            | 64 bits   | Two 32-bit ints (used in CUDA vector types)      |
| `float2`              | 8            | 64 bits   | Two 32-bit floats                                |
| `float4`              | 16           | 128 bits  | Common in graphics and CUDA memory coalescing    |
| `double2`             | 16           | 128 bits  | Two doubles                                      |
| `__half2`             | 4            | 32 bits   | Two FP16s packed (used in vectorized operations) |

### Binary Unit Table

| Unit | Size(Bytes) |
| ------------- | -------------- |
| 1 KB| 1 KiB = 1,024(2^10) Bytes |
| 1 MB| 1 MiB = 1,048,576(2^20) Bytes |
| 1 GB| 1 GiB = 1,073,741,824(2^30) Bytes |
| 1 TB| 1 TiB = 1,099,511,627,776(2^40) Bytes |
| 1 PB| 1 PiB = 1,125,899,906,842,624(2^50) Bytes |
| 1 EB| 2^60 Bytes |

## Acknowledgements

Some of the quizzes and scripts are inspired by the [reference-kernel's problems](https://github.com/gpu-mode/reference-kernels). Huge thanks to the amazing GPU MODE community!
