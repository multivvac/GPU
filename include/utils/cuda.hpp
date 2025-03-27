#ifndef CUDA_H
#define CUDA_H

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

#define CUDA_CHECK(call)                                                      \
    do {                                                                      \
        cudaError_t err = call;                                               \
        if (err != cudaSuccess) {                                             \
            fprintf(stderr, "CUDA Error at %s:%d in %s(): %s\n",             \
                    __FILE__, __LINE__, __func__, cudaGetErrorString(err));  \
            exit(EXIT_FAILURE);                                               \
        }                                                                     \
    } while (0)

#endif // CUDA_H

