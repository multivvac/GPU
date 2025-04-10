#ifndef LENET_KERNEL_H
#define LENET_KERNEL_H

__global__ void conv_forward();
__global__ void liear_forward();
__global__ void conbackward();
__global__ void conv_backward();

#endif
