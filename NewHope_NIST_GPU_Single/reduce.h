
// @Author: Naina Gupta
// Adapted from NewHope Reference Codebase and Parallelized using CUDA
// Updated : August 2019

#ifndef REDUCE_H
#define REDUCE_H

#include <stdint.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

uint16_t montgomery_reduce(uint32_t a);
__device__ uint16_t montgomery_reduce_device(uint32_t a);
uint16_t barrett_reduce(uint16_t a);
__device__ uint16_t barrett_reduce_device(uint16_t a);

#endif
