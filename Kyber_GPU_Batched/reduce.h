
// @Author: Arpan Jati
// Adapted from NewHope Reference Codebase and Parallelized using CUDA
// Updated : August 2019

#ifndef REDUCE_H
#define REDUCE_H

#include <stdint.h>

#define MONT 2285 // 2^16 % Q
#define QINV 62209 // q^(-1) mod 2^16

__device__ int16_t montgomery_reduce(int32_t a);

__device__ int16_t barrett_reduce(int16_t a);

__device__ int16_t csubq(int16_t x);

#endif
