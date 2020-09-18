
// @Author: Naina Gupta
// Adapted from FrodoKEM Reference Codebase and Parallelized using CUDA
// Updated : August 2019

#ifndef FIPS202_H
#define FIPS202_H

#include <stdint.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define SHAKE128_RATE 168
#define SHAKE256_RATE 136

__device__ void shake128_Sh(unsigned char *output, unsigned long long outlen, const unsigned char *input, unsigned long long inlen, uint64_t *s, int threadIdx, int blockId);

  void shake256_absorb(uint64_t *s, const unsigned char *input, unsigned int inputByteLen);
  void shake256_squeezeblocks(unsigned char *output, unsigned long long nblocks, uint64_t *s);
  void shake256(unsigned char *output, unsigned long long outlen, const unsigned char *input,  unsigned long long inlen);
  __global__ void shake256_kernel(unsigned char *output, unsigned long long outlen, unsigned char *input, unsigned long long inlen);
  __global__ void shake256_kernel_single(unsigned char *output, unsigned long long outlen, unsigned char *input, unsigned long long inlen, unsigned char const_byte);
  __global__ void shake256_kernel1(unsigned char* output, unsigned long long outlen, unsigned char* input, unsigned long long inlen, unsigned char const_byte);

#endif
