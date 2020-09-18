
// @Author: Arpan Jati
// Adapted from NewHope Reference Codebase and Parallelized using CUDA
// Updated : August 2019

#ifndef FIPS202_H
#define FIPS202_H

#include <stdint.h>

#define SHAKE128_RATE 168
#define SHAKE256_RATE 136
#define SHA3_256_RATE 136
#define SHA3_512_RATE  72

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

__device__ void shake128_absorb(uint64_t *s, unsigned char *input,
	unsigned int inputByteLen);

__device__ void shake128_squeezeblocks(unsigned char *output, 
	unsigned long long nblocks, uint64_t *s);

__device__ void shake256(unsigned char *output, unsigned long long outlen,
	unsigned char *input,  unsigned long long inlen);

__device__ void sha3_256(unsigned char *output, unsigned char *input,  
	unsigned long long inlen);

__device__ void sha3_512(unsigned char *output, unsigned char *input, 
	unsigned long long inlen);

__global__ void sha3_256_n(int COUNT, unsigned char* output,
		unsigned char* input, unsigned long long inlen);

__global__ void sha3_512_n(int COUNT, unsigned char* output,
		unsigned char* input, unsigned long long inlen);

#endif
