
// @Author: Arpan Jati
// Adapted from NewHope Reference Codebase and Parallelized using CUDA
// Updated : August 2019

#ifndef FIPS202_H
#define FIPS202_H

#include <stdint.h>

#include "main.h"

#define SHAKE128_RATE 168
#define SHAKE256_RATE 136

__device__ void shake128_absorb(
	uint64_t * s,
	unsigned char* input,
	unsigned long long inputByteLen);

__device__ void shake128_squeezeblocks(
	unsigned char* output,
	unsigned long long nblocks,
	uint64_t* s);

__device__ void shake256(unsigned char* output,
	unsigned long long outputByteLen,
	unsigned char* input,
	unsigned long long inputByteLen);

__global__ void shake256_n(int COUNT, unsigned char* output,
	unsigned long long outlen,
	unsigned char* input,
	unsigned long long inlen, unsigned char* largeTemp);

/*__device__ void shake128_squeezeblocks_coal(unsigned char* output,
	unsigned long long nblocks, uint64_t* s);*/

#endif
