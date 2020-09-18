
// @Author: Naina Gupta
// Adapted from NewHope Reference Codebase and Parallelized using CUDA
// Updated : August 2019

#ifndef FIPS202_H
#define FIPS202_H

#include <stdint.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define SHAKE128_RATE 168
#define SHAKE256_RATE 136

extern __constant__ uint8_t noise_seed_const[32];
extern __constant__ uint8_t public_seed_const[32];

void keccak_absorb_shake256_host(uint64_t *s, unsigned int r, unsigned char *m, unsigned long long int mlen, unsigned char p);
void keccak_squeezeblocks_host(unsigned char *h, unsigned long long int nblocks, uint64_t *s, unsigned int r);
void shake256_host(unsigned char* output, unsigned long long outputByteLen, unsigned char* input, unsigned long long inputByteLen);

__device__ void KeccakF1600_StatePermute_sh128_comb(unsigned char domainSep, unsigned char p, uint64_t*state, int blockSz);

__device__ void KeccakF1600_StatePermute_sh_comb(unsigned char nonce, unsigned char domainSep, unsigned char p, uint64_t* state, int threadIdx, int blockSz);
#endif
