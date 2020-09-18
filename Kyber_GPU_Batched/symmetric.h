
// @Author: Arpan Jati
// Adapted from NewHope Reference Codebase and Parallelized using CUDA
// Updated : August 2019

#ifndef SYMMETRIC_H
#define SYMMETRIC_H

#include "params.h"

#include "fips202.h"

typedef struct {
  uint64_t s[25];
} keccak_state;

__device__ void kyber_shake128_absorb(keccak_state *s, unsigned char *input, 
	unsigned char x, unsigned char y);

__device__ void kyber_shake128_squeezeblocks(unsigned char *output, 
	unsigned long long nblocks, keccak_state *s);

__device__ void shake256_prf(unsigned char *output, 
	unsigned long long outlen, unsigned char *key, unsigned char nonce);

//#define hash_h(OUT, IN, INBYTES) sha3_256(OUT, IN, INBYTES)
//#define hash_g(OUT, IN, INBYTES) sha3_512(OUT, IN, INBYTES)

//#define xof_absorb(STATE, IN, X, Y) kyber_shake128_absorb(STATE, IN, X, Y)
//#define xof_squeezeblocks(OUT, OUTBLOCKS, STATE) kyber_shake128_squeezeblocks(OUT, OUTBLOCKS, STATE)

//#define prf(OUT, OUTBYTES, KEY, NONCE) shake256_prf(OUT, OUTBYTES, KEY, NONCE)
//#define kdf(OUT, IN, INBYTES) shake256(OUT, KYBER_SSBYTES, IN, INBYTES)

#define XOF_BLOCKBYTES 168

typedef keccak_state xof_state;


#endif /* SYMMETRIC_H */
