
// @Author: Arpan Jati
// Adapted from NewHope Reference Codebase and Parallelized using CUDA
// Updated : August 2019

#ifndef POLY_H
#define POLY_H

#include <stdint.h>
#include "params.h"

typedef struct {
	int16_t threads[N_TESTS];
} threads;

/*
 * Elements of R_q = Z_q[X]/(X^n + 1). Represents polynomial
 * coeffs[0] + X*coeffs[1] + X^2*xoeffs[2] + ... + X^{n-1}*coeffs[n-1]
 */
typedef struct{
	threads coeffs[KYBER_N];
} poly;

__device__ void poly_compress(unsigned char *r, poly *a);
__device__ void poly_decompress(poly *r, unsigned char *a);

__device__ void poly_tobytes(unsigned char *r, poly *a);
__device__ void poly_frombytes(poly *r, unsigned char *a);

__global__  void poly_frommsg_n(int COUNT, poly* r,  unsigned char* msg);
__global__  void poly_tomsg_n(int COUNT, unsigned char* msg, poly* a);

__device__ void poly_frommsg(poly* r,  unsigned char* msg);
__device__ void poly_tomsg(unsigned char* msg, poly* a);

__global__ void poly_getnoise(int COUNT, poly *r, unsigned char *seed, unsigned char nonce);

__device__ void poly_ntt(poly *r);
__global__ void poly_ntt_n(int COUNT, poly* r);
__device__ void poly_invntt(poly *r);
__global__ void poly_invntt_n(int COUNT, poly* r);

__device__ void poly_basemul(poly *r,  poly *a, poly *b);
__global__ void poly_frommont_n(int COUNT, poly *r);

__global__ void poly_reduce_n(int COUNT, poly *r);
__device__ void poly_reduce(poly* r);

__device__ void poly_csubq(poly *r);

__global__  void poly_add_n(int COUNT, poly *r,  poly *a,  poly *b);
__global__  void poly_sub_n(int COUNT, poly *r,  poly *a,  poly *b);

__device__  void poly_add(poly* r,  poly* a,  poly* b);
__device__  void poly_sub(poly* r,  poly* a,  poly* b);

#endif
