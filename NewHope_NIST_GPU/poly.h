
// @Author: Arpan Jati
// Adapted from NewHope Reference Codebase and Parallelized using CUDA
// Updated : August 2019

#ifndef POLY_H
#define POLY_H

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdint.h>
#include "params.h"

/* 
 * Elements of R_q = Z_q[X]/(X^n + 1). Represents polynomial
 * coeffs[0] + X*coeffs[1] + X^2*xoeffs[2] + ... + X^{n-1}*coeffs[n-1] 
 */

typedef struct {
	uint16_t threads[N_TESTS];
} threads;


#ifdef _WIN32
__declspec(align(32))  typedef struct {
	threads coeffs[NEWHOPE_N];
} poly;

#else
typedef struct {
	threads coeffs[NEWHOPE_N];
} poly __attribute__((aligned(32)));


#endif // _WIN32

__device__ void poly_uniform(poly *a,  unsigned char *seed);
//__device__ void poly_uniform_coal(poly* a, unsigned char* seed, unsigned char* temp_reg);

__global__ void poly_sample(int COUNT, poly *r,  unsigned char *seed, unsigned char nonce);
__global__ void poly_add(int COUNT, poly *r,  poly *a,  poly *b);
__global__ void poly_sub(int COUNT, poly* r,  poly* a,  poly* b);

__global__ void poly_ntt(int COUNT, poly *r);
__global__ void poly_invntt(int COUNT, poly *r);
__global__ void poly_mul_pointwise(int COUNT, poly *r,  poly *a,  poly *b);

__device__ void poly_frombytes(poly *r,  unsigned char *a);
__global__ void poly_frombytes_n(int COUNT, poly* r,  unsigned char* a);

__device__ void poly_tobytes(unsigned char *r,  poly *p);
__global__ void poly_tobytes_n(int COUNT, unsigned char* r,  poly* p);


__device__ void poly_compress(unsigned char *r,  poly *p);
__device__ void poly_decompress(poly *r,  unsigned char *a);

__global__ void poly_frommsg_n(int COUNT, poly *r,  unsigned char *msg);
__global__ void poly_tomsg_n(int COUNT, unsigned char *msg,  poly *x);


#endif
