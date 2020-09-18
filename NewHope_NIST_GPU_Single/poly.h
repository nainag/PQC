
// @Author: Naina Gupta
// Adapted from NewHope Reference Codebase and Parallelized using CUDA
// Updated : August 2019

#ifndef POLY_H
#define POLY_H

#include <stdint.h>
#include "params.h"

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

/* 
 * Elements of R_q = Z_q[X]/(X^n + 1). Represents polynomial
 * coeffs[0] + X*coeffs[1] + X^2*xoeffs[2] + ... + X^{n-1}*coeffs[n-1] 
 */


#ifdef _WIN32
__declspec(align(32)) typedef struct {
	uint16_t coeffs[NEWHOPE_N];
} poly;
#else
typedef struct {
	uint16_t coeffs[NEWHOPE_N];
} poly __attribute__((aligned(32)));
#endif // _WIN32

void poly_frombytes(poly *r,  unsigned char *a);
void poly_tobytes(unsigned char *r,  poly *p);
void poly_compress(unsigned char *r,  poly *p);
void poly_decompress(poly *r,  unsigned char *a);

void poly_frommsg(poly *r,  unsigned char *msg);
void poly_tomsg(unsigned char *msg,  poly *x);

__global__ void poly_tobytes_kernel(unsigned char* r, poly* p);
__global__ void poly_sub_kernel(poly* r, poly* a, poly* b);
__global__ void poly_mul_pointwise_kernel(poly* r, poly* a, poly* b);
__global__ void poly_add_kernel(poly* r, poly* a, poly* b);
__global__ void poly_sample_kernel_parallel(poly* r, unsigned char nonce);
__global__ void poly_uniform_kernel_parallel_sh_comb(poly* a);

#endif
