
// @Author: Arpan Jati
// Adapted from NewHope Reference Codebase and Parallelized using CUDA
// Updated : August 2019

#ifndef POLYVEC_H
#define POLYVEC_H

#include "params.h"
#include "poly.h"

typedef struct {
	poly vec[KYBER_K];
} polyvec;

__device__ void polyvec_compress(unsigned char* r, polyvec* a);
__device__ void polyvec_decompress(polyvec* r,  unsigned char* a);

__device__ void polyvec_tobytes(unsigned char* r, polyvec* a);
__device__ void polyvec_frombytes(polyvec* r,  unsigned char* a);

__global__ void polyvec_ntt_n(int COUNT, polyvec* r);
__global__ void polyvec_invntt_n(int COUNT, polyvec* r);

__global__ void polyvec_pointwise_acc_n(int COUNT, poly* r,  polyvec* a,  polyvec* b, poly* temp);

__global__ void polyvec_reduce_n(int COUNT, polyvec* r);
__device__ void polyvec_reduce(polyvec* r);

__device__ void polyvec_csubq(polyvec* r);

__global__ void polyvec_add_n(int COUNT, polyvec* r,  polyvec* a,  polyvec* b);
__device__ void polyvec_add(polyvec* r,  polyvec* a,  polyvec* b);

#endif
