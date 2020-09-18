
// @Author: Arpan Jati
// Adapted from NewHope Reference Codebase and Parallelized using CUDA
// Updated : August 2019

#ifndef NTT_H
#define NTT_H

#include "poly.h"
#include <stdint.h>

__device__ extern int16_t zetas[128];
__device__ extern int16_t zetasinv[128];

//void ntt(int16_t *poly);
//void invntt(int16_t *poly);

__device__ void ntt_p(poly* r);
__device__ void invntt_p(poly* r);

//__device__ void basemul(int16_t r[2], int16_t a[2],  int16_t b[2], int16_t zeta);

__device__ void basemul2(threads r[2], threads a[2], threads b[2], int16_t zeta);

#endif
