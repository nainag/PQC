
// @Author: Arpan Jati
// Adapted from NewHope Reference Codebase and Parallelized using CUDA
// Updated : August 2019

#ifndef NTT_H
#define NTT_H

#include "inttypes.h"
#include "poly.h"

__device__ extern uint16_t omegas_inv_bitrev_montgomery[];
__device__ extern uint16_t gammas_bitrev_montgomery[];
__device__ extern uint16_t gammas_inv_montgomery[];

__device__ void bitrev_vector(poly* poly);
__device__ void mul_coefficients(poly* poly, const uint16_t* factors);
__device__ void ntt(poly* poly, const uint16_t* omegas);

#endif
