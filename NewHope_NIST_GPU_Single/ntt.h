
// @Author: Naina Gupta
// Adapted from NewHope Reference Codebase and Parallelized using CUDA
// Updated : August 2019

#ifndef NTT_H
#define NTT_H

#include "inttypes.h"
#include "params.h"

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

extern uint16_t omegas_inv_bitrev_montgomery[];
extern uint16_t gammas_bitrev_montgomery[];
extern uint16_t gammas_inv_montgomery[];

void bitrev_vector(uint16_t* poly);
void mul_coefficients(uint16_t* poly,  uint16_t* factors);
void ntt(uint16_t* poly,  uint16_t* omegas);

__global__ void mul_coefficients_invntt_kernel(uint16_t* poly);
__global__ void mul_coefficients_ntt_kernel(uint16_t* poly);
__global__ void bitrev_vector_kernel(uint16_t* poly);
__global__  void ntt_kernel_parallel_shared(uint16_t * a);
__global__  void invntt_kernel_parallel_shared(uint16_t * a);

#endif
