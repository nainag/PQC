
// @Author: Arpan Jati
// Adapted from NewHope Reference Codebase and Parallelized using CUDA
// Updated : August 2019

#ifndef INDCPA_H
#define INDCPA_H

#include "main.h"

__device__ void print_data_device(unsigned char* data, int length);
__global__ void print_poly_device(poly* poly);

void cpapke_keypair(int COUNT, poly_set4* ps, unsigned char* pk,
	unsigned char* sk, unsigned char* rng_buf, cudaStream_t stream);

void cpapke_enc(int COUNT, poly_set4* ps, unsigned char* c,
	 unsigned char* m,
	 unsigned char* pk,
	 unsigned char* coin, cudaStream_t stream);

void cpapke_dec(int COUNT, poly_set4* ps, unsigned char* m,
	 unsigned char* c,
	 unsigned char* sk, cudaStream_t stream);

#endif
