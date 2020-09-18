
// @Author: Arpan Jati
// Adapted from NewHope Reference Codebase and Parallelized using CUDA
// Updated : August 2019

#ifndef MAIN_H
#define MAIN_H

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "poly.h"

#define LARGE_BUFFER_SZ 1024

void HandleError(cudaError_t err, const char* file, int line);

#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ ))

typedef struct _poly_set4
{
	// Length 1
	poly* a;
	poly* b;
	poly* c;
	poly* d;
	poly* e;
	poly* f;
	poly* g;
	poly* h;

	// Length [2 * KYBER_SYMBYTES]
	unsigned char* seed;

	unsigned char* seed_2x;

	// LARGE_BUFFER_SZ = Length in bytes
	unsigned char* large_buffer_a;
	//unsigned char* large_buffer_b;
} poly_set4;


#endif