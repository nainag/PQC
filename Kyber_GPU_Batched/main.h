
// @Author: Arpan Jati
// Adapted from NewHope Reference Codebase and Parallelized using CUDA
// Updated : August 2019

#ifndef MAIN_H
#define MAIN_H

#include "poly.h"
#include "polyvec.h"

#define LARGE_BUFFER_SZ 1024

typedef struct _poly_set4
{
	// Length 1
	poly* a;
	poly* b;
	poly* c;
	poly* d;

	// Length 4
	polyvec* AV;

	// Length 1
	polyvec* av;
	polyvec* bv;
	polyvec* cv;
	polyvec* dv;
	polyvec* ev;
	polyvec* fv;

	// Length [2 * KYBER_SYMBYTES]
	unsigned char* seed;
	
	// LARGE_BUFFER_SZ = Length in bytes
	unsigned char* large_buffer_a;
	unsigned char* large_buffer_b;
} poly_set4;


#endif