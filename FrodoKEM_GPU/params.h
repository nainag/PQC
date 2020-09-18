
// @Author: Naina Gupta
// Adapted from FrodoKEM Reference Codebase and Parallelized using CUDA
// Updated : August 2019

#ifndef _PARAMS_H_
#define _PARAMS_H_

#include <stdint.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define USE_SHAKE128_FOR_A

// Parameters for "FrodoKEM-976"
#define PARAMS_N 976
#define PARAMS_NBAR 8
#define PARAMS_LOGQ 16
#define PARAMS_Q (1 << PARAMS_LOGQ)
#define PARAMS_EXTRACTED_BITS 3
#define PARAMS_STRIPE_STEP 8
#define PARAMS_PARALLEL 4
#define BYTES_SEED_A 16
#define BYTES_MU (PARAMS_EXTRACTED_BITS*PARAMS_NBAR*PARAMS_NBAR)/8
#define BYTES_PKHASH CRYPTO_BYTES

// Selecting SHAKE XOF function for the KEM and noise sampling
#define shake     shake256
#define THR_PER_BLK 16

#endif
