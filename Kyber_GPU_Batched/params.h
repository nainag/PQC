
// @Author: Arpan Jati
// Adapted from NewHope Reference Codebase and Parallelized using CUDA
// Updated : August 2019

#ifndef PARAMS_H
#define PARAMS_H

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#ifndef KYBER_K
#define KYBER_K 4 /* Change this for different security strengths */
#endif

//#define ANALYSIS_MODE

#define N_TESTS 16384

#define BLOCK_SIZE 32

/* Don't change parameters below this line */

#define KYBER_N 256
#define KYBER_Q 3329

#define KYBER_ETA 2

#define KYBER_SYMBYTES 32   /* size in bytes of hashes, and seeds */
#define KYBER_SSBYTES  32   /* size in bytes of shared key */

#define KYBER_POLYBYTES              384
#define KYBER_POLYVECBYTES           (KYBER_K * KYBER_POLYBYTES)

#define KYBER_POLYCOMPRESSEDBYTES    160
#define KYBER_POLYVECCOMPRESSEDBYTES (KYBER_K * 352)

#define KYBER_INDCPA_MSGBYTES       KYBER_SYMBYTES
#define KYBER_INDCPA_PUBLICKEYBYTES (KYBER_POLYVECBYTES + KYBER_SYMBYTES)
#define KYBER_INDCPA_SECRETKEYBYTES (KYBER_POLYVECBYTES)
#define KYBER_INDCPA_BYTES          (KYBER_POLYVECCOMPRESSEDBYTES + KYBER_POLYCOMPRESSEDBYTES)

#define KYBER_PUBLICKEYBYTES  (KYBER_INDCPA_PUBLICKEYBYTES)
#define KYBER_SECRETKEYBYTES  (KYBER_INDCPA_SECRETKEYBYTES +  KYBER_INDCPA_PUBLICKEYBYTES + 2*KYBER_SYMBYTES) /* 32 bytes of additional space to save H(pk) */
#define KYBER_CIPHERTEXTBYTES  KYBER_INDCPA_BYTES

///////////////////

extern int SELECTED_GPU;
extern int MP_COUNT;

#define GPU_G1060  0
#define GPU_P6000  1
#define GPU_940MX  2
#define GPU_V100   3

#define GPU_G1060_N   "GeForce GTX 1060 3GB"
#define GPU_P6000_N   "Quadro P6000"
#define GPU_940MX_N   "GeForce 940MX"
#define GPU_V100_N    "GeForce V100"

//#define SELECTED_GPU GPU_G1060

#define SELECTED_GPU_NAME ( (SELECTED_GPU == GPU_G1060) ? GPU_G1060_N :  \
							(SELECTED_GPU == GPU_P6000) ? GPU_P6000_N :  \
							(SELECTED_GPU == GPU_940MX) ? GPU_940MX_N :  \
														  GPU_V100_N )

#endif
