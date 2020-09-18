
// @Author: Arpan Jati
// Adapted from NewHope Reference Codebase and Parallelized using CUDA
// Updated : August 2019

#ifndef PARAMS_H
#define PARAMS_H

#ifndef NEWHOPE_N
#define NEWHOPE_N 1024
#endif

//#define ANALYSIS_MODE

#define N_TESTS 16384

#define BLOCK_SIZE 32

/* Don't change parameters below this line */

#define NEWHOPE_Q 12289 
#define NEWHOPE_K 8           /* used in noise sampling */

#define NEWHOPE_SYMBYTES (uint64_t)32   /* size of shared key, seeds/coins, and hashes */

#define NEWHOPE_POLYBYTES            ((uint64_t)(14*NEWHOPE_N)/8)
#define NEWHOPE_POLYCOMPRESSEDBYTES  ((uint64_t)( 3*NEWHOPE_N)/8)

#define NEWHOPE_CPAPKE_PUBLICKEYBYTES  (NEWHOPE_POLYBYTES + NEWHOPE_SYMBYTES)
#define NEWHOPE_CPAPKE_SECRETKEYBYTES  (NEWHOPE_POLYBYTES)
#define NEWHOPE_CPAPKE_CIPHERTEXTBYTES (NEWHOPE_POLYBYTES + NEWHOPE_POLYCOMPRESSEDBYTES)

#define NEWHOPE_CPAKEM_PUBLICKEYBYTES NEWHOPE_CPAPKE_PUBLICKEYBYTES
#define NEWHOPE_CPAKEM_SECRETKEYBYTES NEWHOPE_CPAPKE_SECRETKEYBYTES
#define NEWHOPE_CPAKEM_CIPHERTEXTBYTES NEWHOPE_CPAPKE_CIPHERTEXTBYTES

#define NEWHOPE_CCAKEM_PUBLICKEYBYTES NEWHOPE_CPAPKE_PUBLICKEYBYTES
#define NEWHOPE_CCAKEM_SECRETKEYBYTES (NEWHOPE_CPAPKE_SECRETKEYBYTES + NEWHOPE_CPAPKE_PUBLICKEYBYTES + (uint64_t)2*NEWHOPE_SYMBYTES)
#define NEWHOPE_CCAKEM_CIPHERTEXTBYTES (NEWHOPE_CPAPKE_CIPHERTEXTBYTES + NEWHOPE_SYMBYTES)  /* Second part is for Targhi-Unruh */

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
