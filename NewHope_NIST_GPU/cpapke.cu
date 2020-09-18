
// @Author: Arpan Jati
// Adapted from NewHope Reference Codebase and Parallelized using CUDA
// Updated : August 2019
// DEFINE 'ANALYSIS_MODE' to calculate detailed analysis results. 

#include <stdio.h>
#include "api.h"
#include "poly.h"
#include "rng.h"
#include "fips202.h"
#include "main.h"

#include <stdlib.h>
#include <float.h>

int BLK_SZ[10] = { 4, 8, 16, 24, 32, 48, 64, 128, 192, 256 };

int best_index = -1; float lowest_time = FLT_MAX;

int MP_COUNT = 9;

#ifdef ANALYSIS_MODE

int N_EXPTS = 100;

int COUNT_TO_EXPTS(int COUNT)
{
	if (COUNT >= 32768) return 10 * MP_COUNT;
	else if (COUNT >= 16384) return 20 * MP_COUNT;
	else if (COUNT >= 8192) return 40 * MP_COUNT;
	else if (COUNT >= 4096) return 80 * MP_COUNT;
	else if (COUNT >= 1024) return 160 * MP_COUNT;
	else if (COUNT >= 512) return 320 * MP_COUNT;
	else  return 400 * MP_COUNT;
}

#define TIMING_START(GP_1060_SZ,GP_P6000_SZ,GP_940MX_SZ,GP3_V100_SZ) \
 best_index = -1; lowest_time = FLT_MAX; \
    N_EXPTS = COUNT_TO_EXPTS(COUNT);\
for (int HH = 0; HH < 10; HH++)\
{\
	cudaEvent_t start, stop;\
	cudaEventCreate(&start);\
	cudaEventCreate(&stop);\
	int blockSize = BLK_SZ[HH];\
	int gridSize = (COUNT + blockSize - 1) / blockSize;\
	cudaEventRecord(start);\
	for (int KK = 0; KK < N_EXPTS; KK++)\
	{\

#define TIMING_END(NAME) \
	}\
	cudaEventRecord(stop);\
	cudaEventSynchronize(stop);\
	float milliseconds = 0; \
	cudaEventElapsedTime(&milliseconds, start, stop); \
	milliseconds /= N_EXPTS; \
	if( milliseconds < lowest_time)	{		\
		lowest_time = milliseconds;		\
		best_index = HH;	\
	} \
	printf("\n\n NAME %s | blockSize = %d | Time Elapsed: %f ms. K/s: %d ",NAME, blockSize, milliseconds, \
	(int)((((double)COUNT)* (double)1000.0) / (double)milliseconds)); \
	}\
	printf("\n\n ---BEST BLK SZ = %d | N_EXPTS = %d \n\n", BLK_SZ[best_index], N_EXPTS);

#else // Analysis mode

#define TIMING_START(GP_1060_SZ,GP_P6000_SZ,GP_940MX_SZ,GP3_V100_SZ)  \
	 blockSize = (SELECTED_GPU == GPU_G1060) ? GP_1060_SZ :  \
				 (SELECTED_GPU == GPU_P6000) ? GP_P6000_SZ : \
			     (SELECTED_GPU == GPU_940MX) ? GP_940MX_SZ:  \
											   GP3_V100_SZ;  \
	 gridSize = (COUNT + blockSize - 1) / blockSize;

#define TIMING_END(NAME)

#endif

__device__ void print_data_device(unsigned char* data, int length)
{
	int X = threadIdx.x + blockIdx.x * blockDim.x;
	if (X == 0)
	{
		printf("\nGPU DATA\n");

		for (int i = 0; i < length; i++)
		{
			printf("%02X", data[i]);

			if ((i + 1) % 2 == 0)
			{
				printf(" ");
			}

			if ((i + 1) % 32 == 0)
			{
				printf("\n");
			}
		}

		printf("\n");
	}
}

__global__ void print_poly_device(poly* poly)
{
	int X = threadIdx.x + blockIdx.x * blockDim.x;
	if (X == 0)
	{
		printf("\nGPU POLY\n");

		for (int i = 0; i < 1024; i++)
		{
			printf("%d ", poly->coeffs[i].threads[X]);

			if ((i + 1) % 16 == 0)
			{
				printf("\n");
			}
		}

		printf("\n");

	}
}

/*************************************************
* Name:        encode_pk
*
* Description: Serialize the public key as concatenation of the
*              serialization of the polynomial pk and the public seed
*              used to generete the polynomial a.
*
* Arguments:   unsigned char *r:          pointer to the output serialized public key
*              const poly *pk:            pointer to the input public-key polynomial
*              const unsigned char *seed: pointer to the input public seed
**************************************************/
__global__ void encode_pk(int COUNT, unsigned char* r,  poly* pk,
	 unsigned char* seed)
{
	int X = threadIdx.x + blockIdx.x * blockDim.x;
	if (X < COUNT)
	{
		int o_r = X * NEWHOPE_CPAPKE_PUBLICKEYBYTES;
		int o_seed = X * NEWHOPE_SYMBYTES;

		int i;
		poly_tobytes(r + o_r, pk);
		for (i = 0; i < NEWHOPE_SYMBYTES; i++)
		{
			(r + o_r)[NEWHOPE_POLYBYTES + i] = (seed + o_seed)[i];
		}
	}
}

/*************************************************
* Name:        decode_pk
*
* Description: De-serialize the public key; inverse of encode_pk
*
* Arguments:   poly *pk:               pointer to output public-key polynomial
*              unsigned char *seed:    pointer to output public seed
*              const unsigned char *r: pointer to input byte array
**************************************************/
__global__ void decode_pk(int COUNT, poly* pk, unsigned char* seed,
	 unsigned char* r)
{
	int X = threadIdx.x + blockIdx.x * blockDim.x;
	if (X < COUNT)
	{
		int o_r = X * NEWHOPE_CPAPKE_PUBLICKEYBYTES;
		int o_seed = X * NEWHOPE_SYMBYTES;

		poly_frombytes(pk, r + o_r);
		for (int i = 0; i < NEWHOPE_SYMBYTES; i++)
			(seed + o_seed)[i] = (r + o_r)[NEWHOPE_POLYBYTES + i];
	}
}

/*************************************************
* Name:        encode_c
*
* Description: Serialize the ciphertext as concatenation of the
*              serialization of the polynomial b and serialization
*              of the compressed polynomial v
*
* Arguments:   - unsigned char *r: pointer to the output serialized ciphertext
*              - const poly *b:    pointer to the input polynomial b
*              - const poly *v:    pointer to the input polynomial v
**************************************************/
__global__ void encode_c(int COUNT, unsigned char* r,
	 poly* b,  poly* v)
{
	int X = threadIdx.x + blockIdx.x * blockDim.x;
	if (X < COUNT)
	{
		int o_r = X * NEWHOPE_CPAPKE_CIPHERTEXTBYTES;

		poly_tobytes(r + o_r, b);
		poly_compress(r + o_r + NEWHOPE_POLYBYTES, v);
	}
}

/*************************************************
* Name:        decode_c
*
* Description: de-serialize the ciphertext; inverse of encode_c
*
* Arguments:   - poly *b:                pointer to output polynomial b
*              - poly *v:                pointer to output polynomial v
*              - const unsigned char *r: pointer to input byte array
**************************************************/
__global__ void decode_c(int COUNT, poly* b, poly* v,
	 unsigned char* r)
{
	int X = threadIdx.x + blockIdx.x * blockDim.x;
	if (X < COUNT)
	{
		int o_r = X * NEWHOPE_CPAPKE_CIPHERTEXTBYTES;

		poly_frombytes(b, r + o_r);
		poly_decompress(v, r + o_r + NEWHOPE_POLYBYTES);
	}
}

/*************************************************
* Name:        gen_a
*
* Description: Deterministically generate public polynomial a from seed
*
* Arguments:   - poly *a:                   pointer to output polynomial a
*              - const unsigned char *seed: pointer to input seed
**************************************************/
__global__ void gen_a(int COUNT, poly* a,
	 unsigned char* seed)
{
	int X = threadIdx.x + blockIdx.x * blockDim.x;
	if (X < COUNT)
	{
		int o_seed = X * NEWHOPE_SYMBYTES;

		poly_uniform(a, seed + o_seed);
		//poly_uniform_coal(a, seed + o_seed);
	}
}

int blockSize = BLOCK_SIZE;
int gridSize = (N_TESTS + blockSize - 1) / blockSize;

//int blockSize = 1;
//int gridSize = 1;// (N_TESTS + blockSize - 1) / blockSize;

/*************************************************
* Name:        cpapke_keypair
*
* Description: Generates public and private key
*              for the CPA public-key encryption scheme underlying
*              the NewHope KEMs
*
* Arguments:   - unsigned char *pk: pointer to output public key
*              - unsigned char *sk: pointer to output private key
**************************************************/
void cpapke_keypair(int COUNT, poly_set4* ps,
	unsigned char* pk,
	unsigned char* sk, 
	unsigned char* rng_buf, cudaStream_t stream)
{
	poly* ahat = ps->a;
	poly* ehat = ps->b;
	poly* ahat_shat = ps->c;
	poly* bhat = ps->d;
	poly* shat = ps->e;

	unsigned char* largeTemp = ps->large_buffer_a;

	unsigned char* temp_2 = ps->seed_2x;

	//printf("\n PARAM_N = %d \n", N_TESTS);
	//printf("\n BLOCK_SIZE = %d \n", BLOCK_SIZE);
	
	unsigned char* publicseed = temp_2;
	unsigned char* noiseseed = temp_2 + (NEWHOPE_SYMBYTES * N_TESTS);
	
	TIMING_START(256, 192, 32, 8)
	shake256_n << <gridSize, blockSize, 0, stream >> > (COUNT, temp_2, 2 * NEWHOPE_SYMBYTES, rng_buf, NEWHOPE_SYMBYTES, largeTemp);
	TIMING_END("shake256_n")

	TIMING_START(128, 32, 32, 128)
	gen_a << <gridSize, blockSize, 0, stream >> > (COUNT, ahat, publicseed);
	TIMING_END("gen_a")
	//gen_a_coal << <gridSize, blockSize, 0, stream >> > (ahat, publicseed, temp); -- > NOT WORKING

	//printf("\nAHAT:\n");
	//print_poly_device << <1, 1 >> > (ahat);
	TIMING_START(32, 128, 32, 128)
	poly_sample << <gridSize, blockSize, 0, stream >> > (COUNT, shat, noiseseed, 0);
	TIMING_END("poly_sample")

	TIMING_START(128, 256, 32, 256)
	poly_ntt << <gridSize, blockSize, 0, stream >> > (COUNT, shat);
	TIMING_END("poly_ntt")

	//printf("\nSHAT:\n");
	//print_poly_device << <1, 1 >> > (shat);
	TIMING_START(32, 128, 32, 128)
	poly_sample << <gridSize, blockSize, 0, stream >> > (COUNT, ehat, noiseseed, 1);
	TIMING_END("poly_sample")

	TIMING_START(128, 256, 32, 256)
	poly_ntt << <gridSize, blockSize, 0, stream >> > (COUNT, ehat);
	TIMING_END("poly_ntt")
	
	//printf("\nEHAT:\n");
	//print_poly_device << <1, 1 >> > (ehat);
	TIMING_START(128, 256, 32, 24)
	poly_mul_pointwise << <gridSize, blockSize, 0, stream >> > (COUNT, ahat_shat, shat, ahat);
	TIMING_END("poly_mul_pointwise")

	TIMING_START(128, 256, 32, 24)
	poly_add << <gridSize, blockSize, 0, stream >> > (COUNT, bhat, ehat, ahat_shat);
	TIMING_END("poly_add")

	TIMING_START(8, 8, 32, 16)
	poly_tobytes_n << <gridSize, blockSize, 0, stream >> > (COUNT, sk, shat);
	TIMING_END("poly_tobytes_n")

	//printf("\nSHAT:\n");
	//print_poly_device << <1, 1 >> > (shat);

	TIMING_START(8, 4, 32, 8)
	encode_pk << <gridSize, blockSize, 0, stream >> > (COUNT, pk, bhat, publicseed);
	TIMING_END("encode_pk")

	//printf("\nBHAT:\n");
	//print_poly_device << <1, 1 >> > (bhat);
}

/*************************************************
* Name:        cpapke_enc
*
* Description: Encryption function of
*              the CPA public-key encryption scheme underlying
*              the NewHope KEMs
*
* Arguments:   - unsigned char *c:          pointer to output ciphertext
*              - const unsigned char *m:    pointer to input message (of length NEWHOPE_SYMBYTES bytes)
*              - const unsigned char *pk:   pointer to input public key
*              - const unsigned char *coin: pointer to input random coins used as seed
*                                           to deterministically generate all randomness
**************************************************/
void cpapke_enc(int COUNT, poly_set4* ps,
	unsigned char* c,
	 unsigned char* m,
	 unsigned char* pk,
	 unsigned char* coin, cudaStream_t stream)
{
	poly* sprime = ps->a;
	poly* eprime = ps->b;
	poly* vprime = ps->c;
	poly* ahat = ps->d;
	poly* bhat = ps->e;
	poly* eprimeprime = ps->f;
	poly* uhat = ps->g;
	poly* v = ps->h;
	
	unsigned char* publicseed = ps->seed;

	TIMING_START(128, 128, 32, 256)
	poly_frommsg_n << <gridSize, blockSize, 0, stream >> > (COUNT, v, m);
	TIMING_END("poly_frommsg_n")

	TIMING_START(16, 8, 32, 32)
	decode_pk << <gridSize, blockSize, 0, stream >> > (COUNT, bhat, publicseed, pk);
	TIMING_END("decode_pk")

	TIMING_START(128, 32, 32, 128)
	gen_a << <gridSize, blockSize, 0, stream >> > (COUNT, ahat, publicseed);
	TIMING_END("gen_a")

	TIMING_START(32, 128, 32, 128)
	poly_sample << <gridSize, blockSize, 0, stream >> > (COUNT, sprime, coin, 0);
	TIMING_END("poly_sample")

	TIMING_START(32, 128, 32, 128)
	poly_sample << <gridSize, blockSize, 0, stream >> > (COUNT, eprime, coin, 1);
	TIMING_END("poly_sample")

	TIMING_START(32, 128, 32, 128)
	poly_sample << <gridSize, blockSize, 0, stream >> > (COUNT, eprimeprime, coin, 2);
	TIMING_END("poly_sample")

	TIMING_START(128, 256, 32, 256)
	poly_ntt << <gridSize, blockSize, 0, stream >> > (COUNT, sprime);
	TIMING_END("poly_ntt")

	TIMING_START(128, 256, 32, 256)
	poly_ntt << <gridSize, blockSize, 0, stream >> > (COUNT, eprime);
	TIMING_END("poly_ntt")

	TIMING_START(128, 256, 32, 24)
	poly_mul_pointwise << <gridSize, blockSize, 0, stream >> > (COUNT, uhat, ahat, sprime);
	TIMING_END("poly_mul_pointwise")

	TIMING_START(128, 256, 32, 24)
	poly_add << <gridSize, blockSize, 0, stream >> > (COUNT, uhat, uhat, eprime);
	TIMING_END("poly_add")

	TIMING_START(128, 256, 32, 24)
	poly_mul_pointwise << <gridSize, blockSize, 0, stream >> > (COUNT, vprime, bhat, sprime);
	TIMING_END("poly_mul_pointwise")

	TIMING_START(64, 256, 32, 256)
	poly_invntt << <gridSize, blockSize, 0, stream >> > (COUNT, vprime);
	TIMING_END("poly_invntt")

	TIMING_START(128, 256, 32, 24)
	poly_add << <gridSize, blockSize, 0, stream >> > (COUNT, vprime, vprime, eprimeprime);
	TIMING_END("poly_add")

	TIMING_START(128, 256, 32, 24)
	poly_add << <gridSize, blockSize, 0, stream >> > (COUNT, vprime, vprime, v); // add message
	TIMING_END("poly_add")

	TIMING_START(16, 4, 32, 8)
	encode_c << <gridSize, blockSize, 0, stream >> > (COUNT, c, uhat, vprime);
	TIMING_END("encode_c")
}


/*************************************************
* Name:        cpapke_dec
*
* Description: Decryption function of
*              the CPA public-key encryption scheme underlying
*              the NewHope KEMs
*
* Arguments:   - unsigned char *m:        pointer to output decrypted message
*              - const unsigned char *c:  pointer to input ciphertext
*              - const unsigned char *sk: pointer to input secret key
**************************************************/
void cpapke_dec(int COUNT, poly_set4* ps,
	unsigned char* m,
	 unsigned char* c,
	 unsigned char* sk, cudaStream_t stream)
{
	poly* vprime = ps->a;
	poly* uhat = ps->b;
	poly* tmp = ps->c;
	poly* shat = ps->d;

	TIMING_START(16, 8, 32, 32)
	poly_frombytes_n << <gridSize, blockSize, 0, stream >> > (COUNT, shat, sk);
	TIMING_END("poly_frombytes_n")

	TIMING_START(16, 4, 32, 8)
	decode_c << <gridSize, blockSize, 0, stream >> > (COUNT, uhat, vprime, c);
	TIMING_END("decode_c")

	TIMING_START(128, 256, 32, 24)
	poly_mul_pointwise << <gridSize, blockSize, 0, stream >> > (COUNT, tmp, shat, uhat);
	TIMING_END("poly_mul_pointwise")

	TIMING_START(64, 256, 32, 256)
	poly_invntt << <gridSize, blockSize, 0, stream >> > (COUNT, tmp);
	TIMING_END("poly_invntt")

	TIMING_START(256, 256, 32, 32)
	poly_sub << <gridSize, blockSize, 0, stream >> > (COUNT, tmp, tmp, vprime);
	TIMING_END("poly_sub")

	TIMING_START(128, 128, 32, 32)
	poly_tomsg_n << <gridSize, blockSize, 0, stream >> > (COUNT, m, tmp);
	TIMING_END("poly_tomsg_n")
}
