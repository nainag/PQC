
// @Author: Arpan Jati
// Adapted from NewHope Reference Codebase and Parallelized using CUDA
// Updated : August 2019

#include <stdint.h>
#include "params.h"
#include "poly.h"
#include "ntt.h"
#include "reduce.h"
#include "cbd.h"
#include "symmetric.h"

/*************************************************
* Name:        poly_compress
*
* Description: Compression and subsequent serialization of a polynomial
*
* Arguments:   - unsigned char *r: pointer to output byte array (needs space for KYBER_POLYCOMPRESSEDBYTES bytes)
*              -  poly *a:    pointer to input polynomial
**************************************************/
__device__ void poly_compress(unsigned char* r, poly* a)
{
	uint8_t t[8];
	int i, j, k = 0;

	int X = threadIdx.x + blockIdx.x * blockDim.x;

	poly_csubq(a);

	for (i = 0; i < KYBER_N; i += 8)
	{
		for (j = 0; j < 8; j++)
			t[j] = ((((uint32_t)a->coeffs[i + j].threads[X] << 5) + KYBER_Q / 2) / KYBER_Q) & 31;

		r[k] = t[0] | (t[1] << 5);
		r[k + 1] = (t[1] >> 3) | (t[2] << 2) | (t[3] << 7);
		r[k + 2] = (t[3] >> 1) | (t[4] << 4);
		r[k + 3] = (t[4] >> 4) | (t[5] << 1) | (t[6] << 6);
		r[k + 4] = (t[6] >> 2) | (t[7] << 3);
		k += 5;
	}
}

/*************************************************
* Name:        poly_decompress
*
* Description: De-serialization and subsequent decompression of a polynomial;
*              approximate inverse of poly_compress
*
* Arguments:   - poly *r:                pointer to output polynomial
*              -  unsigned char *a: pointer to input byte array (of length KYBER_POLYCOMPRESSEDBYTES bytes)
**************************************************/
__device__ void poly_decompress(poly* r, unsigned char* a)
{
	int X = threadIdx.x + blockIdx.x * blockDim.x;
	for (int i = 0; i < KYBER_N; i += 8)
	{
		r->coeffs[i + 0].threads[X] = (((a[0] & 31) * KYBER_Q) + 16) >> 5;
		r->coeffs[i + 1].threads[X] = ((((a[0] >> 5) | ((a[1] & 3) << 3)) * KYBER_Q) + 16) >> 5;
		r->coeffs[i + 2].threads[X] = ((((a[1] >> 2) & 31) * KYBER_Q) + 16) >> 5;
		r->coeffs[i + 3].threads[X] = ((((a[1] >> 7) | ((a[2] & 15) << 1)) * KYBER_Q) + 16) >> 5;
		r->coeffs[i + 4].threads[X] = ((((a[2] >> 4) | ((a[3] & 1) << 4)) * KYBER_Q) + 16) >> 5;
		r->coeffs[i + 5].threads[X] = ((((a[3] >> 1) & 31) * KYBER_Q) + 16) >> 5;
		r->coeffs[i + 6].threads[X] = ((((a[3] >> 6) | ((a[4] & 7) << 2)) * KYBER_Q) + 16) >> 5;
		r->coeffs[i + 7].threads[X] = (((a[4] >> 3) * KYBER_Q) + 16) >> 5;
		a += 5;
	}
}

/*************************************************
* Name:        poly_tobytes
*
* Description: Serialization of a polynomial
*
* Arguments:   - unsigned char *r: pointer to output byte array (needs space for KYBER_POLYBYTES bytes)
*              -  poly *a:    pointer to input polynomial
**************************************************/
__device__ void poly_tobytes(unsigned char* r, poly* a)
{
	int X = threadIdx.x + blockIdx.x * blockDim.x;

	int i;
	uint16_t t0, t1;

	poly_csubq(a);

	for (i = 0; i < KYBER_N / 2; i++) {
		t0 = a->coeffs[2 * i].threads[X];
		t1 = a->coeffs[2 * i + 1].threads[X];
		r[3 * i] = t0 & 0xff;
		r[3 * i + 1] = (t0 >> 8) | ((t1 & 0xf) << 4);
		r[3 * i + 2] = t1 >> 4;
	}
}

/*************************************************
* Name:        poly_frombytes
*
* Description: De-serialization of a polynomial;
*              inverse of poly_tobytes
*
* Arguments:   - poly *r:                pointer to output polynomial
*              -  unsigned char *a: pointer to input byte array (of KYBER_POLYBYTES bytes)
**************************************************/
__device__ void poly_frombytes(poly* r, unsigned char* a)
{
	int i;

	int X = threadIdx.x + blockIdx.x * blockDim.x;

	for (i = 0; i < KYBER_N / 2; i++) {
		r->coeffs[2 * i].threads[X] = a[3 * i] | ((uint16_t)a[3 * i + 1] & 0x0f) << 8;
		r->coeffs[2 * i + 1].threads[X] = a[3 * i + 1] >> 4 | ((uint16_t)a[3 * i + 2] & 0xff) << 4;
	}
}

/*************************************************
* Name:        poly_getnoise
*
* Description: Sample a polynomial deterministically from a seed and a nonce,
*              with output polynomial close to centered binomial distribution
*              with parameter KYBER_ETA
*
* Arguments:   - poly *r:                   pointer to output polynomial
*              -  unsigned char *seed: pointer to input seed (pointing to array of length KYBER_SYMBYTES bytes)
*              - unsigned char nonce:       one-byte input nonce
**************************************************/
__global__ void poly_getnoise(int COUNT, poly* r, unsigned char* seed, unsigned char nonce)
{
	int X = threadIdx.x + blockIdx.x * blockDim.x;
	if (X < COUNT)
	{
		unsigned char buf[KYBER_ETA * KYBER_N / 4];

		//int buf_o = (KYBER_ETA * KYBER_N / 4) * X;
		int seed_o = X * KYBER_SYMBYTES;

		shake256_prf(buf, KYBER_ETA * KYBER_N / 4, seed + seed_o, nonce);
		cbd(r, buf);
	}
}

/*************************************************
* Name:        poly_ntt
*
* Description: Computes negacyclic number-theoretic transform (NTT) of
*              a polynomial in place;
*              inputs assumed to be in normal order, output in bitreversed order
*
* Arguments:   - uint16_t *r: pointer to in/output polynomial
**************************************************/
__device__ void poly_ntt(poly* r)
{
	ntt_p(r);
	poly_reduce(r);
}

__global__ void poly_ntt_n(int COUNT, poly* r)
{
	int X = threadIdx.x + blockIdx.x * blockDim.x;
	if (X < COUNT)
	{
		ntt_p(r);
		poly_reduce(r);
	}
}

/*************************************************
* Name:        poly_invntt
*
* Description: Computes inverse of negacyclic number-theoretic transform (NTT) of
*              a polynomial in place;
*              inputs assumed to be in bitreversed order, output in normal order
*
* Arguments:   - uint16_t *a: pointer to in/output polynomial
**************************************************/
__device__ void poly_invntt(poly* r)
{
	invntt_p(r);
}

__global__ void poly_invntt_n(int COUNT, poly* r)
{
	int X = threadIdx.x + blockIdx.x * blockDim.x;
	if (X < COUNT)
	{
		invntt_p(r);
	}
}




/*************************************************
* Name:        poly_basemul
*
* Description: Multiplication of two polynomials in NTT domain
*
* Arguments:   - poly *r:       pointer to output polynomial
*              -  poly *a: pointer to first input polynomial
*              -  poly *b: pointer to second input polynomial
**************************************************/
__device__ void poly_basemul(poly* r, poly* a, poly* b)
{
	//int X = threadIdx.x + blockIdx.x * blockDim.x;

	uint32_t addrSh_1 = 0, addrSh_2 = 0;

	for (int i = 0; i < KYBER_N / 4; ++i)
	{
		addrSh_1 = (4 * i);
		addrSh_2 = (4 * i + 2);

		basemul2(r->coeffs + addrSh_1,
			a->coeffs + addrSh_1,
			b->coeffs + addrSh_1,
			zetas[64 + i]);

		basemul2(r->coeffs + addrSh_2,
			a->coeffs + addrSh_2,
			b->coeffs + addrSh_2,
			-zetas[64 + i]);
	}
}

/*************************************************
* Name:        poly_frommont
*
* Description: Inplace conversion of all coefficients of a polynomial
*              from Montgomery domain to normal domain
*
* Arguments:   - poly *r:       pointer to input/output polynomial
**************************************************/
__global__ void poly_frommont_n(int COUNT, poly* r)
{
	int X = threadIdx.x + blockIdx.x * blockDim.x;
	if (X < COUNT)
	{
		int16_t f = (1ULL << 32) % KYBER_Q;

		for (int i = 0; i < KYBER_N; i++) {
			r->coeffs[i].threads[X] =
				montgomery_reduce((int32_t)r->coeffs[i].threads[X] * f);
		}
	}
}

/*************************************************
* Name:        poly_reduce
*
* Description: Applies Barrett reduction to all coefficients of a polynomial
*              for details of the Barrett reduction see comments in reduce.c
*
* Arguments:   - poly *r:       pointer to input/output polynomial
**************************************************/
__global__ void poly_reduce_n(int COUNT, poly* r)
{
	int X = threadIdx.x + blockIdx.x * blockDim.x;
	if (X < COUNT)
	{
		for (int i = 0; i < KYBER_N; i++)
		{
			r->coeffs[i].threads[X] = barrett_reduce(r->coeffs[i].threads[X]);
		}
	}
}

__device__  void poly_reduce(poly* r)
{
	int X = threadIdx.x + blockIdx.x * blockDim.x;

	for (int i = 0; i < KYBER_N; i++)
	{
		r->coeffs[i].threads[X] = barrett_reduce(r->coeffs[i].threads[X]);
	}
}

/*************************************************
* Name:        poly_csubq
*
* Description: Applies conditional subtraction of q to each coefficient of a polynomial
*              for details of conditional subtraction of q see comments in reduce.c
*
* Arguments:   - poly *r:       pointer to input/output polynomial
**************************************************/
__device__ void poly_csubq(poly* r)
{
	int X = threadIdx.x + blockIdx.x * blockDim.x;

	for (int i = 0; i < KYBER_N; i++)
	{
		r->coeffs[i].threads[X] = csubq(r->coeffs[i].threads[X]);
	}
}

/*************************************************
* Name:        poly_add
*
* Description: Add two polynomials
*
* Arguments: - poly *r:       pointer to output polynomial
*            -  poly *a: pointer to first input polynomial
*            -  poly *b: pointer to second input polynomial
**************************************************/
__device__ void poly_add(poly* r, poly* a, poly* b)
{
	int X = threadIdx.x + blockIdx.x * blockDim.x;

	for (int i = 0; i < KYBER_N; i++)
	{
		r->coeffs[i].threads[X] = a->coeffs[i].threads[X] + b->coeffs[i].threads[X];
	}
}

__global__  void poly_add_n(int COUNT, poly* r, poly* a, poly* b)
{
	int X = threadIdx.x + blockIdx.x * blockDim.x;
	if (X < COUNT)
	{
		for (int i = 0; i < KYBER_N; i++)
		{
			r->coeffs[i].threads[X] = a->coeffs[i].threads[X] + b->coeffs[i].threads[X];
		}
	}
}

/*************************************************
* Name:        poly_sub
*
* Description: Subtract two polynomials
*
* Arguments: - poly *r:       pointer to output polynomial
*            -  poly *a: pointer to first input polynomial
*            -  poly *b: pointer to second input polynomial
**************************************************/
__global__  void poly_sub_n(int COUNT, poly* r, poly* a, poly* b)
{
	int X = threadIdx.x + blockIdx.x * blockDim.x;
	if (X < COUNT)
	{
		for (int i = 0; i < KYBER_N; i++)
		{
			r->coeffs[i].threads[X] = a->coeffs[i].threads[X] - b->coeffs[i].threads[X];
		}
	}
}

__device__ void poly_sub(poly* r, poly* a, poly* b)
{
	int X = threadIdx.x + blockIdx.x * blockDim.x;

	for (int i = 0; i < KYBER_N; i++)
	{
		r->coeffs[i].threads[X] = a->coeffs[i].threads[X] - b->coeffs[i].threads[X];
	}
}

/*************************************************
* Name:        poly_frommsg
*
* Description: Convert 32-byte message to polynomial
*
* Arguments:   - poly *r:                  pointer to output polynomial
*              -  unsigned char *msg: pointer to input message
**************************************************/
__global__  void poly_frommsg_n(int COUNT, poly* r, unsigned char* msg)
{
	int X = threadIdx.x + blockIdx.x * blockDim.x;
	if (X < COUNT)
	{
		uint16_t mask;

		int o_msg = X * KYBER_SYMBYTES;

		for (int i = 0; i < KYBER_SYMBYTES; i++)
		{
			for (int j = 0; j < 8; j++)
			{
				mask = -(((msg + o_msg)[i] >> j) & 1);
				r->coeffs[8 * i + j].threads[X] = mask & ((KYBER_Q + 1) / 2);
			}
		}

	}
}

/*************************************************
* Name:        poly_tomsg
*
* Description: Convert polynomial to 32-byte message
*
* Arguments:   - unsigned char *msg: pointer to output message
*              -  poly *a:      pointer to input polynomial
**************************************************/
__global__  void poly_tomsg_n(int COUNT, unsigned char* msg, poly* a)
{
	int X = threadIdx.x + blockIdx.x * blockDim.x;
	if (X < COUNT)
	{
		int o_msg = X * KYBER_SYMBYTES;

		uint16_t t;

		poly_csubq(a);

		for (int i = 0; i < KYBER_SYMBYTES; i++)
		{
			(msg + o_msg)[i] = 0;
			for (int j = 0; j < 8; j++)
			{
				t = (((a->coeffs[8 * i + j].threads[X] << 1) + KYBER_Q / 2) / KYBER_Q) & 1;
				(msg + o_msg)[i] |= t << j;
			}
		}
	}
}



/*************************************************
* Name:        poly_frommsg
*
* Description: Convert 32-byte message to polynomial
*
* Arguments:   - poly *r:                  pointer to output polynomial
*              -  unsigned char *msg: pointer to input message
**************************************************/
__device__ void poly_frommsg(poly* r, unsigned char* msg)
{
	int X = threadIdx.x + blockIdx.x * blockDim.x;

	uint16_t mask;

	int o_msg = X * KYBER_SYMBYTES;

	for (int i = 0; i < KYBER_SYMBYTES; i++)
	{
		for (int j = 0; j < 8; j++)
		{
			mask = -(((msg + o_msg)[i] >> j) & 1);
			r->coeffs[8 * i + j].threads[X] = mask & ((KYBER_Q + 1) / 2);
		}
	}
}

/*************************************************
* Name:        poly_tomsg
*
* Description: Convert polynomial to 32-byte message
*
* Arguments:   - unsigned char *msg: pointer to output message
*              -  poly *a:      pointer to input polynomial
**************************************************/
__device__ void poly_tomsg(unsigned char* msg, poly* a)
{
	int X = threadIdx.x + blockIdx.x * blockDim.x;

	int o_msg = X * KYBER_SYMBYTES;

	uint16_t t;

	poly_csubq(a);

	for (int i = 0; i < KYBER_SYMBYTES; i++)
	{
		(msg + o_msg)[i] = 0;
		for (int j = 0; j < 8; j++)
		{
			t = (((a->coeffs[8 * i + j].threads[X] << 1) + KYBER_Q / 2) / KYBER_Q) & 1;
			(msg + o_msg)[i] |= t << j;
		}
	}
}
