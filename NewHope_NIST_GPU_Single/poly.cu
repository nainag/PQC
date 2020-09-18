
// @Author: Naina Gupta
// Adapted from NewHope Reference Codebase and Parallelized using CUDA
// Updated : August 2019

#include "poly.h"
#include "ntt.h"
#include "reduce.h"
#include "fips202.h"

/*************************************************
* Name:        coeff_freeze
*
* Description: Fully reduces an integer modulo q in constant time
*
* Arguments:   uint16_t x: input integer to be reduced
*
* Returns integer in {0,...,q-1} congruent to x modulo q
**************************************************/
uint16_t coeff_freeze(uint16_t x)
{
	uint16_t m, r;
	int16_t c;
	r = x % NEWHOPE_Q;

	m = r - NEWHOPE_Q;
	c = m;
	c >>= 15;
	r = m ^ ((r ^ m) & c);

	return r;
}

__device__ uint16_t coeff_freeze_device(uint16_t x)
{
	uint16_t m, r;
	int16_t c;
	r = x % NEWHOPE_Q;

	m = r - NEWHOPE_Q;
	c = m;
	c >>= 15;
	r = m ^ ((r ^ m) & c);

	return r;
}


/*************************************************
* Name:        flipabs
*
* Description: Computes |(x mod q) - Q/2|
*
* Arguments:   uint16_t x: input coefficient
*
* Returns |(x mod q) - Q/2|
**************************************************/
uint16_t flipabs(uint16_t x)
{
	int16_t r, m;
	r = coeff_freeze(x);

	r = r - NEWHOPE_Q / 2;
	m = r >> 15;
	return (r + m) ^ m;
}

/*************************************************
* Name:        poly_frombytes
*
* Description: De-serialization of a polynomial
*
* Arguments:   - poly *r:                pointer to output polynomial
*              -  unsigned char *a: pointer to input byte array
**************************************************/
void poly_frombytes(poly* r, unsigned char* a)
{
	int i;
	for (i = 0; i < NEWHOPE_N / 4; i++)
	{
		r->coeffs[4 * i + 0] = a[7 * i + 0] | (((uint16_t)a[7 * i + 1] & 0x3f) << 8);
		r->coeffs[4 * i + 1] = (a[7 * i + 1] >> 6) | (((uint16_t)a[7 * i + 2]) << 2) | (((uint16_t)a[7 * i + 3] & 0x0f) << 10);
		r->coeffs[4 * i + 2] = (a[7 * i + 3] >> 4) | (((uint16_t)a[7 * i + 4]) << 4) | (((uint16_t)a[7 * i + 5] & 0x03) << 12);
		r->coeffs[4 * i + 3] = (a[7 * i + 5] >> 2) | (((uint16_t)a[7 * i + 6]) << 6);
	}
}

/*************************************************
* Name:        poly_tobytes
*
* Description: Serialization of a polynomial
*
* Arguments:   - unsigned char *r: pointer to output byte array
*              -  poly *p:    pointer to input polynomial
**************************************************/
void poly_tobytes(unsigned char* r, poly* p)
{
	int i;
	uint16_t t0, t1, t2, t3;
	for (i = 0; i < NEWHOPE_N / 4; i++)
	{
		t0 = coeff_freeze(p->coeffs[4 * i + 0]);
		t1 = coeff_freeze(p->coeffs[4 * i + 1]);
		t2 = coeff_freeze(p->coeffs[4 * i + 2]);
		t3 = coeff_freeze(p->coeffs[4 * i + 3]);

		r[7 * i + 0] = t0 & 0xff;
		r[7 * i + 1] = (t0 >> 8) | (t1 << 6);
		r[7 * i + 2] = (t1 >> 2);
		r[7 * i + 3] = (t1 >> 10) | (t2 << 4);
		r[7 * i + 4] = (t2 >> 4);
		r[7 * i + 5] = (t2 >> 12) | (t3 << 2);
		r[7 * i + 6] = (t3 >> 6);
	}
}

__global__ void poly_tobytes_kernel(unsigned char* r, poly* p)
{
	int i;
	uint16_t t0, t1, t2, t3;
	int threadId = blockIdx.x * blockDim.x + threadIdx.x;

	if (threadId < NEWHOPE_N / 64)  // 16 threads
	{
		t0 = coeff_freeze_device(p->coeffs[4 * threadId + 0]);
		t1 = coeff_freeze_device(p->coeffs[4 * threadId + 1]);
		t2 = coeff_freeze_device(p->coeffs[4 * threadId + 2]);
		t3 = coeff_freeze_device(p->coeffs[4 * threadId + 3]);

		r[7 * threadId + 0] = t0 & 0xff;
		r[7 * threadId + 1] = (t0 >> 8) | (t1 << 6);
		r[7 * threadId + 2] = (t1 >> 2);
		r[7 * threadId + 3] = (t1 >> 10) | (t2 << 4);
		r[7 * threadId + 4] = (t2 >> 4);
		r[7 * threadId + 5] = (t2 >> 12) | (t3 << 2);
		r[7 * threadId + 6] = (t3 >> 6);
	}
}


/*************************************************
* Name:        poly_compress
*
* Description: Compression and subsequent serialization of a polynomial
*
* Arguments:   - unsigned char *r: pointer to output byte array
*              -  poly *p:    pointer to input polynomial
**************************************************/
void poly_compress(unsigned char* r, poly* p)
{
	unsigned int i, j, k = 0;

	uint32_t t[8];

	for (i = 0; i < NEWHOPE_N; i += 8)
	{
		for (j = 0; j < 8; j++)
		{
			t[j] = coeff_freeze(p->coeffs[i + j]);
			t[j] = (((t[j] << 3) + NEWHOPE_Q / 2) / NEWHOPE_Q) & 0x7;
		}

		r[k] = t[0] | (t[1] << 3) | (t[2] << 6);
		r[k + 1] = (t[2] >> 2) | (t[3] << 1) | (t[4] << 4) | (t[5] << 7);
		r[k + 2] = (t[5] >> 1) | (t[6] << 2) | (t[7] << 5);
		k += 3;
	}
}

/*************************************************
* Name:        poly_decompress
*
* Description: De-serialization and subsequent decompression of a polynomial;
*              approximate inverse of poly_compress
*
* Arguments:   - poly *r:                pointer to output polynomial
*              -  unsigned char *a: pointer to input byte array
**************************************************/
void poly_decompress(poly* r, unsigned char* a)
{
	unsigned int i, j;
	for (i = 0; i < NEWHOPE_N; i += 8)
	{
		r->coeffs[i + 0] = a[0] & 7;
		r->coeffs[i + 1] = (a[0] >> 3) & 7;
		r->coeffs[i + 2] = (a[0] >> 6) | ((a[1] << 2) & 4);
		r->coeffs[i + 3] = (a[1] >> 1) & 7;
		r->coeffs[i + 4] = (a[1] >> 4) & 7;
		r->coeffs[i + 5] = (a[1] >> 7) | ((a[2] << 1) & 6);
		r->coeffs[i + 6] = (a[2] >> 2) & 7;
		r->coeffs[i + 7] = (a[2] >> 5);
		a += 3;
		for (j = 0; j < 8; j++)
			r->coeffs[i + j] = ((uint32_t)r->coeffs[i + j] * NEWHOPE_Q + 4) >> 3;
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
void poly_frommsg(poly* r, unsigned char* msg)
{
	unsigned int i, j, mask;
	for (i = 0; i < 32; i++) // XXX: MACRO for 32
	{
		for (j = 0; j < 8; j++)
		{
			mask = -((msg[i] >> j) & 1);
			r->coeffs[8 * i + j + 0] = mask & (NEWHOPE_Q / 2);
			r->coeffs[8 * i + j + 256] = mask & (NEWHOPE_Q / 2);
#if (NEWHOPE_N == 1024)
			r->coeffs[8 * i + j + 512] = mask & (NEWHOPE_Q / 2);
			r->coeffs[8 * i + j + 768] = mask & (NEWHOPE_Q / 2);
#endif
		}
	}
}

/*************************************************
* Name:        poly_tomsg
*
* Description: Convert polynomial to 32-byte message
*
* Arguments:   - unsigned char *msg: pointer to output message
*              -  poly *x:      pointer to input polynomial
**************************************************/
void poly_tomsg(unsigned char* msg, poly* x)
{
	unsigned int i;
	uint16_t t;

	for (i = 0; i < 32; i++)
		msg[i] = 0;

	for (i = 0; i < 256; i++)
	{
		t = flipabs(x->coeffs[i + 0]);
		t += flipabs(x->coeffs[i + 256]);
#if (NEWHOPE_N == 1024)
		t += flipabs(x->coeffs[i + 512]);
		t += flipabs(x->coeffs[i + 768]);
		t = ((t - NEWHOPE_Q));
#else
		t = ((t - NEWHOPE_Q / 2));
#endif

		t >>= 15;
		msg[i >> 3] |= t << (i & 7);
	}
}



/*************************************************
* Name:        hw
*
* Description: Compute the Hamming weight of a byte
*
* Arguments:   - unsigned char a: input byte
**************************************************/
__constant__ unsigned char HW_Matrix[256] =
{
	0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4,
	1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 5,
	1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 5,
	2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6,
	1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 5,
	2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6,
	2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6,
	3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7,
	1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 5,
	2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6,
	2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6,
	3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7,
	2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6,
	3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7,
	3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7,
	4, 5, 5, 6, 5, 6, 6, 7, 5, 6, 6, 7, 6, 7, 7, 8
};



#include <stdio.h>

/*************************************************
* Name:        poly_uniform
*
* Description: Sample a polynomial deterministically from a seed,
*              with output polynomial looking uniformly random
*
* Arguments:   - poly *a:                   pointer to output polynomial
*              -  unsigned char *seed: pointer to input seed
**************************************************/
__global__ void poly_uniform_kernel_parallel_sh_comb(poly* a)
{
	int threadId = blockIdx.x * blockDim.x + threadIdx.x;

#define gridSz  4  //gridDim.x
#define blockSz 32  //blockDim.x
#define StSz blockSz * 25
	__shared__  uint64_t state[StSz];

	if (threadId < 128)  // 16 threads 128
	{
		unsigned int ctr = 0, pos = 0;
		uint16_t val;
		uint8_t buf[SHAKE128_RATE];
		uint16_t buf1[SHAKE128_RATE];
		int i, j;
		int threadIdxi = threadIdx.x;

		KeccakF1600_StatePermute_sh128_comb(threadId, 0x1F, state, blockSz);

		int inner_lp = 0;
		pos = 0;

		for (int i = 0; i < 21; i++)
		{
			int address = threadIdx.x + i * blockSz;

			auto newRandomV = state[address]; // read 4 stuffs (buf+8*i)

			uint16_t* newRandom = (uint16_t*)(&newRandomV);

			inner_lp = 0;

			while (1)
			{

				uint16_t newVal = newRandom[inner_lp++];

				if (newVal < 5 * NEWHOPE_Q)
				{
					buf1[ctr++] = newVal; 
				}

				if (ctr == 8) //7
					goto jump_out;

				if (inner_lp == 4)
					break;
			}
		}

		jump_out:

		for (int i = 0; i < 8; i++)
		{
			a->coeffs[threadId + (i * 128)] = buf1[i];
		}
		
	}
}

/*************************************************
* Name:        poly_sample
*
* Description: Sample a polynomial deterministically from a seed and a nonce,
*              with output polynomial close to centered binomial distribution
*              with parameter k=8
*
* Arguments:   - poly *r:                   pointer to output polynomial
*              -  unsigned char *seed: pointer to input seed
*              - unsigned char nonce:       one-byte input nonce
**************************************************/
__global__ void poly_sample_kernel_parallel(poly* r, unsigned char nonce)
{
	int threadId = blockIdx.x * blockDim.x + threadIdx.x;
#define blockSz 32

	__shared__ uint64_t state[blockSz * 25];

	if (threadId < 256)  // 16 threads NEWHOPE_N /64
	{
		unsigned char a, b;
		int j;

		KeccakF1600_StatePermute_sh_comb(nonce, threadId, 0x1F, state, threadIdx.x, blockSz);

		int inner_lp = 0;
		int ctr = 0;

		for (int i = 0; i < 1; i++) //21
		{
		int address = threadIdx.x +i * blockSz;

			auto newRandomV = state[address]; // read 4 stuffs

			uint8_t* newRandom = (uint8_t*)(&newRandomV);

			inner_lp = 0;

			while (1)
			{
				a = newRandom[inner_lp++];
				b = newRandom[inner_lp++];

				r->coeffs[threadId + (ctr++ * 256)] = HW_Matrix[a] + NEWHOPE_Q - HW_Matrix[b]; //256

				if (inner_lp == 8)
					break;
			}
		}

	}
}



/*************************************************
* Name:        poly_pointwise
*
* Description: Multiply two polynomials pointwise (i.e., coefficient-wise).
*
* Arguments:   - poly *r:       pointer to output polynomial
*              -  poly *a: pointer to first input polynomial
*              -  poly *b: pointer to second input polynomial
**************************************************/
__global__ void poly_mul_pointwise_kernel(poly* r, poly* a, poly* b)
{
	//int i;
	uint16_t t;

	int threadId = blockIdx.x * blockDim.x + threadIdx.x;

	//for (i = 0; i < NEWHOPE_N; i++)
	if (threadId < NEWHOPE_N)
	{
		t = montgomery_reduce_device(3186 * b->coeffs[threadId]); /* t is now in Montgomery domain */
		r->coeffs[threadId] = montgomery_reduce_device(a->coeffs[threadId] * t);  /* r->coeffs[i] is back in normal domain */
	}
}

/*************************************************
* Name:        poly_add
*
* Description: Add two polynomials
*
* Arguments:   - poly *r:       pointer to output polynomial
*              -  poly *a: pointer to first input polynomial
*              -  poly *b: pointer to second input polynomial
**************************************************/
__global__ void poly_add_kernel(poly* r, poly* a, poly* b)
{
	//int i;
	//for (i = 0; i < NEWHOPE_N; i++)
	int threadId = blockIdx.x * blockDim.x + threadIdx.x;

	if (threadId < NEWHOPE_N)
	{
		r->coeffs[threadId] = (a->coeffs[threadId] + b->coeffs[threadId]) % NEWHOPE_Q;
	}


}

/*************************************************
* Name:        poly_sub
*
* Description: Subtract two polynomials
*
* Arguments:   - poly *r:       pointer to output polynomial
*              -  poly *a: pointer to first input polynomial
*              -  poly *b: pointer to second input polynomial
**************************************************/
__global__ void poly_sub_kernel(poly* r, poly* a, poly* b)
{
	//int i;
	//for (i = 0; i < NEWHOPE_N; i++)

	int threadId = blockIdx.x * blockDim.x + threadIdx.x;

	if (threadId < NEWHOPE_N)
	{
		r->coeffs[threadId] = (a->coeffs[threadId] + 3 * NEWHOPE_Q - b->coeffs[threadId]) % NEWHOPE_Q;
	}
}


/*************************************************
* Name:        poly_ntt
*
* Description: Forward NTT transform of a polynomial in place
*              Input is assumed to have coefficients in bitreversed order
*              Output has coefficients in normal order
*
* Arguments:   - poly *r: pointer to in/output polynomial
**************************************************/


/*************************************************
* Name:        poly_invntt
*
* Description: Inverse NTT transform of a polynomial in place
*              Input is assumed to have coefficients in normal order
*              Output has coefficients in normal order
*
* Arguments:   - poly *r: pointer to in/output polynomial
**************************************************/

