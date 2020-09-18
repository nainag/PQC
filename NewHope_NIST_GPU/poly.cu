
//  @Author: Arpan Jati
//  Adapted from NewHope Reference Codebase and Parallelized using CUDA
//  Updated: August 2019

#include "poly.h"
#include "ntt.h"
#include "reduce.h"
#include "fips202.h"
#include "params.h"

/*************************************************
* Name:        coeff_freeze
*
* Description: Fully reduces an integer modulo q in constant time
*
* Arguments:   uint16_t x: input integer to be reduced
*
* Returns integer in {0,...,q-1} congruent to x modulo q
**************************************************/
__device__ static uint16_t coeff_freeze(uint16_t x)
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
__device__ static uint16_t flipabs(uint16_t x)
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
*              - const unsigned char *a: pointer to input byte array
**************************************************/
__device__ void poly_frombytes(poly* r, unsigned char* a)
{
	int X = threadIdx.x + blockIdx.x * blockDim.x;

	int i;
	for (i = 0; i < NEWHOPE_N / 4; i++)
	{
		r->coeffs[4 * i + 0].threads[X] = a[7 * i + 0] | (((uint16_t)a[7 * i + 1] & 0x3f) << 8);
		r->coeffs[4 * i + 1].threads[X] = (a[7 * i + 1] >> 6) | (((uint16_t)a[7 * i + 2]) << 2) | (((uint16_t)a[7 * i + 3] & 0x0f) << 10);
		r->coeffs[4 * i + 2].threads[X] = (a[7 * i + 3] >> 4) | (((uint16_t)a[7 * i + 4]) << 4) | (((uint16_t)a[7 * i + 5] & 0x03) << 12);
		r->coeffs[4 * i + 3].threads[X] = (a[7 * i + 5] >> 2) | (((uint16_t)a[7 * i + 6]) << 6);
	}
}

__global__ void poly_frombytes_n(int COUNT, poly* r, unsigned char* a)
{
	int X = threadIdx.x + blockIdx.x * blockDim.x;
	if (X < COUNT)
	{
		int X = threadIdx.x + blockIdx.x * blockDim.x;

		int o_a = X * NEWHOPE_CPAPKE_SECRETKEYBYTES;

		int i;
		for (i = 0; i < NEWHOPE_N / 4; i++)
		{
			r->coeffs[4 * i + 0].threads[X] = (a + o_a)[7 * i + 0] | (((uint16_t)(a + o_a)[7 * i + 1] & 0x3f) << 8);
			r->coeffs[4 * i + 1].threads[X] = ((a + o_a)[7 * i + 1] >> 6) | (((uint16_t)(a + o_a)[7 * i + 2]) << 2) | (((uint16_t)(a + o_a)[7 * i + 3] & 0x0f) << 10);
			r->coeffs[4 * i + 2].threads[X] = ((a + o_a)[7 * i + 3] >> 4) | (((uint16_t)(a + o_a)[7 * i + 4]) << 4) | (((uint16_t)(a + o_a)[7 * i + 5] & 0x03) << 12);
			r->coeffs[4 * i + 3].threads[X] = ((a + o_a)[7 * i + 5] >> 2) | (((uint16_t)(a + o_a)[7 * i + 6]) << 6);
		}
	}
}


/*************************************************
* Name:        poly_tobytes
*
* Description: Serialization of a polynomial
*
* Arguments:   - unsigned char *r: pointer to output byte array
*              - const poly *p:    pointer to input polynomial
**************************************************/
__device__ void poly_tobytes(unsigned char* r, poly* p)
{
	int X = threadIdx.x + blockIdx.x * blockDim.x;

	int i;
	uint16_t t0, t1, t2, t3;
	for (i = 0; i < NEWHOPE_N / 4; i++)
	{
		t0 = coeff_freeze(p->coeffs[4 * i + 0].threads[X]);
		t1 = coeff_freeze(p->coeffs[4 * i + 1].threads[X]);
		t2 = coeff_freeze(p->coeffs[4 * i + 2].threads[X]);
		t3 = coeff_freeze(p->coeffs[4 * i + 3].threads[X]);

		r[7 * i + 0] = t0 & 0xff;
		r[7 * i + 1] = (t0 >> 8) | (t1 << 6);
		r[7 * i + 2] = (t1 >> 2);
		r[7 * i + 3] = (t1 >> 10) | (t2 << 4);
		r[7 * i + 4] = (t2 >> 4);
		r[7 * i + 5] = (t2 >> 12) | (t3 << 2);
		r[7 * i + 6] = (t3 >> 6);
	}

}

__global__ void poly_tobytes_n(int COUNT, unsigned char* r, poly* p)
{
	int X = threadIdx.x + blockIdx.x * blockDim.x;
	if (X < COUNT)
	{
		int o_r = X * NEWHOPE_CPAPKE_SECRETKEYBYTES;

		int i;
		uint16_t t0, t1, t2, t3;
		for (i = 0; i < NEWHOPE_N / 4; i++)
		{
			t0 = coeff_freeze(p->coeffs[4 * i + 0].threads[X]);
			t1 = coeff_freeze(p->coeffs[4 * i + 1].threads[X]);
			t2 = coeff_freeze(p->coeffs[4 * i + 2].threads[X]);
			t3 = coeff_freeze(p->coeffs[4 * i + 3].threads[X]);

			(r + o_r)[7 * i + 0] = t0 & 0xff;
			(r + o_r)[7 * i + 1] = (t0 >> 8) | (t1 << 6);
			(r + o_r)[7 * i + 2] = (t1 >> 2);
			(r + o_r)[7 * i + 3] = (t1 >> 10) | (t2 << 4);
			(r + o_r)[7 * i + 4] = (t2 >> 4);
			(r + o_r)[7 * i + 5] = (t2 >> 12) | (t3 << 2);
			(r + o_r)[7 * i + 6] = (t3 >> 6);
		}
	}
}


/*************************************************
* Name:        poly_compress
*
* Description: Compression and subsequent serialization of a polynomial
*
* Arguments:   - unsigned char *r: pointer to output byte array
*              - const poly *p:    pointer to input polynomial
**************************************************/
__device__ void poly_compress(unsigned char* r, poly* p)
{
	int X = threadIdx.x + blockIdx.x * blockDim.x;

	unsigned int i, j, k = 0;

	uint32_t t[8];

	for (i = 0; i < NEWHOPE_N; i += 8)
	{
		for (j = 0; j < 8; j++)
		{
			t[j] = coeff_freeze(p->coeffs[i + j].threads[X]);
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
*              - const unsigned char *a: pointer to input byte array
**************************************************/
__device__ void poly_decompress(poly* r, unsigned char* a)
{
	int X = threadIdx.x + blockIdx.x * blockDim.x;

	unsigned int i, j;
	for (i = 0; i < NEWHOPE_N; i += 8)
	{
		r->coeffs[i + 0].threads[X] = a[0] & 7;
		r->coeffs[i + 1].threads[X] = (a[0] >> 3) & 7;
		r->coeffs[i + 2].threads[X] = (a[0] >> 6) | ((a[1] << 2) & 4);
		r->coeffs[i + 3].threads[X] = (a[1] >> 1) & 7;
		r->coeffs[i + 4].threads[X] = (a[1] >> 4) & 7;
		r->coeffs[i + 5].threads[X] = (a[1] >> 7) | ((a[2] << 1) & 6);
		r->coeffs[i + 6].threads[X] = (a[2] >> 2) & 7;
		r->coeffs[i + 7].threads[X] = (a[2] >> 5);
		a += 3;
		for (j = 0; j < 8; j++)
			r->coeffs[i + j].threads[X] = ((uint32_t)r->coeffs[i + j].threads[X] * NEWHOPE_Q + 4) >> 3;
	}
}

/*************************************************
* Name:        poly_frommsg
*
* Description: Convert 32-byte message to polynomial
*
* Arguments:   - poly *r:                  pointer to output polynomial
*              - const unsigned char *msg: pointer to input message
**************************************************/
__global__ void poly_frommsg_n(int COUNT, poly* r, unsigned char* msg)
{
	int X = threadIdx.x + blockIdx.x * blockDim.x;

	if (X < COUNT)
	{
		unsigned int i, j, mask;

		int o_msg = NEWHOPE_SYMBYTES * X;


		for (i = 0; i < 32; i++) // XXX: MACRO for 32
		{
			for (j = 0; j < 8; j++)
			{
				mask = -(((msg + o_msg)[i] >> j) & 1);
				r->coeffs[8 * i + j + 0].threads[X] = mask & (NEWHOPE_Q / 2);
				r->coeffs[8 * i + j + 256].threads[X] = mask & (NEWHOPE_Q / 2);
#if (NEWHOPE_N == 1024)
				r->coeffs[8 * i + j + 512].threads[X] = mask & (NEWHOPE_Q / 2);
				r->coeffs[8 * i + j + 768].threads[X] = mask & (NEWHOPE_Q / 2);
#endif
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
*              - const poly *x:      pointer to input polynomial
**************************************************/
__global__ void poly_tomsg_n(int COUNT, unsigned char* msg, poly* x)
{
	int X = threadIdx.x + blockIdx.x * blockDim.x;

	if (X < COUNT)
	{
		int o_msg = NEWHOPE_SYMBYTES * X;

		unsigned int i;
		uint16_t t;

		for (i = 0; i < 32; i++)
			(msg + o_msg)[i] = 0;

		for (i = 0; i < 256; i++)
		{
			t = flipabs(x->coeffs[i + 0].threads[X]);
			t += flipabs(x->coeffs[i + 256].threads[X]);
#if (NEWHOPE_N == 1024)
			t += flipabs(x->coeffs[i + 512].threads[X]);
			t += flipabs(x->coeffs[i + 768].threads[X]);
			t = ((t - NEWHOPE_Q));
#else
			t = ((t - NEWHOPE_Q / 2));
#endif

			t >>= 15;
			(msg + o_msg)[i >> 3] |= t << (i & 7);
		}
	}
}

/*
__device__ void poly_uniform_coal(poly* a, unsigned char* seed, unsigned char* temp_reg)
{
	int X = threadIdx.x + blockIdx.x * blockDim.x;

	unsigned int ctr = 0;
	uint16_t val;
	uint64_t state[25];

	// uint8_t buf[SHAKE128_RATE];
	
	uint8_t extseed[NEWHOPE_SYMBYTES + 1];
	int i, j;

	for (i = 0; i < NEWHOPE_SYMBYTES; i++)
		extseed[i] = (seed + X)[i * N_TESTS];

	for (i = 0; i < NEWHOPE_N / 64; i++) 
	{
		ctr = 0;
		extseed[NEWHOPE_SYMBYTES] = i; 
		shake128_absorb(state, extseed, NEWHOPE_SYMBYTES + 1);
		while (ctr < 64) 
		{
			shake128_squeezeblocks_coal(temp_reg, 1, state);
			for (j = 0; j < SHAKE128_RATE && ctr < 64; j += 2)
			{
				val = (temp_reg[((j) * N_TESTS + (X*2))] | ((uint16_t)temp_reg[((j+1) * N_TESTS + X*2)] << 8));
				if (val < 5 * NEWHOPE_Q)
				{
					a->coeffs[i * 64 + ctr].threads[X] = val;
					ctr++;
				}
			}
		}
	}
}
*/

__device__ void poly_uniform(poly* a, unsigned char* seed)
{
	int X = threadIdx.x + blockIdx.x * blockDim.x;

	unsigned int ctr = 0;
	uint16_t val;
	uint64_t state[25];
	uint8_t buf[SHAKE128_RATE];

	uint8_t extseed[NEWHOPE_SYMBYTES + 1];
	int i, j;

	for (i = 0; i < NEWHOPE_SYMBYTES; i++)
		extseed[i] = seed[i];

	for (i = 0; i < NEWHOPE_N / 64; i++) /* generate a in blocks of 64 coefficients */
	{
		ctr = 0;
		extseed[NEWHOPE_SYMBYTES] = i; /* domain-separate the 16 independent calls */
		shake128_absorb(state, extseed, NEWHOPE_SYMBYTES + 1);
		while (ctr < 64) /* Very unlikely to run more than once */
		{
			shake128_squeezeblocks(buf, 1, state);
			for (j = 0; j < SHAKE128_RATE && ctr < 64; j += 2)
			{
				val = (buf[j] | ((uint16_t)buf[j + 1] << 8));
				if (val < 5 * NEWHOPE_Q)
				{
					a->coeffs[i * 64 + ctr].threads[X] = val;
					ctr++;
				}
			}
		}
	}
}


/*************************************************
* Name:        hw
*
* Description: Compute the Hamming weight of a byte
*
* Arguments:   - unsigned char a: input byte
**************************************************/
__device__  unsigned char hw(unsigned char a)
{
	//int X = threadIdx.x + blockIdx.x * blockDim.x;
	unsigned char i, r = 0;
	for (i = 0; i < 8; i++)
		r += (a >> i) & 1;
	return r;
}

/*************************************************
* Name:        poly_sample
*
* Description: Sample a polynomial deterministically from a seed and a nonce,
*              with output polynomial close to centered binomial distribution
*              with parameter k=8
*
* Arguments:   - poly *r:                   pointer to output polynomial
*              - const unsigned char *seed: pointer to input seed
*              - unsigned char nonce:       one-byte input nonce
**************************************************/
__global__ void poly_sample(int COUNT, poly* r, unsigned char* seed, unsigned char nonce)
{
	int X = threadIdx.x + blockIdx.x * blockDim.x;

	if (X < COUNT)
	{

#if NEWHOPE_K != 8
#error "poly_sample in poly.c only supports k=8"
#endif
		unsigned char buf[128], a, b;
		//  uint32_t t, d, a, b, c;
		int i, j;

		int o_seed = X * 32;

		unsigned char extseed[NEWHOPE_SYMBYTES + 2];

		for (i = 0; i < NEWHOPE_SYMBYTES; i++)
			extseed[i] = (seed + o_seed)[i];

		extseed[NEWHOPE_SYMBYTES] = nonce;

		for (i = 0; i < NEWHOPE_N / 64; i++) /* Generate noise in blocks of 64 coefficients */
		{
			extseed[NEWHOPE_SYMBYTES + 1] = i;
			shake256(buf, 128, extseed, NEWHOPE_SYMBYTES + 2);
			for (j = 0; j < 64; j++)
			{
				a = buf[2 * j];
				b = buf[2 * j + 1];
				r->coeffs[64 * i + j].threads[X] = hw(a) + NEWHOPE_Q - hw(b);
				/*
				t = buf[j] | ((uint32_t)buf[j+1] << 8) | ((uint32_t)buf[j+2] << 16) | ((uint32_t)buf[j+3] << 24);
				d = 0;
				for(k=0;k<8;k++)
				  d += (t >> k) & 0x01010101;
				a = d & 0xff;
				b = ((d >>  8) & 0xff);
				c = ((d >> 16) & 0xff);
				d >>= 24;
				r->coeffs[64*i+j/2]   = a + NEWHOPE_Q - b;
				r->coeffs[64*i+j/2+1] = c + NEWHOPE_Q - d;
				*/
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
*              - const poly *a: pointer to first input polynomial
*              - const poly *b: pointer to second input polynomial
**************************************************/
__global__ void poly_mul_pointwise(int COUNT, poly* r, poly* a, poly* b)
{
	int X = threadIdx.x + blockIdx.x * blockDim.x;
	if (X < COUNT)
	{
		int i;
		uint16_t t;
		for (i = 0; i < NEWHOPE_N; i++)
		{
			t = montgomery_reduce(3186 * b->coeffs[i].threads[X]); /* t is now in Montgomery domain */
			r->coeffs[i].threads[X] = montgomery_reduce(a->coeffs[i].threads[X] * t);  /* r->coeffs[i] is back in normal domain */
		}
	}
}

/*************************************************
* Name:        poly_add
*
* Description: Add two polynomials
*
* Arguments:   - poly *r:       pointer to output polynomial
*              - const poly *a: pointer to first input polynomial
*              - const poly *b: pointer to second input polynomial
**************************************************/
__global__ void poly_add(int COUNT, poly* r, poly* a, poly* b)
{
	int X = threadIdx.x + blockIdx.x * blockDim.x;
	if (X < COUNT)
	{
		int i;
		for (i = 0; i < NEWHOPE_N; i++)
			r->coeffs[i].threads[X] = (a->coeffs[i].threads[X] + b->coeffs[i].threads[X]) % NEWHOPE_Q;
	}
}

/*************************************************
* Name:        poly_sub
*
* Description: Subtract two polynomials
*
* Arguments:   - poly *r:       pointer to output polynomial
*              - const poly *a: pointer to first input polynomial
*              - const poly *b: pointer to second input polynomial
**************************************************/
__global__ void poly_sub(int COUNT, poly* r, poly* a, poly* b)
{
	int X = threadIdx.x + blockIdx.x * blockDim.x;
	if (X < COUNT)
	{
		int i;
		for (i = 0; i < NEWHOPE_N; i++)
			r->coeffs[i].threads[X] = (a->coeffs[i].threads[X] + 3 * NEWHOPE_Q - b->coeffs[i].threads[X]) % NEWHOPE_Q;
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
__global__ void poly_ntt(int COUNT, poly* r)
{
	int X = threadIdx.x + blockIdx.x * blockDim.x;
	if (X < COUNT)
	{
		mul_coefficients(r, gammas_bitrev_montgomery);
		ntt(r, gammas_bitrev_montgomery);
	}
}

/*************************************************
* Name:        poly_invntt
*
* Description: Inverse NTT transform of a polynomial in place
*              Input is assumed to have coefficients in normal order
*              Output has coefficients in normal order
*
* Arguments:   - poly *r: pointer to in/output polynomial
**************************************************/
__global__ void poly_invntt(int COUNT, poly* r)
{
	int X = threadIdx.x + blockIdx.x * blockDim.x;
	if (X < COUNT)
	{
		bitrev_vector(r);
		ntt(r, omegas_inv_bitrev_montgomery);
		mul_coefficients(r, gammas_inv_montgomery);
	}
}

