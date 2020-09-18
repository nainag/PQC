
// @Author: Arpan Jati
// Adapted from NewHope Reference Codebase and Parallelized using CUDA
// Updated : August 2019

#include <stdint.h>
#include "params.h"
#include "cbd.h"

/*************************************************
* Name:        load32_littleendian
*
* Description: load bytes into a 32-bit integer
*              in little-endian order
*
* Arguments:   -  unsigned char *x: pointer to input byte array
*
* Returns 32-bit unsigned integer loaded from x
**************************************************/
__device__  uint32_t load32_littleendian( unsigned char* x)
{
	/*uint32_t r;
	r = (uint32_t)x[0];
	r |= (uint32_t)x[1] << 8;
	r |= (uint32_t)x[2] << 16;
	r |= (uint32_t)x[3] << 24;
	return r;*/

	return ((uint32_t*)x)[0];
}

/*************************************************
* Name:        cbd
*
* Description: Given an array of uniformly random bytes, compute
*              polynomial with coefficients distributed according to
*              a centered binomial distribution with parameter KYBER_ETA
*
* Arguments:   - poly *r:                  pointer to output polynomial
*              -  unsigned char *buf: pointer to input byte array
**************************************************/
__device__ void cbd(poly* r,  unsigned char* buf)
{
	uint32_t d, t;
	int16_t a, b;
	int i, j;

	int X = threadIdx.x + blockIdx.x * blockDim.x;

	for (i = 0; i < KYBER_N / 8; i++)
	{
		t = load32_littleendian(buf + (4 * i));
		d = t & 0x55555555;
		d += (t >> 1) & 0x55555555;

		for (j = 0; j < 8; j++)
		{
			a = (d >> 4 * j) & 0x3;
			b = (d >> (4 * j + 2)) & 0x3;
			r->coeffs[8 * i + j].threads[X] = a - b;
		}
	}
}
