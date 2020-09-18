/*
	Based on the public domain implementation in
	crypto_hash/keccakc512/simple/ from http://bench.cr.yp.to/supercop.html
	by Ronny Van Keer
	and the public domain "TweetFips202" implementation
	from https://twitter.com/tweetfips202
	by Gilles Van Assche, Daniel J. Bernstein, and Peter Schwabe 

	Adapted from NewHope Reference Codebase and Parallelized by Naina Gupta 
 */

#include <stdint.h>
#include <assert.h>
#include <string.h>
#include "fips202.h"

#define NROUNDS 24
#define MIN(a, b) ((a) < (b) ? (a) : (b))
#define ROL(a, offset) ((a << offset) ^ (a >> (64-offset)))

__constant__ uint8_t noise_seed_const[32];

__constant__ uint8_t public_seed_const[32];

 /*************************************************
 * Name:        load64
 *
 * Description: Load 8 bytes into uint64_t in little-endian order
 *
 * Arguments:   -  unsigned char *x: pointer to input byte array
 *
 * Returns the loaded 64-bit unsigned integer
 **************************************************/
__device__ static uint64_t load64(unsigned char *x)
{
	return ((uint64_t*)x)[0];
}

static uint64_t load64_host(unsigned char* x)
{
	return ((uint64_t*)x)[0];
}

/*************************************************
* Name:        store64
*
* Description: Store a 64-bit integer to a byte array in little-endian order
*
* Arguments:   - uint8_t *x: pointer to the output byte array
*              - uint64_t u: input 64-bit unsigned integer
**************************************************/
__device__ void store64(uint8_t *x, uint64_t u)
{
	((uint64_t*)x)[0] = u;
}

static void store64_host(uint8_t* x, uint64_t u)
{
	((uint64_t*)x)[0] = u;
}

/* Keccak round constants */
__device__ static const uint64_t KeccakF_RoundConstants[NROUNDS] =
{
	(uint64_t)0x0000000000000001ULL,
	(uint64_t)0x0000000000008082ULL,
	(uint64_t)0x800000000000808aULL,
	(uint64_t)0x8000000080008000ULL,
	(uint64_t)0x000000000000808bULL,
	(uint64_t)0x0000000080000001ULL,
	(uint64_t)0x8000000080008081ULL,
	(uint64_t)0x8000000000008009ULL,
	(uint64_t)0x000000000000008aULL,
	(uint64_t)0x0000000000000088ULL,
	(uint64_t)0x0000000080008009ULL,
	(uint64_t)0x000000008000000aULL,
	(uint64_t)0x000000008000808bULL,
	(uint64_t)0x800000000000008bULL,
	(uint64_t)0x8000000000008089ULL,
	(uint64_t)0x8000000000008003ULL,
	(uint64_t)0x8000000000008002ULL,
	(uint64_t)0x8000000000000080ULL,
	(uint64_t)0x000000000000800aULL,
	(uint64_t)0x800000008000000aULL,
	(uint64_t)0x8000000080008081ULL,
	(uint64_t)0x8000000000008080ULL,
	(uint64_t)0x0000000080000001ULL,
	(uint64_t)0x8000000080008008ULL
};

 static const uint64_t KeccakF_RoundConstants_host[NROUNDS] =
{
	(uint64_t)0x0000000000000001ULL,
	(uint64_t)0x0000000000008082ULL,
	(uint64_t)0x800000000000808aULL,
	(uint64_t)0x8000000080008000ULL,
	(uint64_t)0x000000000000808bULL,
	(uint64_t)0x0000000080000001ULL,
	(uint64_t)0x8000000080008081ULL,
	(uint64_t)0x8000000000008009ULL,
	(uint64_t)0x000000000000008aULL,
	(uint64_t)0x0000000000000088ULL,
	(uint64_t)0x0000000080008009ULL,
	(uint64_t)0x000000008000000aULL,
	(uint64_t)0x000000008000808bULL,
	(uint64_t)0x800000000000008bULL,
	(uint64_t)0x8000000000008089ULL,
	(uint64_t)0x8000000000008003ULL,
	(uint64_t)0x8000000000008002ULL,
	(uint64_t)0x8000000000000080ULL,
	(uint64_t)0x000000000000800aULL,
	(uint64_t)0x800000008000000aULL,
	(uint64_t)0x8000000080008081ULL,
	(uint64_t)0x8000000000008080ULL,
	(uint64_t)0x0000000080000001ULL,
	(uint64_t)0x8000000080008008ULL
};


__device__ static const int keccakf_rotc[24] = {
   1,  3,  6,  10, 15, 21, 28, 36, 45, 55, 2,  14,
   27, 41, 56, 8,  25, 43, 62, 18, 39, 61, 20, 44
};

__device__ static const int keccakf_piln[24] = {
	10, 7,  11, 17, 18, 3, 5,  16, 8,  21, 24, 4,
	15, 23, 19, 13, 12, 2, 20, 14, 22, 9,  6,  1
};

 static const int keccakf_rotc_host[24] = {
   1,  3,  6,  10, 15, 21, 28, 36, 45, 55, 2,  14,
   27, 41, 56, 8,  25, 43, 62, 18, 39, 61, 20, 44
};

 static const int keccakf_piln_host[24] = {
	10, 7,  11, 17, 18, 3, 5,  16, 8,  21, 24, 4,
	15, 23, 19, 13, 12, 2, 20, 14, 22, 9,  6,  1
};

 /*************************************************
* Name:        KeccakF1600_StatePermute
*
* Description: The Keccak F1600 Permutation
*
* Arguments:   - uint64_t * state: pointer to in/output Keccak state
**************************************************/
 void KeccakF1600_StatePermute_host(uint64_t* st_io)
{
	// variables
	int i, j, r;
	uint64_t t, bc[5];

	uint64_t st[25] = { 0 };

	for (int i = 0; i < 25; i++)
	{
		st[i] = st_io[i];
	}

	// actual iteration
	for (r = 0; r < NROUNDS; r++) {

		// Theta
		for (i = 0; i < 5; i++)
			bc[i] = st[i] ^ st[i + 5] ^ st[i + 10] ^ st[i + 15] ^ st[i + 20];

		for (i = 0; i < 5; i++) {
			t = bc[(i + 4) % 5] ^ ROL(bc[(i + 1) % 5], 1);
			for (j = 0; j < 25; j += 5)
				st[j + i] ^= t;
		}

		// Rho Pi
		t = st[1];
		for (i = 0; i < 24; i++) {
			j = keccakf_piln_host[i];
			bc[0] = st[j];
			st[j] = ROL(t, keccakf_rotc_host[i]);
			t = bc[0];
		}

		//  Chi
		for (j = 0; j < 25; j += 5)
		{
			for (i = 0; i < 5; i++)
				bc[i] = st[j + i];

			for (i = 0; i < 5; i++)
			{
				st[j + i] ^= (~bc[(i + 1) % 5]) & bc[(i + 2) % 5];

			}
		}

		//  Iota
		st[0] ^= KeccakF_RoundConstants_host[r];
	}
	for (int i = 0; i < 25; i++)
	{
		st_io[i] = st[i];
	}

}


/*************************************************
* Name:        shake128_absorb
*
* Description: Absorb step of the SHAKE128 XOF.
*              non-incremental, starts by zeroeing the state.
*
* Arguments:   - uint64_t *s:                     pointer to (uninitialized) output Keccak state
*              -  unsigned char *input:      pointer to input to be absorbed into s
*              - unsigned long long inputByteLen: length of input in bytes
**************************************************/

/*************************************************
* Name:        shake128_squeezeblocks
*
* Description: Squeeze step of SHAKE128 XOF. Squeezes full blocks of SHAKE128_RATE bytes each.
*              Modifies the state. Can be called multiple times to keep squeezing,
*              i.e., is incremental.
*
* Arguments:   - unsigned char *output:      pointer to output blocks
*              - unsigned long long nblocks: number of blocks to be squeezed (written to output)
*              - uint64_t *s:                pointer to in/output Keccak state
**************************************************/

__device__ void KeccakF1600_StatePermute_sh_comb(unsigned char nonce, unsigned char domainSep, unsigned char p, uint64_t*state , int threadIdx,
	int blockSz )
{
	// variables
	int i, j, r;
	uint64_t t, bc[5];

	uint64_t st[25] = { 0 };

	st[0] = ((uint64_t*)noise_seed_const)[0];
	st[1] = ((uint64_t*)noise_seed_const)[1];
	st[2] = ((uint64_t*)noise_seed_const)[2];
	st[3] = ((uint64_t*)noise_seed_const)[3];

	uint64_t temp = (p << 16) | (domainSep << 8) | nonce;
	st[4] = temp;

	st[((SHAKE256_RATE / 8) - 1)] = 0x8000000000000000;

// actual iteration
	for (r = 0; r < NROUNDS; r++) {

		// Theta
		bc[0] = st[0] ^ st[5] ^ st[10] ^ st[15] ^ st[20];
		bc[1] = st[1] ^ st[6] ^ st[11] ^ st[16] ^ st[21];
		bc[2] = st[2] ^ st[7] ^ st[12] ^ st[17] ^ st[22];
		bc[3] = st[3] ^ st[8] ^ st[13] ^ st[18] ^ st[23];
		bc[4] = st[4] ^ st[9] ^ st[14] ^ st[19] ^ st[24];

		t = bc[4] ^ ROL(bc[1], 1);
		st[0] = st[0] ^ t;
		st[5] = st[5] ^ t;
		st[10] = st[10] ^ t;
		st[15] = st[15] ^ t;
		st[20] = st[20] ^ t;
		t = bc[0] ^ ROL(bc[2], 1);
		st[1] = st[1] ^ t;
		st[6] = st[6] ^ t;
		st[11] = st[11] ^ t;
		st[16] = st[16] ^ t;
		st[21] = st[21] ^ t;
		t = bc[1] ^ ROL(bc[3], 1);
		st[2] = st[2] ^ t;
		st[7] = st[7] ^ t;
		st[12] = st[12] ^ t;
		st[17] = st[17] ^ t;
		st[22] = st[22] ^ t;
		t = bc[2] ^ ROL(bc[4], 1);
		st[3] = st[3] ^ t;
		st[8] = st[8] ^ t;
		st[13] = st[13] ^ t;
		st[18] = st[18] ^ t;
		st[23] = st[23] ^ t;
		t = bc[3] ^ ROL(bc[0], 1);
		st[4] = st[4] ^ t;
		st[9] = st[9] ^ t;
		st[14] = st[14] ^ t;
		st[19] = st[19] ^ t;
		st[24] = st[24] ^ t;


		// Rho Pi
		t = st[1];

		j = 10;
		bc[0] = st[j];
		st[j] = ROL(t, 1);
		t = bc[0];

		j = 7;
		bc[0] = st[j];
		st[j] = ROL(t, 3);
		t = bc[0];

		j = 11;
		bc[0] = st[j];
		st[j] = ROL(t, 6);
		t = bc[0];

		j = 17;
		bc[0] = st[j];
		st[j] = ROL(t, 10);
		t = bc[0];

		j = 18;
		bc[0] = st[j];
		st[j] = ROL(t, 15);
		t = bc[0];

		j = 3;
		bc[0] = st[j];
		st[j] = ROL(t, 21);
		t = bc[0];

		j = 5;
		bc[0] = st[j];
		st[j] = ROL(t, 28);
		t = bc[0];

		j = 16;
		bc[0] = st[j];
		st[j] = ROL(t, 36);
		t = bc[0];

		j = 8;
		bc[0] = st[j];
		st[j] = ROL(t, 45);
		t = bc[0];

		j = 21;
		bc[0] = st[j];
		st[j] = ROL(t, 55);
		t = bc[0];

		j = 24;
		bc[0] = st[j];
		st[j] = ROL(t, 2);
		t = bc[0];

		j = 4;
		bc[0] = st[j];
		st[j] = ROL(t, 14);
		t = bc[0];

		j = 15;
		bc[0] = st[j];
		st[j] = ROL(t, 27);
		t = bc[0];

		j = 23;
		bc[0] = st[j];
		st[j] = ROL(t, 41);
		t = bc[0];

		j = 19;
		bc[0] = st[j];
		st[j] = ROL(t, 56);
		t = bc[0];

		j = 13;
		bc[0] = st[j];
		st[j] = ROL(t, 8);
		t = bc[0];

		j = 12;
		bc[0] = st[j];
		st[j] = ROL(t, 25);
		t = bc[0];

		j = 2;
		bc[0] = st[j];
		st[j] = ROL(t, 43);
		t = bc[0];

		j = 20;
		bc[0] = st[j];
		st[j] = ROL(t, 62);
		t = bc[0];

		j = 14;
		bc[0] = st[j];
		st[j] = ROL(t, 18);
		t = bc[0];

		j = 22;
		bc[0] = st[j];
		st[j] = ROL(t, 39);
		t = bc[0];

		j = 9;
		bc[0] = st[j];
		st[j] = ROL(t, 61);
		t = bc[0];

		j = 6;
		bc[0] = st[j];
		st[j] = ROL(t, 20);
		t = bc[0];

		j = 1;
		bc[0] = st[j];
		st[j] = ROL(t, 44);
		t = bc[0];

		//  Chi
		for (j = 0; j < 25; j += 5)
		{
			for (i = 0; i < 5; i++)
				bc[i] = st[j + i];

			for (i = 0; i < 5; i++)
			{
				switch (i) {
				case 0: st[j + i] ^= (~bc[1]) & bc[2];
					break;      
				case 1: st[j + i] ^= (~bc[2]) & bc[3];
					break;
				case 2: st[j + i] ^= (~bc[3]) & bc[4];
					break;       
				case 3: st[j + i] ^= (~bc[4]) & bc[0];
					break;
				case 4: st[j + i] ^= (~bc[0]) & bc[1];
					break;
				}
			}
		}

		//  Iota
		st[0] ^= KeccakF_RoundConstants[r];
	}

	for (int i = 0; i < 25; i++)
	{
		state[threadIdx + i * blockSz] = st[i];
	}
	
}

__device__ void KeccakF1600_StatePermute_sh128_comb(unsigned char domainSep, unsigned char p, uint64_t*state, int blockSz)
{
	// variables
	int i, j, r;
	uint64_t t, bc[5];

	uint64_t st[25] = { 0 };

	st[0] = ((uint64_t*)public_seed_const)[0];
	st[1] = ((uint64_t*)public_seed_const)[1];
	st[2] = ((uint64_t*)public_seed_const)[2];
	st[3] = ((uint64_t*)public_seed_const)[3];

	uint64_t temp = (p << 8) | (domainSep) ;
	st[4] = temp;

	st[((SHAKE128_RATE / 8) - 1)] = 0x8000000000000000;

	// actual iteration
	for (r = 0; r < NROUNDS; r++) {

		// Theta
		bc[0] = st[0] ^ st[5] ^ st[10] ^ st[15] ^ st[20];
		bc[1] = st[1] ^ st[6] ^ st[11] ^ st[16] ^ st[21];
		bc[2] = st[2] ^ st[7] ^ st[12] ^ st[17] ^ st[22];
		bc[3] = st[3] ^ st[8] ^ st[13] ^ st[18] ^ st[23];
		bc[4] = st[4] ^ st[9] ^ st[14] ^ st[19] ^ st[24];

		t = bc[4] ^ ROL(bc[1], 1);
		st[0] = st[0] ^ t;
		st[5] = st[5] ^ t;
		st[10] = st[10] ^ t;
		st[15] = st[15] ^ t;
		st[20] = st[20] ^ t;
		t = bc[0] ^ ROL(bc[2], 1);
		st[1] = st[1] ^ t;
		st[6] = st[6] ^ t;
		st[11] = st[11] ^ t;
		st[16] = st[16] ^ t;
		st[21] = st[21] ^ t;
		t = bc[1] ^ ROL(bc[3], 1);
		st[2] = st[2] ^ t;
		st[7] = st[7] ^ t;
		st[12] = st[12] ^ t;
		st[17] = st[17] ^ t;
		st[22] = st[22] ^ t;
		t = bc[2] ^ ROL(bc[4], 1);
		st[3] = st[3] ^ t;
		st[8] = st[8] ^ t;
		st[13] = st[13] ^ t;
		st[18] = st[18] ^ t;
		st[23] = st[23] ^ t;
		t = bc[3] ^ ROL(bc[0], 1);
		st[4] = st[4] ^ t;
		st[9] = st[9] ^ t;
		st[14] = st[14] ^ t;
		st[19] = st[19] ^ t;
		st[24] = st[24] ^ t;


		// Rho Pi
		t = st[1];

		j = 10;
		bc[0] = st[j];
		st[j] = ROL(t, 1);
		t = bc[0];

		j = 7;
		bc[0] = st[j];
		st[j] = ROL(t, 3);
		t = bc[0];

		j = 11;
		bc[0] = st[j];
		st[j] = ROL(t, 6);
		t = bc[0];

		j = 17;
		bc[0] = st[j];
		st[j] = ROL(t, 10);
		t = bc[0];

		j = 18;
		bc[0] = st[j];
		st[j] = ROL(t, 15);
		t = bc[0];

		j = 3;
		bc[0] = st[j];
		st[j] = ROL(t, 21);
		t = bc[0];

		j = 5;
		bc[0] = st[j];
		st[j] = ROL(t, 28);
		t = bc[0];

		j = 16;
		bc[0] = st[j];
		st[j] = ROL(t, 36);
		t = bc[0];

		j = 8;
		bc[0] = st[j];
		st[j] = ROL(t, 45);
		t = bc[0];

		j = 21;
		bc[0] = st[j];
		st[j] = ROL(t, 55);
		t = bc[0];

		j = 24;
		bc[0] = st[j];
		st[j] = ROL(t, 2);
		t = bc[0];

		j = 4;
		bc[0] = st[j];
		st[j] = ROL(t, 14);
		t = bc[0];

		j = 15;
		bc[0] = st[j];
		st[j] = ROL(t, 27);
		t = bc[0];

		j = 23;
		bc[0] = st[j];
		st[j] = ROL(t, 41);
		t = bc[0];

		j = 19;
		bc[0] = st[j];
		st[j] = ROL(t, 56);
		t = bc[0];

		j = 13;
		bc[0] = st[j];
		st[j] = ROL(t, 8);
		t = bc[0];

		j = 12;
		bc[0] = st[j];
		st[j] = ROL(t, 25);
		t = bc[0];

		j = 2;
		bc[0] = st[j];
		st[j] = ROL(t, 43);
		t = bc[0];

		j = 20;
		bc[0] = st[j];
		st[j] = ROL(t, 62);
		t = bc[0];

		j = 14;
		bc[0] = st[j];
		st[j] = ROL(t, 18);
		t = bc[0];

		j = 22;
		bc[0] = st[j];
		st[j] = ROL(t, 39);
		t = bc[0];

		j = 9;
		bc[0] = st[j];
		st[j] = ROL(t, 61);
		t = bc[0];

		j = 6;
		bc[0] = st[j];
		st[j] = ROL(t, 20);
		t = bc[0];

		j = 1;
		bc[0] = st[j];
		st[j] = ROL(t, 44);
		t = bc[0];

		//  Chi
		for (j = 0; j < 25; j += 5)
		{
			for (i = 0; i < 5; i++)
				bc[i] = st[j + i];

			for (i = 0; i < 5; i++)
			{
				switch (i) {
				case 0: st[j + i] ^= (~bc[1]) & bc[2];
					break;      
				case 1: st[j + i] ^= (~bc[2]) & bc[3];
					break;
				case 2: st[j + i] ^= (~bc[3]) & bc[4];
					break;     
				case 3: st[j + i] ^= (~bc[4]) & bc[0];
					break;
				case 4: st[j + i] ^= (~bc[0]) & bc[1];
					break;
				}


			}
		}

		//  Iota
		st[0] ^= KeccakF_RoundConstants[r];
	}

	for (int i = 0; i < 25; i++)
	{
		state[threadIdx.x + i * blockSz] = st[i];
	}

}

/*************************************************
* Name:        keccak_absorb
*
* Description: Absorb step of Keccak;
*              non-incremental, starts by zeroeing the state.
*
* Arguments:   - uint64_t *s:             pointer to (uninitialized) output Keccak state
*              - unsigned int r:          rate in bytes (e.g., 168 for SHAKE128)
*              -  unsigned char *m:  pointer to input to be absorbed into s
*              - unsigned long long mlen: length of input in bytes
*              - unsigned char p:         domain-separation byte for different Keccak-derived functions
**************************************************/

void keccak_absorb_shake256_host(uint64_t* s,
	unsigned int r,
	unsigned char* m, unsigned long long int mlen,
	unsigned char p)
{
	unsigned long long i;
	unsigned char t[200];

	for (i = 0; i < 25; ++i)
		s[i] = 0;

	while (mlen >= r)
	{
		for (i = 0; i < r / 8; ++i)
			s[i] ^= load64_host(m + 8 * i);

		KeccakF1600_StatePermute_host(s);
		mlen -= r;
		m += r;
	}

	for (i = 0; i < r; ++i)
		t[i] = 0;
	for (i = 0; i < mlen; ++i)
		t[i] = m[i];
	t[i] = p;
	t[r - 1] |= 128;
	for (i = 0; i < r / 8; ++i)
		s[i] ^= load64_host(t + 8 * i);
}


/*************************************************
* Name:        keccak_squeezeblocks
*
* Description: Squeeze step of Keccak. Squeezes full blocks of r bytes each.
*              Modifies the state. Can be called multiple times to keep squeezing,
*              i.e., is incremental.
*
* Arguments:   - unsigned char *h:               pointer to output blocks
*              - unsigned long long int nblocks: number of blocks to be squeezed (written to h)
*              - uint64_t *s:                    pointer to in/output Keccak state
*              - unsigned int r:                 rate in bytes (e.g., 168 for SHAKE128)
**************************************************/

void keccak_squeezeblocks_host(unsigned char* h, unsigned long long int nblocks,
	uint64_t* s, unsigned int r)
{
	unsigned int i;
	while (nblocks > 0)
	{
		KeccakF1600_StatePermute_host(s);
		for (i = 0; i < (r >> 3); i++)
		{
			store64_host(h + 8 * i, s[i]);
		}
		h += r;
		nblocks--;
	}
}

/*************************************************
* Name:        shake256
*
* Description: SHAKE256 XOF with non-incremental API
*
* Arguments:   - unsigned char *output:      pointer to output
*              - unsigned long long outlen:  requested output length in bytes
			   -  unsigned char *input: pointer to input
			   - unsigned long long inlen:   length of input in bytes
**************************************************/
void shake256_host(unsigned char* output, unsigned long long outlen,
	unsigned char* input, unsigned long long inlen)
{
	uint64_t s[25];
	unsigned char t[SHAKE256_RATE];
	unsigned long long nblocks = outlen / SHAKE256_RATE;
	size_t i;

	for (i = 0; i < 25; ++i)
		s[i] = 0;

	/* Absorb input */
	keccak_absorb_shake256_host(s, SHAKE256_RATE, input, inlen, 0x1F);

	/* Squeeze output */
	keccak_squeezeblocks_host(output, nblocks, s, SHAKE256_RATE);

	output += nblocks * SHAKE256_RATE;
	outlen -= nblocks * SHAKE256_RATE;

	if (outlen)
	{
		keccak_squeezeblocks_host(t, 1, s, SHAKE256_RATE);
		for (i = 0; i < outlen; i++)
			output[i] = t[i];
	}
}