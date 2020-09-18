/*
	Based on the public domain implementation in
    crypto_hash/keccakc512/simple/ from http://bench.cr.yp.to/supercop.html
    by Ronny Van Keer
    and the public domain "TweetFips202" implementation
    from https://twitter.com/tweetfips202
    by Gilles Van Assche, Daniel J. Bernstein, and Peter Schwabe

	Adapted from NewHope Reference Codebase and Parallelized by Arpan Jati
 */

#include <stdint.h>
#include <assert.h>
#include <string.h>

#include "fips202.h"
#include "main.h"

#define NROUNDS 24
#define ROL(a, offset) ((a << offset) ^ (a >> (64-offset)))

#define MIN(a, b) ((a) < (b) ? (a) : (b))

/*
	static uint64_t load64(const unsigned char* x)
	{
		unsigned long long r = 0, i;

		for (i = 0; i < 8; ++i) {
			r |= (unsigned long long)x[i] << 8 * i;
		}
		return r;
	}

	static void store64(uint8_t* x, uint64_t u)
	{
		unsigned int i;

		for (i = 0; i < 8; ++i) {
			x[i] = u;
			u >>= 8;
		}
	}
*/

__device__ static uint64_t load64(const unsigned char* x)
{
	return ((uint64_t*)x)[0];
}

__device__ static void store64(uint8_t* x, uint64_t u)
{
	((uint64_t*)x)[0] = u;
}

/* Keccak round constants */
__device__  const uint64_t KeccakF_RoundConstants[NROUNDS] =
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

/*************************************************
* Name:        KeccakF1600_StatePermute
*
* Description: The Keccak F1600 Permutation
*
* Arguments:   - uint64_t * state: pointer to in/output Keccak state
**************************************************/
/*__device__ void KeccakF1600_StatePermute(uint64_t* state)
{
	int round;

	uint64_t Aba, Abe, Abi, Abo, Abu;
	uint64_t Aga, Age, Agi, Ago, Agu;
	uint64_t Aka, Ake, Aki, Ako, Aku;
	uint64_t Ama, Ame, Ami, Amo, Amu;
	uint64_t Asa, Ase, Asi, Aso, Asu;
	uint64_t BCa, BCe, BCi, BCo, BCu;
	uint64_t Da, De, Di, Do, Du;
	uint64_t Eba, Ebe, Ebi, Ebo, Ebu;
	uint64_t Ega, Ege, Egi, Ego, Egu;
	uint64_t Eka, Eke, Eki, Eko, Eku;
	uint64_t Ema, Eme, Emi, Emo, Emu;
	uint64_t Esa, Ese, Esi, Eso, Esu;

	//copyFromState(A, state)
	Aba = state[0];
	Abe = state[1];
	Abi = state[2];
	Abo = state[3];
	Abu = state[4];
	Aga = state[5];
	Age = state[6];
	Agi = state[7];
	Ago = state[8];
	Agu = state[9];
	Aka = state[10];
	Ake = state[11];
	Aki = state[12];
	Ako = state[13];
	Aku = state[14];
	Ama = state[15];
	Ame = state[16];
	Ami = state[17];
	Amo = state[18];
	Amu = state[19];
	Asa = state[20];
	Ase = state[21];
	Asi = state[22];
	Aso = state[23];
	Asu = state[24];

	for (round = 0; round < NROUNDS; round += 2)
	{
		//    prepareTheta
		BCa = Aba ^ Aga ^ Aka ^ Ama ^ Asa;
		BCe = Abe ^ Age ^ Ake ^ Ame ^ Ase;
		BCi = Abi ^ Agi ^ Aki ^ Ami ^ Asi;
		BCo = Abo ^ Ago ^ Ako ^ Amo ^ Aso;
		BCu = Abu ^ Agu ^ Aku ^ Amu ^ Asu;

		//thetaRhoPiChiIotaPrepareTheta(round  , A, E)
		Da = BCu ^ ROL(BCe, 1);
		De = BCa ^ ROL(BCi, 1);
		Di = BCe ^ ROL(BCo, 1);
		Do = BCi ^ ROL(BCu, 1);
		Du = BCo ^ ROL(BCa, 1);

		Aba ^= Da;
		BCa = Aba;
		Age ^= De;
		BCe = ROL(Age, 44);
		Aki ^= Di;
		BCi = ROL(Aki, 43);
		Amo ^= Do;
		BCo = ROL(Amo, 21);
		Asu ^= Du;
		BCu = ROL(Asu, 14);
		Eba = BCa ^ ((~BCe) & BCi);
		Eba ^= (uint64_t)KeccakF_RoundConstants[round];
		Ebe = BCe ^ ((~BCi) & BCo);
		Ebi = BCi ^ ((~BCo) & BCu);
		Ebo = BCo ^ ((~BCu) & BCa);
		Ebu = BCu ^ ((~BCa) & BCe);

		Abo ^= Do;
		BCa = ROL(Abo, 28);
		Agu ^= Du;
		BCe = ROL(Agu, 20);
		Aka ^= Da;
		BCi = ROL(Aka, 3);
		Ame ^= De;
		BCo = ROL(Ame, 45);
		Asi ^= Di;
		BCu = ROL(Asi, 61);
		Ega = BCa ^ ((~BCe) & BCi);
		Ege = BCe ^ ((~BCi) & BCo);
		Egi = BCi ^ ((~BCo) & BCu);
		Ego = BCo ^ ((~BCu) & BCa);
		Egu = BCu ^ ((~BCa) & BCe);

		Abe ^= De;
		BCa = ROL(Abe, 1);
		Agi ^= Di;
		BCe = ROL(Agi, 6);
		Ako ^= Do;
		BCi = ROL(Ako, 25);
		Amu ^= Du;
		BCo = ROL(Amu, 8);
		Asa ^= Da;
		BCu = ROL(Asa, 18);
		Eka = BCa ^ ((~BCe) & BCi);
		Eke = BCe ^ ((~BCi) & BCo);
		Eki = BCi ^ ((~BCo) & BCu);
		Eko = BCo ^ ((~BCu) & BCa);
		Eku = BCu ^ ((~BCa) & BCe);

		Abu ^= Du;
		BCa = ROL(Abu, 27);
		Aga ^= Da;
		BCe = ROL(Aga, 36);
		Ake ^= De;
		BCi = ROL(Ake, 10);
		Ami ^= Di;
		BCo = ROL(Ami, 15);
		Aso ^= Do;
		BCu = ROL(Aso, 56);
		Ema = BCa ^ ((~BCe) & BCi);
		Eme = BCe ^ ((~BCi) & BCo);
		Emi = BCi ^ ((~BCo) & BCu);
		Emo = BCo ^ ((~BCu) & BCa);
		Emu = BCu ^ ((~BCa) & BCe);

		Abi ^= Di;
		BCa = ROL(Abi, 62);
		Ago ^= Do;
		BCe = ROL(Ago, 55);
		Aku ^= Du;
		BCi = ROL(Aku, 39);
		Ama ^= Da;
		BCo = ROL(Ama, 41);
		Ase ^= De;
		BCu = ROL(Ase, 2);
		Esa = BCa ^ ((~BCe) & BCi);
		Ese = BCe ^ ((~BCi) & BCo);
		Esi = BCi ^ ((~BCo) & BCu);
		Eso = BCo ^ ((~BCu) & BCa);
		Esu = BCu ^ ((~BCa) & BCe);

		//    prepareTheta
		BCa = Eba ^ Ega ^ Eka ^ Ema ^ Esa;
		BCe = Ebe ^ Ege ^ Eke ^ Eme ^ Ese;
		BCi = Ebi ^ Egi ^ Eki ^ Emi ^ Esi;
		BCo = Ebo ^ Ego ^ Eko ^ Emo ^ Eso;
		BCu = Ebu ^ Egu ^ Eku ^ Emu ^ Esu;

		//thetaRhoPiChiIotaPrepareTheta(round+1, E, A)
		Da = BCu ^ ROL(BCe, 1);
		De = BCa ^ ROL(BCi, 1);
		Di = BCe ^ ROL(BCo, 1);
		Do = BCi ^ ROL(BCu, 1);
		Du = BCo ^ ROL(BCa, 1);

		Eba ^= Da;
		BCa = Eba;
		Ege ^= De;
		BCe = ROL(Ege, 44);
		Eki ^= Di;
		BCi = ROL(Eki, 43);
		Emo ^= Do;
		BCo = ROL(Emo, 21);
		Esu ^= Du;
		BCu = ROL(Esu, 14);
		Aba = BCa ^ ((~BCe) & BCi);
		Aba ^= (uint64_t)KeccakF_RoundConstants[round + 1];
		Abe = BCe ^ ((~BCi) & BCo);
		Abi = BCi ^ ((~BCo) & BCu);
		Abo = BCo ^ ((~BCu) & BCa);
		Abu = BCu ^ ((~BCa) & BCe);

		Ebo ^= Do;
		BCa = ROL(Ebo, 28);
		Egu ^= Du;
		BCe = ROL(Egu, 20);
		Eka ^= Da;
		BCi = ROL(Eka, 3);
		Eme ^= De;
		BCo = ROL(Eme, 45);
		Esi ^= Di;
		BCu = ROL(Esi, 61);
		Aga = BCa ^ ((~BCe) & BCi);
		Age = BCe ^ ((~BCi) & BCo);
		Agi = BCi ^ ((~BCo) & BCu);
		Ago = BCo ^ ((~BCu) & BCa);
		Agu = BCu ^ ((~BCa) & BCe);

		Ebe ^= De;
		BCa = ROL(Ebe, 1);
		Egi ^= Di;
		BCe = ROL(Egi, 6);
		Eko ^= Do;
		BCi = ROL(Eko, 25);
		Emu ^= Du;
		BCo = ROL(Emu, 8);
		Esa ^= Da;
		BCu = ROL(Esa, 18);
		Aka = BCa ^ ((~BCe) & BCi);
		Ake = BCe ^ ((~BCi) & BCo);
		Aki = BCi ^ ((~BCo) & BCu);
		Ako = BCo ^ ((~BCu) & BCa);
		Aku = BCu ^ ((~BCa) & BCe);

		Ebu ^= Du;
		BCa = ROL(Ebu, 27);
		Ega ^= Da;
		BCe = ROL(Ega, 36);
		Eke ^= De;
		BCi = ROL(Eke, 10);
		Emi ^= Di;
		BCo = ROL(Emi, 15);
		Eso ^= Do;
		BCu = ROL(Eso, 56);
		Ama = BCa ^ ((~BCe) & BCi);
		Ame = BCe ^ ((~BCi) & BCo);
		Ami = BCi ^ ((~BCo) & BCu);
		Amo = BCo ^ ((~BCu) & BCa);
		Amu = BCu ^ ((~BCa) & BCe);

		Ebi ^= Di;
		BCa = ROL(Ebi, 62);
		Ego ^= Do;
		BCe = ROL(Ego, 55);
		Eku ^= Du;
		BCi = ROL(Eku, 39);
		Ema ^= Da;
		BCo = ROL(Ema, 41);
		Ese ^= De;
		BCu = ROL(Ese, 2);
		Asa = BCa ^ ((~BCe) & BCi);
		Ase = BCe ^ ((~BCi) & BCo);
		Asi = BCi ^ ((~BCo) & BCu);
		Aso = BCo ^ ((~BCu) & BCa);
		Asu = BCu ^ ((~BCa) & BCe);
	}

	//copyToState(state, A)
	state[0] = Aba;
	state[1] = Abe;
	state[2] = Abi;
	state[3] = Abo;
	state[4] = Abu;
	state[5] = Aga;
	state[6] = Age;
	state[7] = Agi;
	state[8] = Ago;
	state[9] = Agu;
	state[10] = Aka;
	state[11] = Ake;
	state[12] = Aki;
	state[13] = Ako;
	state[14] = Aku;
	state[15] = Ama;
	state[16] = Ame;
	state[17] = Ami;
	state[18] = Amo;
	state[19] = Amu;
	state[20] = Asa;
	state[21] = Ase;
	state[22] = Asi;
	state[23] = Aso;
	state[24] = Asu;

#undef    round
}*/

__device__ static const int keccakf_rotc[24] = {
	1,  3,  6,  10, 15, 21, 28, 36, 45, 55, 2,  14,
	27, 41, 56, 8,  25, 43, 62, 18, 39, 61, 20, 44
};

__device__ static const int keccakf_piln[24] = {
	10, 7,  11, 17, 18, 3, 5,  16, 8,  21, 24, 4,
	15, 23, 19, 13, 12, 2, 20, 14, 22, 9,  6,  1
};

#define ROL(a, offset) ((a << offset) ^ (a >> (64-offset)))

__device__ void KeccakF1600_StatePermute(uint64_t* st_io)
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
			j = keccakf_piln[i];
			bc[0] = st[j];
			st[j] = ROL(t, keccakf_rotc[i]);
			t = bc[0];
		}

		//  Chi
		for (j = 0; j < 25; j += 5) {
			for (i = 0; i < 5; i++)
				bc[i] = st[j + i];
			for (i = 0; i < 5; i++)
				st[j + i] ^= (~bc[(i + 1) % 5]) & bc[(i + 2) % 5];
		}

		//  Iota
		st[0] ^= KeccakF_RoundConstants[r];
	}

	for (int i = 0; i < 25; i++)
	{
		st_io[i] = st[i];
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
*              - const unsigned char *m:  pointer to input to be absorbed into s
*              - unsigned long long mlen: length of input in bytes
*              - unsigned char p:         domain-separation byte for different Keccak-derived functions
**************************************************/
/*__device__  void keccak_absorb(uint64_t* s,
	unsigned int r,
	const unsigned char* m, unsigned long long int mlen,
	unsigned char p)
{
	unsigned long long i;
	unsigned char t[200];

	for (i = 0; i < 25; ++i)
		s[i] = 0;

	while (mlen >= r)
	{
		for (i = 0; i < r / 8; ++i)
			s[i] ^= load64(m + 8 * i);

		KeccakF1600_StatePermute(s);
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
		s[i] ^= load64(t + 8 * i);
}
*/

__device__  void keccak_absorb(
	uint64_t* s,
	unsigned int r,
	unsigned char* m,
	unsigned long long int mlen,
	unsigned char p)
{
	unsigned long long i;
	unsigned char t[200];

	// unsigned char* t = (unsigned char*) malloc(200);

	// Zero state
	for (i = 0; i < 25; ++i)
		s[i] = 0;

	while (mlen >= r)
	{
		for (i = 0; i < r / 8; ++i)
			s[i] ^= load64(m + 8 * i);

		KeccakF1600_StatePermute(s);
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
		s[i] ^= load64(t + 8 * i);

	//free(t);
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
__device__  void keccak_squeezeblocks(
	unsigned char* h, 
	unsigned long long int nblocks,
	uint64_t* s, unsigned int r)
{
	unsigned int i;
	while (nblocks > 0)
	{
		KeccakF1600_StatePermute(s);
		for (i = 0; i < (r >> 3); i++)
		{
			store64((h + 8 * i), s[i]);
		}
		h += r;
		nblocks--;
	}
}

/*
__device__  void keccak_squeezeblocks_coal(unsigned char* h,
	unsigned long long int nblocks,
	uint64_t* s, unsigned int r)
{
	int X = threadIdx.x + blockIdx.x * blockDim.x;
	unsigned int i;
	while (nblocks > 0)
	{
		KeccakF1600_StatePermute(s);
		for (i = 0; i < (r >> 3); i++)
		{
			store64((h + 8 * (i * X)), s[i]);
		}
		h += r;
		nblocks--;
	}
}
*/

/*************************************************
* Name:        shake128_absorb
*
* Description: Absorb step of the SHAKE128 XOF.
*              non-incremental, starts by zeroeing the state.
*
* Arguments:   - uint64_t *s:                     pointer to (uninitialized) output Keccak state
*              - const unsigned char *input:      pointer to input to be absorbed into s
*              - unsigned long long inputByteLen: length of input in bytes
**************************************************/
__device__ void shake128_absorb(uint64_t* s, unsigned char* input, 
	unsigned long long inputByteLen)
{
	keccak_absorb(s, SHAKE128_RATE, input, inputByteLen, 0x1F);
}

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

__device__ void shake128_squeezeblocks(unsigned char* output, 
	unsigned long long nblocks, uint64_t* s)
{
	keccak_squeezeblocks(output, nblocks, s, SHAKE128_RATE);
}

/*__device__ void shake128_squeezeblocks_coal(unsigned char* output,
	unsigned long long nblocks, uint64_t* s)
{
	keccak_squeezeblocks_coal(output, nblocks, s, SHAKE128_RATE);
}*/

__device__ void shake256(unsigned char* output, unsigned long long outlen,
	unsigned char* input, unsigned long long inlen)
{
	uint64_t s[25];
	unsigned char t[SHAKE256_RATE];
	unsigned long long nblocks = outlen / SHAKE256_RATE;
	size_t i;

	for (i = 0; i < 25; ++i)
		s[i] = 0;

	/* Absorb input */
	keccak_absorb(s, SHAKE256_RATE, (input), inlen, 0x1F);

	/* Squeeze output */
	keccak_squeezeblocks(output, nblocks, s, SHAKE256_RATE);

	output += nblocks * SHAKE256_RATE;
	outlen -= nblocks * SHAKE256_RATE;

	if (outlen)
	{
		keccak_squeezeblocks(t, 1, s, SHAKE256_RATE);
		for (i = 0; i < outlen; i++)
			output[i] = t[i];
	}
}

/*************************************************
* Name:        shake256
*
* Description: SHAKE256 XOF with non-incremental API
*
* Arguments:   - unsigned char *output:      pointer to output
*              - unsigned long long outlen:  requested output length in bytes
			   - const unsigned char *input: pointer to input
			   - unsigned long long inlen:   length of input in bytes
**************************************************/
__global__ void shake256_n(int COUNT, unsigned char* output, unsigned long long outlen,
	 unsigned char* input, unsigned long long inlen, unsigned char* largeTemp)
{
	int X = threadIdx.x + blockIdx.x * blockDim.x;
	if (X < COUNT)
	{
		int o_input = X * inlen;
		int o_output = X * outlen;

		//// --- make use of external temp variables --- /////////////////
		  
		int o_tempA = X * 1024;
		int o_tempB = X * 1024 + 512;

		unsigned char* t = (largeTemp + o_tempA);
		uint64_t* s = (uint64_t *)(largeTemp + o_tempB);

		//// --- make use of external temp variables  --- /////////////////

		// uint64_t s[25];
		// unsigned char t[SHAKE256_RATE];

		unsigned long long nblocks = outlen / SHAKE256_RATE;
		size_t i;

		for (i = 0; i < 25; ++i)
			s[i] = 0;

		// Absorb input 
		keccak_absorb(s, SHAKE256_RATE, (input + o_input), inlen, 0x1F);

		// Squeeze output 
		keccak_squeezeblocks(output, nblocks, s, SHAKE256_RATE);

		output += nblocks * SHAKE256_RATE;
		outlen -= nblocks * SHAKE256_RATE;

		if (outlen)
		{
			keccak_squeezeblocks(t, 1, s, SHAKE256_RATE);
			for (i = 0; i < outlen; i++)
				(output + o_output)[i] = t[i];
		}
	}

}
