
//  @Author: Arpan Jati
//  Adapted from NewHope Reference Codebase and Parallelized using CUDA
//  Modified to generate constant output for debugging. Do not use in actual application.
//  Updated: August 2019

#include <string.h>
#include "rng.h"

#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>

#ifdef _WIN32

#include <Windows.h>

static int _initialized = 0;

static HCRYPTPROV hProvider = 0;

void InitializeRandomProviders()
{
	CryptAcquireContextW(&hProvider, 0, 0, PROV_RSA_FULL, CRYPT_VERIFYCONTEXT | CRYPT_SILENT);
	_initialized = 1;
}

int randombytes_real(unsigned char* pbBuffer, unsigned long dwLength)
{
	if (!_initialized)
	{		
		InitializeRandomProviders();
	}

	if (hProvider == 0) return RNG_BAD_PROVIDER;

	if (!CryptGenRandom(hProvider, dwLength, pbBuffer))
	{
		CryptReleaseContext(hProvider, 0);
		return RNG_BAD_REQ_LEN;
	}

	return RNG_SUCCESS;
}


#else

#include <exception>

int randombytes_real(unsigned char* pbBuffer, unsigned long dwLength)
{
	printf("RNG NOT SUPPORTED");
	exit(1);
}

#endif

#include <stdint.h>

int randombytes(unsigned char* pbBuffer, unsigned long dwLength)
{
	for (unsigned int i = 0; i < dwLength / 2; i++)
	{
		((uint16_t*)pbBuffer)[i] = i;
	}

	return 0;
}

#ifndef RAND_MAX
#define RAND_MAX ((int) ((unsigned) ~0 >> 1))
#endif

int do_rand(unsigned long* ctx)
{
#ifdef  USE_WEAK_SEEDING
	/*
	* Historic implementation compatibility.
	* The random sequences do not vary much with the seed,
	* even with overflowing.
	*/
	return ((*ctx = *ctx * 1103515245 + 12345) % ((u_long)RAND_MAX + 1));
#else   /* !USE_WEAK_SEEDING */
	/*
	* Compute x = (7^5 * x) mod (2^31 - 1)
	* without overflowing 31 bits:
	*      (2^31 - 1) = 127773 * (7^5) + 2836
	* From "Random number generators: good ones are hard to find",
	* Park and Miller, Communications of the ACM, vol. 31, no. 10,
	* October 1988, p. 1195.
	*/
	long hi, lo, x;

	/* Can't be initialized with 0, so use another value. */
	if (*ctx == 0)
		* ctx = 123459876;
	hi = *ctx / 127773;
	lo = *ctx % 127773;
	x = 16807 * lo - 2836 * hi;
	if (x < 0)
		x += 0x7fffffff;
	return ((*ctx = x) % ((unsigned long)RAND_MAX + 1));
#endif  /* !USE_WEAK_SEEDING */
}



