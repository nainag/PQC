/********************************************************************************************
* FrodoKEM: Learning with Errors Key Encapsulation
*
* Abstract: setting parameters to test FrodoKEM-976

// @Author: Naina Gupta
// Adapted from FrodoKEM Reference Codebase and Parallelized using CUDA
// Updated : August 2019

*********************************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <stdbool.h>
#include "ds_benchmark.h"
#include "api.h"


#define SYSTEM_NAME    "FrodoKEM-976"

/********************************************************************************************
* FrodoKEM: Learning with Errors Key Encapsulation
*
* Abstract: benchmarking/testing KEM scheme
*********************************************************************************************/

#define KEM_TEST_ITERATIONS 1
#define KEM_BENCH_SECONDS     1


static int kem_test(const char *named_parameters, int iterations)
{
	uint8_t pk[CRYPTO_PUBLICKEYBYTES];
	uint8_t sk[CRYPTO_SECRETKEYBYTES];
	uint8_t ss_encap[CRYPTO_BYTES], ss_decap[CRYPTO_BYTES];
	uint8_t ct[CRYPTO_CIPHERTEXTBYTES];

	printf("\n");
	printf("=======================================================================================================================\n");
	printf("  GPU | Testing correctness of key encapsulation mechanism (KEM), system %s, tests for %d iterations\n", named_parameters, iterations);
	printf("=======================================================================================================================\n");

	for (int i = 0; i < iterations; i++) {
		crypto_kem_keypair_gpu(pk, sk);

		crypto_kem_enc(ct, ss_encap, pk);

		for (int j = 0; j < 24; j++)
		{
			printf("%02X ", ss_encap[j]);
		}
		printf("\n");

		crypto_kem_dec(ss_decap, ct, sk);

		for (int j = 0; j < 24; j++)
		{
			printf("%02X ", ss_decap[j]);
		}
		printf("\n");
		if (memcmp(ss_encap, ss_decap, CRYPTO_BYTES) != 0) {
			printf("\n ERROR!\n");
			return true;
		}


	}
	printf("Tests PASSED. All session keys matched.\n");
	printf("\n\n");

	return true;
}


static void kem_bench(const int seconds)
{
	uint8_t pk[CRYPTO_PUBLICKEYBYTES];
	uint8_t sk[CRYPTO_SECRETKEYBYTES];
	uint8_t ss_encap[CRYPTO_BYTES], ss_decap[CRYPTO_BYTES];
	uint8_t ct[CRYPTO_CIPHERTEXTBYTES];

	TIME_OPERATION_SECONDS({ crypto_kem_keypair_gpu(pk, sk); }, "Key generation", seconds);

	crypto_kem_keypair_gpu(pk, sk);
	TIME_OPERATION_SECONDS({ crypto_kem_enc(ct, ss_encap, pk); }, "KEM encapsulate", seconds);

	crypto_kem_enc(ct, ss_encap, pk);
	TIME_OPERATION_SECONDS({ crypto_kem_dec(ss_decap, ct, sk); }, "KEM decapsulate", seconds);
}


int main()
{
	int OK = true;

	OK = kem_test(SYSTEM_NAME, KEM_TEST_ITERATIONS);
	if (OK != true) {
		goto exit;
	}

	// PRINT_TIMER_HEADER
	//	kem_bench(KEM_BENCH_SECONDS);

exit:
	return (OK == true) ? EXIT_SUCCESS : EXIT_FAILURE;
}

