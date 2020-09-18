
// @Author: Naina Gupta
// Adapted from NewHope Reference Codebase and Parallelized using CUDA
// Updated : August 2019
// -------------------------------------------------------------
// CODE FOR PERFORMANCE COMPARISON. NOT FOR ACTUAL DEPLOYMENT
// -------------------------------------------------------------

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include "rng.h"
#include "api.h"

#include "cpapke.h"

#define NTESTS 1

int main()
{
	printf("NewHope-1024-CPA | CPU \n");

	unsigned char pk[ NEWHOPE_CPAKEM_PUBLICKEYBYTES];
	unsigned char sk[NEWHOPE_CPAKEM_SECRETKEYBYTES];

	unsigned char ct[NEWHOPE_CPAKEM_CIPHERTEXTBYTES];

	unsigned char msg1[NEWHOPE_SYMBYTES];
	unsigned char msg2[NEWHOPE_SYMBYTES];

	memset(msg2, 0, 32);

	unsigned char coins[NEWHOPE_SYMBYTES];

	randombytes(msg1, NEWHOPE_SYMBYTES);

	//print_data("msg1", msg1, NEWHOPE_SYMBYTES);

	randombytes(coins, NEWHOPE_SYMBYTES);

	//print_data("coins", coins, NEWHOPE_SYMBYTES);

	cpapke_keypair(pk, sk);

	//print_data("PK", pk, NEWHOPE_CPAKEM_PUBLICKEYBYTES);
	//print_data("SK", sk, NEWHOPE_CPAKEM_SECRETKEYBYTES);

	cpapke_enc(ct, msg1, pk, coins);

	//print_data("CT", ct, NEWHOPE_CPAKEM_CIPHERTEXTBYTES);

	print_data("msg1", msg1, NEWHOPE_SYMBYTES);

	cpapke_dec(msg2, ct, sk);

	//print_data("CT", ct, NEWHOPE_CPAKEM_CIPHERTEXTBYTES);

	print_data("msg2", msg2, NEWHOPE_SYMBYTES);


	if (memcmp(msg1, msg2, NEWHOPE_SYMBYTES)) {
		printf("ERROR : < MESSAGE VERIFICATION > !! \n");		
	}
	else
	{
		printf("Test passed !! \n");
	}
	printf("\n\nDONE. ");

	return 0;
}