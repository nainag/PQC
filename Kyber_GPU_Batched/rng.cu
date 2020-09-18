//
//  rng.c
//
//  Created by Bassham, Lawrence E (Fed) on 8/29/17.
//  Copyright © 2017 Bassham, Lawrence E (Fed). All rights reserved.
// 
//  @Author: Arpan Jati
//  Adapted from NewHope Reference Codebase and Parallelized using CUDA
//  Modified to generate constant output for debugging. Do not use in actual application.
//  Updated: August 2019
//

#include <string.h>
#include "rng.h"

void
randombytes_init(unsigned char *entropy_input,
                 unsigned char *personalization_string,
                 int security_strength)
{
   
}

 __inline void delay(unsigned int count)
{
	while (count--) {}
}


int randombytes(unsigned char* x, unsigned long long xlen)
{ // Generation of "nbytes" of random values

	for (unsigned int i = 0; i < xlen; i++)
	{
		x[i] = i;
	}

	return 0;
}












