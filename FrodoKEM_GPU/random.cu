/********************************************************************************************
* Hardware-based random number generation function using /dev/urandom

//  @Author: Arpan Jati
//  Adapted from FrodoKEM Reference Codebase and Parallelized using CUDA
//  Modified to generate constant output for debugging. Do not use in actual application.
//  Updated: August 2019
//
*********************************************************************************************/ 

#include "random.h"
#include <stdlib.h>
//#include <unistd.h>
#include <fcntl.h>
static int lock = -1;


static __inline void delay(unsigned int count)
{
	while (count--) {}
}


int randombytes_init(unsigned char* random_array, unsigned int nbytes, unsigned int start)
{ // Generation of "nbytes" of random values

	for (unsigned int i = 0; i < nbytes; i++)
	{
		random_array[i] = start++;
	}

	return 0;
}

int randombytes(unsigned char* random_array, unsigned int nbytes)
{ // Generation of "nbytes" of random values

	for (unsigned int i = 0; i < nbytes; i++)
	{
		random_array[i] = i;
	}

	return 0;
}