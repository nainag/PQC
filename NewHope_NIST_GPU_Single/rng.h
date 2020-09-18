//
//  rng.h
//
//  Created by Bassham, Lawrence E (Fed) on 8/29/17.
//  Copyright © 2017 Bassham, Lawrence E (Fed). All rights reserved.
//
//  @Author: Arpan Jati
//  Adapted from NewHope Reference Codebase and Parallelized using CUDA// 
//  Modified to generate constant output for debugging. Do not use in actual application.
//  Updated: August 2019

#ifndef rng_h
#define rng_h

#include <stdio.h>

#define RNG_SUCCESS      0
#define RNG_BAD_MAXLEN  -1
#define RNG_BAD_OUTBUF  -2
#define RNG_BAD_REQ_LEN -3

void InitializeRandomProviders();

int randombytes(unsigned char* x, unsigned long xlen);

int randombytes_real(unsigned char* pbBuffer, unsigned long dwLength);


#endif /* rng_h */
