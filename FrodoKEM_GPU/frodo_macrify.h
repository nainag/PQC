/********************************************************************************************
* FrodoKEM: Learning with Errors Key Encapsulation
*
* Abstract: header for internal functions

// @Author: Naina Gupta
// Adapted from FrodoKEM Reference Codebase and Parallelized using CUDA
// Updated : August 2019

*********************************************************************************************/

#ifndef _FRODO_MACRIFY_H_
#define _FRODO_MACRIFY_H_

#include <stddef.h>
#include <stdint.h>
#include "config.h"

void frodo_pack(unsigned char *out, const size_t outlen, const uint16_t *in, const size_t inlen, const unsigned char lsb);
void frodo_unpack(uint16_t *out, const size_t outlen, const unsigned char *in, const size_t inlen, const unsigned char lsb);
void frodo_sample_n(uint16_t *s, const size_t n);
void clear_bytes(uint8_t *mem, size_t n);

int frodo_mul_add_as_plus_e_gpu(uint16_t *out_d, uint16_t *s_d, uint16_t *e_d, const uint8_t *seed_A, uint16_t *A_d);
int frodo_mul_add_sa_plus_e_gpu(uint16_t *out_d, uint16_t *s_d, uint16_t *e_d, const uint8_t *seed_A, uint16_t *A_d);

__global__ void frodo_sub_kernel(uint16_t *out,  uint16_t *a,  uint16_t *b);
__global__ void frodo_add_kernel(uint16_t *out,  uint16_t *a,  uint16_t *b);
__global__ void frodo_key_encode_kernel(uint16_t *out,  uint16_t *in);
__global__ void frodo_key_decode_kernel(uint16_t *out,  uint16_t *in);
__global__ void frodo_mul_bs_kernel(uint16_t *out,  uint16_t *b,  uint16_t *s);
__global__ void frodo_mul_add_sb_plus_e_kernel(uint16_t *out, uint16_t *b, uint16_t *s, uint16_t *e);

#endif
