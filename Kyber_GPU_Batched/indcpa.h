
// @Author: Arpan Jati
// Adapted from NewHope Reference Codebase and Parallelized using CUDA
// Updated : August 2019

#ifndef INDCPA_H
#define INDCPA_H

#include "main.h"

void HandleError(cudaError_t err, const char* file, int line);

#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ ))


void indcpa_keypair(int COUNT, poly_set4* ps, unsigned char *pk,
                    unsigned char *sk, unsigned char* rng_buf, cudaStream_t stream);

void indcpa_enc(int COUNT, poly_set4* ps, unsigned char *c,
                unsigned char *m,
                unsigned char *pk,
                unsigned char *coins, cudaStream_t stream);

void indcpa_dec(int COUNT, poly_set4* ps, unsigned char *m,
                unsigned char *c,
                unsigned char *sk, cudaStream_t stream);

void print_data(const char* text, unsigned char* data, int length);

#endif
