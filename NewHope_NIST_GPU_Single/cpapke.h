
// @Author: Naina Gupta
// Adapted from NewHope Reference Codebase and Parallelized using CUDA
// Updated : August 2019

#ifndef INDCPA_H
#define INDCPA_H

void print_data(char* text, unsigned char* data, int length);

void cpapke_keypair(unsigned char *pk, 
                    unsigned char *sk);

void cpapke_enc(unsigned char *c,
                unsigned char *m,
                unsigned char *pk,
                unsigned char *coins);

void cpapke_dec(unsigned char *m,
                unsigned char *c,
                unsigned char *sk);

#endif
