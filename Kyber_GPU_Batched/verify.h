
#ifndef VERIFY_H
#define VERIFY_H

#include <stdio.h>

int verify( unsigned char *a,  unsigned char *b, size_t len);

void cmov(unsigned char *r,  unsigned char *x, size_t len, unsigned char b);

#endif
