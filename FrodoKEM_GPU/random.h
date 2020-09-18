#ifndef __RANDOM_H__
#define __RANDOM_H__


// Generate random bytes and output the result to random_array
int randombytes(unsigned char* random_array, unsigned int nbytes);
int randombytes_init(unsigned char* random_array, unsigned int nbytes, unsigned int start);

#endif
