
// @Author: Naina Gupta
// Adapted from NewHope Reference Codebase and Parallelized using CUDA
// Updated : August 2019

#include <stdio.h>
#include "api.h"
#include "poly.h"
#include "rng.h"
#include "fips202.h"
#include "ntt.h"

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

void print_data(char* text, unsigned char* data, int length)
{
	printf("%s\n", text);

	for (int i = 0; i < length; i++)
	{
		printf("%02X", data[i]);

		if ((i + 1) % 2 == 0)
		{
			printf(" ");
		}

		if ((i + 1) % 32 == 0)
		{
			printf("\n");
		}
	}

	printf("\n");
}

void print_poly(char* text, poly* poly)
{
	printf("%s\n", text);

	for (int i = 0; i < NEWHOPE_N; i++)
	{
		printf("%d ", poly->coeffs[i]);

		//if ((i + 1) % 2 == 0)
		//{
		//	printf(" ");
		//}

		if ((i + 1) % 16 == 0)
		{
			printf("\n");
		}
	}

	printf("\n");
}


/*************************************************
* Name:        encode_pk
*
* Description: Serialize the public key as concatenation of the
*              serialization of the polynomial pk and the public seed
*              used to generete the polynomial a.
*
* Arguments:   unsigned char *r:          pointer to the output serialized public key
*               poly *pk:            pointer to the input public-key polynomial
*               unsigned char *seed: pointer to the input public seed
**************************************************/
void encode_pk(unsigned char* r, poly* pk, unsigned char* seed)
{
	int i;
	poly_tobytes(r, pk);
	for (i = 0; i < NEWHOPE_SYMBYTES; i++)
		r[NEWHOPE_POLYBYTES + i] = seed[i];
}

/*************************************************
* Name:        decode_pk
*
* Description: De-serialize the public key; inverse of encode_pk
*
* Arguments:   poly *pk:               pointer to output public-key polynomial
*              unsigned char *seed:    pointer to output public seed
*               unsigned char *r: pointer to input byte array
**************************************************/
void decode_pk(poly* pk, unsigned char* seed, unsigned char* r)
{
	int i;
	poly_frombytes(pk, r);
	for (i = 0; i < NEWHOPE_SYMBYTES; i++)
		seed[i] = r[NEWHOPE_POLYBYTES + i];
}

/*************************************************
* Name:        encode_c
*
* Description: Serialize the ciphertext as concatenation of the
*              serialization of the polynomial b and serialization
*              of the compressed polynomial v
*
* Arguments:   - unsigned char *r: pointer to the output serialized ciphertext
*              -  poly *b:    pointer to the input polynomial b
*              -  poly *v:    pointer to the input polynomial v
**************************************************/
void encode_c(unsigned char* r, poly* b, poly* v)
{
	poly_tobytes(r, b);
	poly_compress(r + NEWHOPE_POLYBYTES, v);
}

/*************************************************
* Name:        decode_c
*
* Description: de-serialize the ciphertext; inverse of encode_c
*
* Arguments:   - poly *b:                pointer to output polynomial b
*              - poly *v:                pointer to output polynomial v
*              -  unsigned char *r: pointer to input byte array
**************************************************/
void decode_c(poly* b, poly* v, unsigned char* r)
{
	poly_frombytes(b, r);
	poly_decompress(v, r + NEWHOPE_POLYBYTES);
}

/*************************************************
* Name:        gen_a
*
* Description: Deterministically generate public polynomial a from seed
*
* Arguments:   - poly *a:                   pointer to output polynomial a
*              -  unsigned char *seed: pointer to input seed
**************************************************/
// void gen_a(poly* a,  unsigned char* seed)
//{
//	poly_uniform(a, seed);
//}
//


#include <stdlib.h>

static void HandleError2(cudaError_t err,
	const char* file,
	int line) {
	if (err != cudaSuccess) {
		printf("%s in %s at line %d\n", cudaGetErrorString(err),
			file, line);
		exit(EXIT_FAILURE);
	}
}

#define HANDLE_ERROR2( err ) (HandleError2( err, __FILE__, __LINE__ ))

#define TIMING_START \
{\
auto start = std::chrono::high_resolution_clock::now(); \
int cnt = 0; \
while (1)\
{

//gen_a(&ahat, publicseed);

#define TIMING_END(NAME)\
	auto finish = std::chrono::high_resolution_clock::now();\
	cnt++;\
	auto s_passed = std::chrono::duration_cast<std::chrono::microseconds>(finish - start).count();\
	if (s_passed > 2000000)\
	{\
		std::cout << "NAME = " << NAME <<" | TIME (us): " << ((float)s_passed / (float)cnt) << "exec time us\n";\
		break;\
	}\
}\
}


#include <chrono>
#include <iostream>

using namespace std;

/*************************************************
* Name:        cpapke_keypair
*
* Description: Generates public and private key
*              for the CPA public-key encryption scheme underlying
*              the NewHope KEMs
*
* Arguments:   - unsigned char *pk: pointer to output public key
*              - unsigned char *sk: pointer to output private key
**************************************************/
void cpapke_keypair(unsigned char* pk,
	unsigned char* sk)
{
	cudaEvent_t start, stop;

	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	poly* ahat, * ehat, * ahat_shat, * bhat, * shat;
	poly* shat_h, * bhat_h;

	shat_h = (poly*)(malloc(sizeof(poly)));
	bhat_h = (poly*)(malloc(sizeof(poly)));

	unsigned char z[2 * NEWHOPE_SYMBYTES];
	unsigned char* publicseed = z;
	unsigned char* noiseseed = z + NEWHOPE_SYMBYTES;

	// Choose which GPU to run on, change this on a multi-GPU system.

	HANDLE_ERROR2(cudaSetDevice(0));

	unsigned char* z_d;
	HANDLE_ERROR2(cudaMalloc((void**)&z_d, 2 * NEWHOPE_SYMBYTES));

	unsigned char* seed_d = z_d;
	unsigned char* noiseseed_d = (z_d + NEWHOPE_SYMBYTES);

	HANDLE_ERROR2(cudaMalloc((void**)&ahat, sizeof(poly)));
	HANDLE_ERROR2(cudaMalloc((void**)&ehat, sizeof(poly)));
	HANDLE_ERROR2(cudaMalloc((void**)&ahat_shat, sizeof(poly)));
	HANDLE_ERROR2(cudaMalloc((void**)&bhat, sizeof(poly)));
	HANDLE_ERROR2(cudaMalloc((void**)&shat, sizeof(poly)));

	unsigned char* sk_d;
	HANDLE_ERROR2(cudaMalloc((void**)&sk_d, NEWHOPE_CPAKEM_SECRETKEYBYTES));

	cudaEventRecord(start, 0);

	randombytes(z, NEWHOPE_SYMBYTES);

	shake256_host(z, 2 * NEWHOPE_SYMBYTES, z, NEWHOPE_SYMBYTES);

	HANDLE_ERROR2(cudaMemcpyToSymbol(noise_seed_const, noiseseed, NEWHOPE_SYMBYTES * sizeof(unsigned char), 0, cudaMemcpyHostToDevice));
	HANDLE_ERROR2(cudaMemcpyToSymbol(public_seed_const, publicseed, NEWHOPE_SYMBYTES * sizeof(unsigned char), 0, cudaMemcpyHostToDevice));

	poly_uniform_kernel_parallel_sh_comb << <4, 32 >> > (ahat);

	poly_sample_kernel_parallel << <8, 32 >> > (shat, 0);

	mul_coefficients_ntt_kernel << <16, 64 >> > (shat->coeffs);

	ntt_kernel_parallel_shared << <1, 512 >> > (shat->coeffs);

	poly_sample_kernel_parallel << < 8, 32 >> > (ehat, 1);

	mul_coefficients_ntt_kernel << < 16, 64 >> > (ehat->coeffs);

	ntt_kernel_parallel_shared << <1, 512 >> > (ehat->coeffs);

	poly_mul_pointwise_kernel << <16, 64 >> > (ahat_shat, shat, ahat);

	poly_add_kernel << <16, 64 >> > (bhat, ehat, ahat_shat);

	HANDLE_ERROR2(cudaMemcpy(shat_h, shat, sizeof(poly), cudaMemcpyDeviceToHost));
	HANDLE_ERROR2(cudaMemcpy(bhat_h, bhat, sizeof(poly), cudaMemcpyDeviceToHost));

	cudaEventRecord(stop, 0); 

	cudaEventSynchronize(stop);

	poly_tobytes(sk, shat_h);

	encode_pk(pk, bhat_h, publicseed);

	float time;
	cudaEventElapsedTime(&time, start, stop);

	fprintf(stdout, "Time Taken in 1000 KeyGen: %f ms\n", time);

	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	free(shat_h);
	free(bhat_h);

	cudaFree(z_d);

	cudaFree(ahat);
	cudaFree(ehat);
	cudaFree(ahat_shat);
	cudaFree(bhat);
	cudaFree(shat);
}

/*************************************************
* Name:        cpapke_enc
*
* Description: Encryption function of
*              the CPA public-key encryption scheme underlying
*              the NewHope KEMs
*
* Arguments:   - unsigned char *c:          pointer to output ciphertext
*              -  unsigned char *m:    pointer to input message (of length NEWHOPE_SYMBYTES bytes)
*              -  unsigned char *pk:   pointer to input public key
*              -  unsigned char *coin: pointer to input random coins used as seed
*                                           to deterministically generate all randomness
**************************************************/
void cpapke_enc(unsigned char* c,
	unsigned char* m,
	unsigned char* pk,
	unsigned char* coin)
{
	cudaEvent_t start, stop;

	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	poly* sprime, * eprime, * vprime, * ahat, * bhat, * eprimeprime, * uhat, * v;

	poly* uhat_h, * bhat_h, * vprime_h, * v_h;

	//poly* temp;
	//temp = (poly*)(malloc(sizeof(poly)));

	v_h = (poly*)(malloc(sizeof(poly)));
	uhat_h = (poly*)(malloc(sizeof(poly)));
	bhat_h = (poly*)(malloc(sizeof(poly)));
	vprime_h = (poly*)(malloc(sizeof(poly)));

	// Choose which GPU to run on, change this on a multi-GPU system.
	HANDLE_ERROR2(cudaSetDevice(0));

	HANDLE_ERROR2(cudaMalloc((void**)&sprime, sizeof(poly)));
	HANDLE_ERROR2(cudaMalloc((void**)&eprime, sizeof(poly)));
	HANDLE_ERROR2(cudaMalloc((void**)&vprime, sizeof(poly)));
	HANDLE_ERROR2(cudaMalloc((void**)&ahat, sizeof(poly)));
	HANDLE_ERROR2(cudaMalloc((void**)&bhat, sizeof(poly)));
	HANDLE_ERROR2(cudaMalloc((void**)&eprimeprime, sizeof(poly)));
	HANDLE_ERROR2(cudaMalloc((void**)&uhat, sizeof(poly)));
	HANDLE_ERROR2(cudaMalloc((void**)&v, sizeof(poly)));

	unsigned char* sk_d;
	HANDLE_ERROR2(cudaMalloc((void**)&sk_d, NEWHOPE_CPAKEM_SECRETKEYBYTES));

	//TIMING_START

	unsigned char publicseed[NEWHOPE_SYMBYTES];

	cudaEventRecord(start, 0);

	poly_frommsg(v_h, m);
	decode_pk(bhat_h, publicseed, pk);

	HANDLE_ERROR2(cudaMemcpyToSymbol(noise_seed_const, coin, NEWHOPE_SYMBYTES * sizeof(unsigned char), 0, cudaMemcpyHostToDevice));
	HANDLE_ERROR2(cudaMemcpyToSymbol(public_seed_const, publicseed, NEWHOPE_SYMBYTES * sizeof(unsigned char), 0, cudaMemcpyHostToDevice));
	HANDLE_ERROR2(cudaMemcpy(v, v_h, sizeof(poly), cudaMemcpyHostToDevice));
	HANDLE_ERROR2(cudaMemcpy(bhat, bhat_h, sizeof(poly), cudaMemcpyHostToDevice));

	poly_uniform_kernel_parallel_sh_comb << <4, 32 >> > (ahat);
	//HANDLE_ERROR2(cudaMemcpy(temp, ahat, sizeof(poly), cudaMemcpyDeviceToHost));

	poly_sample_kernel_parallel << <8, 32 >> > (sprime, 0); //8, 16 

	poly_sample_kernel_parallel << <8, 32 >> > (eprime, 1); //8, 16

	poly_sample_kernel_parallel << <8, 32 >> > (eprimeprime, 2); //8, 16

	mul_coefficients_ntt_kernel << <16, 64 >> > (sprime->coeffs);

	ntt_kernel_parallel_shared << <1, 512 >> > (sprime->coeffs);

	mul_coefficients_ntt_kernel << <16, 64 >> > (eprime->coeffs);

	ntt_kernel_parallel_shared << <1, 512 >> > (eprime->coeffs);

	poly_mul_pointwise_kernel << <16, 64 >> > (uhat, ahat, sprime);

	poly_add_kernel << <16, 64 >> > (uhat, uhat, eprime);

	poly_mul_pointwise_kernel << <16, 64 >> > (vprime, bhat, sprime);

	bitrev_vector_kernel << <16, 64 >> > (vprime->coeffs);

	invntt_kernel_parallel_shared << <1, 512 >> > (vprime->coeffs);

	mul_coefficients_invntt_kernel << <16, 64 >> > (vprime->coeffs);

	poly_add_kernel << <16, 64 >> > (vprime, vprime, eprimeprime);

	poly_add_kernel << <16, 64 >> > (vprime, vprime, v);

	HANDLE_ERROR2(cudaMemcpy(uhat_h, uhat, sizeof(poly), cudaMemcpyDeviceToHost));
	HANDLE_ERROR2(cudaMemcpy(vprime_h, vprime, sizeof(poly), cudaMemcpyDeviceToHost));

	encode_c(c, uhat_h, vprime_h);

	cudaEventRecord(stop, 0);

	cudaEventSynchronize(stop);

	float time;
	cudaEventElapsedTime(&time, start, stop);

	fprintf(stdout, "Time Taken in cpapke_enc: %f ms\n", time);

	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	free(uhat_h);
	free(bhat_h);
	free(vprime_h);
	free(v_h);

	cudaFree(sprime);
	cudaFree(eprime);
	cudaFree(vprime);
	cudaFree(ahat);
	cudaFree(bhat);
	cudaFree(eprimeprime);
	cudaFree(uhat);
	cudaFree(v);
}

/*************************************************
* Name:        cpapke_dec
*
* Description: Decryption function of
*              the CPA public-key encryption scheme underlying
*              the NewHope KEMs
*
* Arguments:   - unsigned char *m:        pointer to output decrypted message
*              -  unsigned char *c:  pointer to input ciphertext
*              -  unsigned char *sk: pointer to input secret key
**************************************************/
void cpapke_dec(unsigned char* m,
	unsigned char* c,
	unsigned char* sk)
{
	cudaEvent_t start, stop;

	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	poly* vprime, * uhat, * tmp, * shat;
	poly* vprime_h, * uhat_h, * tmp_h, * shat_h;

	vprime_h = (poly*)(malloc(sizeof(poly)));
	uhat_h = (poly*)(malloc(sizeof(poly)));
	tmp_h = (poly*)(malloc(sizeof(poly)));
	shat_h = (poly*)(malloc(sizeof(poly)));

	// Choose which GPU to run on, change this on a multi-GPU system.
	HANDLE_ERROR2(cudaSetDevice(0));

	HANDLE_ERROR2(cudaMalloc((void**)&vprime, sizeof(poly)));
	HANDLE_ERROR2(cudaMalloc((void**)&uhat, sizeof(poly)));
	HANDLE_ERROR2(cudaMalloc((void**)&tmp, sizeof(poly)));
	HANDLE_ERROR2(cudaMalloc((void**)&shat, sizeof(poly)));

	// TIMING_START

	cudaEventRecord(start, 0);

	poly_frombytes(shat_h, sk);
	decode_c(uhat_h, vprime_h, c);

	HANDLE_ERROR2(cudaMemcpy(shat, shat_h, sizeof(poly), cudaMemcpyHostToDevice));
	HANDLE_ERROR2(cudaMemcpy(vprime, vprime_h, sizeof(poly), cudaMemcpyHostToDevice));
	HANDLE_ERROR2(cudaMemcpy(uhat, uhat_h, sizeof(poly), cudaMemcpyHostToDevice));

	poly_mul_pointwise_kernel << <16, 64 >> > (tmp, shat, uhat);

	bitrev_vector_kernel << <16, 64 >> > (tmp->coeffs);
	invntt_kernel_parallel_shared << <1, 512 >> > (tmp->coeffs);

	mul_coefficients_invntt_kernel << <16, 64 >> > (tmp->coeffs);

	poly_sub_kernel << <16, 64 >> > (tmp, tmp, vprime);

	HANDLE_ERROR2(cudaMemcpy(tmp_h, tmp, sizeof(poly), cudaMemcpyDeviceToHost));

	poly_tomsg(m, tmp_h);

	cudaEventRecord(stop, 0);

	cudaEventSynchronize(stop);

	float time;
	cudaEventElapsedTime(&time, start, stop);

	fprintf(stdout, "Time Taken in cpapke_dec: %f ms\n", time);

	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	free(vprime_h);
	free(uhat_h);
	free(tmp_h);
	free(shat_h);

	cudaFree(vprime);
	cudaFree(uhat);
	cudaFree(tmp);
	cudaFree(shat);

}
