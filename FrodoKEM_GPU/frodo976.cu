/********************************************************************************************
* FrodoKEM: Learning with Errors Key Encapsulation
*
* Abstract: functions for FrodoKEM-976
*           Instantiates "frodo_macrify.c" with the necessary matrix arithmetic functions

// @Author: Naina Gupta
// Adapted from FrodoKEM Reference Codebase and Parallelized using CUDA
// Updated : August 2019

*********************************************************************************************/

#include "api.h"
#include "params.h"
#include "frodo_macrify.h"

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

// CDF table
__constant__ uint16_t CDF_TABLE[11] = { 5638, 15915, 23689, 28571, 31116, 32217, 32613, 32731, 32760, 32766, 32767 };


/********************************************************************************************
* FrodoKEM: Learning with Errors Key Encapsulation
*
* Abstract: Key Encapsulation Mechanism (KEM) based on Frodo
*********************************************************************************************/

#include <stdio.h>
#include <string.h>
#include "fips202.h"
#include "random.h"

__global__ void frodo_sample_n_Kernel(uint16_t *s, const size_t n);


void print_data(const char* text, unsigned char* data, int length)
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


__device__ void print_data_d(const char* text, unsigned char* data, int length)
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


int time_n = 1;

int crypto_kem_keypair_gpu(unsigned char* pk, unsigned char* sk)
{ // FrodoKEM's key generation
  // Outputs: public key pk (               BYTES_SEED_A + (PARAMS_LOGQ*PARAMS_N*PARAMS_NBAR)/8 bytes)
  //          secret key sk (CRYPTO_BYTES + BYTES_SEED_A + (PARAMS_LOGQ*PARAMS_N*PARAMS_NBAR)/8 + 2*PARAMS_N*PARAMS_NBAR + BYTES_PKHASH bytes)

	cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeEightByte);

	uint8_t *pk_seedA = &pk[0];
	uint8_t *pk_b = &pk[BYTES_SEED_A];
	uint8_t *sk_s = &sk[0];
	uint8_t *sk_pk = &sk[CRYPTO_BYTES];
	uint8_t *sk_S = &sk[CRYPTO_BYTES + CRYPTO_PUBLICKEYBYTES];
	uint8_t *sk_pkh = &sk[CRYPTO_BYTES + CRYPTO_PUBLICKEYBYTES + 2 * PARAMS_N*PARAMS_NBAR];
	uint16_t B[PARAMS_N*PARAMS_NBAR] = { 0 };
	uint8_t randomness_s[CRYPTO_BYTES];						  // contains secret data
	uint8_t randomness_z[BYTES_SEED_A];

	uint8_t shake_input_seedSE_h[1 + CRYPTO_BYTES];           // contains secret data

	uint16_t *s_d = 0;										
	uint16_t *e_d = 0;
	uint16_t* A_d = 0;
	uint16_t* out_d = 0;
	uint8_t *randomness_seedSE_d = 0;

	cudaError_t cudaStatus;
	cudaEvent_t start, stop;
	float time;

	// Choose which GPU to run on, change this on a multi-GPU system.
	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");

	}

	cudaStatus = cudaMalloc((void**)&randomness_seedSE_d, (1 + CRYPTO_BYTES) * sizeof(uint8_t));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
	}

	cudaStatus = cudaMalloc((void**)&s_d, (2 * PARAMS_N*PARAMS_NBAR + 24) * sizeof(uint16_t));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
	}

	e_d = (uint16_t *)&s_d[PARAMS_N*PARAMS_NBAR];

	cudaStatus = cudaMalloc(&out_d, PARAMS_NBAR * PARAMS_N * sizeof(uint16_t));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
	}

	cudaStatus = cudaMalloc(&A_d, PARAMS_N * PARAMS_N * sizeof(uint16_t));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
	}

	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	cudaEventRecord(start, 0);


	// Generate the secret value s, the seed for S and E, and the seed for the seed for A. Add seed_A to the public key

	for (int t = 0; t < time_n; t++)
	{

		randombytes_init(randomness_z, BYTES_SEED_A, 2 * CRYPTO_BYTES);
		shake(pk_seedA, BYTES_SEED_A, randomness_z, BYTES_SEED_A);

		randombytes_init(randomness_s, CRYPTO_BYTES, 0);

		randombytes_init(&shake_input_seedSE_h[1], CRYPTO_BYTES, CRYPTO_BYTES);

		// later this can be ignored using cuda random number generator
		cudaStatus = cudaMemcpy(&randomness_seedSE_d[1], &shake_input_seedSE_h[1], (CRYPTO_BYTES) * sizeof(uint8_t), cudaMemcpyHostToDevice);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMemcpy failed!");
		}

		//shake256_kernel1 << <1, 1 >> > ((uint8_t*)(s_d), 2 * PARAMS_N * PARAMS_NBAR * sizeof(uint16_t), randomness_seedSE_d, 1 + CRYPTO_BYTES, 0x5F);

		shake256_kernel_single << <7, 32 >> > ((uint8_t*)(s_d), (2 * PARAMS_N*PARAMS_NBAR) * sizeof(uint16_t),
			randomness_seedSE_d, 1 + CRYPTO_BYTES, 0x5F);

		frodo_sample_n_Kernel << <61, 128 >> > (s_d, PARAMS_N*PARAMS_NBAR);

		frodo_sample_n_Kernel << <61, 128 >> > (e_d, PARAMS_N*PARAMS_NBAR);

		frodo_mul_add_as_plus_e_gpu(out_d, s_d, e_d, pk, A_d);

		cudaStatus = cudaGetLastError();
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "frodo_mul_add_as_plus_e launch failed: %s\n", cudaGetErrorString(cudaStatus));
		}

		cudaStatus = cudaMemcpy(B, out_d, PARAMS_NBAR * PARAMS_N * sizeof(uint16_t), cudaMemcpyDeviceToHost);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMemcpy failed! %d", t);
		}

		cudaStatus = cudaMemcpy(sk_S, s_d, PARAMS_N*PARAMS_NBAR * sizeof(uint16_t), cudaMemcpyDeviceToHost);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMemcpy failed!");
		}

		// Encode the second part of the public key
		frodo_pack(pk_b, CRYPTO_PUBLICKEYBYTES - BYTES_SEED_A, B, PARAMS_N*PARAMS_NBAR, PARAMS_LOGQ);

		// Add s, pk and S to the secret key
		memcpy(sk_s, randomness_s, CRYPTO_BYTES);
		memcpy(sk_pk, pk, CRYPTO_PUBLICKEYBYTES);

		// Add H(pk) to the secret key
		shake(sk_pkh, BYTES_PKHASH, pk, CRYPTO_PUBLICKEYBYTES);
	}

	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);

	cudaEventElapsedTime(&time, start, stop);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	fprintf(stdout, "Time Taken in KeyGen x %d: %f ms\n", time_n, time);

	cudaFree(A_d);
	cudaFree(s_d);
	cudaFree(e_d);
	cudaFree(out_d);
	cudaFree(randomness_seedSE_d);

	// Cleanup:
	clear_bytes(shake_input_seedSE_h, 1 + CRYPTO_BYTES);
	return 0;
}

int crypto_kem_enc(unsigned char *ct, unsigned char *ss, const unsigned char *pk)
{ // FrodoKEM's key encapsulation
	const uint8_t *pk_seedA = &pk[0];
	const uint8_t *pk_b = &pk[BYTES_SEED_A];
	uint8_t *ct_c1 = &ct[0];
	uint8_t *ct_c2 = &ct[(PARAMS_LOGQ*PARAMS_N*PARAMS_NBAR) / 8];
	uint16_t B[PARAMS_N*PARAMS_NBAR] = { 0 };
	uint16_t V[PARAMS_NBAR*PARAMS_NBAR] = { 0 };                 // contains secret data
	uint16_t C[PARAMS_NBAR*PARAMS_NBAR] = { 0 };
	uint16_t Bp[PARAMS_N*PARAMS_NBAR] = { 0 };
	uint16_t Sp[(2 * PARAMS_N + PARAMS_NBAR)*PARAMS_NBAR] = { 0 };  // contains secret data
	uint16_t *Ep = (uint16_t *)&Sp[PARAMS_N*PARAMS_NBAR];     // contains secret data
	uint16_t *Epp = (uint16_t *)&Sp[2 * PARAMS_N*PARAMS_NBAR];  // contains secret data
	uint8_t G2in[BYTES_PKHASH + BYTES_MU];                    // contains secret data via mu
	uint8_t *pkh = &G2in[0];
	uint8_t *mu = &G2in[BYTES_PKHASH];                        // contains secret data
	uint8_t G2out[2 * CRYPTO_BYTES];                            // contains secret data
	uint8_t *seedSE = &G2out[0];                              // contains secret data
	uint8_t *k = &G2out[CRYPTO_BYTES];                        // contains secret data
	uint8_t Fin[CRYPTO_CIPHERTEXTBYTES + CRYPTO_BYTES];       // contains secret data via Fin_k
	uint8_t *Fin_ct = &Fin[0];
	uint8_t *Fin_k = &Fin[CRYPTO_CIPHERTEXTBYTES];            // contains secret data
	uint8_t shake_input_seedSE_h[1 + CRYPTO_BYTES];             // contains secret data

	uint16_t *sp_d = 0;
	uint16_t *ep_d = 0;
	uint16_t *epp_d = 0;
	uint8_t *randomness_seedSE_d = 0;
	uint16_t *V_d = 0;
	uint16_t *C_d = 0;
	uint16_t *B_d = 0;
	uint8_t *mu_d = 0;

	cudaError_t cudaStatus;
	cudaEvent_t start, stop;
	float time;

	uint16_t *A_d = 0;

	uint16_t *out_d = 0;

	cudaStatus = cudaMalloc((void**)&randomness_seedSE_d, (1 + CRYPTO_BYTES) * sizeof(uint8_t));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "KEM-ENC: cudaMalloc failed! 1");
	}

	cudaStatus = cudaMalloc((void**)&sp_d, (2 * PARAMS_N + PARAMS_NBAR)*PARAMS_NBAR * sizeof(uint16_t));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "KEM-ENC: cudaMalloc failed! 2");
	}

	ep_d = (uint16_t *)&sp_d[PARAMS_N*PARAMS_NBAR];
	epp_d = (uint16_t *)&sp_d[2 * PARAMS_N*PARAMS_NBAR];

	cudaStatus = cudaMalloc(&out_d, PARAMS_NBAR * PARAMS_N * sizeof(uint16_t));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "KEM-ENC: cudaMalloc failed! 3");
	}

	cudaStatus = cudaMalloc(&A_d, PARAMS_N * PARAMS_N * sizeof(uint16_t));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "KEM-ENC: cudaMalloc failed!");
	}

	cudaStatus = cudaMalloc(&V_d, PARAMS_NBAR * PARAMS_NBAR * sizeof(uint16_t));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "KEM-ENC: cudaMalloc failed!");
	}

	cudaStatus = cudaMalloc(&C_d, PARAMS_NBAR * PARAMS_NBAR * sizeof(uint16_t));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "KEM-ENC: cudaMalloc failed!");
	}

	cudaStatus = cudaMalloc(&B_d, PARAMS_N * PARAMS_NBAR * sizeof(uint16_t));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "KEM-ENC: cudaMalloc failed!");
	}

	cudaStatus = cudaMalloc(&mu_d, BYTES_MU * sizeof(uint8_t));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "KEM-ENC: cudaMalloc failed!");
	}

	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	cudaEventRecord(start, 0);

	for (int t = 0; t < time_n; t++)
	{
		// pkh <- G_1(pk), generate random mu, compute (seedSE || k) = G_2(pkh || mu)
		shake(pkh, BYTES_PKHASH, pk, CRYPTO_PUBLICKEYBYTES);
		randombytes(mu, BYTES_MU);

		shake(G2out, CRYPTO_BYTES + CRYPTO_BYTES, G2in, BYTES_PKHASH + BYTES_MU);

		frodo_unpack(B, PARAMS_N*PARAMS_NBAR, pk_b, CRYPTO_PUBLICKEYBYTES - BYTES_SEED_A, PARAMS_LOGQ);

		// Generate Sp and Ep, and compute Bp = Sp*A + Ep. Generate A on-the-fly
		memcpy(&shake_input_seedSE_h[1], seedSE, CRYPTO_BYTES);

		cudaStatus = cudaMemcpy(&randomness_seedSE_d[1], &shake_input_seedSE_h[1], (CRYPTO_BYTES) * sizeof(uint8_t), cudaMemcpyHostToDevice);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMemcpy failed!");
		}

		shake256_kernel_single << <7, 32 >> > ((uint8_t*)(sp_d), (2 * PARAMS_N*PARAMS_NBAR + 64) * sizeof(uint16_t),
			randomness_seedSE_d, 1 + CRYPTO_BYTES, 0x96);

		//shake256_kernel1 << <1, 1 >> > ((uint8_t*)(sp_d), (2 * PARAMS_N * PARAMS_NBAR + 64) * sizeof(uint16_t), randomness_seedSE_d, 1 + CRYPTO_BYTES, 0x96);

		frodo_sample_n_Kernel << <61, 128 >> > (sp_d, PARAMS_N*PARAMS_NBAR);
		
		frodo_sample_n_Kernel << <61, 128 >> > (ep_d, PARAMS_N*PARAMS_NBAR);
		
		frodo_mul_add_sa_plus_e_gpu(out_d, sp_d, ep_d, pk_seedA, A_d);

		frodo_sample_n_Kernel << <1, 64 >> > (epp_d, PARAMS_NBAR*PARAMS_NBAR);

		cudaStatus = cudaMemcpy(B_d, B, PARAMS_N*PARAMS_NBAR * sizeof(uint16_t), cudaMemcpyHostToDevice);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMemcpy failed!");
		}

		cudaStatus = cudaMemcpy(mu_d, mu, BYTES_MU * sizeof(uint8_t), cudaMemcpyHostToDevice);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMemcpy failed!");
		}

		frodo_mul_add_sb_plus_e_kernel<<<8,8>>>(V_d, B_d, sp_d, epp_d);

		// Encode mu, and compute C = V + enc(mu) (mod q)
		frodo_key_encode_kernel << <1, 8 >> > (C_d, (uint16_t*)mu_d);

		frodo_add_kernel << <8, 8 >> > (C_d, V_d, C_d);

		cudaStatus = cudaMemcpy(V, V_d, PARAMS_NBAR * PARAMS_NBAR * sizeof(uint16_t), cudaMemcpyDeviceToHost);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "KEM-ENC: cudaMemcpy failed!");
		}

		cudaStatus = cudaMemcpy(Bp, out_d, PARAMS_NBAR * PARAMS_N * sizeof(uint16_t), cudaMemcpyDeviceToHost);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "KEM-ENC: cudaMemcpy failed!");
		}

		cudaStatus = cudaMemcpy(C, C_d, PARAMS_NBAR*PARAMS_NBAR * sizeof(uint16_t), cudaMemcpyDeviceToHost);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "KEM-ENC: cudaMemcpy failed!");
		}

		// Generate Epp, and compute V = Sp*B + Epp
		
		frodo_pack(ct_c1, (PARAMS_LOGQ*PARAMS_N*PARAMS_NBAR) / 8, Bp, PARAMS_N*PARAMS_NBAR, PARAMS_LOGQ);
		frodo_pack(ct_c2, (PARAMS_LOGQ*PARAMS_NBAR*PARAMS_NBAR) / 8, C, PARAMS_NBAR*PARAMS_NBAR, PARAMS_LOGQ);

		// Compute ss = F(ct||KK)
		memcpy(Fin_ct, ct, CRYPTO_CIPHERTEXTBYTES);
		memcpy(Fin_k, k, CRYPTO_BYTES);

		shake(ss, CRYPTO_BYTES, Fin, CRYPTO_CIPHERTEXTBYTES + CRYPTO_BYTES);

	}

	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);

	cudaEventElapsedTime(&time, start, stop);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	fprintf(stdout, "KEM-ENC: Time Taken: %f ms\n", time);

	cudaFree(A_d);
	cudaFree(sp_d);
	cudaFree(out_d);
	cudaFree(randomness_seedSE_d);
	cudaFree(V_d);
	cudaFree(C_d);
	cudaFree(B_d);
	cudaFree(mu_d);

	// Cleanup:
	clear_bytes((uint8_t *)V, PARAMS_NBAR*PARAMS_NBAR * sizeof(uint16_t));
	clear_bytes((uint8_t *)Sp, PARAMS_N*PARAMS_NBAR * sizeof(uint16_t));
	clear_bytes((uint8_t *)Ep, PARAMS_N*PARAMS_NBAR * sizeof(uint16_t));
	clear_bytes((uint8_t *)Epp, PARAMS_NBAR*PARAMS_NBAR * sizeof(uint16_t));
	clear_bytes(mu, BYTES_MU);
	clear_bytes(G2out, 2 * CRYPTO_BYTES);
	clear_bytes(Fin_k, CRYPTO_BYTES);
	clear_bytes(shake_input_seedSE_h, 1 + CRYPTO_BYTES);
	return 0;
}


int crypto_kem_dec(unsigned char *ss, const unsigned char *ct, const unsigned char *sk)
{ // FrodoKEM's key decapsulation
	uint16_t B[PARAMS_N*PARAMS_NBAR] = { 0 };
	uint16_t Bp[PARAMS_N*PARAMS_NBAR] = { 0 };
	uint16_t W[PARAMS_NBAR*PARAMS_NBAR] = { 0 };                // contains secret data
	uint16_t C[PARAMS_NBAR*PARAMS_NBAR] = { 0 };
	uint16_t CC[PARAMS_NBAR*PARAMS_NBAR] = { 0 };
	uint16_t BBp[PARAMS_N*PARAMS_NBAR] = { 0 };
	uint16_t Sp[(2 * PARAMS_N + PARAMS_NBAR)*PARAMS_NBAR] = { 0 };  // contains secret data
	uint16_t *Ep = (uint16_t *)&Sp[PARAMS_N*PARAMS_NBAR];     // contains secret data
	uint16_t *Epp = (uint16_t *)&Sp[2 * PARAMS_N*PARAMS_NBAR];  // contains secret data
	const uint8_t *ct_c1 = &ct[0];
	const uint8_t *ct_c2 = &ct[(PARAMS_LOGQ*PARAMS_N*PARAMS_NBAR) / 8];
	const uint8_t *sk_s = &sk[0];
	const uint8_t *sk_pk = &sk[CRYPTO_BYTES];
	const uint16_t *sk_S = (uint16_t *)&sk[CRYPTO_BYTES + CRYPTO_PUBLICKEYBYTES];
	const uint8_t *sk_pkh = &sk[CRYPTO_BYTES + CRYPTO_PUBLICKEYBYTES + 2 * PARAMS_N*PARAMS_NBAR];
	const uint8_t *pk_seedA = &sk_pk[0];
	const uint8_t *pk_b = &sk_pk[BYTES_SEED_A];
	uint8_t G2in[BYTES_PKHASH + BYTES_MU];                   // contains secret data via muprime
	uint8_t *pkh = &G2in[0];
	uint8_t *muprime = &G2in[BYTES_PKHASH];                  // contains secret data
	uint8_t G2out[2 * CRYPTO_BYTES];                           // contains secret data
	uint8_t *seedSEprime = &G2out[0];                        // contains secret data
	uint8_t *kprime = &G2out[CRYPTO_BYTES];                  // contains secret data
	uint8_t Fin[CRYPTO_CIPHERTEXTBYTES + CRYPTO_BYTES];      // contains secret data via Fin_k
	uint8_t *Fin_ct = &Fin[0];
	uint8_t *Fin_k = &Fin[CRYPTO_CIPHERTEXTBYTES];           // contains secret data
	uint8_t shake_input_seedSEprime[1 + CRYPTO_BYTES];       // contains secret data

	uint16_t *Bp_d = 0;
	uint16_t *C_d = 0;
	uint16_t *W_d = 0;
	uint16_t *sk_S_d = 0;
	uint8_t *G2in_d = 0;
	uint8_t *pkh_d = 0;
	uint8_t *muprime_d = 0;
	uint8_t *G2out_d = 0;
	uint8_t *seedSEprime_d = 0;
	uint8_t *kprime_d = 0;
	uint16_t *Sp_d = 0;
	uint16_t *Ep_d = 0;
	uint16_t *Epp_d = 0;
	uint8_t *shake_input_seedSEprime_d = 0;
	uint16_t *A_d = 0;
	uint16_t *BBp_d = 0; //
	uint16_t *B_d = 0;
	uint16_t *CC_d = 0;


	cudaError_t cudaStatus;
	cudaEvent_t start, stop;
	float time;

	cudaStatus = cudaMalloc((void**)&Bp_d, PARAMS_N*PARAMS_NBAR * sizeof(uint16_t));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "KEM-DEC: cudaMalloc failed!");
	}

	cudaStatus = cudaMalloc((void**)&sk_S_d, PARAMS_N*PARAMS_NBAR * sizeof(uint16_t));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "KEM-DEC: cudaMalloc failed!");
	}

	cudaStatus = cudaMalloc((void**)&C_d, PARAMS_NBAR*PARAMS_NBAR * sizeof(uint16_t));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "KEM-DEC: cudaMalloc failed!");
	}

	cudaStatus = cudaMalloc((void**)&W_d, PARAMS_NBAR*PARAMS_NBAR * sizeof(uint16_t));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "KEM-DEC: cudaMalloc failed!");
	}

	cudaStatus = cudaMalloc((void**)&G2in_d, BYTES_PKHASH + BYTES_MU * sizeof(uint8_t));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "KEM-DEC: cudaMalloc failed!");
	}

	cudaStatus = cudaMalloc((void**)&G2out_d, 2 * CRYPTO_BYTES * sizeof(uint8_t));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "KEM-DEC: cudaMalloc failed!");
	}

	cudaStatus = cudaMalloc((void**)&shake_input_seedSEprime_d, (1 + CRYPTO_BYTES) * sizeof(uint8_t));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "KEM-DEC: cudaMalloc failed!");
	}

	cudaStatus = cudaMalloc((void**)&Sp_d, (2 * PARAMS_N + PARAMS_NBAR)*PARAMS_NBAR * sizeof(uint16_t));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "KEM-DEC: cudaMalloc failed!");
	}

	cudaStatus = cudaMalloc(&BBp_d, PARAMS_NBAR * PARAMS_N * sizeof(uint16_t));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "KEM-DEC: cudaMalloc failed!");
	}

	cudaStatus = cudaMalloc(&A_d, PARAMS_N * PARAMS_N * sizeof(uint16_t));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "KEM-ENC: cudaMalloc failed!");
	}

	cudaStatus = cudaMalloc(&B_d, PARAMS_N*PARAMS_NBAR * sizeof(uint16_t));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "KEM-DEC: cudaMalloc failed!");
	}

	cudaStatus = cudaMalloc(&CC_d, PARAMS_NBAR*PARAMS_NBAR * sizeof(uint16_t));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "KEM-DEC: cudaMalloc failed!");
	}

	pkh_d = (uint8_t *)&G2in_d[0];
	muprime_d = (uint8_t *)&G2in_d[BYTES_PKHASH];
	seedSEprime_d = (uint8_t *)&G2out_d[0];                        // contains secret data
	kprime_d = (uint8_t *)&G2out_d[CRYPTO_BYTES];
	Ep_d = (uint16_t *)&Sp_d[PARAMS_N*PARAMS_NBAR];
	Epp_d = (uint16_t *)&Sp_d[2 * PARAMS_N*PARAMS_NBAR];


	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	cudaEventRecord(start, 0);


	for (int t = 0; t < time_n; t++)
	{
		// Compute W = C - Bp*S (mod q), and decode the randomness mu
		frodo_unpack(Bp, PARAMS_N*PARAMS_NBAR, ct_c1, (PARAMS_LOGQ*PARAMS_N*PARAMS_NBAR) / 8, PARAMS_LOGQ);
		frodo_unpack(C, PARAMS_NBAR*PARAMS_NBAR, ct_c2, (PARAMS_LOGQ*PARAMS_NBAR*PARAMS_NBAR) / 8, PARAMS_LOGQ);
		frodo_unpack(B, PARAMS_N*PARAMS_NBAR, pk_b, CRYPTO_PUBLICKEYBYTES - BYTES_SEED_A, PARAMS_LOGQ);

		cudaStatus = cudaMemcpy(Bp_d, Bp, PARAMS_N*PARAMS_NBAR * sizeof(uint16_t), cudaMemcpyHostToDevice);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMemcpy failed!");
		}

		cudaStatus = cudaMemcpy(C_d, C, PARAMS_NBAR*PARAMS_NBAR * sizeof(uint16_t), cudaMemcpyHostToDevice);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMemcpy failed!");
		}

		cudaStatus = cudaMemcpy(sk_S_d, sk_S, PARAMS_N*PARAMS_NBAR * sizeof(uint16_t), cudaMemcpyHostToDevice);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMemcpy failed!");
		}

		frodo_mul_bs_kernel << <8, 8 >> > (W_d, Bp_d, sk_S_d);

		frodo_sub_kernel<<<8,8>>>(W_d, C_d, W_d);

		frodo_key_decode_kernel<<<1,8>>>((uint16_t*)muprime_d, W_d);

		// Generate (seedSE' || k') = G_2(pkh || mu')
		cudaStatus = cudaMemcpy(pkh_d, sk_pkh, BYTES_PKHASH * sizeof(uint8_t), cudaMemcpyHostToDevice);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMemcpy failed!");
		}

		shake256_kernel << <1, 1 >> > (G2out_d, CRYPTO_BYTES + CRYPTO_BYTES, G2in_d, BYTES_PKHASH + BYTES_MU);

		cudaStatus = cudaMemcpy(G2out, G2out_d, CRYPTO_BYTES + CRYPTO_BYTES, cudaMemcpyDeviceToHost);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMemcpy failed!");
		}


		// Generate Sp and Ep, and compute BBp = Sp*A + Ep. Generate A on-the-fly
		cudaStatus = cudaMemcpy(&shake_input_seedSEprime_d[1], seedSEprime_d, (CRYPTO_BYTES) * sizeof(uint8_t), cudaMemcpyDeviceToDevice);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMemcpy failed!");
		}

		shake256_kernel_single << <7, 32 >> > ((uint8_t*)(Sp_d), (2 * PARAMS_N*PARAMS_NBAR + 64) * sizeof(uint16_t),
			shake_input_seedSEprime_d, 1 + CRYPTO_BYTES, 0x96);

		//shake256_kernel1 << <1, 1 >> > ((uint8_t*)(Sp_d), (2 * PARAMS_N * PARAMS_NBAR + 64) * sizeof(uint16_t), shake_input_seedSEprime_d, 1 + CRYPTO_BYTES, 0x96);

		frodo_sample_n_Kernel << <61, 128 >> > (Sp_d, PARAMS_N*PARAMS_NBAR);

		frodo_sample_n_Kernel << <61, 128 >> > (Ep_d, PARAMS_N*PARAMS_NBAR);

		frodo_mul_add_sa_plus_e_gpu(BBp_d, Sp_d, Ep_d, pk_seedA, A_d);

		// Generate Epp, and compute W = Sp*B + Epp
		frodo_sample_n_Kernel << <1, 64 >> > (Epp_d, PARAMS_NBAR*PARAMS_NBAR);
		
		cudaStatus = cudaMemcpy(B_d, B, PARAMS_N*PARAMS_NBAR * sizeof(uint16_t), cudaMemcpyHostToDevice);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMemcpy failed!");
		}

		frodo_mul_add_sb_plus_e_kernel << <8, 8 >> > (W_d, B_d, Sp_d, Epp_d);

		// Encode mu, and compute CC = W + enc(mu') (mod q)
		frodo_key_encode_kernel<<<1,8>>>(CC_d, (uint16_t*)muprime_d);

		frodo_add_kernel<<<8,8>>>(CC_d, W_d, CC_d);

		cudaStatus = cudaMemcpy(BBp, BBp_d, PARAMS_N*PARAMS_NBAR * sizeof(uint16_t), cudaMemcpyDeviceToHost);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMemcpy failed!");
		}

		cudaStatus = cudaMemcpy(CC, CC_d, PARAMS_NBAR*PARAMS_NBAR * sizeof(uint16_t), cudaMemcpyDeviceToHost);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMemcpy failed!");
		}

		cudaStatus = cudaMemcpy(kprime, kprime_d, CRYPTO_BYTES * sizeof(uint8_t), cudaMemcpyDeviceToHost);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMemcpy failed!");
		}

		// Prepare input to F
		memcpy(Fin_ct, ct, CRYPTO_CIPHERTEXTBYTES);

		// Reducing BBp modulo q
		for (int i = 0; i < PARAMS_N*PARAMS_NBAR; i++) BBp[i] = BBp[i] & ((1 << PARAMS_LOGQ) - 1);

		// Is (Bp == BBp & C == CC) = true
		if (memcmp(Bp, BBp, 2 * PARAMS_N*PARAMS_NBAR) == 0 && memcmp(C, CC, 2 * PARAMS_NBAR*PARAMS_NBAR) == 0) {
			// Load k' to do ss = F(ct || k')
			memcpy(Fin_k, kprime, CRYPTO_BYTES);
		}
		else {
			// Load s to do ss = F(ct || s)
			memcpy(Fin_k, sk_s, CRYPTO_BYTES);
		}
		shake(ss, CRYPTO_BYTES, Fin, CRYPTO_CIPHERTEXTBYTES + CRYPTO_BYTES);

	}

	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);

	cudaEventElapsedTime(&time, start, stop);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	fprintf(stdout, "KEM-DEC: Time Taken: %f ms\n", time);

	cudaFree(Bp_d);
	cudaFree(sk_S_d);
	cudaFree(C_d);
	cudaFree(W_d);
	cudaFree(G2in_d);
	cudaFree(G2out_d);
	cudaFree(shake_input_seedSEprime_d);
	cudaFree(Sp_d);
	cudaFree(BBp_d);
	cudaFree(A_d);
	cudaFree(B_d);
	cudaFree(CC_d);

	// Cleanup:
	clear_bytes((uint8_t *)W, PARAMS_NBAR*PARAMS_NBAR * sizeof(uint16_t));
	clear_bytes((uint8_t *)Sp, PARAMS_N*PARAMS_NBAR * sizeof(uint16_t));
	clear_bytes((uint8_t *)Ep, PARAMS_N*PARAMS_NBAR * sizeof(uint16_t));
	clear_bytes((uint8_t *)Epp, PARAMS_NBAR*PARAMS_NBAR * sizeof(uint16_t));
	clear_bytes(muprime, BYTES_MU);
	clear_bytes(G2out, 2 * CRYPTO_BYTES);
	clear_bytes(Fin_k, CRYPTO_BYTES);
	clear_bytes(shake_input_seedSEprime, 1 + CRYPTO_BYTES);
	return 0;
}


__global__ void frodo_sample_n_Kernel(uint16_t *s, const size_t n)
{ // Fills vector s with n samples from the noise distribution which requires 16 bits to sample. 
  // The distribution is specified by its CDF.
  // Input: pseudo-random values (2*n bytes) passed in s. The input is overwritten by the output.
	unsigned int  j;
	uint16_t CDF_TABLE_LEN = 11;

	unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < n) {
		uint8_t sample = 0;
		uint16_t prnd = s[idx] >> 1;    // Drop the least significant bit
		uint8_t sign = s[idx] & 0x1;    // Pick the least significant bit

		// No need to compare with the last value.
		for (j = 0; j < CDF_TABLE_LEN - 1; j++) { //(CDF_TABLE_LEN - 1)
			// Constant time comparison: 1 if CDF_TABLE[j] < s, 0 otherwise. Uses the fact that CDF_TABLE[j] and s fit in 15 bits.
			sample += (uint16_t)(CDF_TABLE[j] - prnd) >> 15;
		}
		// Assuming that sign is either 0 or 1, flips sample iff sign = 1
		s[idx] = ((-sign) ^ sample) + sign;
	}
}
