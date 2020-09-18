
// @Author: Arpan Jati
// Adapted from NewHope Reference Codebase and Parallelized using CUDA
// Updated : August 2019
// -------------------------------------------------------------
// CODE FOR PERFORMANCE COMPARISON. NOT FOR ACTUAL DEPLOYMENT
// -------------------------------------------------------------

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include "rng.h"
#include "api.h"

#define	MAX_MARKER_LEN		50
#define KAT_SUCCESS          0
#define KAT_FILE_OPEN_ERROR -1
#define KAT_DATA_ERROR      -3
#define KAT_CRYPTO_FAILURE  -4


/*

int main_ind_cca()
{
	int                 ret_val;

	unsigned char ct[CRYPTO_CIPHERTEXTBYTES];
	unsigned char ss[CRYPTO_BYTES];
	unsigned char ss1[CRYPTO_BYTES];

	unsigned char pk[CRYPTO_PUBLICKEYBYTES];
	unsigned char sk[CRYPTO_SECRETKEYBYTES];

	ret_val = crypto_kem_keypair(pk, sk);

	if (ret_val != 0) {
		printf("crypto_kem_keypair returned <%d>\n", ret_val);
		return KAT_CRYPTO_FAILURE;
	}

	ret_val = crypto_kem_enc(ct, ss, pk);

	if (ret_val != 0) {
		printf("crypto_kem_enc returned <%d>\n", ret_val);
		return KAT_CRYPTO_FAILURE;
	}

	ret_val = crypto_kem_dec(ss1, ct, sk);

	if (ret_val != 0) {
		printf("crypto_kem_dec returned <%d>\n", ret_val);
		return KAT_CRYPTO_FAILURE;
	}

	if (memcmp(ss, ss1, CRYPTO_BYTES)) {
		printf("crypto_kem_dec returned bad 'ss' value\n");
		return KAT_CRYPTO_FAILURE;
	}

	return 0;
}
*/

#include "params.h"
#include "indcpa.h"
#include "main.h"

#include <chrono>
#include <iostream>

using namespace std;

// MAIN CPA 

void allocatePolySet(poly_set4* polySet)
{
	HANDLE_ERROR(cudaMalloc(&(polySet->a), sizeof(poly)));
	HANDLE_ERROR(cudaMalloc(&(polySet->b), sizeof(poly)));
	HANDLE_ERROR(cudaMalloc(&(polySet->c), sizeof(poly)));
	HANDLE_ERROR(cudaMalloc(&(polySet->d), sizeof(poly)));

	HANDLE_ERROR(cudaMalloc(&(polySet->av), sizeof(polyvec)));
	HANDLE_ERROR(cudaMalloc(&(polySet->bv), sizeof(polyvec)));
	HANDLE_ERROR(cudaMalloc(&(polySet->cv), sizeof(polyvec)));
	HANDLE_ERROR(cudaMalloc(&(polySet->dv), sizeof(polyvec)));
	HANDLE_ERROR(cudaMalloc(&(polySet->ev), sizeof(polyvec)));
	HANDLE_ERROR(cudaMalloc(&(polySet->fv), sizeof(polyvec)));

	HANDLE_ERROR(cudaMalloc(&(polySet->AV), sizeof(polyvec) * 4));

	HANDLE_ERROR(cudaMalloc(&(polySet->seed), (KYBER_SYMBYTES * 2) * N_TESTS));

	HANDLE_ERROR(cudaMalloc(&(polySet->large_buffer_a), LARGE_BUFFER_SZ * N_TESTS));
	HANDLE_ERROR(cudaMalloc(&(polySet->large_buffer_b), LARGE_BUFFER_SZ * N_TESTS));
}

void freePolySet(poly_set4* polySet)
{
	HANDLE_ERROR(cudaFree(polySet->a));
	HANDLE_ERROR(cudaFree(polySet->b));
	HANDLE_ERROR(cudaFree(polySet->c));
	HANDLE_ERROR(cudaFree(polySet->d));

	HANDLE_ERROR(cudaFree(polySet->av));
	HANDLE_ERROR(cudaFree(polySet->bv));
	HANDLE_ERROR(cudaFree(polySet->cv));
	HANDLE_ERROR(cudaFree(polySet->dv));
	HANDLE_ERROR(cudaFree(polySet->ev));
	HANDLE_ERROR(cudaFree(polySet->fv));

	HANDLE_ERROR(cudaFree(polySet->seed));

	HANDLE_ERROR(cudaFree(polySet->large_buffer_a));
	HANDLE_ERROR(cudaFree(polySet->large_buffer_b));
}


int Hardware()
{
	cudaDeviceProp prop;
	int count;
	HANDLE_ERROR(cudaGetDeviceCount(&count));
	for (int i = 0; i < count; i++) {
		HANDLE_ERROR(cudaGetDeviceProperties(&prop, i));
		printf(" --- General Information for device %d ---\n", i);
		printf("Name: %s\n", prop.name);
		printf("Compute capability: %d.%d\n", prop.major, prop.minor);
		printf("Clock rate: %d\n", prop.clockRate);
		printf("Device copy overlap: ");
		if (prop.deviceOverlap)
			printf("Enabled\n");
		else
			printf("Disabled\n");
		printf("Kernel execition timeout : ");
		if (prop.kernelExecTimeoutEnabled)
			printf("Enabled\n");
		else
			printf("Disabled\n");
		printf(" --- Memory Information for device %d ---\n", i);
		printf("Total global mem: %llu\n", prop.totalGlobalMem);
		printf("Total constant Mem: %llu\n", prop.totalConstMem);
		printf("Max mem pitch: %zd\n", prop.memPitch);
		printf("Texture Alignment: %zd\n", prop.textureAlignment);
		printf(" --- MP Information for device %d ---\n", i);
		printf("Multiprocessor count: %d\n", prop.multiProcessorCount);

		MP_COUNT = prop.multiProcessorCount;

		printf("Shared mem per mp: %zd\n", prop.sharedMemPerBlock);
		printf("Registers per mp: %d\n", prop.regsPerBlock);
		printf("Threads in warp: %d\n", prop.warpSize);
		printf("Max threads per block: %d\n", prop.maxThreadsPerBlock);
		printf("Max thread dimensions: (%d, %d, %d)\n", prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
		printf("Max grid dimensions: (%d, %d, %d)\n", prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
		printf("\n");
	}

	return count;
}

int SELECTED_GPU = GPU_G1060;

#ifdef ANALYSIS_MODE
int NORMAL_COUNTS[4] = { 1024,4096,16384,32768 };
#else
int NORMAL_COUNTS[15] = { 4,8,16,32,64,128,256,512,1024,2048,4096,8192,16384,20000,32768 };
#endif

int COUNT = N_TESTS;

int main()
{
	printf("\n KYBER GPU BATCHED | N_TESTS = %d ", N_TESTS);

	printf("\nN_TESTS: %d \n", N_TESTS);

	int gpu_count = Hardware();

	printf("\n SELECT GPU TYPE (for optimization): ");
	printf("\n		GPU_G1060 = 0");
	printf("\n		GPU_P6000 = 1");
	printf("\n		GPU_940MX = 2");
	printf("\n		GPU_V100  = 3");
	printf("\n		    (0-3) = ");

	char buffer[20];

	auto str = fgets(buffer, 20, stdin);
	//auto str = "3";

	if (str != NULL)
	{
		int v = atoi(str);

		if (v < 0 || v >3)
		{
			printf("\n NO_SUCH_GPU !!");
			exit(1);
		}
		else
		{
			SELECTED_GPU = v;
		}
	}
	else
	{
		printf("\n INVALID INPUT !!");
		exit(1);
	}

	printf("\n SELECTED GPU %s \n\n", SELECTED_GPU_NAME);

	cudaEvent_t start, stop;

	int SELECTED_GPU_ID = 0;

	if (gpu_count > 1)
	{
		printf("\nSELECT GPU ID (for execution) (0-%d): ", (gpu_count - 1));
		auto str = fgets(buffer, 20, stdin);

		if (str != NULL)
		{
			int v = atoi(str);

			if (v < 0 || v > 3)
			{
				printf("\n NO_SUCH_GPU_ID !!");
				exit(1);
			}
			else
			{
				SELECTED_GPU_ID = v;
			}
		}
	}

	printf("\n SELECTED GPU ID = %d \n", SELECTED_GPU_ID);

	// Choose which GPU to run on, change this on a multi-GPU system.
	HANDLE_ERROR(cudaSetDevice(SELECTED_GPU_ID));

	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	//////////////////////////////

	printf("\n SERVER MODE (TYPE '1' to enable): ");

	int serverMode = 0;

	auto ch = getchar();

	//auto ch = '0';
	if (ch == '1')
	{
		serverMode = 1;
		printf("\n SERVER MODE ENABLED [A-C]");
		printf("\n // -------------------------------------------------------------------");
		printf("\n // NOTE THAT FOR SERVER MODE, THE RESULTS WILL BE INCORRECT !! ");
		printf("\n // FURTHER, THIS IS NOT THE EXACT PERFORMANCE AS WE EMULATE THE SERVER BEHAVIOR");
		printf("\n // AND COMPUTE TOTAL AMOUNT OF COMPUTATIONS FOR A SATURATED SERVER.");
		printf("\n // Keeping 'SERVERMODE = 0' performs all THREE stages of the key exchange together.");
		printf("\n // Not realistic ! But, lets us know the performance improvements by comparing GPU");
		printf("\n // results with a CPU based implementation.");
		printf("\n // -------------------------------------------------------------------");
	}
	else
	{
		printf("\n SERVER MODE DISABLED [A-B-C]");
	}

	// int N_TESTS_D = (N_TESTS);

	poly_set4 tempPoly_0[4];
	poly_set4 tempPoly_1[4];

	allocatePolySet(&tempPoly_0[0]);
	allocatePolySet(&tempPoly_1[0]);

	unsigned char* pk_h_0;
	unsigned char* sk_h_0;
	unsigned char* ct_h_0;
	unsigned char* msg1_h_0;
	unsigned char* msg2_h_0;
	unsigned char* coins_h_0;
	unsigned char* rng_buf_h_0;

	unsigned char* pk_h_1;
	unsigned char* sk_h_1;
	unsigned char* ct_h_1;
	unsigned char* msg1_h_1;
	unsigned char* msg2_h_1;
	unsigned char* coins_h_1;
	unsigned char* rng_buf_h_1;

	HANDLE_ERROR(cudaHostAlloc((void**)&pk_h_0, KYBER_INDCPA_PUBLICKEYBYTES * N_TESTS, cudaHostAllocDefault));
	HANDLE_ERROR(cudaHostAlloc((void**)&sk_h_0, KYBER_INDCPA_SECRETKEYBYTES * N_TESTS, cudaHostAllocDefault));
	HANDLE_ERROR(cudaHostAlloc((void**)&ct_h_0, KYBER_INDCPA_BYTES * N_TESTS, cudaHostAllocDefault));
	HANDLE_ERROR(cudaHostAlloc((void**)&msg1_h_0, KYBER_INDCPA_MSGBYTES * N_TESTS, cudaHostAllocDefault));
	HANDLE_ERROR(cudaHostAlloc((void**)&msg2_h_0, KYBER_INDCPA_MSGBYTES * N_TESTS, cudaHostAllocDefault));
	HANDLE_ERROR(cudaHostAlloc((void**)&coins_h_0, KYBER_SYMBYTES * N_TESTS, cudaHostAllocDefault));
	HANDLE_ERROR(cudaHostAlloc((void**)&rng_buf_h_0, KYBER_SYMBYTES * 2 * N_TESTS, cudaHostAllocDefault));

	HANDLE_ERROR(cudaHostAlloc((void**)&pk_h_1, KYBER_INDCPA_PUBLICKEYBYTES * N_TESTS, cudaHostAllocDefault));
	HANDLE_ERROR(cudaHostAlloc((void**)&sk_h_1, KYBER_INDCPA_SECRETKEYBYTES * N_TESTS, cudaHostAllocDefault));
	HANDLE_ERROR(cudaHostAlloc((void**)&ct_h_1, KYBER_INDCPA_BYTES * N_TESTS, cudaHostAllocDefault));
	HANDLE_ERROR(cudaHostAlloc((void**)&msg1_h_1, KYBER_INDCPA_MSGBYTES * N_TESTS, cudaHostAllocDefault));
	HANDLE_ERROR(cudaHostAlloc((void**)&msg2_h_1, KYBER_INDCPA_MSGBYTES * N_TESTS, cudaHostAllocDefault));
	HANDLE_ERROR(cudaHostAlloc((void**)&coins_h_1, KYBER_SYMBYTES * N_TESTS, cudaHostAllocDefault));
	HANDLE_ERROR(cudaHostAlloc((void**)&rng_buf_h_1, KYBER_SYMBYTES * 2 * N_TESTS, cudaHostAllocDefault));

	//
	unsigned char* pk_d_0;
	unsigned char* sk_d_0;
	unsigned char* ct_d_0;
	unsigned char* msg1_d_0;
	unsigned char* msg2_d_0;
	unsigned char* coins_d_0;
	unsigned char* rng_buf_d_0;


	unsigned char* pk_d_1;
	unsigned char* sk_d_1;
	unsigned char* ct_d_1;
	unsigned char* msg1_d_1;
	unsigned char* msg2_d_1;
	unsigned char* coins_d_1;
	unsigned char* rng_buf_d_1;

	HANDLE_ERROR(cudaMalloc((void**)&pk_d_0, KYBER_INDCPA_PUBLICKEYBYTES * N_TESTS));
	HANDLE_ERROR(cudaMalloc((void**)&sk_d_0, KYBER_INDCPA_SECRETKEYBYTES * N_TESTS));
	HANDLE_ERROR(cudaMalloc((void**)&ct_d_0, KYBER_INDCPA_BYTES * N_TESTS));
	HANDLE_ERROR(cudaMalloc((void**)&msg1_d_0, KYBER_INDCPA_MSGBYTES * N_TESTS));
	HANDLE_ERROR(cudaMalloc((void**)&msg2_d_0, KYBER_INDCPA_MSGBYTES * N_TESTS));
	HANDLE_ERROR(cudaMalloc((void**)&coins_d_0, KYBER_SYMBYTES * N_TESTS));
	HANDLE_ERROR(cudaMalloc((void**)&rng_buf_d_0, KYBER_SYMBYTES * 2 * N_TESTS));

	HANDLE_ERROR(cudaMalloc((void**)&pk_d_1, KYBER_INDCPA_PUBLICKEYBYTES * N_TESTS));
	HANDLE_ERROR(cudaMalloc((void**)&sk_d_1, KYBER_INDCPA_SECRETKEYBYTES * N_TESTS));
	HANDLE_ERROR(cudaMalloc((void**)&ct_d_1, KYBER_INDCPA_BYTES * N_TESTS));
	HANDLE_ERROR(cudaMalloc((void**)&msg1_d_1, KYBER_INDCPA_MSGBYTES * N_TESTS));
	HANDLE_ERROR(cudaMalloc((void**)&msg2_d_1, KYBER_INDCPA_MSGBYTES * N_TESTS));
	HANDLE_ERROR(cudaMalloc((void**)&coins_d_1, KYBER_SYMBYTES * N_TESTS));
	HANDLE_ERROR(cudaMalloc((void**)&rng_buf_d_1, KYBER_SYMBYTES * 2 * N_TESTS));

	memset(msg1_h_0, 0, KYBER_SYMBYTES * N_TESTS);
	randombytes(msg1_h_0, KYBER_SYMBYTES * N_TESTS);
	randombytes(coins_h_0, KYBER_SYMBYTES * N_TESTS);
	randombytes(rng_buf_h_0, KYBER_SYMBYTES * N_TESTS * 2);

	memset(msg1_h_1, 0, KYBER_SYMBYTES * N_TESTS);
	randombytes(msg1_h_1, KYBER_SYMBYTES * N_TESTS);
	randombytes(coins_h_1, KYBER_SYMBYTES * N_TESTS);
	randombytes(rng_buf_h_1, KYBER_SYMBYTES * N_TESTS * 2);


	cudaStream_t stream_0;
	cudaStream_t stream_1;

	HANDLE_ERROR(cudaStreamCreate(&stream_0));
	HANDLE_ERROR(cudaStreamCreate(&stream_1));

	// stream_1 = stream_0;

#ifdef ANALYSIS_MODE
	for (int i = 0; i < 4; i++)
#else
	for (int i = 0; i < 13; i++)
#endif
	{
		COUNT = NORMAL_COUNTS[i];

		cudaEventRecord(start);

		HANDLE_ERROR(cudaMemcpyAsync(pk_d_0, pk_h_0, KYBER_INDCPA_PUBLICKEYBYTES * COUNT, cudaMemcpyHostToDevice, stream_0));
		HANDLE_ERROR(cudaMemcpyAsync(sk_d_0, sk_h_0, KYBER_INDCPA_SECRETKEYBYTES * COUNT, cudaMemcpyHostToDevice, stream_0));
		HANDLE_ERROR(cudaMemcpyAsync(ct_d_0, ct_h_0, KYBER_INDCPA_BYTES * COUNT, cudaMemcpyHostToDevice, stream_0));
		HANDLE_ERROR(cudaMemcpyAsync(msg1_d_0, msg1_h_0, KYBER_INDCPA_MSGBYTES * COUNT, cudaMemcpyHostToDevice, stream_0));
		HANDLE_ERROR(cudaMemcpyAsync(msg2_d_0, msg2_h_0, KYBER_INDCPA_MSGBYTES * COUNT, cudaMemcpyHostToDevice, stream_0));
		HANDLE_ERROR(cudaMemcpyAsync(coins_d_0, coins_h_0, KYBER_SYMBYTES * COUNT, cudaMemcpyHostToDevice, stream_0));
		HANDLE_ERROR(cudaMemcpyAsync(rng_buf_d_0, rng_buf_h_0, KYBER_SYMBYTES * 2 * COUNT, cudaMemcpyHostToDevice, stream_0));

		HANDLE_ERROR(cudaMemcpyAsync(pk_d_1, pk_h_1, KYBER_INDCPA_PUBLICKEYBYTES * COUNT, cudaMemcpyHostToDevice, stream_1));
		HANDLE_ERROR(cudaMemcpyAsync(sk_d_1, sk_h_1, KYBER_INDCPA_SECRETKEYBYTES * COUNT, cudaMemcpyHostToDevice, stream_1));
		HANDLE_ERROR(cudaMemcpyAsync(ct_d_1, ct_h_1, KYBER_INDCPA_BYTES * COUNT, cudaMemcpyHostToDevice, stream_1));
		HANDLE_ERROR(cudaMemcpyAsync(msg1_d_1, msg1_h_1, KYBER_INDCPA_MSGBYTES * COUNT, cudaMemcpyHostToDevice, stream_1));
		HANDLE_ERROR(cudaMemcpyAsync(msg2_d_1, msg2_h_1, KYBER_INDCPA_MSGBYTES * COUNT, cudaMemcpyHostToDevice, stream_1));
		HANDLE_ERROR(cudaMemcpyAsync(coins_d_1, coins_h_1, KYBER_SYMBYTES * COUNT, cudaMemcpyHostToDevice, stream_1));
		HANDLE_ERROR(cudaMemcpyAsync(rng_buf_d_1, rng_buf_h_1, KYBER_SYMBYTES * 2 * COUNT, cudaMemcpyHostToDevice, stream_1));

		indcpa_keypair(COUNT, &tempPoly_0[0], pk_d_0, sk_d_0, rng_buf_d_0, stream_0);
		indcpa_keypair(COUNT, &tempPoly_1[0], pk_d_1, sk_d_1, rng_buf_d_1, stream_1);

		if (!serverMode)
		{
			indcpa_enc(COUNT, &tempPoly_0[0], ct_d_0, msg1_d_0, pk_d_0, coins_d_0, stream_0);
			indcpa_enc(COUNT, &tempPoly_1[0], ct_d_1, msg1_d_1, pk_d_1, coins_d_1, stream_1);
		}

		indcpa_dec(COUNT, &tempPoly_0[0], msg2_d_0, ct_d_0, sk_d_0, stream_0);
		indcpa_dec(COUNT, &tempPoly_1[0], msg2_d_1, ct_d_1, sk_d_1, stream_1);

		HANDLE_ERROR(cudaMemcpyAsync(pk_h_0, pk_d_0, KYBER_INDCPA_PUBLICKEYBYTES * COUNT, cudaMemcpyDeviceToHost, stream_0));
		HANDLE_ERROR(cudaMemcpyAsync(sk_h_0, sk_d_0, KYBER_INDCPA_SECRETKEYBYTES * COUNT, cudaMemcpyDeviceToHost, stream_0));
		HANDLE_ERROR(cudaMemcpyAsync(ct_h_0, ct_d_0, KYBER_INDCPA_BYTES * COUNT, cudaMemcpyDeviceToHost, stream_0));
		HANDLE_ERROR(cudaMemcpyAsync(msg1_h_0, msg1_d_0, KYBER_INDCPA_MSGBYTES * COUNT, cudaMemcpyDeviceToHost, stream_0));
		HANDLE_ERROR(cudaMemcpyAsync(msg2_h_0, msg2_d_0, KYBER_INDCPA_MSGBYTES * COUNT, cudaMemcpyDeviceToHost, stream_0));
		HANDLE_ERROR(cudaMemcpyAsync(coins_h_0, coins_d_0, KYBER_SYMBYTES * COUNT, cudaMemcpyDeviceToHost, stream_0));
		HANDLE_ERROR(cudaMemcpyAsync(rng_buf_h_0, rng_buf_d_0, KYBER_SYMBYTES * 2 * COUNT, cudaMemcpyDeviceToHost, stream_0));

		HANDLE_ERROR(cudaMemcpyAsync(pk_h_1, pk_d_1, KYBER_INDCPA_PUBLICKEYBYTES * COUNT, cudaMemcpyDeviceToHost, stream_1));
		HANDLE_ERROR(cudaMemcpyAsync(sk_h_1, sk_d_1, KYBER_INDCPA_SECRETKEYBYTES * COUNT, cudaMemcpyDeviceToHost, stream_1));
		HANDLE_ERROR(cudaMemcpyAsync(ct_h_1, ct_d_1, KYBER_INDCPA_BYTES * COUNT, cudaMemcpyDeviceToHost, stream_1));
		HANDLE_ERROR(cudaMemcpyAsync(msg1_h_1, msg1_d_1, KYBER_INDCPA_MSGBYTES * COUNT, cudaMemcpyDeviceToHost, stream_1));
		HANDLE_ERROR(cudaMemcpyAsync(msg2_h_1, msg2_d_1, KYBER_INDCPA_MSGBYTES * COUNT, cudaMemcpyDeviceToHost, stream_1));
		HANDLE_ERROR(cudaMemcpyAsync(coins_h_1, coins_d_1, KYBER_SYMBYTES * COUNT, cudaMemcpyDeviceToHost, stream_1));
		HANDLE_ERROR(cudaMemcpyAsync(rng_buf_h_1, rng_buf_d_1, KYBER_SYMBYTES * 2 * COUNT, cudaMemcpyDeviceToHost, stream_1));

		cudaEventRecord(stop);
		cudaEventSynchronize(stop);

		///////////////////

		float milliseconds = 0;
		cudaEventElapsedTime(&milliseconds, start, stop);

		auto KXps = (int)((((double)(COUNT * 2)) * (double)1000.0) / (double)milliseconds);

		printf("\n =================--------------================------------==================");
		printf("\n COUNT=%d | Time Elapsed: %f ms. K/s: %d ", COUNT * 2, milliseconds, KXps);
		printf("\n =================--------------================------------================== \n");


		int error = -1;
		int match_count = 0;

		for (int i = 0; i < KYBER_INDCPA_MSGBYTES * COUNT; i++)
		{
			if (msg1_h_0[i] != msg2_h_0[i])
			{
				error = i;
				break;
			}
		}

		for (int i = 0; i < KYBER_INDCPA_MSGBYTES * COUNT; i++)
		{
			if (msg1_h_0[i] == msg2_h_0[i])
			{
				match_count++;
			}
		}

		int max_match = KYBER_INDCPA_MSGBYTES * COUNT;

		if (error != -1)
		{
			printf(" ERROR : < MESSAGE VERIFICATION #0 > \n IDX: %d \n INST-ID: %d \n ERROR-COUNT: %d \n MAX-MATCH: %d \n MATCH-RATIO: %f\n",
				error, (error / KYBER_INDCPA_MSGBYTES), max_match - match_count, max_match, (match_count * 100.0F) / max_match);
		}
		else
		{
			printf(" MESSAGE VERIFICATION COMPLETE #0 !! \n");
		}

		error = -1;
		match_count = 0;

		for (int i = 0; i < KYBER_INDCPA_MSGBYTES * COUNT; i++)
		{
			if (msg1_h_1[i] != msg2_h_1[i])
			{
				error = i;
				break;
			}
		}

		for (int i = 0; i < KYBER_INDCPA_MSGBYTES * COUNT; i++)
		{
			if (msg1_h_1[i] == msg2_h_1[i])
			{
				match_count++;
			}
		}

		max_match = KYBER_INDCPA_MSGBYTES * COUNT;

		if (error != -1)
		{
			printf(" ERROR : < MESSAGE VERIFICATION #1 > \n IDX: %d \n INST-ID: %d \n ERROR-COUNT: %d \n MAX-MATCH: %d \n MATCH-RATIO: %f\n",
				error, (error / KYBER_INDCPA_MSGBYTES), max_match - match_count, max_match, (match_count * 100.0F) / max_match);
		}
		else
		{
			printf(" MESSAGE VERIFICATION COMPLETE #1 !! \n");
		}

		printf(" =================--------------================------------================== \n");

	}

	/*
		print_data("PK", pk_h, KYBER_INDCPA_PUBLICKEYBYTES);
		print_data("SK", sk_h, KYBER_INDCPA_SECRETKEYBYTES);
		print_data("CT", ct_h, KYBER_INDCPA_BYTES);

		print_data("msg1", msg1_h, KYBER_INDCPA_MSGBYTES);
		print_data("msg2", msg2_h + KYBER_INDCPA_MSGBYTES, KYBER_INDCPA_MSGBYTES);

		print_data("coins", coins_h, KYBER_SYMBYTES);
	*/

	// printf("\n\n");

	//print_data("msg1_0", msg1_h_0, KYBER_INDCPA_MSGBYTES);
	//print_data("msg2_0", msg2_h_0, KYBER_INDCPA_MSGBYTES);

	//print_data("msg1_1", msg1_h_1, KYBER_INDCPA_MSGBYTES);
	//print_data("msg2_1", msg2_h_1, KYBER_INDCPA_MSGBYTES);


	/////////////////////////////////////////

	HANDLE_ERROR(cudaStreamDestroy(stream_0));
	HANDLE_ERROR(cudaStreamDestroy(stream_1));

	HANDLE_ERROR(cudaFree(pk_d_0));
	HANDLE_ERROR(cudaFree(sk_d_0));
	HANDLE_ERROR(cudaFree(ct_d_0));
	HANDLE_ERROR(cudaFree(msg1_d_0));
	HANDLE_ERROR(cudaFree(msg2_d_0));
	HANDLE_ERROR(cudaFree(coins_d_0));
	HANDLE_ERROR(cudaFree(rng_buf_d_0));

	HANDLE_ERROR(cudaFree(pk_d_1));
	HANDLE_ERROR(cudaFree(sk_d_1));
	HANDLE_ERROR(cudaFree(ct_d_1));
	HANDLE_ERROR(cudaFree(msg1_d_1));
	HANDLE_ERROR(cudaFree(msg2_d_1));
	HANDLE_ERROR(cudaFree(coins_d_1));
	HANDLE_ERROR(cudaFree(rng_buf_d_1));

	HANDLE_ERROR(cudaFreeHost(pk_h_0));
	HANDLE_ERROR(cudaFreeHost(sk_h_0));
	HANDLE_ERROR(cudaFreeHost(ct_h_0));
	HANDLE_ERROR(cudaFreeHost(msg1_h_0));
	HANDLE_ERROR(cudaFreeHost(msg2_h_0));
	HANDLE_ERROR(cudaFreeHost(coins_h_0));
	HANDLE_ERROR(cudaFreeHost(rng_buf_h_0));

	HANDLE_ERROR(cudaFreeHost(pk_h_1));
	HANDLE_ERROR(cudaFreeHost(sk_h_1));
	HANDLE_ERROR(cudaFreeHost(ct_h_1));
	HANDLE_ERROR(cudaFreeHost(msg1_h_1));
	HANDLE_ERROR(cudaFreeHost(msg2_h_1));
	HANDLE_ERROR(cudaFreeHost(coins_h_1));
	HANDLE_ERROR(cudaFreeHost(rng_buf_h_1));

	freePolySet(&tempPoly_0[0]);
	freePolySet(&tempPoly_1[0]);


	// Check for any errors launching the kernel
	HANDLE_ERROR(cudaGetLastError());

	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	HANDLE_ERROR(cudaDeviceSynchronize());


	HANDLE_ERROR(cudaDeviceReset());

	printf("\n\nDONE. ");

	//getchar();

	return 0;
}