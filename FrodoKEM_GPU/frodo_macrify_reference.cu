/********************************************************************************************
* FrodoKEM: Learning with Errors Key Encapsulation
*
* Abstract: matrix arithmetic functions used by the KEM

// @Author: Naina Gupta
// Adapted from FrodoKEM Reference Codebase and Parallelized using CUDA
// Updated : August 2019

*********************************************************************************************/

#include "api.h"
#include "params.h"
#include <stdio.h>
#include <stdlib.h>
#include <memory.h>

#include "fips202.h"

#if defined(USE_AES128_FOR_A)
#include "aes/aes.h"
#elif defined (USE_SHAKE128_FOR_A)
#include "fips202.h"
#endif    


__device__ unsigned short atomicAddShort(unsigned short* address, unsigned short val)

{
	unsigned int* base_address = (unsigned int*)((char*)address - ((size_t)address & 2));	//tera's revised version (showtopic=201975)

	unsigned int long_val = ((size_t)address & 2) ? ((unsigned int)val << 16) : (unsigned short)val;

	unsigned int long_old = atomicAdd(base_address, long_val);

	if ((size_t)address & 2) {

		return (unsigned short)(long_old >> 16);

	}
	else {

		unsigned int overflow = ((long_old & 0xffff) + long_val) & 0xffff0000;

		if (overflow)

			atomicSub(base_address, overflow);

		return (unsigned short)(long_old & 0xffff);

	}

}

__global__ void matMulKernelTile(uint16_t* res_d, uint16_t* a_d, uint16_t* b_d)
{
	__shared__ uint16_t aRow[PARAMS_N];// , bTile[TILE_DIM1];
	__shared__ uint16_t sumSH[8][16];

	unsigned int idx = blockIdx.x + gridDim.x * blockIdx.y;

	if (idx < PARAMS_N)
	{
		__syncthreads();

		int tid = blockDim.y * threadIdx.x + threadIdx.y;

		aRow[tid] = a_d[(idx * PARAMS_N) + tid];
		aRow[tid + 128] = a_d[(idx * PARAMS_N) + tid + 128];
		aRow[tid + 256] = a_d[(idx * PARAMS_N) + tid + 256];
		aRow[tid + 384] = a_d[(idx * PARAMS_N) + tid + 384];
		aRow[tid + 512] = a_d[(idx * PARAMS_N) + tid + 512];
		aRow[tid + 640] = a_d[(idx * PARAMS_N) + tid + 640];
		aRow[tid + 768] = a_d[(idx * PARAMS_N) + tid + 768];

		if (tid < 80)
			aRow[tid + 896] = a_d[(idx * PARAMS_N) + tid + 896];

		__syncthreads();

		uint16_t sum = 0;

		int i = threadIdx.y;
		int k = threadIdx.x;

		for (int j = 0; j < PARAMS_N / 16; j++)
		{
			int loc = j * 16 + i;

			sum += aRow[loc] * b_d[(k * PARAMS_N) + loc];

		}

		sumSH[k][i] = sum;
		atomicAddShort(&res_d[idx * PARAMS_NBAR + k], sumSH[k][i]);

	}

}


__global__ void matMulKernelSA(uint16_t* res_d, uint16_t* a_d, uint16_t* b_d)
{
	__shared__ uint16_t bCol[PARAMS_N];

	unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < PARAMS_N)
	{
		int k = blockIdx.y;

		__syncthreads();

		int tid = threadIdx.x;

		if (blockIdx.x < 7)
		{
			bCol[tid] = b_d[(k * PARAMS_N) + tid];
			bCol[tid + 128] = b_d[(k * PARAMS_N) + tid + 128];
			bCol[tid + 256] = b_d[(k * PARAMS_N) + tid + 256];
			bCol[tid + 384] = b_d[(k * PARAMS_N) + tid + 384];
			bCol[tid + 512] = b_d[(k * PARAMS_N) + tid + 512];
			bCol[tid + 640] = b_d[(k * PARAMS_N) + tid + 640];
			bCol[tid + 768] = b_d[(k * PARAMS_N) + tid + 768];

			if (tid < 80)
				bCol[tid + 896] = b_d[(k * PARAMS_N) + tid + 896];
		}
		else
		{
			bCol[tid] = b_d[(k * PARAMS_N) + tid];
			bCol[tid + 80] = b_d[(k * PARAMS_N) + tid + 80];
			bCol[tid + 160] = b_d[(k * PARAMS_N) + tid + 160];
			bCol[tid + 240] = b_d[(k * PARAMS_N) + tid + 240];
			bCol[tid + 320] = b_d[(k * PARAMS_N) + tid + 320];
			bCol[tid + 400] = b_d[(k * PARAMS_N) + tid + 400];
			bCol[tid + 480] = b_d[(k * PARAMS_N) + tid + 480];
			bCol[tid + 560] = b_d[(k * PARAMS_N) + tid + 560];
			bCol[tid + 640] = b_d[(k * PARAMS_N) + tid + 640];
			bCol[tid + 720] = b_d[(k * PARAMS_N) + tid + 720];
			bCol[tid + 800] = b_d[(k * PARAMS_N) + tid + 800];
			bCol[tid + 880] = b_d[(k * PARAMS_N) + tid + 880];


			if (tid < 16)
				bCol[tid + 960] = b_d[(k * PARAMS_N) + tid + 960];
		}

		__syncthreads();

		uint16_t sum = 0;

		for (int j = 0; j < PARAMS_N; j++)
		{
			sum += a_d[(j * PARAMS_N) + idx] * bCol[j];
		}
		res_d[k * PARAMS_N + idx] += sum;
	}

}


__constant__ uint8_t seed_A_const[16];

__global__ void generateMatA2(uint16_t* A)
{
	__shared__ uint64_t stateS[THR_PER_BLK * 25];

	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	uint8_t seed_A_th[18] = { 0 };

	for (int i = 0; i < 25; i++)
	{
		stateS[threadIdx.x + i * THR_PER_BLK] = 0;
	}

	__syncthreads();

	if (idx < PARAMS_N) { //

		for (int i = 0; i < 16; i++)
		{
			seed_A_th[i + 2] = seed_A_const[i];
		}

		((uint16_t*)seed_A_th)[0] = (uint16_t)idx;
		shake128_Sh((unsigned char*)(A + idx * PARAMS_N), (unsigned long long)(2 * PARAMS_N), (const unsigned char*)(seed_A_th), (unsigned long long)(2 + BYTES_SEED_A), stateS, threadIdx.x, blockIdx.x);
	}
}

int frodo_mul_add_as_plus_e_gpu(uint16_t* out_d, uint16_t* s_d, uint16_t* e_d, const uint8_t* seed_A, uint16_t* A_d)
{ // Generate-and-multiply: generate matrix A (N x N) row-wise, multiply by s on the right.
  // Inputs: s, e (N x N_BAR)
  // Output: out = A*s + e (N x N_BAR)
	int i, j, k;

	cudaError_t cudaStatus;

	cudaStatus = cudaMemset(A_d, 0, PARAMS_N * PARAMS_N * sizeof(uint16_t));

	cudaStatus = cudaMemcpyToSymbol(seed_A_const, seed_A, BYTES_SEED_A * sizeof(uint8_t), 0, cudaMemcpyHostToDevice);

	generateMatA2 << <64, 16 >> > (A_d); 

	cudaStatus = cudaMemcpy(out_d, e_d, PARAMS_NBAR * PARAMS_N * sizeof(uint16_t), cudaMemcpyDeviceToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "ERROR: %s\n", cudaGetErrorString(cudaStatus));
	}

	matMulKernelTile << <dim3(128, 8, 1), dim3(8, 16, 1) >> > (out_d, A_d, s_d);

	return 1;
}


int frodo_mul_add_sa_plus_e_gpu(uint16_t* out_d, uint16_t* s_d, uint16_t* e_d, const uint8_t* seed_A, uint16_t* A_d)
{ // Generate-and-multiply: generate matrix A (N x N) column-wise, multiply by s' on the left.
  // Inputs: s', e' (N_BAR x N)
  // Output: out = s'*A + e' (N_BAR x N)
	int i, j, k;

	cudaError_t cudaStatus;

	cudaStatus = cudaMemset(A_d, 0, PARAMS_N * PARAMS_N * sizeof(uint16_t));

	cudaStatus = cudaMemcpyToSymbol(seed_A_const, seed_A, BYTES_SEED_A * sizeof(uint8_t), 0, cudaMemcpyHostToDevice);

	generateMatA2 << <64, 16 >> > (A_d);  

	cudaStatus = cudaMemcpy(out_d, e_d, PARAMS_NBAR * PARAMS_N * sizeof(uint16_t), cudaMemcpyDeviceToDevice);

	matMulKernelSA << <dim3(8, 8, 1), dim3(128, 1, 1) >> > (out_d, A_d, s_d);

	return 1;
}


__global__ void frodo_mul_bs_kernel(uint16_t* out, uint16_t* b, uint16_t* s)
{ // Multiply by s on the right
  // Inputs: b (N_BAR x N), s (N x N_BAR)
  // Output: out = b*s (N_BAR x N_BAR)

	int i = blockIdx.x;
	int j = threadIdx.x;

	int k;

	int sum = 0;

	out[i * PARAMS_NBAR + j] = 0;

	for (k = 0; k < PARAMS_N; k++)
	{
		sum += b[i * PARAMS_N + k] * s[j * PARAMS_N + k];
	}

	out[i * PARAMS_NBAR + j] = (uint32_t)(sum) & ((1 << PARAMS_LOGQ) - 1);

}


__global__ void frodo_mul_add_sb_plus_e_kernel(uint16_t* out, uint16_t* b, uint16_t* s, uint16_t* e)
{ // Multiply by s on the left
  // Inputs: b (N x N_BAR), s (N_BAR x N), e (N_BAR x N_BAR)
  // Output: out = s*b + e (N_BAR x N_BAR)

	int k = blockIdx.x;
	int i = threadIdx.x;

	int j;

	int sum = 0;

	sum = e[k * PARAMS_NBAR + i];

	for (j = 0; j < PARAMS_N; j++)
	{
		sum += s[k * PARAMS_N + j] * b[j * PARAMS_NBAR + i];
	}

	out[k * PARAMS_NBAR + i] = (uint32_t)(sum) & ((1 << PARAMS_LOGQ) - 1);
}


__global__ void frodo_add_kernel(uint16_t* out, uint16_t* a, uint16_t* b)
{ // Add a and b
  // Inputs: a, b (N_BAR x N_BAR)
  // Output: c = a + b

	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	//for (int i = 0; i < (PARAMS_NBAR*PARAMS_NBAR); i++) 

	if (idx < (PARAMS_NBAR * PARAMS_NBAR))
	{
		out[idx] = (a[idx] + b[idx]) & ((1 << PARAMS_LOGQ) - 1);
	}
}

__global__ void frodo_sub_kernel(uint16_t* out, uint16_t* a, uint16_t* b)
{ // Subtract a and b
  // Inputs: a, b (N_BAR x N_BAR)
  // Output: c = a - b

	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	//for (int i = 0; i < (PARAMS_NBAR*PARAMS_NBAR); i++) 
	if (idx < (PARAMS_NBAR * PARAMS_NBAR))
	{
		out[idx] = (a[idx] - b[idx]) & ((1 << PARAMS_LOGQ) - 1);
	}
}

__global__ void frodo_key_encode_kernel(uint16_t* out, uint16_t* in)
{
	unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
	//for (int i = 0; i < (PARAMS_NBAR*PARAMS_NBAR); i++) 

	unsigned int j, npieces_word = 8;
	unsigned int nwords = (PARAMS_NBAR * PARAMS_NBAR) / 8;
	uint64_t temp, mask = ((uint64_t)1 << PARAMS_EXTRACTED_BITS) - 1;
	uint16_t* pos = out + idx * npieces_word;

	//for (i = 0; i < nwords; i++) 
	if (idx < nwords)
	{
		temp = 0;
		for (j = 0; j < PARAMS_EXTRACTED_BITS; j++)
			temp |= ((uint64_t)((uint8_t*)in)[idx * PARAMS_EXTRACTED_BITS + j]) << (8 * j);
		for (j = 0; j < npieces_word; j++) {
			*pos = (uint16_t)((temp & mask) << (PARAMS_LOGQ - PARAMS_EXTRACTED_BITS));
			temp >>= PARAMS_EXTRACTED_BITS;
			pos++;
		}
	}

}

__global__ void frodo_key_decode_kernel(uint16_t* out, uint16_t* in)
{ // Decoding

	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	//for (i = 0; i < nwords; i++) 

	if (idx < ((PARAMS_NBAR * PARAMS_NBAR) / 8))
	{
		unsigned int  j, npieces_word = 8;
		//unsigned int nwords = (PARAMS_NBAR * PARAMS_NBAR) / 8;
		uint16_t temp, maskex = ((uint16_t)1 << PARAMS_EXTRACTED_BITS) - 1, maskq = ((uint16_t)1 << PARAMS_LOGQ) - 1;
		uint8_t* pos = (uint8_t*)out;
		uint64_t templong;
		unsigned int  index = idx * npieces_word;

		templong = 0;
		for (j = 0; j < npieces_word; j++) {  // temp = floor(in*2^{-11}+0.5)
			temp = ((in[index] & maskq) + (1 << (PARAMS_LOGQ - PARAMS_EXTRACTED_BITS - 1))) >> (PARAMS_LOGQ - PARAMS_EXTRACTED_BITS);
			templong |= ((uint64_t)(temp & maskex)) << (PARAMS_EXTRACTED_BITS * j);
			index++;
		}
		for (j = 0; j < PARAMS_EXTRACTED_BITS; j++)
			pos[idx * PARAMS_EXTRACTED_BITS + j] = (templong >> (8 * j)) & 0xFF;
	}
}