
// @Author: Arpan Jati
// Adapted from NewHope Reference Codebase and Parallelized using CUDA
// Updated : August 2019

#include <stdint.h>
#include <malloc.h>
#include "indcpa.h"
#include "poly.h"
#include "polyvec.h"
#include "rng.h"
#include "ntt.h"
#include "symmetric.h"

#include <stdlib.h>
#include <float.h>

#include "main.h"


int BLK_SZ[10] = { 4, 8, 16, 24, 32, 48, 64, 128, 192, 256 };

int best_index = -1; float lowest_time = FLT_MAX;

int MP_COUNT = 9;

#ifdef ANALYSIS_MODE

int N_EXPTS = 100;

int COUNT_TO_EXPTS(int COUNT)
{
	if (COUNT >= 32768) return 10 * MP_COUNT;
	else if (COUNT >= 16384) return 20 * MP_COUNT;
	else if (COUNT >= 8192) return 40 * MP_COUNT;
	else if (COUNT >= 4096) return 80 * MP_COUNT;
	else if (COUNT >= 1024) return 160 * MP_COUNT;
	else if (COUNT >= 512) return 320 * MP_COUNT;
	else  return 400 * MP_COUNT;
}

#define TIMING_START(GP_1060_SZ,GP_P6000_SZ,GP_940MX_SZ,GP3_V100_SZ) \
 best_index = -1; lowest_time = FLT_MAX; \
    N_EXPTS = COUNT_TO_EXPTS(COUNT);\
for (int HH = 0; HH < 10; HH++)\
{\
	cudaEvent_t start, stop;\
	cudaEventCreate(&start);\
	cudaEventCreate(&stop);\
	int blockSize = BLK_SZ[HH];\
	int gridSize = (COUNT + blockSize - 1) / blockSize;\
	cudaEventRecord(start);\
	for (int KK = 0; KK < N_EXPTS; KK++)\
	{\

#define TIMING_END(NAME) \
	}\
	cudaEventRecord(stop);\
	cudaEventSynchronize(stop);\
	float milliseconds = 0; \
	cudaEventElapsedTime(&milliseconds, start, stop); \
	milliseconds /= N_EXPTS; \
	if( milliseconds < lowest_time)	{		\
		lowest_time = milliseconds;		\
		best_index = HH;	\
	} \
	printf("\n\n NAME %s | blockSize = %d | Time Elapsed: %f ms. K/s: %d ",NAME, blockSize, milliseconds, \
	(int)((((double)COUNT)* (double)1000.0) / (double)milliseconds)); \
	}\
	printf("\n\n ---BEST BLK SZ = %d | N_EXPTS = %d \n\n", BLK_SZ[best_index], N_EXPTS);

#else // Analysis mode

#define TIMING_START(GP_1060_SZ,GP_P6000_SZ,GP_940MX_SZ,GP3_V100_SZ)  \
	 blockSize = (SELECTED_GPU == GPU_G1060) ? GP_1060_SZ :  \
				 (SELECTED_GPU == GPU_P6000) ? GP_P6000_SZ : \
			     (SELECTED_GPU == GPU_940MX) ? GP_940MX_SZ:  \
											   GP3_V100_SZ;  \
	 gridSize = (COUNT + blockSize - 1) / blockSize;

#define TIMING_END(NAME)


#endif

void HandleError(cudaError_t err, const char* file, int line)
{
	if (err != cudaSuccess) {
		printf("%s in %s at line %d\n", cudaGetErrorString(err),
			file, line);
		exit(EXIT_FAILURE);
	}
}

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



/*************************************************
* Name:        pack_pk
*
* Description: Serialize the public key as concatenation of the
*              serialized vector of polynomials pk
*              and the public seed used to generate the matrix A.
*
* Arguments:   unsigned char *r:          pointer to the output serialized public key
*               poly *pk:            pointer to the input public-key polynomial
*               unsigned char *seed: pointer to the input public seed
**************************************************/
__global__  void pack_pk_n(int COUNT, unsigned char* r, polyvec* pk, unsigned char* seed)
{
	int X = threadIdx.x + blockIdx.x * blockDim.x;
	if (X < COUNT)
	{
		int o_seed = KYBER_SYMBYTES * X;
		int o_r = KYBER_INDCPA_PUBLICKEYBYTES * X;

		int i;
		polyvec_tobytes(r + o_r, pk);
		for (i = 0; i < KYBER_SYMBYTES; i++)
			(r + o_r)[i + KYBER_POLYVECBYTES] = (seed + o_seed)[i];
	}
}

/*************************************************
* Name:        unpack_pk
*
* Description: De-serialize public key from a byte array;
*              approximate inverse of pack_pk
*
* Arguments:   - polyvec *pk:                   pointer to output public-key vector of polynomials
*              - unsigned char *seed:           pointer to output seed to generate matrix A
*              -  unsigned char *packedpk: pointer to input serialized public key
**************************************************/
__global__  void unpack_pk_n(int COUNT, polyvec* pk, unsigned char* seed, unsigned char* packedpk)
{
	int X = threadIdx.x + blockIdx.x * blockDim.x;
	if (X < COUNT)
	{
		int o_packedpk = KYBER_INDCPA_PUBLICKEYBYTES * X;
		int o_seed = KYBER_SYMBYTES * X;

		int i;
		polyvec_frombytes(pk, packedpk + o_packedpk);
		for (i = 0; i < KYBER_SYMBYTES; i++)
			(seed + o_seed)[i] = (packedpk + o_packedpk)[i + KYBER_POLYVECBYTES];
	}
}

/*************************************************
* Name:        pack_sk
*
* Description: Serialize the secret key
*
* Arguments:   - unsigned char *r:  pointer to output serialized secret key
*              -  polyvec *sk: pointer to input vector of polynomials (secret key)
**************************************************/
__global__  void pack_sk_n(int COUNT, unsigned char* r, polyvec* sk)
{
	int X = threadIdx.x + blockIdx.x * blockDim.x;
	if (X < COUNT)
	{
		int o_r = KYBER_INDCPA_SECRETKEYBYTES * X;

		polyvec_tobytes(r + o_r, sk);
	}
}

/*************************************************
* Name:        unpack_sk
*
* Description: De-serialize the secret key;
*              inverse of pack_sk
*
* Arguments:   - polyvec *sk:                   pointer to output vector of polynomials (secret key)
*              -  unsigned char *packedsk: pointer to input serialized secret key
**************************************************/
__global__  void unpack_sk_n(int COUNT, polyvec* sk, unsigned char* packedsk)
{
	int X = threadIdx.x + blockIdx.x * blockDim.x;
	if (X < COUNT)
	{
		int o_r = KYBER_INDCPA_SECRETKEYBYTES * X;

		polyvec_frombytes(sk, packedsk + o_r);
	}
}

/*************************************************
* Name:        pack_ciphertext
*
* Description: Serialize the ciphertext as concatenation of the
*              compressed and serialized vector of polynomials b
*              and the compressed and serialized polynomial v
*
* Arguments:   unsigned char *r:          pointer to the output serialized ciphertext
*               poly *pk:            pointer to the input vector of polynomials b
*               unsigned char *seed: pointer to the input polynomial v
**************************************************/
__global__  void pack_ciphertext_n(int COUNT, unsigned char* r, polyvec* b, poly* v)
{
	int X = threadIdx.x + blockIdx.x * blockDim.x;
	if (X < COUNT)
	{
		int o_r = X * KYBER_INDCPA_BYTES;

		polyvec_compress((r + o_r), b);
		poly_compress((r + o_r) + KYBER_POLYVECCOMPRESSEDBYTES, v);
	}
}

/*************************************************
* Name:        unpack_ciphertext
*
* Description: De-serialize and decompress ciphertext from a byte array;
*              approximate inverse of pack_ciphertext
*
* Arguments:   - polyvec *b:             pointer to the output vector of polynomials b
*              - poly *v:                pointer to the output polynomial v
*              -  unsigned char *c: pointer to the input serialized ciphertext
**************************************************/
__global__  void unpack_ciphertext_n(int COUNT, polyvec* b, poly* v, unsigned char* c)
{
	int X = threadIdx.x + blockIdx.x * blockDim.x;
	if (X < COUNT)
	{
		int o_c = X * KYBER_INDCPA_BYTES;

		polyvec_decompress(b, (c + o_c));
		poly_decompress(v, (c + o_c) + KYBER_POLYVECCOMPRESSEDBYTES);
	}
}

/*************************************************
* Name:        rej_uniform
*
* Description: Run rejection sampling on uniform random bytes to generate
*              uniform random integers mod q
*
* Arguments:   - int16_t *r:               pointer to output buffer
*              - unsigned int len:         requested number of 16-bit integers (uniform mod q)
*              - unsigned char *buf: pointer to input buffer (assumed to be uniform random bytes)
*              - unsigned int buflen:      length of input buffer in bytes
*
* Returns number of sampled 16-bit integers (at most len)
**************************************************/
__device__  unsigned int rej_uniform(poly* r, unsigned int len,
	unsigned char* buf, unsigned int buflen)
{
	int X = threadIdx.x + blockIdx.x * blockDim.x;

	unsigned int ctr, pos;
	uint16_t val;

	//printf("\n UNIFORM G:\n ");

	ctr = pos = 0;
	while (ctr < len && pos + 2 <= buflen)
	{
		val = buf[pos] | ((uint16_t)buf[pos + 1] << 8);
		pos += 2;

		if (val < 19 * KYBER_Q)
		{
			val -= (val >> 12) * KYBER_Q; // Barrett reduction
			r->coeffs[ctr++].threads[X] = (int16_t)val;

			//printf(" %5d = %5d | ", (ctr - 1), val);

			//if ((((ctr - 1) + 1) % 8) == 0)
			//{
				//printf("\n");
			//}

		}
	}

	return ctr;
}

//#define gen_a(A,B)  gen_matrix(A,B,0)
//#define gen_at(A,B) gen_matrix(A,B,1)

/*************************************************
* Name:        gen_matrix
*
* Description: Deterministically generate matrix A (or the transpose of A)
*              from a seed. Entries of the matrix are polynomials that look
*              uniformly random. Performs rejection sampling on output of
*              a XOF
*
* Arguments:   - polyvec *a:                pointer to ouptput matrix A
*              -  unsigned char *seed: pointer to input seed
*              - int transposed:            boolean deciding whether A or A^T is generated
**************************************************/
__global__ void gen_matrix_n(int COUNT, polyvec* a,
	unsigned char* seed, int transposed, unsigned char* large_bufA) // Not static for benchmarking
{
	int X = threadIdx.x + blockIdx.x * blockDim.x;
	if (X < COUNT)
	{
		unsigned int ctr, i, j;
		unsigned int maxnblocks = (530 + XOF_BLOCKBYTES) / XOF_BLOCKBYTES; /* 530 is expected number of required bytes */

	   // int buf_bytes = XOF_BLOCKBYTES * maxnblocks + 1;

		int o_largeBuffer = X * LARGE_BUFFER_SZ;


		unsigned char* buf = (large_bufA + o_largeBuffer);//(unsigned char*)malloc(buf_bytes);

		xof_state state;

		//printf("\n gen_matrix START ---------------------\n ");

		int o_seed = KYBER_SYMBYTES * X;

		for (i = 0; i < KYBER_K; i++)
		{
			for (j = 0; j < KYBER_K; j++)
			{
				if (transposed)
				{
					kyber_shake128_absorb(&state, seed + o_seed, i, j);
				}
				else
				{
					kyber_shake128_absorb(&state, seed + o_seed, j, i);
				}

				kyber_shake128_squeezeblocks(buf, maxnblocks, &state);

				ctr = rej_uniform(&(a[i].vec[j]), KYBER_N, buf, maxnblocks * XOF_BLOCKBYTES);

				//printf("\n I:%d | J: %d | CTR: %d", i, j, ctr);

				while (ctr < KYBER_N)
				{
					kyber_shake128_squeezeblocks(buf, 1, &state);

					ctr += rej_uniform(&(a[i].vec[j]) + (ctr * N_TESTS),
						KYBER_N - ctr, buf, XOF_BLOCKBYTES);

					//printf("\n I:%d | J: %d | CTR: %d", i, j, ctr);
				}

				// print_data_d("\n\n\n buf", buf, XOF_BLOCKBYTES * maxnblocks + 1);

				//__syncthreads();

			}
		}

		//free(buf);

		//printf("\n gen_matrix END ---------------------\n ");

	}

}

int blockSize = BLOCK_SIZE;
int gridSize = (N_TESTS + blockSize - 1) / blockSize;

//int blockSize = N_TESTS;
//int gridSize = 1;

__global__ void print_poly(int COUNT, poly* poly)
{
	int X = threadIdx.x + blockIdx.x * blockDim.x;
	if (X < COUNT)
	{
		printf("GPU | PRINT POLY\n");

		for (int i = 0; i < 256; i++)
		{
			printf("%d ", poly->coeffs[i].threads[0]);

			//if ((i + 1) % 2 == 0)
			//{
			//	printf(" ");
			//}

			if ((i + 1) % 8 == 0)
			{
				printf("\n");
			}
		}

		printf("\n");

	}
}


__global__ void print_polyvec(int COUNT, polyvec* polyvec)
{
	int X = threadIdx.x + blockIdx.x * blockDim.x;
	if (X < COUNT)
	{
		printf("GPU POLYVEC ---------------------------\n");

		for (int v = 0; v < KYBER_K; v++)
		{
			printf("VEC---- %d\n", v);

			for (int i = 0; i < 256; i++)
			{
				printf("%d ", (polyvec->vec[v]).coeffs[i].threads[X]);

				if ((i + 1) % 8 == 0)
				{
					printf("\n");
				}
			}
		}

		printf("\n");
	}
}

/*************************************************
* Name:        indcpa_keypair
*
* Description: Generates public and private key for the CPA-secure
*              public-key encryption scheme underlying Kyber
*
* Arguments:   - unsigned char *pk: pointer to output public key (of length KYBER_INDCPA_PUBLICKEYBYTES bytes)
*              - unsigned char *sk: pointer to output private key (of length KYBER_INDCPA_SECRETKEYBYTES bytes)
**************************************************/
void indcpa_keypair(int COUNT, poly_set4* ps, unsigned char* pk, unsigned char* sk,
	unsigned char* rng_buf, cudaStream_t stream)
{
	//printf("\nCOUNT: %d \n", COUNT);
	//printf("DEFAULT BLOCK SIZE: %d \n", blockSize);

	polyvec* a = ps->AV; // [KYBER_K]

	polyvec* e = ps->av;
	polyvec* pkpv = ps->bv;
	polyvec* skpv = ps->cv;

	poly* poly_temp = ps->a;

	// unsigned char* buf = ps->seed;

	unsigned char* publicseed = rng_buf;
	unsigned char* noiseseed = rng_buf + KYBER_SYMBYTES;
	int i;
	unsigned char nonce = 0;

	unsigned char* large_bufA = ps->large_buffer_a;

	//randombytes_device(buf, KYBER_SYMBYTES);

	TIMING_START(64, 32, 32, 32)
		sha3_512_n << < gridSize, blockSize, 0, stream >> > (COUNT, rng_buf, rng_buf, KYBER_SYMBYTES);
	TIMING_END("sha3_512_n")

		TIMING_START(128, 192, 192, 256)
		gen_matrix_n << <gridSize, blockSize, 0, stream >> > (COUNT, a, publicseed, 0, large_bufA);
	TIMING_END("gen_matrix_n")

		for (i = 0; i < KYBER_K; i++)
		{
			TIMING_START(64, 192, 192, 128)
				poly_getnoise << <gridSize, blockSize, 0, stream >> > (COUNT, skpv->vec + i, noiseseed, nonce++);
			TIMING_END("poly_getnoise")
		}

	for (i = 0; i < KYBER_K; i++)
	{
		TIMING_START(64, 192, 192, 128)
			poly_getnoise << <gridSize, blockSize, 0, stream >> > (COUNT, e->vec + i, noiseseed, nonce++);
		TIMING_END("poly_getnoise")
	}

	//cudaFuncSetCacheConfig(polyvec_ntt_n, cudaFuncCachePreferL1);
	TIMING_START(256, 256, 256, 128)
		polyvec_ntt_n << <gridSize, blockSize, 0, stream >> > (COUNT, skpv);
	TIMING_END("polyvec_ntt_n")

		TIMING_START(256, 256, 256, 128)
		polyvec_ntt_n << <gridSize, blockSize, 0, stream >> > (COUNT, e);
	TIMING_END("polyvec_ntt_n")

		//print_polyvec << <gridSize, blockSize >> > (skpv);
		//print_polyvec << <gridSize, blockSize >> > (e);

		// matrix-vector multiplication
		for (i = 0; i < KYBER_K; i++)
		{
			TIMING_START(128, 256, 256, 64)
				polyvec_pointwise_acc_n << <gridSize, blockSize, 0, stream >> > (COUNT, &(pkpv->vec[i]), &(a[i]), skpv, poly_temp);
			TIMING_END("polyvec_pointwise_acc_n")

			TIMING_START(256, 256, 256, 32)
				poly_frommont_n << <gridSize, blockSize, 0, stream >> > (COUNT, &(pkpv->vec[i]));
			TIMING_END("poly_frommont_n")
				//print_poly << <gridSize, blockSize,  0, stream>> > (&(pkpv->vec[i]));
		}

	//print_polyvec << <gridSize, blockSize,  0, stream>> > (pkpv);
	//print_polyvec << <gridSize, blockSize,  0, stream>> > (e);

	TIMING_START(256, 256, 256, 32)
		polyvec_add_n << <gridSize, blockSize, 0, stream >> > (COUNT, pkpv, pkpv, e);
	TIMING_END("polyvec_add_n")

	TIMING_START(128, 256, 256, 32)
		polyvec_reduce_n << <gridSize, blockSize, 0, stream >> > (COUNT, pkpv);
	TIMING_END("polyvec_reduce_n")

	TIMING_START(16, 8, 8, 64)
		pack_sk_n << <gridSize, blockSize, 0, stream >> > (COUNT, sk, skpv);
	TIMING_END("pack_sk_n")

	TIMING_START(16, 8, 8, 16)
		pack_pk_n << <gridSize, blockSize, 0, stream >> > (COUNT, pk, pkpv, publicseed);
	TIMING_END("pack_pk_n")
}

/*************************************************
* Name:        indcpa_enc
*
* Description: Encryption function of the CPA-secure
*              public-key encryption scheme underlying Kyber.
*
* Arguments:   - unsigned char *c:          pointer to output ciphertext (of length KYBER_INDCPA_BYTES bytes)
*              -  unsigned char *m:    pointer to input message (of length KYBER_INDCPA_MSGBYTES bytes)
*              -  unsigned char *pk:   pointer to input public key (of length KYBER_INDCPA_PUBLICKEYBYTES bytes)
*              -  unsigned char *coin: pointer to input random coins used as seed (of length KYBER_SYMBYTES bytes)
*                                           to deterministically generate all randomness
**************************************************/
void indcpa_enc(int COUNT, poly_set4* ps, unsigned char* c,
	unsigned char* m,
	unsigned char* pk,
	unsigned char* coins, cudaStream_t stream)
{
	polyvec* at = ps->AV; // [KYBER_K]

	polyvec* sp = ps->av;
	polyvec* pkpv = ps->bv;
	polyvec* ep = ps->cv;
	polyvec* bp = ps->dv;

	poly* v = ps->a;
	poly* k = ps->b;
	poly* epp = ps->c;

	poly* poly_temp = ps->d;

	unsigned char* seed = ps->seed;
	int i;
	unsigned char nonce = 0;

	unsigned char* large_bufA = ps->large_buffer_a;

	TIMING_START(16, 8, 8, 256)
		unpack_pk_n << <gridSize, blockSize, 0, stream >> > (COUNT, pkpv, seed, pk);
	TIMING_END("unpack_pk_n")

	TIMING_START(16, 256, 256, 32)
		poly_frommsg_n << <gridSize, blockSize, 0, stream >> > (COUNT, k, m);
	TIMING_END("poly_frommsg_n")

	TIMING_START(128, 192, 192, 256)
		gen_matrix_n << <gridSize, blockSize, 0, stream >> > (COUNT, at, seed, 1, large_bufA);
	TIMING_END("gen_matrix_n")

		for (i = 0; i < KYBER_K; i++)
		{
			TIMING_START(64, 192, 192, 128)
				poly_getnoise << <gridSize, blockSize,  0, stream >> > (COUNT, sp->vec + i, coins, nonce++);
			TIMING_END("poly_getnoise")
		}

	for (i = 0; i < KYBER_K; i++)
	{
		TIMING_START(64, 192, 192, 128)
			poly_getnoise << <gridSize, blockSize, 0, stream >> > (COUNT, ep->vec + i, coins, nonce++);
		TIMING_END("poly_getnoise")
	}

	TIMING_START(64, 192, 192, 128)
		poly_getnoise << <gridSize, blockSize, 0, stream >> > (COUNT, epp, coins, nonce++);
	TIMING_END("poly_getnoise")

	TIMING_START(256, 256, 256, 128)
		polyvec_ntt_n << <gridSize, blockSize, 0, stream >> > (COUNT, sp);
	TIMING_END("polyvec_ntt_n")

		// matrix-vector multiplication
		for (i = 0; i < KYBER_K; i++)
		{
			TIMING_START(128, 256, 256, 64)
				polyvec_pointwise_acc_n << <gridSize, blockSize, 0, stream >> > (COUNT, &(bp->vec[i]), &(at[i]), sp, poly_temp);
			TIMING_END("polyvec_pointwise_acc_n")
		}

	TIMING_START(128, 256, 256, 64)
		polyvec_pointwise_acc_n << <gridSize, blockSize, 0, stream >> > (COUNT, v, pkpv, sp, poly_temp);
	TIMING_END("polyvec_pointwise_acc_n")

	TIMING_START(128, 256, 256, 256)
		polyvec_invntt_n << <gridSize, blockSize, 0, stream >> > (COUNT, bp);
	TIMING_END("polyvec_invntt_n")

	TIMING_START(128, 256, 256, 256)
		poly_invntt_n << <gridSize, blockSize, 0, stream >> > (COUNT, v);
	TIMING_END("poly_invntt_n")

	TIMING_START(128, 128, 128, 32)
		polyvec_add_n << <gridSize, blockSize, 0, stream >> > (COUNT, bp, bp, ep);
	TIMING_END("polyvec_add_n")

	TIMING_START(128, 128, 128, 32)
		poly_add_n << <gridSize, blockSize, 0, stream >> > (COUNT, v, v, epp);
	TIMING_END("poly_add_n")

	TIMING_START(128, 128, 128, 32)
		poly_add_n << <gridSize, blockSize, 0, stream >> > (COUNT, v, v, k);
	TIMING_END("poly_add_n")

	TIMING_START(128, 256, 256, 32)
		polyvec_reduce_n << <gridSize, blockSize, 0, stream >> > (COUNT, bp);
	TIMING_END("polyvec_reduce_n")

	TIMING_START(128, 256, 256, 32)
		poly_reduce_n << <gridSize, blockSize, 0, stream >> > (COUNT, v);
	TIMING_END("poly_reduce_n")

	TIMING_START(32, 8, 8, 16)
		pack_ciphertext_n << <gridSize, blockSize, 0, stream >> > (COUNT, c, bp, v);
	TIMING_END("pack_ciphertext_n")
}

/*************************************************
* Name:        indcpa_dec
*
* Description: Decryption function of the CPA-secure
*              public-key encryption scheme underlying Kyber.
*
* Arguments:   - unsigned char *m:        pointer to output decrypted message (of length KYBER_INDCPA_MSGBYTES)
*              -  unsigned char *c:  pointer to input ciphertext (of length KYBER_INDCPA_BYTES)
*              -  unsigned char *sk: pointer to input secret key (of length KYBER_INDCPA_SECRETKEYBYTES)
**************************************************/
void indcpa_dec(int COUNT, poly_set4* ps, unsigned char* m,
	unsigned char* c,
	unsigned char* sk, cudaStream_t stream)
{
	polyvec* bp = ps->av;
	polyvec* skpv = ps->bv;

	poly* v = ps->a;
	poly* mp = ps->b;
	poly* poly_temp = ps->c;

	TIMING_START(16, 8, 8, 256)
		unpack_ciphertext_n << <gridSize, blockSize, 0, stream >> > (COUNT, bp, v, c);
	TIMING_END("unpack_ciphertext_n")

	TIMING_START(16, 8, 8, 32)
		unpack_sk_n << <gridSize, blockSize, 0, stream >> > (COUNT, skpv, sk);
	TIMING_END("unpack_sk_n")

	TIMING_START(128, 256, 256, 128)
		polyvec_ntt_n << <gridSize, blockSize, 0, stream >> > (COUNT, bp);
	TIMING_END("polyvec_ntt_n")

	TIMING_START(128, 256, 256, 64)
		polyvec_pointwise_acc_n << <gridSize, blockSize, 0, stream >> > (COUNT, mp, skpv, bp, poly_temp);
	TIMING_END("polyvec_pointwise_acc_n")

	TIMING_START(128, 256, 256, 256)
		poly_invntt_n << <gridSize, blockSize, 0, stream >> > (COUNT, mp);
	TIMING_END("poly_invntt_n")

	TIMING_START(128, 128, 128, 32)
		poly_sub_n << <gridSize, blockSize, 0, stream >> > (COUNT, mp, v, mp);
	TIMING_END("poly_sub_n")

	TIMING_START(128, 256, 256, 32)
		poly_reduce_n << <gridSize, blockSize, 0, stream >> > (COUNT, mp);
	TIMING_END("poly_reduce_n")

	TIMING_START(64, 128, 128, 32)
		poly_tomsg_n << <gridSize, blockSize, 0, stream >> > (COUNT, m, mp);
	TIMING_END("poly_tomsg_n")
}
