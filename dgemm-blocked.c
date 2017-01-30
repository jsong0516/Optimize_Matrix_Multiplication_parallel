#include <immintrin.h>

const char* dgemm_desc = "Simple blocked dgemm.";

#if !defined(BLOCK_SIZE)
#define BLOCK_SIZE 64
#endif

#if !defined(BIG_BLOCK_SIZE)
#define BIG_BLOCK_SIZE 180
#endif

#define min(a,b) (((a)<(b))?(a):(b))

static double copy_a[BLOCK_SIZE*BLOCK_SIZE] __attribute__((aligned(32)));
static double copy_b[BLOCK_SIZE*BLOCK_SIZE] __attribute__((aligned(32)));
static double copy_c[BLOCK_SIZE*BLOCK_SIZE] __attribute__((aligned(32)));

inline void copy_to_arr(int lda, int m, int n, double *org, double *des){
  for(int j = 0; j < n; ++j){
    memcpy(des+m*j, org+lda*j, m*8);
  }
}

inline void copy_to_blk(int lda, int m, int n, double *org, double *des){
  for(int j = 0; j < n; ++j)
    memcpy(des+lda*j, org+m*j, m*8);
}

inline void do_fast_4x4_block (int lda, double *A, double *B, double *C){

  register __m256d colc1 = _mm256_load_pd(C);
  register __m256d colc2 = _mm256_load_pd(C+lda);
  register __m256d colc3 = _mm256_load_pd(C+2*lda);
  register __m256d colc4 = _mm256_load_pd(C+3*lda);
  register __m256d cola;
  __m256d brod1, brod2, brod3, brod4;

  for(int i = 0; i < 4; ++i){
    cola = _mm256_load_pd(A+i*lda);
    brod1 = _mm256_set1_pd(B[i]);
    brod2 = _mm256_set1_pd(B[i+lda]);
    brod3 = _mm256_set1_pd(B[i+2*lda]);
    brod4 = _mm256_set1_pd(B[i+3*lda]);
    colc1 = _mm256_add_pd(colc1, _mm256_mul_pd(brod1, cola));
    colc2 = _mm256_add_pd(colc2, _mm256_mul_pd(brod2, cola));
    colc3 = _mm256_add_pd(colc3, _mm256_mul_pd(brod3, cola));
    colc4 = _mm256_add_pd(colc4, _mm256_mul_pd(brod4, cola));
  }

  _mm256_store_pd(C, colc1);
  _mm256_store_pd(C+lda, colc2);
  _mm256_store_pd(C+2*lda, colc3);
  _mm256_store_pd(C+3*lda, colc4);
}

void do_small_square_block (int lda, int M, int N, int K, double* A, double* B, double* C)
{
  for(int j = 0; j < N; j+=4)
    for(int k = 0; k < K; k+=4)
      for(int i = 0; i < M; i+=4)
  do_fast_4x4_block(BLOCK_SIZE, copy_a+i+k*BLOCK_SIZE, copy_b+k+j*BLOCK_SIZE, copy_c+i+j*BLOCK_SIZE);
  copy_to_blk(lda, M, N, copy_c, C);
}

void do_big_block (int lda, int M, int N, int K, double* A, double* B, double* C)
{
  for (int j = 0; j < N; j += BLOCK_SIZE)
    for (int k = 0; k < K; k += BLOCK_SIZE){
      int NN = min (BLOCK_SIZE, N-j);
      int KK = min (BLOCK_SIZE, K-k);
      memset(copy_b, 0, BLOCK_SIZE*BLOCK_SIZE*8);
      copy_to_arr(lda, KK, NN, B+k+j*lda, copy_b);
      for (int i = 0; i < M; i += BLOCK_SIZE)
  {
    int MM = min (BLOCK_SIZE, M-i);
    if(MM+NN+KK == 3*BLOCK_SIZE){
      copy_to_arr(lda, BLOCK_SIZE, BLOCK_SIZE, A+i+k*lda, copy_a);
      copy_to_arr(lda, BLOCK_SIZE, BLOCK_SIZE, C+i+j*lda, copy_c);
      do_small_square_block(lda, BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE, copy_a, copy_b, C+i+j*lda);
    }
    else{
      memset(copy_a, 0, BLOCK_SIZE*BLOCK_SIZE*8);
      memset(copy_c, 0, BLOCK_SIZE*BLOCK_SIZE*8);
      for(int kk = 0; kk < KK; ++kk) memcpy(A+i+(k+kk)*lda, copy_a+kk*BLOCK_SIZE, MM*8);
      for(int jj = 0; jj < NN; ++jj) memcpy(C+i+(j+jj)*lda, copy_c+jj*BLOCK_SIZE, MM*8);
      do_small_square_block(lda, MM-1 + 4 - (MM-1 % 4), NN-1 + 4 - (NN-1 % 4), KK-1 + 4 - (KK-1 % 4), copy_a, copy_b, C+i+j*lda);
    }
  }
    }
}

void square_dgemm (int lda, double* A, double* B, double* C)
{
  for (int j = 0; j < lda; j += BIG_BLOCK_SIZE)
    for (int k = 0; k < lda; k += BIG_BLOCK_SIZE)
      for (int i = 0; i < lda; i += BIG_BLOCK_SIZE)
  {
    int M = min (BIG_BLOCK_SIZE, lda-i);
    int N = min (BIG_BLOCK_SIZE, lda-j);
    int K = min (BIG_BLOCK_SIZE, lda-k);
    do_big_block(lda, M, N, K, A + i + k*lda, B + k + j*lda, C + i + j*lda);
  }
}