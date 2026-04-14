#include <cuda.h>
#include <math.h>
#include <stdio.h>
#include <string.h>
#include <time.h>

void fake_malloc() {
  size_t size = sizeof(float);
  float *u_d;

  cudaMalloc((void **)&u_d, size);
  cudaFree(u_d);
}



__global__ void col_cuda_bvp_set(float *u, int s, int r) {

  int k, m;
  float h = 1.0 / ((float)(s * r));
  float h2 = h * h;

  float pi = 4.0 * (float)atan(1.0);
  float pip = 2.0 * (float)atan(1.0);
  

  m = blockIdx.x * blockDim.x + threadIdx.x;

  for (k = 0; k < s; k++) {
    float x2 = (h * (float)(m * s + k)) * (h * (float)(m * s + k));
    u[m * s + k] = h2 * 20000.0 * exp(-100 * x2) * (1 - 200 * x2);
  }

  if ((blockIdx.x == 0) && (threadIdx.x == 0)) {
    u[0] *= 0.5;
  }
}

__global__ void col_cuda_bvp_1a(float *u, int s, int r) {
  int k, m;
  m = blockIdx.x * blockDim.x + threadIdx.x;

  float sum, e;
  sum = u[m * s];
  e = 0.0f;

  for (k = 1; k < s; k++) {
    float x = u[m * s + k];
    float t = sum + x;

    if (fabsf(sum) >= fabsf(x))
      e += (sum - t) + x;
    else
      e += (x - t) + sum;

    sum = t;
    u[m * s + k] = sum + e;
  }
}


__global__ void col_cuda_bvp_1b(float *u, int s, int r) {
  float x, t;
  float sum, e;
  sum = u[s - 1];
  e = 0.0f;

  int k;
  for (k = 1; k < r; k++) {
    x = u[(k + 1) * s - 1];
    t = sum + x;

    if (fabsf(sum) >= fabsf(x))
      e += (sum - t) + x;
    else
      e += (x - t) + sum;

    sum = t;
    u[(k + 1) * s - 1] = sum + e;
  }
}


__global__ void col_cuda_bvp_1c(float *u, int s, int r) {
  int m, k;
  m = blockIdx.x * blockDim.x + threadIdx.x;

  if (m != 0) {
    float a = u[m * s - 1];
    for (k = 0; k < s - 1; k++)
      u[m * s + k] += a;
  }
}


__global__ void col_cuda_bvp_2a(float *u, int s, int r) {

  int m, k;
  m = blockIdx.x * blockDim.x + threadIdx.x;
  float sum, e;
  sum = u[(m + 1) * s - 1];
  e = 0.0f;

  for (k = s - 2; k >= 0; k--) {
    float x = u[m * s + k];
    float t = sum + x;

    if (fabsf(sum) >= fabsf(x))
      e += (sum - t) + x;
    else
      e += (x - t) + sum;

    sum = t;
    u[m * s + k] = sum + e;
  }
}

__global__ void col_cuda_bvp_2b(float *u, int s, int r) {
  float sum, e;
  sum = u[(r - 1) * s];
  e = 0.0f;

  int k;
  for (k = r - 2; k >= 0; k--) {
    float x = u[k * s];
    float t = sum + x;

    if (fabsf(sum) >= fabsf(x))
      e += (sum - t) + x;
    else
      e += (x - t) + sum;

    sum = t;
    u[k * s] = sum + e;
  }
}

__global__ void col_cuda_bvp_2c(float *u, int s, int r) {
  int m, k;
  m = blockIdx.x * blockDim.x + threadIdx.x;
  if (m < r - 1) {
    float a = u[(m + 1) * s];
    for (k = 1; k < s; k++)
      u[m * s + k] += a;
  }
}

double col_bvp(int s, int r, int block_size, float *u_h) {
  int N = r * s;
  size_t size = N * sizeof(float);
  float *u_d;

  cudaMalloc((void **)&u_d, size); 

  col_cuda_bvp_set<<<r / block_size, block_size>>>(u_d, s, r);
  fake_malloc();
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaEventRecord(start);

  col_cuda_bvp_1a<<<r / block_size, block_size>>>(u_d, s, r);
  col_cuda_bvp_1b<<<1, 1>>>(u_d, s, r);
  col_cuda_bvp_1c<<<r / block_size, block_size>>>(u_d, s, r);
  col_cuda_bvp_2a<<<r / block_size, block_size>>>(u_d, s, r);
  col_cuda_bvp_2b<<<1, 1>>>(u_d, s, r);
  col_cuda_bvp_2c<<<r / block_size, block_size>>>(u_d, s, r);

  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  float ms;
  cudaEventElapsedTime(&ms, start, stop);

  double t = (ms) / 1000.0;
  cudaMemcpy(u_h, u_d, sizeof(float) * N, cudaMemcpyDeviceToHost);
  cudaFree(u_d);

  return t;
}

float bvp_test_P2(float *f_h, int N) {
  float h = 1.0 / ((float)N);
  float diff = 0.0;
  float sumw = 0.0;
  float x2, y, b;
  int i;

  for (i = 0; i < N; i++) {
    x2 = (h * (float)i) * (h * (float)i);
    y = 100 * exp(-100 * x2) - 100 * exp(-100.0);
    b = (f_h[i] - y);
    diff += b * b;
    sumw += y * y;
  }
  return sqrt(diff) / sqrt(sumw);
}


int main(int argc, char *argv[]) {
  int N, r, s, bsize; 
  s = atoi(argv[1]);
  r = atoi(argv[2]);
  bsize = atoi(argv[3]);
  N = r * s;

  size_t size = N * sizeof(float);
  float *c_h = (float *)malloc(size);

  
  double time = col_bvp(s, r, bsize, c_h);
  

  float h = 1.0 / ((float)N);
  
  

  float diff = 0.0;
  float sumw = 0.0;
  
  for (int i = 0; i < N; i++) {
    float x2 = (h * (float)i) * (h * (float)i);
    float y = 100 * exp(-100 * x2) - 100 * exp(-100.0);

    diff += (c_h[i] - y) * (c_h[i] - y);
    sumw += y * y;
  }

  if (strcmp(argv[argc - 1], "test") == 0)
    printf("%.30f", bvp_test_P2(c_h, N));
  else
    printf("%.6lf", time);

  free(c_h);
  return 0;
}
