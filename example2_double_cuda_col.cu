#include <cuda.h>
#include <math.h>
#include <stdio.h>
#include <string.h>
#include <time.h>

void fake_malloc() {
  size_t size = sizeof(double);
  double *u_d;

  cudaMalloc((void **)&u_d, size);
  cudaFree(u_d);
}



__global__ void col_cuda_bvp_set(double *u, int s, int r) {

  int k, m;
  double h = 1.0 / ((double)(s * r));
  double h2 = h * h;

  double pi = 4.0 * (double)atan(1.0);
  double pip = 2.0 * (double)atan(1.0);
  

  m = blockIdx.x * blockDim.x + threadIdx.x;

  for (k = 0; k < s; k++) {
    double x2 = (h * (double)(m * s + k)) * (h * (double)(m * s + k));
    u[m * s + k] = h2 * 20000.0 * exp(-100 * x2) * (1 - 200 * x2);
  }

  if ((blockIdx.x == 0) && (threadIdx.x == 0)) {
    u[0] *= 0.5;
  }
}

__global__ void col_cuda_bvp_1a(double *u, int s, int r) {
  int k, m;
  m = blockIdx.x * blockDim.x + threadIdx.x;

  for (k = 1; k < s; k++)
    u[m * s + k] += u[m * s + k - 1];
}


__global__ void col_cuda_bvp_1b(double *u, int s, int r) {

  int k;
  for (k = 1; k < r; k++) {
    u[(k + 1) * s - 1] += u[k * s - 1];
  }
}


__global__ void col_cuda_bvp_1c(double *u, int s, int r) {
  int m, k;
  m = blockIdx.x * blockDim.x + threadIdx.x;

  if (m != 0) {
    double a = u[m * s - 1];
    for (k = 0; k < s - 1; k++)
      u[m * s + k] += a;
  }
}


__global__ void col_cuda_bvp_2a(double *u, int s, int r) {
  int m, k;
  m = blockIdx.x * blockDim.x + threadIdx.x;
  for (k = s - 2; k >= 0; k--)
    u[m * s + k] += u[m * s + k + 1];
}



__global__ void col_cuda_bvp_2b(double *u, int s, int r) {
  int k;
  for (k = r - 2; k >= 0; k--) {
    u[k * s] += u[(k + 1) * s];
  }
}



__global__ void col_cuda_bvp_2c(double *u, int s, int r) {
  int m, k;
  m = blockIdx.x * blockDim.x + threadIdx.x;
  if (m < r - 1) {
    double a = u[(m + 1) * s];
    for (k = 1; k < s; k++)
      u[m * s + k] += a;
  }
}

double col_bvp(int s, int r, int block_size, double *u_h) {
  int N = r * s;
  size_t size = N * sizeof(double);
  double *u_d;

  cudaMalloc((void **)&u_d, size); 
  fake_malloc();
  col_cuda_bvp_set<<<r / block_size, block_size>>>(u_d, s, r);
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
  cudaMemcpy(u_h, u_d, sizeof(double) * N, cudaMemcpyDeviceToHost);
  cudaFree(u_d);

  return t;
}

double bvp_test_P2(double *f_h, int N) {
  double h = 1.0 / ((double)N);
  double diff = 0.0;
  double sumw = 0.0;
  double x2, y, b;
  int i;

  for (i = 0; i < N; i++) {
    x2 = (h * (double)i) * (h * (double)i);
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
  size_t size = N * sizeof(double);
  double *c_h = (double *)malloc(size);

  
  double time = col_bvp(s, r, bsize, c_h);
  

  if (strcmp(argv[argc - 1], "test") == 0)
    printf("%.30lf", bvp_test_P2(c_h, N));
  else
    printf("%.6lf", time);

  free(c_h);
  return 0;
}
