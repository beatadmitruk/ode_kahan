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


__global__ void row_cuda_bvp_set(double *u, int s, int r) {
  int k, m;
  m = blockIdx.x * blockDim.x + threadIdx.x;
  double h = 1.0 / ((double)(s * r));
  double h2 = h * h;

  double c1 = M_PI * M_PI / 4;
  double c2 = M_PI / 2;

  for (k = 0; k < s; k++) {
    double x = (h * (double)(m * s + k));
    u[m + k * r] = h2 * c1 * cos(c2 * x);
  }

  if ((blockIdx.x == 0) && (threadIdx.x == 0))
    u[0] *= 0.5;
}

__global__ void row_cuda_bvp_1a(double *u, int s, int r) {
  int k, m;
  m = blockIdx.x * blockDim.x + threadIdx.x;

  for (k = 1; k < s; k++)
    u[m + k * r] += u[m + (k - 1) * r];
}

__global__ void row_cuda_bvp_1b(double *u, int s, int r) {
  int k;

  for (k = (s - 1) * r + 1; k < r * s; k++)
    u[k] += u[k - 1];
}

__global__ void row_cuda_bvp_1c2a(double *u, int s, int r) {
  int m, k;
  m = blockIdx.x * blockDim.x + threadIdx.x;
  
  if (m > 0) {
    double a = u[(s - 1) * r + m - 1];
    for (k = 0; k < s - 1; k++)
      u[m + k * r] += a;
  }

  
  for (k = s - 2; k >= 0; k--)
    u[m + k * r] += u[m + (k + 1) * r];
}

__global__ void row_cuda_bvp_2b(double *u, int s, int r) {
  int k;
  for (k = r - 2; k >= 0; k--)
    u[k] += u[k + 1];
}

__global__ void row_cuda_bvp_2c(double *u, int s, int r) {
  int m, k;
  m = blockIdx.x * blockDim.x + threadIdx.x;

  if (m != r - 1) {
    double a = u[m + 1];
    for (k = 1; k < s; k++)
      u[m + k * r] += a;
  }
}

__global__ void row_to_col_cuda(double *u, double *c, int s, int r) {
  int m, k, i;
  m = blockIdx.x * blockDim.x + threadIdx.x;
  for (k = 0; k < s; k++) {
    i = m + k * r;
    c[i] = u[i / s + (i % s) * r];
  }
}

double row_bvp(int s, int r, int block_size, double *u_h) {
  int N = r * s, j;
  size_t size = N * sizeof(double);
  double *vd_tmp, *u_d, *u_c;
  vd_tmp = (double *)malloc(r * sizeof(double));

  cudaMalloc((void **)&u_d, size); 
  cudaMalloc((void **)&u_c, size);
  
  row_cuda_bvp_set<<<r / block_size, block_size>>>(u_d, s, r);
  fake_malloc();
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaEventRecord(start);
  row_cuda_bvp_1a<<<r / block_size, block_size>>>(u_d, s, r);

  
  cudaMemcpy(vd_tmp, &u_d[(s - 1) * r], sizeof(double) * r,
             cudaMemcpyDeviceToHost);
  for (j = 1; j < r; j++)
    vd_tmp[j] = vd_tmp[j] + vd_tmp[j - 1];

  cudaMemcpy(&u_d[(s - 1) * r], vd_tmp, sizeof(double) * r,
             cudaMemcpyHostToDevice);

  row_cuda_bvp_1c2a<<<r / block_size, block_size>>>(u_d, s, r);

  
  cudaMemcpy(vd_tmp, u_d, sizeof(double) * r, cudaMemcpyDeviceToHost);
  for (j = r - 2; j >= 0; j--)
    vd_tmp[j] = vd_tmp[j] + vd_tmp[j + 1];

  cudaMemcpy(u_d, vd_tmp, sizeof(double) * r, cudaMemcpyHostToDevice);

  row_cuda_bvp_2c<<<r / block_size, block_size>>>(u_d, s, r);
  row_to_col_cuda<<<r / block_size, block_size>>>(u_d, u_c, s, r);

  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  float ms;
  cudaEventElapsedTime(&ms, start, stop);

  double t = (ms) / 1000.0;
  cudaMemcpy(u_h, u_c, sizeof(double) * N, cudaMemcpyDeviceToHost);
  cudaFree(u_c);
  cudaFree(u_d);
  free(vd_tmp);
  return t;
  ;
}

double bvp_test_P1(double *f_h, int N) {
  double c = M_PI / 2;
  double h = 1.0 / ((double)N);
  double diff = 0.0;
  double sumw = 0.0;

  for (int i = 0; i < N; i++) {
    double x = (h * (double)i);
    double y = cos(c * x);
    diff += (f_h[i] - y) * (f_h[i] - y);
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
  double *r_h = (double *)malloc(size);

  
  double time = row_bvp(s, r, bsize, r_h);
  

  if (strcmp(argv[argc - 1], "test") == 0)
    printf("%.30lf", bvp_test_P1(r_h, N));
  else
    printf("%.6lf", time);

  free(r_h);
  return 0;
}
