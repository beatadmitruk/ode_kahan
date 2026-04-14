#include <math.h>
#include <omp.h>
#include <openacc.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

double acc_bvp_solve_row(float *u_h, int s, int r) {
  int k, j, N = r * s;
  float h = 1.0 / ((float)N);
  float h2 = h * h;
  float *u = acc_malloc(sizeof(float) * N);
  float c1 = M_PI * M_PI / 4;
  float c2 = M_PI / 2;

#pragma acc parallel deviceptr(u)
  {
#pragma acc loop independent
    for (j = 0; j < r; j++) {
#pragma acc loop independent
      for (k = 0; k < s; k++) {
        float x = (h * (double)(j * s + k));
        u[j + k * r] = h2 * c1 * cos(c2 * x);
      }
    }
  }
#pragma acc parallel num_gangs(1) deviceptr(u)
  {
    u[0] *= 0.5;
  }
  double time = omp_get_wtime();
  
#pragma acc parallel deviceptr(u)
  {

    for (int k = 1; k < s; k++) {
#pragma acc loop independent
      for (int j = 0; j < r; j++) {
        u[k * r + j] += u[(k - 1) * r + j];
      }
    }
  }

  
#pragma acc parallel num_gangs(1) deviceptr(u)
  {

    for (int k = (s - 1) * r; k < N; k++)
      u[k] += u[k - 1];
  }

  
#pragma acc parallel deviceptr(u)
  {
#pragma acc loop independent
    for (int k = 0; k < s - 1; k++) {
      for (int j = 1; j < r; j++)
        u[j + k * r] += u[(s - 1) * r + j - 1];
    }
  }

  
#pragma acc parallel deviceptr(u)
  {

    for (int k = s - 2; k >= 0; k--) {
#pragma acc loop independent
      for (int j = 0; j < r; j++) {
        u[j + k * r] += u[j + (k + 1) * r];
      }
    }
  }

  
#pragma acc parallel num_gangs(1) deviceptr(u)
  {
    for (int k = r - 2; k >= 0; k--)
      u[k] += u[k + 1];
  }

  
#pragma acc parallel deviceptr(u)
  {
#pragma acc loop independent
    for (int k = 1; k < s; k++) {
      for (int j = 0; j < r - 1; j++)
        u[j + k * r] += u[j + 1];
    }
  }
#pragma acc parallel deviceptr(u) present(u_h)
  {
#pragma acc loop independent
    for (int i = 0; i < N; i++)
      u_h[i] = u[i / s + (i % s) * r];
  }
#pragma acc wait
  time = omp_get_wtime() - time;
  acc_free(u);
  return time;
}

float bvp_test_P1(float *f_h, int N) {
  float c = M_PI / 2;
  float h = 1.0 / ((float)N);
  float diff = 0.0;
  float sumw = 0.0;

  for (int i = 0; i < N; i++) {
    float x = (h * (float)i);
    float y = cos(c * x);
    diff += (f_h[i] - y) * (f_h[i] - y);
    sumw += y * y;
  }
  return sqrt(diff) / sqrt(sumw);
}
int main(int argc, char **argv) {

  int N, r, s;
  s = atoi(argv[1]);
  r = atoi(argv[2]);
  N = s * r;

  double time;

  size_t size = N * sizeof(float);

  float *u = (float *)malloc(size);

  acc_init(acc_device_nvidia);

#pragma acc data copyout(u[0 : N])
  {
#pragma acc data present(u)
    {

      time = acc_bvp_solve_row(u, s, r);
    }
  }

  if (strcmp(argv[argc - 1], "test") == 0)
    printf("%.30f", bvp_test_P1(u, N));
  else
    printf("%.6lf", time);
  free(u);

  return 0;
}
