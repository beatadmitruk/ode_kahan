#include "omp.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

void bvp_examp(float *u, int N) {
  int k;
  for (k = 1; k < N; k++)
    u[k] += u[k - 1];

  for (k = N - 2; k >= 0; k--)
    u[k] += u[k + 1];
}

void bvp_set_P1(float *u, int N) {
  int k;
  float h = 1.0 / ((float)N);
  float h2 = h * h;
  float c1 = M_PI * M_PI / 4;
  float c2 = M_PI / 2;

  for (k = 0; k < N; k++) {
    float x = (h * (float)k);
    u[k] = h2 * c1 * cos(c2 * x);
  }
  u[0] *= 0.5;
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

int main(int argc, char *argv[]) {
  int N;
  N = atoi(argv[1]);

  size_t size = N * sizeof(float);
  float *f_h = (float *)malloc(size);
  bvp_set_P1(f_h, N);
  double time = omp_get_wtime();
  bvp_examp(f_h, N);
  time = omp_get_wtime() - time;

  if (strcmp(argv[argc - 1], "test") == 0)
    printf("%.30f", bvp_test_P1(f_h, N));
  else
    printf("%.6lf", time);
  free(f_h);

  return 0;
}
