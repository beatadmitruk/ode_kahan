#include "omp.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

void bvp_set_P2(float *u, int N) {
  int k;
  float h = 1.0 / ((float)N);
  float h2 = h * h;

  for (k = 0; k < N; k++) {
    float x2 = (h * (float)k) * (h * (float)k);
    u[k] = h2 * 20000.0 * exp(-100 * x2) * (1 - 200 * x2);
  }
  u[0] *= 0.5;
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

void bvp_examp(float *u, int N) {
  int k;
  for (k = 1; k < N; k++)
    u[k] += u[k - 1];

  for (k = N - 2; k >= 0; k--)
    u[k] += u[k + 1];
}

int main(int argc, char *argv[]) {
  int N;
  N = atoi(argv[1]);

  size_t size = N * sizeof(float);
  float *f_h = (float *)malloc(size);
  bvp_set_P2(f_h, N);
  double time = omp_get_wtime();

  bvp_examp(f_h, N);
  time = omp_get_wtime() - time;

  if (strcmp(argv[argc - 1], "test") == 0)
    printf("%.30lf", bvp_test_P2(f_h, N));
  else
    printf("%.6lf", time);

  free(f_h);

  return 0;
}
