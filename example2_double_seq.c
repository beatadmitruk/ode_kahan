#include "omp.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

void bvp_set_P2(double *u, int N) {
  int k;
  double h = 1.0 / ((double)N);
  double h2 = h * h;

  for (k = 0; k < N; k++) {
    double x2 = (h * (double)k) * (h * (double)k);
    u[k] = h2 * 20000.0 * exp(-100 * x2) * (1 - 200 * x2);
  }
  u[0] *= 0.5;
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

void bvp_examp(double *u, int N) {
  int k;
  for (k = 1; k < N; k++)
    u[k] += u[k - 1];

  for (k = N - 2; k >= 0; k--)
    u[k] += u[k + 1];
}

int main(int argc, char *argv[]) {
  int N;
  N = atoi(argv[1]);

  size_t size = N * sizeof(double);
  double *f_h = (double *)malloc(size);
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
