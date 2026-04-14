#include "omp.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

void bvp_set_P1(double *u, int N) {
  int k;
  double h = 1.0 / ((double)N);
  double h2 = h * h;
  double c1 = M_PI * M_PI / 4;
  double c2 = M_PI / 2;

  for (k = 0; k < N; k++) {
    double x = (h * (double)k);
    u[k] = h2 * c1 * cos(c2 * x);
  }
  u[0] *= 0.5;
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

void bvp_examp(double *u, int N) {
  int k;
  double sum, e, y, tmp;
  sum = u[0], e = 0;
  for (k = 1; k < N; k++) {
    tmp = sum;
    y = u[k] + e;
    u[k] = sum = tmp + y;
    e = (tmp - sum) + y;
  }
  sum = u[N - 1], e = 0;
  for (k = N - 2; k >= 0; k--) {
    tmp = sum;
    y = u[k] + e;
    u[k] = sum = tmp + y;
    e = (tmp - sum) + y;
  }
}

int main(int argc, char *argv[]) {
  int N;
  N = atoi(argv[1]);

  size_t size = N * sizeof(double);
  double *f_h = (double *)malloc(size);
  bvp_set_P1(f_h, N);
  double time = omp_get_wtime();
  bvp_examp(f_h, N);
  time = omp_get_wtime() - time;

  if (strcmp(argv[argc - 1], "test") == 0)
    printf("%.30lf", bvp_test_P1(f_h, N));
  else
    printf("%.6lf", time);
  free(f_h);

  return 0;
}
