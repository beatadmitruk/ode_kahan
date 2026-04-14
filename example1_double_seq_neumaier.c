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
  for (k = 1; k < N; k++)
    u[k] += u[k - 1];

  for (k = N - 2; k >= 0; k--)
    u[k] += u[k + 1];
}

void bvp_kahan(double *u, int N) {
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

void bvp_neumaier(double *u, int N) {
  int k;
  double s, c, t;

  s = u[0];
  c = 0.0f;

  for (k = 1; k < N; k++) {
    t = s + u[k];
    if (fabsf(s) >= fabsf(u[k]))
      c += (s - t) + u[k];
    else
      c += (u[k] - t) + s;

    s = t;
    u[k] = s + c;
  }

  s = u[N - 1];
  c = 0.0f;

  for (k = N - 2; k >= 0; k--) {
    t = s + u[k];
    if (fabsf(s) >= fabsf(u[k]))
      c += (s - t) + u[k];
    else
      c += (u[k] - t) + s;

    s = t;
    u[k] = s + c;
  }
}

void bvp_klein(double *u, int N) {
  int k;
  double s, c, cc, t, zz;

  s = u[0];
  c = 0.0f;
  cc = 0.0f;

  for (k = 1; k < N; k++) {
    t = s + u[k];
    if (fabs(s) >= fabs(u[k]))
      c += (s - t) + u[k];
    else
      c += (u[k] - t) + s;
    s = t;

    zz = 0.0f;
    t = c + zz;
    if (fabsf(c) >= fabsf(zz))
      zz = (c - t) + zz;
    else
      zz = (zz - t) + c;
    c = t;
    u[k] = s + c + cc;
  }

  s = u[N - 1];
  c = 0.0f;
  cc = 0.0f;

  for (k = N - 2; k >= 0; k--) {
    t = s + u[k];
    if (fabsf(s) >= fabsf(u[k]))
      c += (s - t) + u[k];
    else
      c += (u[k] - t) + s;
    s = t;

    zz = 0.0f;
    t = c + zz;
    if (fabsf(c) >= fabsf(zz))
      zz = (c - t) + zz;
    else
      zz = (zz - t) + c;
    c = t;

    u[k] = s + c + cc;
  }
}

int main(int argc, char *argv[]) {
  int N;
  N = atoi(argv[1]);

  size_t size = N * sizeof(double);
  double *f_h = (double *)malloc(size);
  bvp_set_P1(f_h, N);
  double time = omp_get_wtime();
  bvp_neumaier(f_h, N);
  time = omp_get_wtime() - time;

  if (strcmp(argv[argc - 1], "test") == 0)
    printf("%.30lf", bvp_test_P1(f_h, N));
  else
    printf("%.6lf", time);

  free(f_h);

  return 0;
}
