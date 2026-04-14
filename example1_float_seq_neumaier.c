#include "omp.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

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

void bvp_examp(float *u, int N) {
  int k;
  for (k = 1; k < N; k++)
    u[k] += u[k - 1];

  for (k = N - 2; k >= 0; k--)
    u[k] += u[k + 1];
}

void bvp_kahan(float *u, int N) {
  int k;
  float sum, e, y, tmp;
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

void bvp_neumaier(float *u, int N) {
  int k;
  float s, c, t;

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

void bvp_klein(float *u, int N) {
  int k;
  float s, c, cc, t, zz;

  s = u[0];
  c = 0.0f;
  cc = 0.0f;

  for (k = 1; k < N; k++) {
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

  size_t size = N * sizeof(float);
  float *f_h = (float *)malloc(size);
  bvp_set_P1(f_h, N);
  double time = omp_get_wtime();
  bvp_neumaier(f_h, N);
  time = omp_get_wtime() - time;

  if (strcmp(argv[argc - 1], "test") == 0)
    printf("%.30f", bvp_test_P1(f_h, N));
  else
    printf("%.6lf", time);

  free(f_h);

  return 0;
}
