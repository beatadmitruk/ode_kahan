#include <math.h>
#include <omp.h>
#include <openacc.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

double acc_bvp_solve_row(double *u_h, int s, int r) {
  int k, j, N = r * s;
  double h = 1.0 / ((double)N);
  double h2 = h * h;
  double sum, e, x, t;
  double *u = acc_malloc(sizeof(double) * N);

#pragma acc parallel deviceptr(u)
  {
#pragma acc loop independent
    for (j = 0; j < r; j++) {
#pragma acc loop independent
      for (k = 0; k < s; k++) {
        double x2 = (h * (double)(j * s + k)) * (h * (double)(j * s + k));
        u[j + k * r] = h2 * 20000.0 * exp(-100 * x2) * (1 - 200 * x2);
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
#pragma acc loop independent
    for (int j = 0; j < r; j++) {
      sum = u[j], e = 0;
      for (int k = 1; k < s; k++) {
        x = u[j + k * r];
        t = sum + x;

        if (fabs(sum) >= fabs(x))
          e += (sum - t) + x;
        else
          e += (x - t) + sum;

        sum = t;
        u[j + k * r] = sum + e;
      }
    }
  }

  
#pragma acc parallel num_gangs(1) deviceptr(u)
  {
    sum = u[r * (s - 1) - 1], e = 0;
    for (int k = (s - 1) * r; k < N; k++) {
      x = u[k];
      t = sum + x;

      if (fabs(sum) >= fabs(x))
        e += (sum - t) + x;
      else
        e += (x - t) + sum;

      sum = t;
      u[k] = sum + e;
    }
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
#pragma acc loop independent
    for (int j = 0; j < r; j++) {
      sum = u[r * s + j - r], e = 0;
      for (int k = s - 2; k >= 0; k--) {
        x = u[j + k * r];
        t = sum + x;

        if (fabs(sum) >= fabs(x))
          e += (sum - t) + x;
        else
          e += (x - t) + sum;

        sum = t;
        u[j + k * r] = sum + e;
      }
    }
  }

  
#pragma acc parallel num_gangs(1) deviceptr(u)
  {
    sum = u[r - 1], e = 0;
    for (int k = r - 2; k >= 0; k--) {
      x = u[k];
      t = sum + x;

      if (fabs(sum) >= fabs(x))
        e += (sum - t) + x;
      else
        e += (x - t) + sum;

      sum = t;
      u[k] = sum + e;
    }
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
int main(int argc, char **argv) {

  int N, r, s;
  s = atoi(argv[1]);
  r = atoi(argv[2]);
  N = s * r;

  double time;

  size_t size = N * sizeof(double);

  double *u = (double *)malloc(size);

  acc_init(acc_device_nvidia);

#pragma acc data copyout(u[0 : N])
  {
#pragma acc data present(u)
    {

      time = acc_bvp_solve_row(u, s, r);
    }
  }

  if (strcmp(argv[argc - 1], "test") == 0)
    printf("%.30lf", bvp_test_P2(u, N));
  else
    printf("%.6lf", time);
  free(u);

  return 0;
}
