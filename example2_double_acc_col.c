#include <math.h>
#include <omp.h>
#include <openacc.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

double acc_bvp_solve_col(double *u, int s, int r) {
  int k, N = r * s;
  double h = 1.0 / ((double)N);
  double h2 = h * h;

#pragma acc parallel present(u)
  {
#pragma acc loop independent
    for (k = 0; k < N; k++) {
      double x2 = (h * (double)(k)) * (h * (double)(k));
      u[k] = h2 * 20000.0 * exp(-100 * x2) * (1 - 200 * x2);
    }
  }
#pragma acc parallel num_gangs(1) present(u)
  {
    u[0] *= 0.5;
  }
  double time = omp_get_wtime();


#pragma acc parallel present(u)
  {
#pragma acc loop independent
    for (int j = 0; j < r; j++) {
      for (int k = 1; k < s; k++) {
        u[j * s + k] = u[j * s + k] + u[j * s + k - 1];
      }
    }
  }


#pragma acc parallel num_gangs(1) present(u)
  {
    for (int k = 1; k < r; k++) {
      u[(k + 1) * s - 1] += u[k * s - 1];
    }
  }


#pragma acc parallel present(u)
  {
    for (int j = 1; j < r; j++) {
      double a = u[j * s - 1];
#pragma acc loop independent
      for (int k = 0; k < s - 1; k++)
        u[j * s + k] += a;
    }
  }


#pragma acc parallel present(u)
  {
#pragma acc loop independent
    for (int j = 0; j < r; j++) {
      for (int k = s - 2; k >= 0; k--) {
        u[j * s + k] += u[j * s + k + 1];
      }
    }
  }


#pragma acc parallel num_gangs(1) present(u)
  {
    for (int k = r - 2; k >= 0; k--) {
      u[k * s] += u[k * s + s];
    }
  }


#pragma acc parallel present(u)
  {
    for (int j = 0; j < r - 1; j++) {
      double a = u[j * s + s];
#pragma acc loop independent

      for (int k = 1; k < s; k++)
        u[j * s + k] += a;
    }
  }
#pragma acc wait
  return omp_get_wtime() - time;
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
  double *u_col = (double *)malloc(size);

  acc_init(acc_device_nvidia);

#pragma acc data copy(u_col[0 : N])
  {
#pragma acc data present(u_col)
    {

      time = acc_bvp_solve_col(u_col, s, r);
    }
  }

  if (strcmp(argv[argc - 1], "test") == 0)
    printf("%.30lf", bvp_test_P2(u_col, N));
  else
    printf("%.6lf", time);
  free(u_col);

  return 0;
}
