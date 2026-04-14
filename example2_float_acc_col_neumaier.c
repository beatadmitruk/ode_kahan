#include <math.h>
#include <omp.h>
#include <openacc.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

double acc_bvp_solve_col(float *u, int s, int r) {
  int k, N = r * s;
  float sum, y, e, tmp;
  float x, t;
  float h = 1.0 / ((float)N);
  float h2 = h * h;

#pragma acc parallel present(u)
  {
#pragma acc loop independent
    for (k = 0; k < N; k++) {
      float x2 = (h * (float)(k)) * (h * (float)(k));
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
      sum = u[j * s];
      e = 0.0;
      for (int k = 1; k < s; k++) {
        x = u[j * s + k];
        t = sum + x;
        if (fabsf(sum) >= fabsf(x))
          e += (sum - t) + x;
        else
          e += (x - t) + sum;
        sum = t;
        u[j * s + k] = sum + e;
      }
    }
  }


#pragma acc parallel num_gangs(1) present(u)
  {
    sum = u[s - 1];
    e = 0.0;

    for (int k = 1; k < r; k++) {
      x = u[(k + 1) * s - 1];
      t = sum + x;

      if (fabsf(sum) >= fabsf(x))
        e += (sum - t) + x;
      else
        e += (x - t) + sum;

      sum = t;
      u[(k + 1) * s - 1] = sum + e;
    }
  }


#pragma acc parallel present(u)
  {
    for (int j = 1; j < r; j++) {
      float a = u[j * s - 1];
#pragma acc loop independent
      for (int k = 0; k < s - 1; k++)
        u[j * s + k] += a;
    }
  }


#pragma acc parallel present(u)
  {
#pragma acc loop independent
    for (int j = 0; j < r; j++) {
      sum = u[(j + 1) * s - 1];
      e = 0.0;

      for (int k = s - 2; k >= 0; k--) {
        x = u[j * s + k];
        t = sum + x;

        if (fabsf(sum) >= fabsf(x))
          e += (sum - t) + x;
        else
          e += (x - t) + sum;

        sum = t;
        u[j * s + k] = sum + e;
      }
    }
  }


#pragma acc parallel num_gangs(1) present(u)
  {
    sum = u[(r - 1) * s];
    e = 0.0;

    for (int k = r - 2; k >= 0; k--) {
      x = u[k * s];
      t = sum + x;

      if (fabsf(sum) >= fabsf(x))
        e += (sum - t) + x;
      else
        e += (x - t) + sum;

      sum = t;
      u[k * s] = sum + e;
    }
  }


#pragma acc parallel present(u)
  {
    for (int j = 0; j < r - 1; j++) {
      float a = u[j * s + s];
#pragma acc loop independent

      for (int k = 1; k < s; k++)
        u[j * s + k] += a;
    }
  }
#pragma acc wait
  return omp_get_wtime() - time;
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

int main(int argc, char **argv) {
  int N, r, s;
  s = atoi(argv[1]);
  r = atoi(argv[2]);
  N = s * r;

  double time;

  size_t size = N * sizeof(float);
  float *u_col = (float *)malloc(size);

  acc_init(acc_device_nvidia);

#pragma acc data copy(u_col[0 : N])
  {
#pragma acc data present(u_col)
    {

      time = acc_bvp_solve_col(u_col, s, r);
    }
  }

  if (strcmp(argv[argc - 1], "test") == 0)
    printf("%.30f", bvp_test_P2(u_col, N));
  else
    printf("%.6lf", time);
  free(u_col);

  return 0;
}
