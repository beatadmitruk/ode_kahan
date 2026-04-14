#include <math.h>
#include <omp.h>
#include <openacc.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

double acc_bvp_solve_col(float *u, int s, int r) {
  int k, N = r * s;
  float sum, y, e, tmp;
  float h = 1.0 / ((float)N);
  float h2 = h * h;
  float c1 = M_PI * M_PI / 4;
  float c2 = M_PI / 2;

#pragma acc parallel present(u)
  {
#pragma acc loop independent
    for (k = 0; k < N; k++) {
      float x = (h * (float)(k));
      u[k] = h2 * c1 * cos(c2 * x);
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
      sum = u[j * s], e = 0;
      for (int k = 1; k < s; k++) {
        tmp = sum;
        y = u[j * s + k] + e;
        u[j * s + k] = sum = tmp + y;
        e = (tmp - sum) + y;
      }
    }
  }


#pragma acc parallel num_gangs(1) present(u)
  {
    sum = u[s - 1], e = 0;
    for (int k = 1; k < r; k++) {
      tmp = sum;
      y = u[(k + 1) * s - 1] + e;
      u[(k + 1) * s - 1] = sum = tmp + y;
      e = (tmp - sum) + y;
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
      sum = u[(j + 1) * s - 1], e = 0;
      for (int k = s - 2; k >= 0; k--) {
        tmp = sum;
        y = u[j * s + k] + e;
        u[j * s + k] = sum = tmp + y;
        e = (tmp - sum) + y;
      }
    }
  }


#pragma acc parallel num_gangs(1) present(u)
  {
    sum = u[(r - 1) * s], e = 0;
    for (int k = r - 2; k >= 0; k--) {
      tmp = sum;
      y = u[k * s] + e;
      u[k * s] = sum = tmp + y;
      e = (tmp - sum) + y;
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
    printf("%.30f", bvp_test_P1(u_col, N));
  else
    printf("%.6lf", time);
  free(u_col);

  return 0;
}
