#include <chrono>
#include <dpct/dpct.hpp>
#include <math.h>
#include <stdio.h>
#include <string.h>
#include <sycl/sycl.hpp>
#include <time.h>

void fake_malloc() {
  dpct::device_ext &dev_ct1 = dpct::get_current_device();
  sycl::queue &q_ct1 = dev_ct1.in_order_queue();
  size_t size = sizeof(double);
  double *u_d;

  u_d = (double *)sycl::malloc_device(size, q_ct1);
  sycl::free(u_d, q_ct1);
}

// Kernel that executes on the CUDA device row_wise
SYCL_EXTERNAL void row_cuda_bvp_set(double *u, int s, int r,
                                    const sycl::nd_item<3> &item_ct1) {
  int k, m;
  m = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
      item_ct1.get_local_id(2);
  double h = 1.0 / ((double)(s * r));
  double h2 = h * h;

  double c1 = M_PI * M_PI / 4;
  double c2 = M_PI / 2;

  for (k = 0; k < s; k++) {
    double x = (h * (double)(m * s + k));
    u[m + k * r] = h2 * c1 * sycl::cos(c2 * x);
  }

  if ((item_ct1.get_group(2) == 0) && (item_ct1.get_local_id(2) == 0))
    u[0] *= 0.5;
}

SYCL_EXTERNAL void row_cuda_bvp_1a(double *u, int s, int r,
                                   const sycl::nd_item<3> &item_ct1) {
  int k, m;
  m = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
      item_ct1.get_local_id(2);

  for (k = 1; k < s; k++)
    u[m + k * r] += u[m + (k - 1) * r];
}

void row_cuda_bvp_1b(double *u, int s, int r) {
  int k;

  for (k = (s - 1) * r + 1; k < r * s; k++)
    u[k] += u[k - 1];
}

SYCL_EXTERNAL void row_cuda_bvp_1c2a(double *u, int s, int r,
                                     const sycl::nd_item<3> &item_ct1) {
  int m, k;
  m = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
      item_ct1.get_local_id(2);
  // 1c
  if (m > 0) {
    double a = u[(s - 1) * r + m - 1];
    for (k = 0; k < s - 1; k++)
      u[m + k * r] += a;
  }

  // 2a
  for (k = s - 2; k >= 0; k--)
    u[m + k * r] += u[m + (k + 1) * r];
}

void row_cuda_bvp_2b(double *u, int s, int r) {
  int k;
  for (k = r - 2; k >= 0; k--)
    u[k] += u[k + 1];
}

SYCL_EXTERNAL void row_cuda_bvp_2c(double *u, int s, int r,
                                   const sycl::nd_item<3> &item_ct1) {
  int m, k;
  m = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
      item_ct1.get_local_id(2);

  if (m != r - 1) {
    double a = u[m + 1];
    for (k = 1; k < s; k++)
      u[m + k * r] += a;
  }
}

SYCL_EXTERNAL void row_to_col_cuda(double *u, double *c, int s, int r,
                                   const sycl::nd_item<3> &item_ct1) {
  int m, k, i;
  m = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
      item_ct1.get_local_id(2);
  for (k = 0; k < s; k++) {
    i = m + k * r;
    c[i] = u[i / s + (i % s) * r];
  }
}

double row_bvp(int s, int r, int block_size, double *u_h) {
  dpct::device_ext &dev_ct1 = dpct::get_current_device();
  sycl::queue &q_ct1 = dev_ct1.in_order_queue();
  int N = r * s, j;
  size_t size = N * sizeof(double);
  double *vd_tmp, *u_d, *u_c;
  vd_tmp = (double *)malloc(r * sizeof(double));

  u_d = (double *)sycl::malloc_device(size, q_ct1); // Allocate array on device
  u_c = (double *)sycl::malloc_device(size, q_ct1);
  // double t = omp_get_wtime();
  /*
  DPCT1049:15: The work-group size passed to the SYCL kernel may exceed the
  limit. To get the device limit, query info::device::max_work_group_size.
  Adjust the work-group size if needed.
  */
  {
    dpct::has_capability_or_fail(q_ct1.get_device(), {sycl::aspect::fp64});
    q_ct1.parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, r / block_size) *
                                             sycl::range<3>(1, 1, block_size),
                                         sycl::range<3>(1, 1, block_size)),
                       [=](sycl::nd_item<3> item_ct1) {
                         row_cuda_bvp_set(u_d, s, r, item_ct1);
                       });
  }
  fake_malloc();
  dpct::event_ptr start, stop;
  std::chrono::time_point<std::chrono::steady_clock> start_ct1;
  std::chrono::time_point<std::chrono::steady_clock> stop_ct1;
  start = new sycl::event();
  stop = new sycl::event();

  /*
  DPCT1012:136: Detected kernel execution time measurement pattern and generated
  an initial code for time measurements in SYCL. You can change the way time is
  measured depending on your goals.
  */
  start_ct1 = std::chrono::steady_clock::now();
  /*
  DPCT1049:16: The work-group size passed to the SYCL kernel may exceed the
  limit. To get the device limit, query info::device::max_work_group_size.
  Adjust the work-group size if needed.
  */
  {
    dpct::has_capability_or_fail(q_ct1.get_device(), {sycl::aspect::fp64});
    *stop = q_ct1.parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, r / block_size) *
                              sycl::range<3>(1, 1, block_size),
                          sycl::range<3>(1, 1, block_size)),
        [=](sycl::nd_item<3> item_ct1) {
          row_cuda_bvp_1a(u_d, s, r, item_ct1);
        });
  }

  //---  row_cuda_bvp_1b <<< 1,1 >>> (u_d, s, r);
  q_ct1.memcpy(vd_tmp, &u_d[(s - 1) * r], sizeof(double) * r).wait();
  for (j = 1; j < r; j++)
    vd_tmp[j] = vd_tmp[j] + vd_tmp[j - 1];

  q_ct1.memcpy(&u_d[(s - 1) * r], vd_tmp, sizeof(double) * r).wait();

  /*
  DPCT1049:17: The work-group size passed to the SYCL kernel may exceed the
  limit. To get the device limit, query info::device::max_work_group_size.
  Adjust the work-group size if needed.
  */
  {
    dpct::has_capability_or_fail(q_ct1.get_device(), {sycl::aspect::fp64});
    *stop = q_ct1.parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, r / block_size) *
                              sycl::range<3>(1, 1, block_size),
                          sycl::range<3>(1, 1, block_size)),
        [=](sycl::nd_item<3> item_ct1) {
          row_cuda_bvp_1c2a(u_d, s, r, item_ct1);
        });
  }

  //---  rowcuda_bvp_2b <<< 1,1 >>> (u_d, s, r);
  q_ct1.memcpy(vd_tmp, u_d, sizeof(double) * r).wait();
  for (j = r - 2; j >= 0; j--)
    vd_tmp[j] = vd_tmp[j] + vd_tmp[j + 1];

  q_ct1.memcpy(u_d, vd_tmp, sizeof(double) * r).wait();

  /*
  DPCT1049:18: The work-group size passed to the SYCL kernel may exceed the
  limit. To get the device limit, query info::device::max_work_group_size.
  Adjust the work-group size if needed.
  */
  {
    dpct::has_capability_or_fail(q_ct1.get_device(), {sycl::aspect::fp64});
    *stop = q_ct1.parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, r / block_size) *
                              sycl::range<3>(1, 1, block_size),
                          sycl::range<3>(1, 1, block_size)),
        [=](sycl::nd_item<3> item_ct1) {
          row_cuda_bvp_2c(u_d, s, r, item_ct1);
        });
  }
  /*
  DPCT1049:19: The work-group size passed to the SYCL kernel may exceed the
  limit. To get the device limit, query info::device::max_work_group_size.
  Adjust the work-group size if needed.
  */
  {
    dpct::has_capability_or_fail(q_ct1.get_device(), {sycl::aspect::fp64});
    *stop = q_ct1.parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, r / block_size) *
                              sycl::range<3>(1, 1, block_size),
                          sycl::range<3>(1, 1, block_size)),
        [=](sycl::nd_item<3> item_ct1) {
          row_to_col_cuda(u_d, u_c, s, r, item_ct1);
        });
  }

  /*
  DPCT1012:137: Detected kernel execution time measurement pattern and generated
  an initial code for time measurements in SYCL. You can change the way time is
  measured depending on your goals.
  */
  stop->wait();
  stop_ct1 = std::chrono::steady_clock::now();
  float ms;
  ms = std::chrono::duration<float, std::milli>(stop_ct1 - start_ct1).count();

  double t = (ms) / 1000.0;
  q_ct1.memcpy(u_h, u_c, sizeof(double) * N).wait();
  sycl::free(u_c, q_ct1);
  sycl::free(u_d, q_ct1);
  free(vd_tmp);
  return t;
  ;
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

// main routine that executes on the host
int main(int argc, char *argv[]) {
  int N, r, s, bsize; // Number of elements in arrays
  s = atoi(argv[1]);
  r = atoi(argv[2]);
  bsize = atoi(argv[3]);
  N = r * s;

  size_t size = N * sizeof(double);
  double *r_h = (double *)malloc(size);

  // double time = omp_get_wtime();
  double time = row_bvp(s, r, bsize, r_h);
  // time = omp_get_wtime() - time;

  if (strcmp(argv[argc - 1], "test") == 0)
    printf("%.30lf", bvp_test_P1(r_h, N));
  else
    printf("%.6lf", time);

  free(r_h);
  return 0;
}
