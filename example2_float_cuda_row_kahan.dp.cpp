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
  size_t size = sizeof(float);
  float *u_d;

  u_d = (float *)sycl::malloc_device(size, q_ct1);
  sycl::free(u_d, q_ct1);
}

// Kernel that executes on the CUDA device row_wise
SYCL_EXTERNAL void row_cuda_bvp_set(float *u, int s, int r,
                                    const sycl::nd_item<3> &item_ct1) {
  int k, m;
  m = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
      item_ct1.get_local_id(2);
  float h = 1.0 / ((float)(s * r));
  float h2 = h * h;

  float pi = 4.0 * (float)sycl::atan(1.0);
  float pip = 2.0 * (float)sycl::atan(1.0);
  //  float pi2p4=pi*pi*0.25;

  for (k = 0; k < s; k++) {
    float x2 = (h * (float)(m * s + k)) * (h * (float)(m * s + k));
    u[m + k * r] = h2 * 20000.0 * sycl::exp(-100 * x2) * (1 - 200 * x2);
  }

  if ((item_ct1.get_group(2) == 0) && (item_ct1.get_local_id(2) == 0))
    u[0] *= 0.5;
}

SYCL_EXTERNAL void row_cuda_bvp_1a(float *u, int s, int r,
                                   const sycl::nd_item<3> &item_ct1) {
  int k, m;
  m = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
      item_ct1.get_local_id(2);
  float sum = u[m], e = 0, y, tmp;

  for (k = 1; k < s; k++) {
    // u[m+k*r]+=u[m+(k-1)*r];
    tmp = sum;
    y = u[m + k * r] + e;
    u[m + k * r] = sum = tmp + y;
    e = (tmp - sum) + y;
  }
}

void row_cuda_bvp_1b(float *u, int s, int r) {
  int k;

  for (k = (s - 1) * r + 1; k < r * s; k++)
    u[k] += u[k - 1];
}

SYCL_EXTERNAL void row_cuda_bvp_1c2a(float *u, int s, int r,
                                     const sycl::nd_item<3> &item_ct1) {
  int m, k;
  m = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
      item_ct1.get_local_id(2);
  // 1c
  if (m > 0) {
    float a = u[(s - 1) * r + m - 1];
    for (k = 0; k < s - 1; k++)
      u[m + k * r] += a;
  }

  // 2a
  float sum = u[r * s + m - r], e = 0, y, tmp;
  for (k = s - 2; k >= 0; k--) {
    // u[m+k*r]+=u[m+(k+1)*r];
    tmp = sum;
    y = u[m + k * r] + e;
    u[m + k * r] = sum = tmp + y;
    e = (tmp - sum) + y;
  }
}

void row_cuda_bvp_2b(float *u, int s, int r) {
  int k;
  for (k = r - 2; k >= 0; k--)
    u[k] += u[k + 1];
}

SYCL_EXTERNAL void row_cuda_bvp_2c(float *u, int s, int r,
                                   const sycl::nd_item<3> &item_ct1) {
  int m, k;
  m = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
      item_ct1.get_local_id(2);

  if (m != r - 1) {
    float a = u[m + 1];
    for (k = 1; k < s; k++)
      u[m + k * r] += a;
  }
}

SYCL_EXTERNAL void row_to_col_cuda(float *u, float *c, int s, int r,
                                   const sycl::nd_item<3> &item_ct1) {
  int m, k, i;
  m = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
      item_ct1.get_local_id(2);
  for (k = 0; k < s; k++) {
    i = m + k * r;
    c[i] = u[i / s + (i % s) * r];
  }
}

double row_bvp(int s, int r, int block_size, float *u_h) {
  dpct::device_ext &dev_ct1 = dpct::get_current_device();
  sycl::queue &q_ct1 = dev_ct1.in_order_queue();
  int N = r * s, j;
  size_t size = N * sizeof(float);
  float *vd_tmp, *u_d, *u_c;
  vd_tmp = (float *)malloc(r * sizeof(float));

  u_d = (float *)sycl::malloc_device(size, q_ct1); // Allocate array on device
  u_c = (float *)sycl::malloc_device(size, q_ct1);
  // double t = omp_get_wtime();
  /*
  DPCT1049:115: The work-group size passed to the SYCL kernel may exceed the
  limit. To get the device limit, query info::device::max_work_group_size.
  Adjust the work-group size if needed.
  */
  q_ct1.parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, r / block_size) *
                                           sycl::range<3>(1, 1, block_size),
                                       sycl::range<3>(1, 1, block_size)),
                     [=](sycl::nd_item<3> item_ct1) {
                       row_cuda_bvp_set(u_d, s, r, item_ct1);
                     });
  fake_malloc();
  dpct::event_ptr start, stop;
  std::chrono::time_point<std::chrono::steady_clock> start_ct1;
  std::chrono::time_point<std::chrono::steady_clock> stop_ct1;
  start = new sycl::event();
  stop = new sycl::event();

  /*
  DPCT1012:174: Detected kernel execution time measurement pattern and generated
  an initial code for time measurements in SYCL. You can change the way time is
  measured depending on your goals.
  */
  start_ct1 = std::chrono::steady_clock::now();
  /*
  DPCT1049:116: The work-group size passed to the SYCL kernel may exceed the
  limit. To get the device limit, query info::device::max_work_group_size.
  Adjust the work-group size if needed.
  */
  *stop = q_ct1.parallel_for(
      sycl::nd_range<3>(sycl::range<3>(1, 1, r / block_size) *
                            sycl::range<3>(1, 1, block_size),
                        sycl::range<3>(1, 1, block_size)),
      [=](sycl::nd_item<3> item_ct1) { row_cuda_bvp_1a(u_d, s, r, item_ct1); });

  //---  row_cuda_bvp_1b <<< 1,1 >>> (u_d, s, r);
  q_ct1.memcpy(vd_tmp, &u_d[(s - 1) * r], sizeof(float) * r).wait();

  float sum = vd_tmp[0], e = 0, y, tmp;
  for (j = 1; j < r; j++) {
    // vd_tmp[j] = vd_tmp[j]+ vd_tmp[j-1];
    tmp = sum;
    y = vd_tmp[j] + e;
    vd_tmp[j] = sum = tmp + y;
    e = (tmp - sum) + y;
  }

  q_ct1.memcpy(&u_d[(s - 1) * r], vd_tmp, sizeof(float) * r).wait();

  /*
  DPCT1049:117: The work-group size passed to the SYCL kernel may exceed the
  limit. To get the device limit, query info::device::max_work_group_size.
  Adjust the work-group size if needed.
  */
  *stop = q_ct1.parallel_for(
      sycl::nd_range<3>(sycl::range<3>(1, 1, r / block_size) *
                            sycl::range<3>(1, 1, block_size),
                        sycl::range<3>(1, 1, block_size)),
      [=](sycl::nd_item<3> item_ct1) {
        row_cuda_bvp_1c2a(u_d, s, r, item_ct1);
      });

  //---  rowcuda_bvp_2b <<< 1,1 >>> (u_d, s, r);
  q_ct1.memcpy(vd_tmp, u_d, sizeof(float) * r).wait();

  sum = vd_tmp[r - 1], e = 0;
  for (j = r - 2; j >= 0; j--) {
    // vd_tmp[j] = vd_tmp[j]+vd_tmp[j+1];
    tmp = sum;
    y = vd_tmp[j] + e;
    vd_tmp[j] = sum = tmp + y;
    e = (tmp - sum) + y;
  }

  q_ct1.memcpy(u_d, vd_tmp, sizeof(float) * r).wait();

  /*
  DPCT1049:118: The work-group size passed to the SYCL kernel may exceed the
  limit. To get the device limit, query info::device::max_work_group_size.
  Adjust the work-group size if needed.
  */
  *stop = q_ct1.parallel_for(
      sycl::nd_range<3>(sycl::range<3>(1, 1, r / block_size) *
                            sycl::range<3>(1, 1, block_size),
                        sycl::range<3>(1, 1, block_size)),
      [=](sycl::nd_item<3> item_ct1) { row_cuda_bvp_2c(u_d, s, r, item_ct1); });
  /*
  DPCT1049:119: The work-group size passed to the SYCL kernel may exceed the
  limit. To get the device limit, query info::device::max_work_group_size.
  Adjust the work-group size if needed.
  */
  *stop = q_ct1.parallel_for(
      sycl::nd_range<3>(sycl::range<3>(1, 1, r / block_size) *
                            sycl::range<3>(1, 1, block_size),
                        sycl::range<3>(1, 1, block_size)),
      [=](sycl::nd_item<3> item_ct1) {
        row_to_col_cuda(u_d, u_c, s, r, item_ct1);
      });

  /*
  DPCT1012:175: Detected kernel execution time measurement pattern and generated
  an initial code for time measurements in SYCL. You can change the way time is
  measured depending on your goals.
  */
  stop->wait();
  stop_ct1 = std::chrono::steady_clock::now();
  float ms;
  ms = std::chrono::duration<float, std::milli>(stop_ct1 - start_ct1).count();

  double t = (ms) / 1000.0;
  q_ct1.memcpy(u_h, u_c, sizeof(float) * N).wait();
  sycl::free(u_c, q_ct1);
  sycl::free(u_d, q_ct1);
  free(vd_tmp);
  return t;
  ;
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

// main routine that executes on the host
int main(int argc, char *argv[]) {
  int N, r, s, bsize; // Number of elements in arrays
  s = atoi(argv[1]);
  r = atoi(argv[2]);
  bsize = atoi(argv[3]);
  N = r * s;

  size_t size = N * sizeof(float);
  float *r_h = (float *)malloc(size);

  // double time = omp_get_wtime();
  double time = row_bvp(s, r, bsize, r_h);
  // time = omp_get_wtime() - time;

  if (strcmp(argv[argc - 1], "test") == 0)
    printf("%.30lf", bvp_test_P2(r_h, N));
  else
    printf("%.6lf", time);

  free(r_h);
  return 0;
}
