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

// Kernel that executes on the CUDA device column_wise

SYCL_EXTERNAL void col_cuda_bvp_set(float *u, int s, int r,
                                    const sycl::nd_item<3> &item_ct1) {

  int k, m;
  float h = 1.0 / ((float)(s * r));
  float h2 = h * h;
  float c1 = M_PI * M_PI / 4;
  float c2 = M_PI / 2;

  m = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
      item_ct1.get_local_id(2);

  for (k = 0; k < s; k++) {
    float x = (h * (float)(m * s + k));
    u[m * s + k] = h2 * c1 * sycl::cos(c2 * x);
  }

  if ((item_ct1.get_group(2) == 0) && (item_ct1.get_local_id(2) == 0)) {
    u[0] *= 0.5;
  }
}
// 1A
SYCL_EXTERNAL void col_cuda_bvp_1a(float *u, int s, int r,
                                   const sycl::nd_item<3> &item_ct1) {
  int k, m;
  m = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
      item_ct1.get_local_id(2);

  for (k = 1; k < s; k++)
    u[m * s + k] += u[m * s + k - 1];
}

// 1B
SYCL_EXTERNAL void col_cuda_bvp_1b(float *u, int s, int r) {

  int k;
  for (k = 1; k < r; k++) {
    u[(k + 1) * s - 1] += u[k * s - 1];
  }
}

// 1C
SYCL_EXTERNAL void col_cuda_bvp_1c(float *u, int s, int r,
                                   const sycl::nd_item<3> &item_ct1) {
  int m, k;
  m = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
      item_ct1.get_local_id(2);

  if (m != 0) {
    float a = u[m * s - 1];
    for (k = 0; k < s - 1; k++)
      u[m * s + k] += a;
  }
}

// 2A
SYCL_EXTERNAL void col_cuda_bvp_2a(float *u, int s, int r,
                                   const sycl::nd_item<3> &item_ct1) {
  int m, k;
  m = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
      item_ct1.get_local_id(2);
  for (k = s - 2; k >= 0; k--)
    u[m * s + k] += u[m * s + k + 1];
}

// 2B

SYCL_EXTERNAL void col_cuda_bvp_2b(float *u, int s, int r) {
  int k;
  for (k = r - 2; k >= 0; k--) {
    u[k * s] += u[(k + 1) * s];
  }
}

// 2C

SYCL_EXTERNAL void col_cuda_bvp_2c(float *u, int s, int r,
                                   const sycl::nd_item<3> &item_ct1) {
  int m, k;
  m = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
      item_ct1.get_local_id(2);
  if (m < r - 1) {
    float a = u[(m + 1) * s];
    for (k = 1; k < s; k++)
      u[m * s + k] += a;
  }
}

double col_bvp(int s, int r, int block_size, float *u_h) {
  dpct::device_ext &dev_ct1 = dpct::get_current_device();
  sycl::queue &q_ct1 = dev_ct1.in_order_queue();
  int N = r * s;
  size_t size = N * sizeof(float);
  float *u_d;

  u_d = (float *)sycl::malloc_device(size, q_ct1); // Allocate array on device

  /*
  DPCT1049:30: The work-group size passed to the SYCL kernel may exceed the
  limit. To get the device limit, query info::device::max_work_group_size.
  Adjust the work-group size if needed.
  */
  q_ct1.parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, r / block_size) *
                                           sycl::range<3>(1, 1, block_size),
                                       sycl::range<3>(1, 1, block_size)),
                     [=](sycl::nd_item<3> item_ct1) {
                       col_cuda_bvp_set(u_d, s, r, item_ct1);
                     });
  fake_malloc();
  dpct::event_ptr start, stop;
  std::chrono::time_point<std::chrono::steady_clock> start_ct1;
  std::chrono::time_point<std::chrono::steady_clock> stop_ct1;
  start = new sycl::event();
  stop = new sycl::event();

  /*
  DPCT1012:142: Detected kernel execution time measurement pattern and generated
  an initial code for time measurements in SYCL. You can change the way time is
  measured depending on your goals.
  */
  start_ct1 = std::chrono::steady_clock::now();

  /*
  DPCT1049:31: The work-group size passed to the SYCL kernel may exceed the
  limit. To get the device limit, query info::device::max_work_group_size.
  Adjust the work-group size if needed.
  */
  *stop = q_ct1.parallel_for(
      sycl::nd_range<3>(sycl::range<3>(1, 1, r / block_size) *
                            sycl::range<3>(1, 1, block_size),
                        sycl::range<3>(1, 1, block_size)),
      [=](sycl::nd_item<3> item_ct1) { col_cuda_bvp_1a(u_d, s, r, item_ct1); });
  *stop = q_ct1.parallel_for(
      sycl::nd_range<3>(sycl::range<3>(1, 1, 1), sycl::range<3>(1, 1, 1)),
      [=](sycl::nd_item<3> item_ct1) { col_cuda_bvp_1b(u_d, s, r); });
  /*
  DPCT1049:32: The work-group size passed to the SYCL kernel may exceed the
  limit. To get the device limit, query info::device::max_work_group_size.
  Adjust the work-group size if needed.
  */
  *stop = q_ct1.parallel_for(
      sycl::nd_range<3>(sycl::range<3>(1, 1, r / block_size) *
                            sycl::range<3>(1, 1, block_size),
                        sycl::range<3>(1, 1, block_size)),
      [=](sycl::nd_item<3> item_ct1) { col_cuda_bvp_1c(u_d, s, r, item_ct1); });
  /*
  DPCT1049:33: The work-group size passed to the SYCL kernel may exceed the
  limit. To get the device limit, query info::device::max_work_group_size.
  Adjust the work-group size if needed.
  */
  *stop = q_ct1.parallel_for(
      sycl::nd_range<3>(sycl::range<3>(1, 1, r / block_size) *
                            sycl::range<3>(1, 1, block_size),
                        sycl::range<3>(1, 1, block_size)),
      [=](sycl::nd_item<3> item_ct1) { col_cuda_bvp_2a(u_d, s, r, item_ct1); });
  *stop = q_ct1.parallel_for(
      sycl::nd_range<3>(sycl::range<3>(1, 1, 1), sycl::range<3>(1, 1, 1)),
      [=](sycl::nd_item<3> item_ct1) { col_cuda_bvp_2b(u_d, s, r); });
  /*
  DPCT1049:34: The work-group size passed to the SYCL kernel may exceed the
  limit. To get the device limit, query info::device::max_work_group_size.
  Adjust the work-group size if needed.
  */
  *stop = q_ct1.parallel_for(
      sycl::nd_range<3>(sycl::range<3>(1, 1, r / block_size) *
                            sycl::range<3>(1, 1, block_size),
                        sycl::range<3>(1, 1, block_size)),
      [=](sycl::nd_item<3> item_ct1) { col_cuda_bvp_2c(u_d, s, r, item_ct1); });

  /*
  DPCT1012:143: Detected kernel execution time measurement pattern and generated
  an initial code for time measurements in SYCL. You can change the way time is
  measured depending on your goals.
  */
  stop->wait();
  stop_ct1 = std::chrono::steady_clock::now();
  float ms;
  ms = std::chrono::duration<float, std::milli>(stop_ct1 - start_ct1).count();

  double t = (ms) / 1000.0;
  q_ct1.memcpy(u_h, u_d, sizeof(float) * N).wait();
  sycl::free(u_d, q_ct1);

  return t;
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

// main routine that executes on the host
int main(int argc, char *argv[]) {
  int N, r, s, bsize; // Number of elements in arrays
  s = atoi(argv[1]);
  r = atoi(argv[2]);
  bsize = atoi(argv[3]);
  N = r * s;
  size_t size = N * sizeof(float);
  float *c_h = (float *)malloc(size);

  // double time = omp_get_wtime();
  double time = col_bvp(s, r, bsize, c_h);
  // time = omp_get_wtime() - time;

  if (strcmp(argv[argc - 1], "test") == 0)
    printf("%.30f", bvp_test_P1(c_h, N));
  else
    printf("%.6lf", time);

  free(c_h);
  return 0;
}
