#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include <time.h>
#include <stdio.h>
#include <math.h>
#include "omp.h"



// Kernel that executes on the CUDA device column_wise

void col_cuda_bvp_set(float *u, int s, int r, const sycl::nd_item<3> &item_ct1){

  int k,m;
  float h=1.0/((float)(s*r));
  float h2=h*h;

  float pi = 4.0 * (float)sycl::atan(1.0);
  float pip = 2.0 * (float)sycl::atan(1.0);


  m = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
      item_ct1.get_local_id(2);

  for(k=0;k<s;k++){
    float x2=(h*(float)(m*s+k))*(h*(float)(m*s+k));
    u[m * s + k] = h2 * 20000.0 * sycl::exp(-100 * x2) * (1 - 200 * x2);
  }

  if ((item_ct1.get_group(2) == 0) && (item_ct1.get_local_id(2) == 0)) {
    u[0]*=0.5;
  }



}

void col_cuda_bvp_1a(float *u, int s, int r, const sycl::nd_item<3> &item_ct1){
  int k,m;
  m = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
      item_ct1.get_local_id(2);

  for(k=1;k<s;k++)
    u[m*s+k]+=u[m*s+k-1];
}

//1B
void col_cuda_bvp_1b(float *u, int s, int r){

    int k;
    for (k=1;k<r;k++) {
      u[(k+1)*s-1]+=u[k*s-1];
    }
}


//1C
void col_cuda_bvp_1c(float *u, int s, int r, const sycl::nd_item<3> &item_ct1){
  int m,k;
  m = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
      item_ct1.get_local_id(2);

  if(m!=0){
    float a = u[m*s-1];
    for(k=0;k<s-1;k++)
      u[m*s+k]+=a;
  }

}

//2A
void col_cuda_bvp_2a(float *u, int s, int r, const sycl::nd_item<3> &item_ct1){
  int m,k;
  m = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
      item_ct1.get_local_id(2);
    for(k=s-2;k>=0;k--)
      u[m*s+k]+=u[m*s+k+1];

}



void col_cuda_bvp_2b(float *u, int s, int r){
    int k;
    for (k=r-2;k>=0;k--) {
       u[k*s]+=u[(k+1)*s];
    }
  }




void col_cuda_bvp_2c(float *u, int s, int r, const sycl::nd_item<3> &item_ct1) {
  int m,k;
  m = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
      item_ct1.get_local_id(2);
  if(m<r-1){
    float a=u[(m+1)*s];
    for (k=1;k<s;k++)
      u[m*s+k]+=a;
  }

}

void col_bvp(int s, int r, int block_size, float *u_h) {
    dpct::device_ext &dev_ct1 = dpct::get_current_device();
    sycl::queue &q_ct1 = dev_ct1.in_order_queue();
  int N=r*s;
  size_t size = N * sizeof(float);
  float *u_d;

  u_d = (float *)sycl::malloc_device(size, q_ct1); // Allocate array on device
  //
  /*
  DPCT1049:0: The work-group size passed to the SYCL kernel may exceed the
  limit. To get the device limit, query info::device::max_work_group_size.
  Adjust the work-group size if needed.
  */
  q_ct1.parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, r / block_size) *
                                           sycl::range<3>(1, 1, block_size),
                                       sycl::range<3>(1, 1, block_size)),
                     [=](sycl::nd_item<3> item_ct1) {
                       col_cuda_bvp_set(u_d, s, r, item_ct1);
                     });

  /*
  DPCT1049:1: The work-group size passed to the SYCL kernel may exceed the
  limit. To get the device limit, query info::device::max_work_group_size.
  Adjust the work-group size if needed.
  */

  q_ct1.parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, r / block_size) *
                                           sycl::range<3>(1, 1, block_size),
                                       sycl::range<3>(1, 1, block_size)),
                     [=](sycl::nd_item<3> item_ct1) {
                       col_cuda_bvp_1a(u_d, s, r, item_ct1);
                     });
  q_ct1.parallel_for(
      sycl::nd_range<3>(sycl::range<3>(1, 1, 1), sycl::range<3>(1, 1, 1)),
      [=](sycl::nd_item<3> item_ct1) {
        col_cuda_bvp_1b(u_d, s, r);
      });
  /*
  DPCT1049:2: The work-group size passed to the SYCL kernel may exceed the
  limit. To get the device limit, query info::device::max_work_group_size.
  Adjust the work-group size if needed.
  */
  q_ct1.parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, r / block_size) *
                                           sycl::range<3>(1, 1, block_size),
                                       sycl::range<3>(1, 1, block_size)),
                     [=](sycl::nd_item<3> item_ct1) {
                       col_cuda_bvp_1c(u_d, s, r, item_ct1);
                     });
  /*
  DPCT1049:3: The work-group size passed to the SYCL kernel may exceed the
  limit. To get the device limit, query info::device::max_work_group_size.
  Adjust the work-group size if needed.
  */
  q_ct1.parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, r / block_size) *
                                           sycl::range<3>(1, 1, block_size),
                                       sycl::range<3>(1, 1, block_size)),
                     [=](sycl::nd_item<3> item_ct1) {
                       col_cuda_bvp_2a(u_d, s, r, item_ct1);
                     });
  q_ct1.parallel_for(
      sycl::nd_range<3>(sycl::range<3>(1, 1, 1), sycl::range<3>(1, 1, 1)),
      [=](sycl::nd_item<3> item_ct1) {
        col_cuda_bvp_2b(u_d, s, r);
      });
  /*
  DPCT1049:4: The work-group size passed to the SYCL kernel may exceed the
  limit. To get the device limit, query info::device::max_work_group_size.
  Adjust the work-group size if needed.
  */
  q_ct1.parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, r / block_size) *
                                           sycl::range<3>(1, 1, block_size),
                                       sycl::range<3>(1, 1, block_size)),
                     [=](sycl::nd_item<3> item_ct1) {
                       col_cuda_bvp_2c(u_d, s, r, item_ct1);
                     });

  dev_ct1.queues_wait_and_throw();

  q_ct1.memcpy(u_h, u_d, sizeof(float) * N).wait();
  dpct::dpct_free(u_d, q_ct1);




}

// main routine that executes on the host
int main(int argc, char *argv[]){
    int N,r,s, bsize;   // Number of elements in arrays
    s=atoi(argv[1]);
    r=atoi(argv[2]);
    bsize = atoi(argv[3]);
    N=r*s;

    size_t size = N * sizeof(float);
    float *c_h = (float *)malloc(size);

    double time = omp_get_wtime();
    col_bvp(s,r,bsize,c_h);
    time = omp_get_wtime() - time;

    float h=1.0/((float)N);

    float diff=0.0;
    float sumw=0.0;
    // Print results
    for (int i=0; i<N; i++) {
        float x2=(h*(float)i)*(h*(float)i);
        float y=100*exp(-100*x2)-100*exp(-100.0);

        diff+=(c_h[i]-y)*(c_h[i]-y);
        sumw+=y*y;
    }

    printf("%.30f\n",sqrt(diff)/sqrt(sumw));
    printf("%.6lf",time);

    free(c_h);
    return 0;
}
