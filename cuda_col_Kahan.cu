#include <time.h>
#include <stdio.h>
#include <cuda.h>
#include <math.h>
#include "omp.h"


// Kernel that executes on the CUDA device column_wise

__global__ void col_cuda_bvp_set(float *u, int s, int r){

  int k,m;
  float h=1.0/((float)(s*r));
  float h2=h*h;

  float pi=4.0*(float)atan(1.0);
  float pip=2.0*(float)atan(1.0);
//  float pi2p4=pi*pi*0.25;


  m=blockIdx.x*blockDim.x+threadIdx.x;


  for(k=0;k<s;k++){
    float x2=(h*(float)(m*s+k))*(h*(float)(m*s+k));
    u[m*s+k] = h2*20000.0*exp(-100*x2)*(1-200*x2);
  }


  if((blockIdx.x==0)&&(threadIdx.x==0)) {
    u[0]*=0.5;
  }



}

__global__ void col_cuda_bvp_1a(float *u, int s, int r){
  int k,m;
  m=blockIdx.x*blockDim.x+threadIdx.x;
  float sum, y,e,tmp;
  sum=u[m*s],e=0;
  for(k=1;k<s;k++){
    //u[m*s+k]+=u[m*s+k-1];
    tmp=sum;
    y=u[m*s+k]+e;
    u[m*s+k]=sum=tmp+y;
    e=(tmp-sum)+y;
  }
}

//1B
__global__ void col_cuda_bvp_1b(float *u, int s, int r){
    float sum, y,e,tmp;
    sum=u[s-1],e=0;
    int k;
    for (k=1;k<r;k++) {
      //u[(k+1)*s-1]+=u[k*s-1];
      tmp=sum;
      y=u[(k+1)*s-1]+e;
      u[(k+1)*s-1]=sum=tmp+y;
      e=(tmp-sum)+y;
    }
}


//1C
__global__ void col_cuda_bvp_1c(float *u, int s, int r){
  int m,k;
  m=blockIdx.x*blockDim.x+threadIdx.x;

  if(m!=0){
    float a = u[m*s-1];
    for(k=0;k<s-1;k++)
      u[m*s+k]+=a;
  }

}

//2A
__global__ void col_cuda_bvp_2a(float *u, int s, int r){
  float sum, y,e,tmp;
  int m,k;
  m=blockIdx.x*blockDim.x+threadIdx.x;
  sum=u[(m+1)*s-1],e=0;
    for(k=s-2;k>=0;k--){
      //u[m*s+k]+=u[m*s+k+1];
      tmp=sum;
      y=u[m*s+k]+e;
      u[m*s+k]=sum=tmp+y;
      e=(tmp-sum)+y;
    }

}



__global__ void col_cuda_bvp_2b(float *u, int s, int r){
    float sum, y,e,tmp;
    sum=u[(r-1)*s],e=0;
    int k;
    for (k=r-2;k>=0;k--) {
       //u[k*s]+=u[(k+1)*s];
      tmp=sum;
      y=u[k*s]+e;
      u[k*s]=sum=tmp+y;
      e=(tmp-sum)+y;
    }
  }




__global__ void col_cuda_bvp_2c(float *u, int s, int r) {
  int m,k;
  m=blockIdx.x*blockDim.x+threadIdx.x;
  if(m<r-1){
    float a=u[(m+1)*s];
    for (k=1;k<s;k++)
      u[m*s+k]+=a;
  }

}

void col_bvp(int s, int r, int block_size,float *u_h){
  int N=r*s;
  size_t size = N * sizeof(float);
  float *u_d;


  cudaMalloc((void **) &u_d, size);   // Allocate array on device

  col_cuda_bvp_set <<< r/block_size, block_size >>> (u_d, s, r);
  col_cuda_bvp_1a <<< r/block_size, block_size >>> (u_d, s, r);
  col_cuda_bvp_1b <<< 1,1 >>> (u_d, s, r);
  col_cuda_bvp_1c <<< r/block_size, block_size >>> (u_d, s, r);
  col_cuda_bvp_2a <<< r/block_size, block_size >>> (u_d, s, r);
  col_cuda_bvp_2b <<< 1,1 >>> (u_d, s, r);
  col_cuda_bvp_2c <<< r/block_size, block_size >>> (u_d, s, r);

  cudaDeviceSynchronize();
  cudaMemcpy(u_h, u_d, sizeof(float)*N, cudaMemcpyDeviceToHost);
  cudaFree(u_d);

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
    double time = col_bvp(s,r,bsize,c_h);
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
