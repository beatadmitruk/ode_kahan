#include <time.h>
#include <stdio.h>
#include <cuda.h>
#include <math.h>
#include "omp.h"



// Kernel that executes on the CUDA device row_wise
__global__ void row_cuda_bvp_set(float *u, int s, int r){
  int k,m;
  m=blockIdx.x*blockDim.x+threadIdx.x;
  float h=1.0/((float)(s*r));
  float h2=h*h;

  float pi=4.0*(float)atan(1.0);
  float pip=2.0*(float)atan(1.0);
//  float pi2p4=pi*pi*0.25;


  for(k=0;k<s;k++){
    float x2=(h*(float)(m*s+k))*(h*(float)(m*s+k));
    u[m+k*r] = h2*20000.0*exp(-100*x2)*(1-200*x2);
  }

  if((blockIdx.x==0)&&(threadIdx.x==0))
    u[0]*=0.5;
}

__global__ void row_cuda_bvp_1a(float *u, int s, int r){
  int k,m;
  m=blockIdx.x*blockDim.x+threadIdx.x;
  float sum=u[m], e=0, y,tmp;

  for(k=1;k<s;k++){
    //u[m+k*r]+=u[m+(k-1)*r];
    tmp=sum;
    y=u[m+k*r] + e;
    u[m+k*r] = sum = tmp + y;
    e=(tmp-sum)+y;
  }
}


__global__ void row_cuda_bvp_1b(float *u, int s, int r){
    int k;

    for (k=(s-1)*r+1;k<r*s;k++)
      u[k]+=u[k-1];
}



__global__ void row_cuda_bvp_1c2a(float *u, int s, int r){
  int m,k;
  m=blockIdx.x*blockDim.x+threadIdx.x;
  //1c
  if(m>0) {
    float a=u[(s-1)*r+m-1];
    for (k=0;k<s-1;k++)
       u[m+k*r]+=a;
  }

  //2a
  float sum=u[r*s+m-r], e=0, y,tmp;
  for(k=s-2;k>=0;k--){
    //u[m+k*r]+=u[m+(k+1)*r];
    tmp=sum;
    y=u[m+k*r] + e;
    u[m+k*r] = sum = tmp + y;
    e=(tmp-sum)+y;
  }
}



__global__ void row_cuda_bvp_2b(float *u, int s, int r){
    int k;
    for (k=r-2;k>=0;k--)
       u[k]+=u[k+1];
}




__global__ void row_cuda_bvp_2c(float *u, int s, int r) {
  int m,k;
  m=blockIdx.x*blockDim.x+threadIdx.x;

  if(m!=r-1) {
    float a=u[m+1];
    for (k=1;k<s;k++)
       u[m+k*r]+=a;
  }
}

__global__ void row_to_col_cuda(float *u, float *c, int s, int r){
    int m,k,i;
    m=blockIdx.x*blockDim.x+threadIdx.x;
    for(k=0;k<s;k++){
      i=m+k*r;
      c[i]=u[i/s+(i%s)*r];
    }
}

void row_bvp(int s, int r, int block_size,float *u_h){
  int N=r*s, j;
  size_t size = N * sizeof(float);
  float *vd_tmp, *u_d, *u_c;
  vd_tmp = (float *)malloc(r*sizeof(float));

  cudaMalloc((void **) &u_d, size);   // Allocate array on device
  cudaMalloc((void **) &u_c, size);

  row_cuda_bvp_set <<< r/block_size, block_size >>> (u_d, s, r);

   row_cuda_bvp_1a <<< r/block_size, block_size >>> (u_d, s, r);

//---  row_cuda_bvp_1b <<< 1,1 >>> (u_d, s, r);
  cudaMemcpy(vd_tmp, &u_d[(s-1)*r], sizeof(float)*r, cudaMemcpyDeviceToHost);

  float sum=vd_tmp[0], e=0, y, tmp;
  for(j=1;j<r;j++){
    //vd_tmp[j] = vd_tmp[j]+ vd_tmp[j-1];
    tmp=sum;
    y=vd_tmp[j]+e;
    vd_tmp[j]=sum=tmp+y;
    e=(tmp-sum)+y;
  }

  cudaMemcpy(&u_d[(s-1)*r], vd_tmp, sizeof(float)*r, cudaMemcpyHostToDevice);

  row_cuda_bvp_1c2a <<< r/block_size, block_size >>> (u_d, s, r);

  //---  rowcuda_bvp_2b <<< 1,1 >>> (u_d, s, r);
  cudaMemcpy(vd_tmp, u_d, sizeof(float)*r, cudaMemcpyDeviceToHost);

  sum=vd_tmp[r-1], e=0;
  for(j=r-2;j>=0;j--){
    //vd_tmp[j] = vd_tmp[j]+vd_tmp[j+1];
    tmp=sum;
    y=vd_tmp[j]+e;
    vd_tmp[j]=sum=tmp+y;
    e=(tmp-sum)+y;
  }

  cudaMemcpy(u_d, vd_tmp, sizeof(float)*r, cudaMemcpyHostToDevice);

  row_cuda_bvp_2c <<< r/block_size, block_size >>> (u_d, s, r);
  row_to_col_cuda <<< r/block_size, block_size >>> (u_d, u_c, s,r);

  cudaDeviceSynchronize();
  cudaMemcpy(u_h, u_c, sizeof(float)*N, cudaMemcpyDeviceToHost);
  cudaFree(u_c);
  cudaFree(u_d);
  free(vd_tmp);

}

// main routine that executes on the host
int main(int argc, char *argv[]){
    int N,r,s, bsize;   // Number of elements in arrays
    s=atoi(argv[1]);
    r=atoi(argv[2]);
    bsize = atoi(argv[3]);
    N=r*s;

    size_t size = N * sizeof(float);
    float *r_h = (float *)malloc(size);

    double time = omp_get_wtime();
    row_bvp(s,r,bsize,r_h);
    time = omp_get_wtime() - time;

    float h=1.0/((float)N);

    float diff=0.0;
    float sumw=0.0;
    // Print results
    for (int i=0; i<N; i++) {
        float x2=(h*(float)i)*(h*(float)i);
        float y=100*exp(-100*x2)-100*exp(-100.0);

        diff += (r_h[i]-y)*(r_h[i]-y);
        sumw+=y*y;
    }

    printf("%.30f\n",sqrt(diff)/sqrt(sumw));
    printf("%.6lf",time);

    free(r_h);
    return 0;
}

