#include <time.h>
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include "omp.h"



void bvp_set(float *u, int N){
  int k;
  float h=1.0/((float)N);
  float h2=h*h;


  for(k=0;k<N;k++){
    float x2=(h*(float)k)*(h*(float)k);
    u[k] = h2*20000.0*exp(-100*x2)*(1-200*x2);
  }
  u[0]*=0.5;
}

void bvp_examp1(float *u, int N){
  int k;
  float sum, e, y,tmp;
  sum=u[0], e=0;
  for(k=1;k<N;k++){
    //u[k] += u[k-1];
    tmp=sum;
    y=u[k]+e;
    u[k]=sum=tmp+y;
    e=(tmp-sum)+y;
  }
   sum=u[N-1], e=0;
  for(k=N-2;k>=0;k--){
    //u[k] += u[k+1];
    tmp=sum;
    y=u[k]+e;
    u[k]=sum=tmp+y;
    e=(tmp-sum)+y;
  }
}


int main(int argc, char *argv[]){
    int N;
    N=atoi(argv[1]);

    size_t size = N * sizeof(float);
    float *f_h = (float *)malloc(size);
bvp_set(f_h, N);
    double time = omp_get_wtime();
    
    bvp_examp1(f_h, N);
    time = omp_get_wtime() - time;


    float h=1.0/((float)N);


    float diff=0.0;
    float sumw=0.0;

    for (int i=0; i<N; i++){
        float x2=(h*(float)i)*(h*(float)i);
        float y=100*exp(-100*x2)-100*exp(-100.0);

        diff+=(f_h[i]-y)*(f_h[i]-y);
        sumw+=y*y;

    }

    printf("%.30f\n",sqrt(diff)/sqrt(sumw));
    printf("%.6lf",time);

    free(f_h);

    return 0;
}
