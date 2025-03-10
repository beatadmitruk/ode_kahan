#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <math.h>
#include <openacc.h>




double acc_bvp_solve_row(float *u_h, int s, int r){
    int k,j, N=r*s;
    float h=1.0/((float)N);
    float h2=h*h;
    float sum, e, y,tmp;
    float *u = acc_malloc(sizeof(float)*N);

#pragma acc parallel deviceptr(u)
    {
#pragma acc loop  independent
        for(j=0;j<r;j++) {
            #pragma acc loop  independent
            for(k=0;k<s;k++){
                float x2=(h*(float)(j*s+k))*(h*(float)(j*s+k));
                u[j+k*r] = h2*20000.0*exp(-100*x2)*(1-200*x2);
            }
        }
    }
#pragma acc parallel num_gangs(1) deviceptr(u)
    {
        u[0]*=0.5;
    }
    double time=omp_get_wtime();
    //1A
#pragma acc parallel deviceptr(u)
    {
#pragma acc loop  independent
    for(int j=0;j<r;j++){
        sum=u[j], e=0;
        for(int k=1;k<s;k++){
            //u[k*r+j] += u[(k-1)*r + j];
            tmp=sum;
            y=u[j+k*r] + e;
            u[j+k*r] = sum = tmp + y;
            e=(tmp-sum)+y;
        }
    }
    }

    //1B
#pragma acc parallel num_gangs(1) deviceptr(u)
    {
    sum=u[r*(s-1)-1], e=0;
    for (int k=(s-1)*r;k<N;k++){
        //u[k]+=u[k-1];
        tmp=sum;
        y=u[k]+e;
        u[k]=sum=tmp+y;
        e=(tmp-sum)+y;
    }
    }




    //1C
#pragma acc parallel deviceptr(u)
    {
#pragma acc loop  independent
    for(int k=0;k<s-1;k++){
         for(int j=1;j<r;j++)
            u[j+k*r]+=u[(s-1)*r+j-1];
    }
    }

    //2A
#pragma acc parallel deviceptr(u)
    {
#pragma acc loop  independent
    for(int j=0;j<r;j++){
        sum=u[r*s+j-r], e=0;
        for(int k=s-2;k>=0;k--){
            //u[j+k*r]+=u[j+(k+1)*r];
            tmp=sum;
            y=u[j+k*r] + e;
            u[j+k*r] = sum = tmp + y;
            e=(tmp-sum)+y;
        }
    }
    }
    

    //2B
#pragma acc parallel num_gangs(1) deviceptr(u)
    {
    sum=u[r-1], e=0;
    for (int k=r-2;k>=0;k--){
        //u[k]+=u[k+1];
        tmp=sum;
        y=u[k]+e;
        u[k]=sum=tmp+y;
        e=(tmp-sum)+y;
    }
    }
    

    //2C
#pragma acc parallel deviceptr(u)
    {
#pragma acc loop independent
    for (int k=1;k<s;k++){
        for(int j=0;j<r-1;j++)
            u[j+k*r]+=u[j+1];
    }
    }
#pragma acc parallel deviceptr(u) present(u_h)
    {
#pragma acc loop independent
    for(int i=0;i<N;i++)
        u_h[i]=u[i/s+(i%s)*r];
    }
    time=omp_get_wtime()-time;
    acc_free(u);
    return time;


}



int main(int argc,char **argv){

    int N,r,s;
    s=atoi(argv[1]);
    r=atoi(argv[2]);
    N=s*r;

    double time;

    size_t size = N * sizeof(float);


    float *u = (float*)malloc(size);



    acc_init(acc_device_nvidia);


    
#pragma acc data copyout(u[0:N])
{
#pragma acc data present(u)
{
    
    time = acc_bvp_solve_row(u,s,r);
    
}
}



    float h=1.0/((float)N);

    float diff=0.0;
    float sumw =0.0;


    for (int i=0; i<N; i++) {
        float x2=(h*(float)i)*(h*(float)i);
        float y=100*exp(-100*x2)-100*exp(-100.0);
        sumw += y*y;

        float b=(y-u[i]);
        diff += b*b;
    }


    free(u);
    printf("%.30f\n",sqrt(diff)/sqrt(sumw));
    printf("%.6lf",time);


    return 0;
}
