#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <math.h>
#include <openacc.h>


double acc_bvp_solve_col(float *u, int s, int r){
    int k, N=r*s;
    float h=1.0/((float)N);
    float h2=h*h;
    
#pragma acc parallel present(u)
    {
#pragma acc loop  independent
        for(k=0;k<N;k++){
            float x2=(h*(float)(k))*(h*(float)(k));
            u[k] = h2*20000.0*exp(-100*x2)*(1-200*x2);
        }
    }
#pragma acc parallel num_gangs(1) present(u)
    {
        u[0]*=0.5;
    }
    double time=omp_get_wtime();

// 1A
#pragma acc parallel present(u)
    {
#pragma acc loop  independent
        for(int j=0;j<r;j++) {
            for(int k=1;k<s;k++){
                u[j*s+k]= u[j*s+k]+ u[j*s+k-1];

            }

        }
    }

// 1B
#pragma acc parallel num_gangs(1) present(u)
    {
        for (int k=1;k<r;k++){
            u[(k+1)*s-1]+=u[k*s-1];
        }
    }


// 1C
#pragma acc parallel present(u)
    {
        for(int j=1;j<r;j++) {
            float a=u[j*s-1];
#pragma acc loop  independent
            for (int k=0;k<s-1;k++)
                u[j*s+k]+=a;
        }
    }

// 2A
#pragma acc parallel present(u)
    {
#pragma acc loop  independent
        for(int j=0;j<r;j++){
            for(int k=s-2;k>=0;k--){
                u[j*s+k]+=u[j*s+k+1];
            }
        }
    }

// 2B
#pragma acc parallel num_gangs(1) present(u)
    {
        for (int k=r-2;k>=0;k--){
            u[k*s]+=u[k*s+s];
        }
    }

//2C
#pragma acc parallel present(u)
{
    for(int j=0;j<r-1;j++) {
        float a=u[j*s+s];
#pragma acc loop independent

            for (int k=1;k<s;k++)
                u[j*s+k]+=a;
    }
}
return omp_get_wtime()-time;

}

int main(int argc,char **argv)
{
    int N,r,s;
    s=atoi(argv[1]);
    r=atoi(argv[2]);
    N=s*r;

    double time;

    size_t size = N * sizeof(float);
    float *u_col = (float*)malloc(size);

    acc_init(acc_device_nvidia);
    
#pragma acc data copy(u_col[0:N])
{
#pragma acc data present(u_col)
{
    
    time=acc_bvp_solve_col(u_col,s,r);
    

}
}
    

    float h=1.0/((float)N);
    float diff=0.0;
    float sumw =0.0;



    for (int i=0; i<N; i++) {
        float x2=(h*(float)i)*(h*(float)i);
        float y=100*exp(-100*x2)-100*exp(-100.0);
        sumw += y*y;
        float a= (y-u_col[i]);
        diff += a*a;

    }


    free(u_col);
    printf("%.30f\n",sqrt(diff)/sqrt(sumw));
    printf("%.6lf",time);



    return 0;
}
