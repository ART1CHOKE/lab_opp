#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define N 10
#define t 10e-3
#define eps 10e-6

static void ident(double *result,const double *A){
    for (int i = 0; i <N ; ++i) {
        result[i]=A[i];
    }
}

static void subVectorVector(double *result,const double *A,const double *B){
    for (int i = 0; i <N ; ++i) {
        result[i]=A[i]-B[i];
    }
}

static void mulVectorT(double *result){
    for (int i = 0; i <N ; ++i) {
        result[i]=t*result[i];
    }
}

static double norm(const double *vector){
    double result=0.0;
    for (int i = 0; i <N ; ++i) {
        result+=pow(vector[i],2);
    }
    return sqrt(result);
}

static void mulMatrixVector(double *result,const double *A,const double *B){
    for (int i = 0; i <N ; ++i) {
        for (int j = 0; j <N ; ++j) {
            result[i]+=A[i*N+j]*B[j];
        }
    }
}

int main() {
    double *A = (double *) malloc(sizeof(double) * N * N);
    double *x = (double *) malloc(sizeof(double) * N);
    double *b = (double *) malloc(sizeof(double) * N);

    double *temp = (double *) malloc(sizeof(double) * N);
    double *temp1 = (double *) malloc(sizeof(double) * N);

    for (int i = 0; i < N; ++i) {
        temp[i] = x[i] = 0.0;
        b[i] = N+1.0;
    }

    for (int j = 0; j < N; ++j) {
        for (int i = 0; i < N; ++i) {
            if (i == j) {
                A[j * N + i] = 2.0;
            } else {
                A[j * N + i] = 1.0;
            }
        }
    }

    mulMatrixVector(temp,A,x);
    subVectorVector(temp1,temp,b);

    while(norm(temp1)/norm(b)>=eps){
        mulVectorT(temp1);
        subVectorVector(temp,x,temp1);
        ident(x,temp);
        mulMatrixVector(temp,A,x);
        subVectorVector(temp1,temp,b);
    }

    for (int k = 0; k <N ; ++k) {
        printf("%f\n",x[k]);
    }

    free(A);
    free(x);
    free(b);
    free(temp);
    free(temp1);
}