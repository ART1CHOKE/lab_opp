#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>
#include <time.h>

#define N 6000
#define t 10e-6
#define eps 10e-9

static void ident(double *result, const double *A) {
#pragma omp for
    for (int i = 0; i < N; ++i) {
        result[i] = A[i];
    }
}

static void subVectorVector(double *result, const double *A, const double *B) {
#pragma omp for
    for (int i = 0; i < N; ++i) {
        result[i] = A[i] - B[i];
    }
}

static void mulVectorT(double *result) {
#pragma omp for
    for (int i = 0; i < N; ++i) {
        result[i] = t * result[i];
    }
}

static void mulMatrixVector(double *result, const double *A, const double *B) {
#pragma omp for
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            result[i] += A[i * N + j] * B[j];
        }
    }
}

int main(int argc, char **argv) {

    if (argc != 2) {
        perror("bad number arguments\n");
        return EXIT_FAILURE;
    }

    int quantity_threads = atoi(argv[1]);
    if (quantity_threads == 0) {
        perror("bad string\n");
        return EXIT_FAILURE;
    }

    omp_set_num_threads(quantity_threads);

    struct timespec start, end;

    double *A = (double *) malloc(sizeof(double) * N * N);
    double *x = (double *) malloc(sizeof(double) * N);
    double *b = (double *) malloc(sizeof(double) * N);

    double *temp = (double *) malloc(sizeof(double) * N);
    double *temp1 = (double *) malloc(sizeof(double) * N);

    for (int i = 0; i < N; ++i) {
        temp[i] = x[i] = 0.0;
        b[i] = N + 1.0;
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

    double b_norm=0.0;
    double tmp_norm=0.0;

    clock_gettime(CLOCK_MONOTONIC_RAW, &start);

#pragma omp parallel
    {
        mulMatrixVector(temp, A, x);
        subVectorVector(temp1, temp, b);

#pragma omp for reduction(+:b_norm)
        for (int i = 0; i < N; ++i) {
            b_norm += pow(b[i], 2);
        }
#pragma omp single
        {
            b_norm = sqrt(b_norm);
        }
#pragma omp for reduction(+:tmp_norm)
        for (int i = 0; i < N; ++i) {
            tmp_norm += pow(temp1[i], 2);
        }
#pragma omp single
        {
            tmp_norm = sqrt(tmp_norm);
        }

        while (tmp_norm/b_norm >= eps) {
            mulVectorT(temp1);
            subVectorVector(temp, x, temp1);
            ident(x, temp);
            mulMatrixVector(temp, A, x);
            subVectorVector(temp1, temp, b);
#pragma omp single
            {
                tmp_norm = 0.0;
            }
#pragma omp for reduction(+:tmp_norm)
            for (int i = 0; i < N; ++i) {
                tmp_norm += pow(temp1[i], 2);
            }
#pragma omp single
            {
                tmp_norm = sqrt(tmp_norm);
            }
        }
    }

    clock_gettime(CLOCK_MONOTONIC_RAW, &end);

    printf("Time taken: %lf sec.\n", end.tv_sec - start.tv_sec + 0.000000001 * (end.tv_nsec - start.tv_nsec));

    /*for (int k = 0; k < N; ++k) {
        printf("%f\n", x[k]);
    }*/

    free(A);
    free(x);
    free(b);
    free(temp);
    free(temp1);
}