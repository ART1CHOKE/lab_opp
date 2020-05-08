#include <mpi/mpi.h>
#include <stdio.h>
#include <math.h>
#include <stdlib.h>

#define N 10000
#define t 10e-6
#define eps 10e-9

static void ident(double *result, const double *A, int quantity_str) {
    int i;
    for (i = 0; i < quantity_str; ++i) {
        result[i] = A[i];
    }
}

int sum_prev(const int *vector, int number_proccess) {
    int result = 0;
    int i;
    for (i = 0; i < number_proccess; ++i) {
        result += vector[i];
    }
    return result;
}


static void subVectorVector(double *result, const double *A, const double *B, int quantity_str, int sum_p) {
    int i;
    for (i = 0; i < quantity_str; ++i) {
        result[i] = A[i + sum_p] - B[i];
    }
}

static void mulVectorT(double *result, int quantity_str) {
    int i;
    for (i = 0; i < quantity_str; ++i) {
        result[i] = t * result[i];
    }
}

static double norm2(const double *vector, int quantity_elements) {
    double result = 0.0;
    int i;
    for (i = 0; i < quantity_elements; ++i) {
        result += pow(vector[i], 2);
    }
    return result;
}

static void mulMatrixVector(double *result, const double *A, const double *B, int quantity_str) {
    int i,j;
    for (i = 0; i < quantity_str; ++i) {
        for (j = 0; j < N; ++j) {
            result[i] += A[i * N + j] * B[j];
        }
    }
}

int main(int argc, char **argv) {
    int size, rank;

    MPI_Init(&argc, &argv);

    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    double sum;

    int *quan_str = (int *) malloc(size * sizeof(int));
    if(quan_str==NULL){
        fprintf(stderr,"malloc error quan_str %d\n",rank);
        return EXIT_FAILURE;
    }

    int i,j;

    for (i = 0; i < size; ++i) {
        quan_str[i] = N / size;
    }

    int mod = N % size;

    if (mod > 0) {
        for (i = 0; mod > 0; ++i, mod--) {
            quan_str[i]++;
        }
    }

    int check = 1;
    double start, end;

    double *A = (double *) malloc(sizeof(double) * N * quan_str[rank]);
    if(A==NULL){
        fprintf(stderr,"malloc error A %d\n",rank);
        return EXIT_FAILURE;
    }

    double *b = (double *) malloc(sizeof(double) * N);
    if(b==NULL){
        fprintf(stderr,"malloc error b %d\n",rank);
        return EXIT_FAILURE;
    }

    double *x = (double *) malloc(sizeof(double) * N);
    if(x==NULL){
        fprintf(stderr,"malloc error x %d\n",rank);
        return EXIT_FAILURE;
    }

    double *temp = (double *) malloc(sizeof(double) * quan_str[rank]);
    if(temp==NULL){
        fprintf(stderr,"malloc error temp %d\n",rank);
        return EXIT_FAILURE;
    }

    double *temp1 = (double *) malloc(sizeof(double) * quan_str[rank]);
    if(temp1==NULL){
        fprintf(stderr,"malloc error temp1 %d\n",rank);
        return EXIT_FAILURE;
    }

    int sum_p = sum_prev(quan_str, rank);

    for (i = 0; i < quan_str[rank]; i++) {
        for (j = 0; j < N; j++) {
            if (sum_p + i == j) {
                A[i * N + j] = 2;
            } else {
                A[i * N + j] = 1;
            }
        }
    }

    for ( i = 0; i < N; ++i) {
        b[i] = N + 1.0;
        x[i] = 0.0;
    }

    if(rank==0) {
        start = MPI_Wtime();
    }

    mulMatrixVector(temp, A, x, quan_str[rank]);
    subVectorVector(temp1, temp, b, quan_str[rank], 0);

    while (check != 0) {
        double partnorm = 0;
        mulVectorT(temp1, quan_str[rank]);
        subVectorVector(temp, x, temp1, quan_str[rank], sum_p);
        if (rank != 0) {
            MPI_Send(temp, quan_str[rank], MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
        } else {
            ident(x, temp, quan_str[rank]);
            for (i = 1; i < size; i++) {
                MPI_Recv(&x[sum_prev(quan_str, i)], quan_str[i], MPI_DOUBLE, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }
        }

        MPI_Bcast(x, N, MPI_DOUBLE, 0, MPI_COMM_WORLD);

        mulMatrixVector(temp, A, x, quan_str[rank]);
        subVectorVector(temp1, temp, b, quan_str[rank], 0);

        partnorm = norm2(temp1, quan_str[rank]);
        MPI_Reduce(&partnorm, &sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

        if (rank == 0) {
            if (sqrt(sum) / sqrt(norm2(b, N)) >= eps) {
                check = 1;
            } else {
                check = 0;
            }
        }
        MPI_Bcast(&check, 1, MPI_INT, 0, MPI_COMM_WORLD);
    }

    if (rank == 0) {
        end = MPI_Wtime();
        /*for (i = 0; i < N; ++i) {
            printf("%f\n", x[i]);
        }*/
        printf("%f\n",end-start);
    }


    free(A);
    free(b);
    free(x);
    free(quan_str);
    free(temp);
    free(temp1);

    MPI_Finalize();

    return 0;
}
