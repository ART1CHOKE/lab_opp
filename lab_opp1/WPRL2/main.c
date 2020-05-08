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
    int i;
    int result = 0;
    for (i = 0; i < number_proccess; ++i) {
        result += vector[i];
    }
    return result;
}

static void subVectorVector(double *result, const double *A, const double *B, int quantity_str) {
    int i;
    for (i = 0; i < quantity_str; ++i) {
        result[i] = A[i] - B[i];
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

static double norm3(const double *vector) {
    int i;
    double result = 0.0;
    for (i = 0; i < N; ++i) {
        result += pow(vector[0], 2);
    }
    return result;
}

int main(int argc, char **argv) {
    int size, rank;

    MPI_Init(&argc, &argv);

    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    double sum;
    double start,end;

    int *quan_str = (int *) malloc(size * sizeof(int));
    if(quan_str==NULL){
        fprintf(stderr,"malloc error quan_str %d\n",rank);
        return EXIT_FAILURE;
    }

    int i,j,l;

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

    double *A = (double *) malloc(sizeof(double) * N * quan_str[rank]);
    if(A==NULL){
        fprintf(stderr,"malloc error A %d\n",rank);
        return EXIT_FAILURE;
    }

    double *b = (double *) malloc(sizeof(double) * quan_str[rank]);
    if(b==NULL){
        fprintf(stderr,"malloc error b %d\n",rank);
        return EXIT_FAILURE;
    }

    double *x = (double *) malloc(sizeof(double) * quan_str[rank]);
    if(x==NULL){
        fprintf(stderr,"malloc error x %d\n",rank);
        return EXIT_FAILURE;
    }

    double *temp = (double *) malloc(sizeof(double) * quan_str[0]);
    if(temp==NULL){
        fprintf(stderr,"malloc error temp %d\n",rank);
        return EXIT_FAILURE;
    }
    double *temp1 = (double *) malloc(sizeof(double) * quan_str[0]);
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

    for (i = 0; i < quan_str[rank]; ++i) {
        b[i] = N + 1.0;
        temp[i] = x[i] = 0.0;
    }

    if(rank==0){
        start=MPI_Wtime();
    }

    subVectorVector(temp1, temp, b, quan_str[rank]);

    while (check != 0) {
        double partnorm = 0;
        mulVectorT(temp1, quan_str[rank]);
        subVectorVector(temp, x, temp1, quan_str[rank]);

        ident(x, temp, quan_str[rank]);

        for (j = 0; j < size; ++j) {
            if (j == rank) {
                j++;
                if (j == size) {
                    break;
                }
            }
            MPI_Send(x, quan_str[rank], MPI_DOUBLE, j, 1, MPI_COMM_WORLD);
        }

        for (i = 0; i < quan_str[rank]; ++i) {
            for (l = 0; l < quan_str[rank]; ++l) {
                temp[i] += A[i * N + l + sum_prev(quan_str, rank)] * x[l];
            }
        }


        for (j = 0; j < size; ++j) {
            if (j == rank) {
                j++;
                if (j == size) {
                    break;
                }
            }
            MPI_Recv(temp1, quan_str[j], MPI_DOUBLE, j, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            for (i = 0; i < quan_str[rank]; ++i) {
                for (l = 0; l < quan_str[j]; ++l) {
                    temp[i] += A[i * N + l + sum_prev(quan_str, j)] * temp1[l];
                }
            }
        }

        subVectorVector(temp1, temp, b, quan_str[rank]);
        partnorm = norm2(temp1, quan_str[rank]);

        MPI_Reduce(&partnorm, &sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

        if (rank == 0) {
            if (sqrt(sum) / sqrt(norm3(b)) >= eps) {
                check = 1;
            } else {
                check = 0;
            }
        }
        MPI_Bcast(&check, 1, MPI_INT, 0, MPI_COMM_WORLD);
    }

    if(rank==0){
        end=MPI_Wtime();
        printf("%f\n",end-start);
    }


    /*for (i = 0; i < quan_str[rank]; ++i) {
        printf("%f\n", x[i]);
    }*/


    free(A);
    free(b);
    free(x);
    free(quan_str);
    free(temp);
    free(temp1);

    MPI_Finalize();

    return 0;
}
