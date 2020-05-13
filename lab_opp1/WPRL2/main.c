#include <mpi/mpi.h>
#include <stdio.h>
#include <math.h>
#include <stdlib.h>

#define N 10000
#define t 10e-6
#define eps 10e-9

static void ident(double *result, const double *A,const int quantity_str) {
    int i;
    for (i = 0; i < quantity_str; ++i) {
        result[i] = A[i];
    }
}

int quan_str_get(const int number_proccess,const int size){
    int quan_str = N / size;
    int mod = N % size;
    if (mod >= (number_proccess + 1)) {
        quan_str++;
    }
    return quan_str;
}

int sum_prev(const int number_proccess,const int size) {
    int result = 0;
    int i;
    for (i = 0; i < number_proccess; ++i) {
        result += quan_str_get(i,size);
    }
    return result;
}

static void subVectorVector(double *result, const double *A, const double *B,const int quantity_str) {
    int i;
    for (i = 0; i < quantity_str; ++i) {
        result[i] = A[i] - B[i];
    }
}

static void mulVectorT(double *result,const int quantity_str) {
    int i;
    for (i = 0; i < quantity_str; ++i) {
        result[i] = t * result[i];
    }
}

static double norm2(const double *vector,const int quantity_elements) {
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

    int quan_str = quan_str_get(rank,size);

    int i,j,l;

    int check = 1;

    double *A = (double *) malloc(sizeof(double) * N * quan_str);
    if(A==NULL){
        fprintf(stderr,"malloc error A %d\n",rank);
        return EXIT_FAILURE;
    }

    double *b = (double *) malloc(sizeof(double) * quan_str);
    if(b==NULL){
        fprintf(stderr,"malloc error b %d\n",rank);
        return EXIT_FAILURE;
    }

    double *x = (double *) malloc(sizeof(double) * quan_str);
    if(x==NULL){
        fprintf(stderr,"malloc error x %d\n",rank);
        return EXIT_FAILURE;
    }

    double *temp = (double *) malloc(sizeof(double) * quan_str_get(0,size));
    if(temp==NULL){
        fprintf(stderr,"malloc error temp %d\n",rank);
        return EXIT_FAILURE;
    }
    double *temp1 = (double *) malloc(sizeof(double) * quan_str_get(0,size));
    if(temp1==NULL){
        fprintf(stderr,"malloc error temp1 %d\n",rank);
        return EXIT_FAILURE;
    }

    int sum_p = sum_prev(rank,size);

    for (i = 0; i < quan_str; i++) {
        for (j = 0; j < N; j++) {
            if (sum_p + i == j) {
                A[i * N + j] = 2.0;
            } else {
                A[i * N + j] = 1.0;
            }
        }
    }

    for (i = 0; i < quan_str; ++i) {
        b[i] = N + 1.0;
        temp[i] = x[i] = 0.0;
    }

    if(rank==0){
        start=MPI_Wtime();
    }

    subVectorVector(temp1, temp, b, quan_str);

    while (check != 0) {
        double partnorm = 0;
        mulVectorT(temp1, quan_str);
        subVectorVector(temp, x, temp1, quan_str);

        ident(x, temp, quan_str);

        for (j = 0; j < size; ++j) {
            if (j == rank) {
                j++;
                if (j == size) {
                    break;
                }
            }
            MPI_Send(x, quan_str, MPI_DOUBLE, j, 1, MPI_COMM_WORLD);
        }

        for (i = 0; i < quan_str; ++i) {
            for (l = 0; l < quan_str; ++l) {
                temp[i] += A[i * N + l + sum_prev(rank,size)] * x[l];
            }
        }


        for (j = 0; j < size; ++j) {
            if (j == rank) {
                j++;
                if (j == size) {
                    break;
                }
            }
            int quan_str_j=quan_str_get(j,size);
            MPI_Recv(temp1, quan_str_j, MPI_DOUBLE, j, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            for (i = 0; i < quan_str; ++i) {
                for (l = 0; l < quan_str_j; ++l) {
                    temp[i] += A[i * N + l + sum_prev(j,size)] * temp1[l];
                }
            }
        }

        subVectorVector(temp1, temp, b, quan_str);
        partnorm = norm2(temp1, quan_str);

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


    /*for (i = 0; i < quan_str; ++i) {
        printf("%f\n", x[i]);
    }*/


    free(A);
    free(b);
    free(x);
    free(temp);
    free(temp1);

    MPI_Finalize();

    return 0;
}
