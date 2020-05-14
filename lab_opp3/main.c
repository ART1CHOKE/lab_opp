#include <stdio.h>
#include <stdlib.h>
#include <mpi/mpi.h>

#define P1 2
#define P2 2
#define N1 2
#define N12 2
#define N2 2

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);

    double *A, *B, *C;
    double *sA, *sB, *sC;

    double start, end;

    int dims[2] = {P1, P2}, periods[2] = {0, 0};
    int coords[] = {0, 0};
    int size, rank;


    int i, j, k;

    MPI_Comm comm2d;
    MPI_Comm comm1d_y;
    MPI_Comm comm1d_x;


    MPI_Comm_size(MPI_COMM_WORLD, &size);

    MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, 0, &comm2d);

    MPI_Comm_rank(comm2d, &rank);

    if (rank == 0) {
        A = (double *) malloc(sizeof(double) * N1 * N12);
        B = (double *) malloc(sizeof(double) * N12 * N2);
        C = (double *) malloc(sizeof(double) * N1 * N2);

        for (i = 0; i < N1; ++i) {
            for (j = 0; j < N12; ++j) {
                A[N12 * i + j] = 4.0;
            }
        }
        //храним матрицу B сразу в транспонированном виде,чтобы легче было умножать
        for (i = 0; i < N2; ++i) {
            for (j = 0; j < N12; ++j) {
                B[N2 * i + j] = 3.0;
            }
        }

        for (i = 0; i < N1; ++i) {
            for (j = 0; j < N2; ++j) {
                C[N2 * i + j] = 0.0;
            }
        }

    }
    sA = (double *) malloc(sizeof(double) * N1 / dims[0] * N12);
    sB = (double *) malloc(sizeof(double) * N2 / dims[1] * N12);
    sC = (double *) malloc(sizeof(double) * N1 / dims[0] * N2 / dims[1]);

    if (rank == 0) {
        start = MPI_Wtime();
    }

    int belongs[2] = {1, 0};
    MPI_Cart_sub(comm2d, belongs, &comm1d_y);
    belongs[0] = 0;
    belongs[1] = 1;
    MPI_Cart_sub(comm2d, belongs, &comm1d_x);

    MPI_Cart_coords(comm2d, rank, 2, coords);

    if (coords[1] == 0) {
        MPI_Scatter(A, N1 / dims[0] * N12, MPI_DOUBLE, sA, N1 / dims[0] * N12, MPI_DOUBLE, 0, comm1d_y);
    }

    if (coords[0] == 0) {
        MPI_Scatter(B, N2 / dims[1] * N12, MPI_DOUBLE, sB, N2 / dims[1] * N12, MPI_DOUBLE, 0, comm1d_x);
    }


    MPI_Bcast(sA, N1 / dims[0] * N12, MPI_DOUBLE, 0, comm1d_x);
    MPI_Bcast(sB, N2 / dims[1] * N12, MPI_DOUBLE, 0, comm1d_y);

    for (i = 0; i < N1 / dims[0]; i++) {
        for (j = 0; j < N2 / dims[1]; j++) {
            sC[N2 / dims[1] * i + j] = 0.0;
            for (k = 0; k < N12; k++) {
                sC[N2 / dims[1] * i + j] += sA[N12 * i + k] * sB[N12 * j + k];
            }
        }
    }

    MPI_Datatype sCtype;//тип для хранения матрицы sC(каждая строка матрицы будет сдвинута на N12 элементов)
    MPI_Datatype sCtype_aligned;//выровненный sCtype

    /* Чтобы заполнить матрицу C можно воспользоваться Gather ,но т.к Gather заполняет все последовательно
     * он нам не подходит(матрица С будет построена неккоректно).Чтобы все прекрасно работало нам нужен определенный
     * тип со сдвигами,чтобы части матрицы C определялись корректно.*/

    MPI_Type_vector(N1 / dims[0], N2 / dims[1], N12, MPI_DOUBLE, &sCtype);

    /*определяем наш тип у которого будут N1/dims[0](количество строк матрицы sC) блоков.
     * N2/dims[1](количество столбцов sC) это число элементов базового типа в каждом блоке
     * N12 это расстояние между блоками(от начала одного блока до другого)*/

    MPI_Type_create_resized(sCtype, 0, (int) sizeof(double) * N2 / dims[1], &sCtype_aligned);

    //устанавливает верхнюю границу для нашего sCtype

    MPI_Type_commit(&sCtype_aligned);

    //регистрируем созданный производный тип.

    int quantity_elements_process[dims[0] * dims[1]];

    //содержит количество элементов, которые получены от каждого из процессов

    int displs[dims[0] * dims[1]];

    //содержит элементы i,которые определяют смещение относительно *C, в котором размещаются данные из процесса i

    for (i = 0; i < dims[0]; ++i) {
        for (j = 0; j < dims[1]; ++j) {
            quantity_elements_process[i * dims[1] + j] = 1;
            displs[i * dims[1] + j] = i * N1 + j;
        }
    }

    MPI_Gatherv(sC, N1 / dims[0] * N2 / dims[1], MPI_DOUBLE, C, quantity_elements_process, displs, sCtype_aligned, 0,
                comm2d);

    MPI_Type_free(&sCtype_aligned);

    if (rank == 0) {

        end = MPI_Wtime();

        printf("%f\n", end - start);


        /*for (i = 0; i < N1; i++) {
            for (j = 0; j < N2; j++) {
                printf("%f ", C[N2 * i + j]);
            }
            printf("\n");
        }*/

        free(A);
        free(B);
        free(C);
    }

    free(sA);
    free(sB);
    free(sC);

    MPI_Comm_free(&comm2d);
    MPI_Comm_free(&comm1d_y);
    MPI_Comm_free(&comm1d_x);

    MPI_Finalize();

    return 0;
}

