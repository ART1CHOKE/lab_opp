#include <stdio.h>
#include <mpi/mpi.h>
#include <math.h>
#include <stdlib.h>

#define N 256
#define EPS 1e-8
#define A 1e5
#define D 2.0
#define H (D/(N - 1))
#define HH (H*H)
#define K (1.0/((2.0/HH) + (2.0/HH) + (2.0/HH) + A))


static double phi(double x, double y, double z) {
    return pow(x, 2) + pow(y, 2) + pow(z, 2);
}

static double ro(double x, double y, double z) {
    return 6 - A * phi(x, y, z);
}

static void init(double *part_area, int quan_str_z, int shift) {
    int i, j, k;
    for (k = 0; k < quan_str_z; ++k) {
        for (j = 0; j < N; ++j) {
            for (i = 0; i < N; ++i) {
                if ((k + shift) == 0 || (k + shift) == N - 1 || j == 0 || j == N - 1 || i == 0 || i == N - 1) {
                    part_area[k * N * N + j * N + i] = phi(i * H, j * H, (k + shift) * H);
                } else {
                    part_area[k * N * N + j * N + i] = 0;
                }
            }
        }
    }
}

static void
sendRecvBorders(double *prev_area, double *border_up, double *border_down, int rank, int size, int quan_str_z,
                MPI_Request *requests) {
    if (rank != 0) {
        MPI_Isend(&prev_area[0], N * N, MPI_DOUBLE, rank - 1, 0, MPI_COMM_WORLD, &requests[0]);
        MPI_Irecv(border_up, N * N, MPI_DOUBLE, rank - 1, 1, MPI_COMM_WORLD, &requests[2]);
    }
    if (rank != size - 1) {
        MPI_Isend(&prev_area[(quan_str_z - 1) * N * N], N * N, MPI_DOUBLE, rank + 1, 1, MPI_COMM_WORLD,
                  &requests[1]);
        MPI_Irecv(border_down, N * N, MPI_DOUBLE, rank + 1, 0, MPI_COMM_WORLD, &requests[3]);
    }
}

static void waitBorders(MPI_Request *requests, int rank, int size) {
    if (rank != 0) {
        MPI_Wait(&requests[0], MPI_STATUS_IGNORE);
        MPI_Wait(&requests[2], MPI_STATUS_IGNORE);
    }
    if (rank != size - 1) {
        MPI_Wait(&requests[1], MPI_STATUS_IGNORE);
        MPI_Wait(&requests[3], MPI_STATUS_IGNORE);
    }
}

static void calcMid(const double *prev_area, double *next_area, int quan_str_z, int shift) {
    int i, j, k;
    for (k = 1; k < quan_str_z - 1; ++k) {
        for (j = 1; j < N - 1; ++j) {
            for (i = 1; i < N - 1; ++i) {
                double phi_i =
                        (prev_area[k * N * N + j * N + (i + 1)] + prev_area[k * N * N + j * N + (i - 1)]) / HH;
                double phi_j =
                        (prev_area[k * N * N + (j + 1) * N + i] + prev_area[k * N * N + (j - 1) * N + i]) / HH;
                double phi_k =
                        (prev_area[(k + 1) * N * N + j * N + i] + prev_area[(k - 1) * N * N + j * N + i]) / HH;
                next_area[k * N * N + j * N + i] = K * (phi_i + phi_j + phi_k - ro(i * H, j * H, (k + shift) * H));
            }
        }
    }
}

static void
calcBorders(const double *part_area, double *tmp_area, const double *border_up, const double *border_down, int rank,
            int size,
            int quan_str_z,
            int shift) {
    int i, j;
    for (j = 1; j < N - 1; ++j) {
        for (i = 1; i < N - 1; ++i) {
            if (rank != 0) {
                double phi_i =
                        (part_area[j * N + (i + 1)] + part_area[N * N + j * N + (i - 1)]) / HH;
                double phi_j =
                        (part_area[(j + 1) * N + i] + part_area[(j - 1) * N + i]) / HH;
                double phi_k =
                        (part_area[N * N + j * N + i] + border_up[j * N + i]) / HH;
                tmp_area[j * N + i] = K * (phi_i + phi_j + phi_k - ro(i * H, j * H, shift * H));
            }

            if (rank != size - 1) {
                double phi_i =
                        (part_area[(quan_str_z - 1) * N * N + j * N + (i + 1)] +
                         part_area[(quan_str_z - 1) * N * N + j * N + (i - 1)]) /
                        HH;
                double phi_j =
                        (part_area[(quan_str_z - 1) * N * N + (j + 1) * N + i] +
                         part_area[(quan_str_z - 1) * N * N + (j - 1) * N + i]) /
                        HH;
                double phi_k =
                        (border_down[j * N + i] + part_area[(quan_str_z - 2) * N * N + j * N + i]) /
                        HH;
                tmp_area[(quan_str_z - 1) * N * N + j * N + i] =
                        K * (phi_i + phi_j + phi_k - ro(i * H, j * H, ((quan_str_z - 1) + shift) * H));
            }
        }
    }
}

static int check_max(double *area1, double *area2, int quan_str_z) {
    for (int k = 0; k < quan_str_z; ++k) {
        for (int j = 0; j < N; ++j) {
            for (int i = 1; i < N; ++i) {
                if (fabs(area1[k * N * N + j * N + i] - area2[k * N * N + j * N + i]) > EPS) {
                    return 0;
                }
            }
        }
    }
    return 1;
}

static void calcFault(double *area, int rank, int quan_str_z, int shift) {
    double max = 0.0;
    double global = 0.0;
    int i, k, j;
    for (k = 1; k < quan_str_z; k++) {
        for (j = 1; j < N - 1; j++) {
            for (i = 1; i < N - 1; i++) {
                double tmp = fabs(area[k * N * N + j * N + i] - phi(i * H, j * H, (k + shift) * H));
                if (tmp > max) {
                    max = tmp;
                }
            }
        }
    }
    MPI_Reduce(&max, &global, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Bcast(&global, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    if (rank == 0) {
        printf("%lf\n", global);
    }
}

int main(int argc, char **argv) {
    int size, rank;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (N % size != 0) {
        perror("Bad number of processes");
        return EXIT_FAILURE;
    }

    int quan_str_z = N / size;
    int shift = quan_str_z * rank;

    double *prev_area = (double *) malloc(sizeof(double) * quan_str_z * N * N);

    double *next_area = (double *) malloc(sizeof(double) * quan_str_z * N * N);

    double *border_up;
    double *border_down;

    if (rank != 0) {
        border_up = (double *) malloc(sizeof(double) * N * N);
    }
    if (rank != size - 1) {
        border_down = (double *) malloc(sizeof(double) * N * N);
    }

    init(prev_area, quan_str_z, shift);
    init(next_area, quan_str_z, shift);

    double start, end;
    int check = 0;
    int global = 0;

    MPI_Request requests[4];
    if (rank == 0) {
        start = MPI_Wtime();
    }
    while (global != 1) {
        sendRecvBorders(prev_area, border_up, border_down, rank, size, quan_str_z, requests);
        calcMid(prev_area, next_area, quan_str_z, shift);
        waitBorders(requests, rank, size);
        calcBorders(prev_area, next_area, border_up, border_down, rank, size, quan_str_z, shift);
        check = check_max(next_area, prev_area, quan_str_z);
        MPI_Reduce(&check, &global, 1, MPI_INT, MPI_MIN, 0, MPI_COMM_WORLD);
        MPI_Bcast(&global, 1, MPI_INT, 0, MPI_COMM_WORLD);
        double *tmp = prev_area;
        prev_area = next_area;
        next_area = tmp;
    }

    calcFault(prev_area, rank, quan_str_z, shift);

    if (rank == 0) {
        end = MPI_Wtime();
    }
    if (rank == 0) {
        printf("%f\n", end - start);
    }
    free(prev_area);
    free(next_area);
    if (rank != 0) {
        free(border_up);
    }
    if (rank != size - 1) {
        free(border_down);
    }

    MPI_Finalize();

    return EXIT_SUCCESS;
}
