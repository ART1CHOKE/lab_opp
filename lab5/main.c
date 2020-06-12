#include <stdio.h>
#include <unistd.h>
#include <mpi/mpi.h>
#include <stdlib.h>
#include <pthread.h>

#define QUAN_TASKS 20

int size, rank;
int curTask;
int processTaskNum;
pthread_mutex_t mutex;
pthread_t thrs[2];
int *TaskList;

void *execute(void *me) {
    int time;
    int i;
    
    for (curTask = 0; curTask < processTaskNum;) {
        pthread_mutex_lock(&mutex);
        time = TaskList[curTask];
        ++curTask;
        pthread_mutex_unlock(&mutex);
        sleep(time);
        printf("did Task rank %d\n",rank);
    }
    int ready;
    for (i = 0; i < size; i++) {
        if (i == rank)
            i++;
        if(i==size){
            break;
        }
        ready = 1;
        while (1) {
            MPI_Send(&ready, 1, MPI_INT, i, 0, MPI_COMM_WORLD);
            int Task;
            MPI_Recv(&Task, 1, MPI_INT, i, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            if (Task == -1) break;
            sleep(Task);
            printf("did Task rank %d\n",rank);
        }
    }
    MPI_Barrier(MPI_COMM_WORLD);
    ready = 0;
    //отправляем текущему процессу,чтобы завершить другой поток
    MPI_Send(&ready, 1, MPI_INT, rank, 0, MPI_COMM_WORLD);
}

void *distribute(void *me) {
    MPI_Status st;
    int ready;
    int Task;

    while (1) {
        MPI_Recv(&ready, 1, MPI_INT, MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, &st);
        if (ready == 0) break;
        pthread_mutex_lock(&mutex);

        if (curTask == processTaskNum) {
            Task = -1;
        } else {
            Task = TaskList[curTask];
            ++curTask;
        }

        pthread_mutex_unlock(&mutex);

        MPI_Send(&Task, 1, MPI_INT, st.MPI_SOURCE, 1, MPI_COMM_WORLD);
    }
}

int main(int argc, char **argv) {
    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, NULL);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int i;

    pthread_mutex_init(&mutex, NULL);

    processTaskNum = QUAN_TASKS / size;
    TaskList = (int *) malloc(sizeof(int) * processTaskNum);

    for (i = 0; i < processTaskNum; ++i) {
        TaskList[i] = QUAN_TASKS / size * rank + rand() % (QUAN_TASKS / size);
    }

    pthread_attr_t attrs;
    if (pthread_attr_init(&attrs) != 0) {
        perror("Cannot initialize attributes\n");
        return EXIT_FAILURE;
    }
    if (pthread_attr_setdetachstate(&attrs, PTHREAD_CREATE_JOINABLE) != 0) {
        perror("Error in setting attributes\n");
        return EXIT_FAILURE;
    }


    if (pthread_create(&thrs[0], &attrs, execute, NULL) != 0) {
        perror("Cannot create a DO thread!");
        return EXIT_FAILURE;
    }

    if (pthread_create(&thrs[1], &attrs, distribute, NULL) != 0) {
        perror("Cannot create a SEND thread!");
        return EXIT_FAILURE;
    }

    pthread_attr_destroy(&attrs);

    if (pthread_join(thrs[0], NULL) != 0) {
        perror("Cannot join a thread\n");
        return EXIT_FAILURE;
    }


    if (pthread_join(thrs[1], NULL) != 0) {
        perror("Cannot join a thread\n");
        return EXIT_FAILURE;
    }

    free(TaskList);

    MPI_Finalize();

    return EXIT_SUCCESS;
}
