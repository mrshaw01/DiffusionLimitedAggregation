#include <mpi.h>
#include <stdio.h>
#include <math.h>
#include <omp.h>
#include <string.h>
#include <stdlib.h>
#include <time.h>

#define iter 1500
#define N 128
#define tol 0.001
#define omega 1.6

int i, j, k, r, rank, size, Np;
int dx[4] = {-1, 0, 0, 1}, dy[4] = {0, -1, 1, 0};

float * C, * Cprev, * delta, * all_delta;
int * O, * Obelow, * candidates;
float * Cabove, * Cbelow;
float * nutris, * nutri;

char fpath[40];
FILE * f;
MPI_Status status;

float max(float a, float b) {
    return a > b ? a : b;
}

float r2() {
    return (float) rand() / (float) RAND_MAX;
}

void diffuse() {
    do {
        // delta for stopping criterion
        * delta = 0;

        // r == 0: estimating for red cell
        // r == 1: estimating for black cell
        for (r = 0; r < 2; ++r) {
            for (i = 0; i < N; ++i) {
                * (Cabove + i) = 0;
                * (Cbelow + i) = 0;
            }

            // Send the last row to below
            if (rank != size - 1)
                MPI_Send(C + (Np - 1) * N, N, MPI_FLOAT, rank + 1, 1, MPI_COMM_WORLD);

            // Send the first row to above
            if (rank != 0)
                MPI_Send(C, N, MPI_FLOAT, rank - 1, 0, MPI_COMM_WORLD);

            // Receive from below
            if (rank != size - 1)
                MPI_Recv(Cbelow, N, MPI_FLOAT, rank + 1, 0, MPI_COMM_WORLD, & status);

            // Receive from above
            if (rank != 0)
                MPI_Recv(Cabove, N, MPI_FLOAT, rank - 1, 1, MPI_COMM_WORLD, & status);

            for (i = 0; i < Np; ++i) {
                if (rank == 0 && i == 0)
                    continue;

                for (j = 0; j < N; ++j) {
                    // Equation: (C + i*N + j) = (1-omega) * (C + i*N + j) + (omega/4) * ((C + (i-1)*N + j) + (C + i*N + j-1) + (C + i*N + j+1) + (C + (i+1)*N + j))

                    // If Object
                    if (*(O + i * N + j) == 1)
                        continue;

                    // If false cell's color
                    if ((rank * N + i + j) % 2 == r)
                        continue;

                    // First row, rank 0
                    if (rank == 0 && i == 0)
                        continue;

                    // Last row, rank (size - 1)
                    if (rank == size - 1 && i == Np - 1)
                        continue;

                    // Top left corner
                    if (i == 0 && j == 0)
                        *(C + i * N + j) = (omega / 4) * ( * (Cabove + j) + * (C + i * N + N - 1) + * (C + i * N + j + 1) + * (C + (i + 1) * N + j)) + (1 - omega) * * (C + i * N + j);
                    // Top right corner
                    else  if (i == 0 && j == N - 1) 
                        *(C + i * N + j) = (omega / 4) * ( * (Cabove + j) + * (C + i * N + j - 1) + * (C + i * N + 0) + * (C + (i + 1) * N + j)) + (1 - omega) * * (C + i * N + j);
                    // Bottom left corner
                    else if (i == Np - 1 && j == 0) // 
                        *(C + i * N + j) = (omega / 4) * ( * (C + (i - 1) * N + j) + * (C + i * N + N - 1) + * (C + i * N + j + 1) + * (Cbelow + j)) + (1 - omega) * * (C + i * N + j);
                    // Bottom right corner
                    else if (i == Np - 1 && j == N - 1) 
                        *(C + i * N + j) = (omega / 4) * ( * (C + (i - 1) * N + j) + * (C + i * N + j - 1) + * (C + i * N + 0) + * (Cbelow + j)) + (1 - omega) * * (C + i * N + j);
                    // First column
                    else if (i != 0 && i != Np - 1 && j == 0)
                        *(C + i * N + j) = (omega / 4) * ( * (C + (i - 1) * N + j) + * (C + i * N + N - 1) + * (C + i * N + j + 1) + * (C + (i + 1) * N + j)) + (1 - omega) * * (C + i * N + j);
                    // Last column
                    else if (i != 0 && i != Np - 1 && j == N - 1) 
                        *(C + i * N + j) = (omega / 4) * ( * (C + (i - 1) * N + j) + * (C + i * N + j - 1) + * (C + i * N + 0) + * (C + (i + 1) * N + j)) + (1 - omega) * * (C + i * N + j);
                    // First row
                    else if (i == 0 && j != 0 && j != N - 1)
                        *(C + i * N + j) = (omega / 4) * ( * (Cabove + j) + * (C + i * N + j - 1) + * (C + i * N + j + 1) + * (C + (i + 1) * N + j)) + (1 - omega) * * (C + i * N + j);
                    // Last row
                    else if (i == Np - 1 && j != 0 && j != N - 1)
                        *(C + i * N + j) = (omega / 4) * ( * (C + (i - 1) * N + j) + * (C + i * N + j - 1) + * (C + i * N + j + 1) + * (Cbelow + j)) + (1 - omega) * * (C + i * N + j);
                    // General case: within the boundary
                    else
                        *(C + i * N + j) = (omega / 4) * ( * (C + (i - 1) * N + j) + * (C + i * N + j - 1) + * (C + i * N + j + 1) + * (C + (i + 1) * N + j)) + (1 - omega) * * (C + i * N + j);

                    * delta = max( * delta, fabs( * (C + i * N + j) - * (Cprev + i * N + j)));
                }
            }
        }

        // Gather delta in all_delta 
        MPI_Allgather(delta, 1, MPI_FLOAT, all_delta, 1, MPI_FLOAT, MPI_COMM_WORLD);
        * delta = 0;

        // Retrieve delta
        for (i = 0; i < size; ++i)
            * delta = max( * delta, *(all_delta + i));

        // Update cell previous value
        for (i = 0; i < Np; ++i)
            for (j = 0; j < N; ++j)
                * (Cprev + i * N + j) = * (C + i * N + j);
    } while ( * delta > tol);

    return;
}

int main(int argc, char * argv[]) {
    srand(time(NULL));

    MPI_Init( & argc, & argv);
    MPI_Comm_rank(MPI_COMM_WORLD, & rank);
    MPI_Comm_size(MPI_COMM_WORLD, & size);

    printf("Start process %d\n", rank);

    // Number rows each process
    Np = N / size;
    snprintf(fpath, 40, "output/%d.txt", rank);
    
    f = fopen(fpath, "w");
    printf("Process %d, output path: %s\n", rank, fpath);

    // Release memory
    C = (float * ) malloc(Np * N * sizeof(float));
    Cprev = (float * ) malloc(Np * N * sizeof(float));
    delta = (float * ) malloc(sizeof(float));
    all_delta = (float * ) malloc(size * sizeof(float));

    Cabove = (float * ) malloc(N * sizeof(float));
    Cbelow = (float * ) malloc(N * sizeof(float));

    O = (int * ) malloc(Np * N * sizeof(int));
    candidates = (int * ) malloc(Np * N * sizeof(int));
    Obelow = (int * ) malloc(N * sizeof(int));

    nutris = (float * ) malloc(size * sizeof(float));
    nutri = (float * ) malloc(sizeof(float));

    // Cell Initialization
    for (i = 0; i < Np; ++i)
        for (j = 0; j < N; ++j) {
            if (rank == 0 && i == 0)
                *(C + i * N + j) = 1;
            else
                *(C + i * N + j) = 0;
            *(Cprev + i * N + j) = *(C + i * N + j);
            *(O + i * N + j) = 0;
        }

    // Object initialization
    if (rank == size - 1)
        *(O + (Np - 1) * N + N / 2) = 1;

    // Diffuse protein process
    for (k = 0; k < iter; ++k) {
        diffuse();
        for (i = 0; i < Np; ++i)
            for (j = 0; j < N; ++j)
                *(Cprev + i * N + j) = *(C + i * N + j);

        // Grow object
        for (i = 0; i < N; ++i)
            *(Obelow + i) = 0;

        // Send the first row of object to above
        if (rank != 0)
            MPI_Send(O, N, MPI_INT, rank - 1, 0, MPI_COMM_WORLD);

        // Receive object from below
        if (rank != size - 1)
            MPI_Recv(Obelow, N, MPI_INT, rank + 1, 0, MPI_COMM_WORLD, & status);

        // Initialize candidates
        for (i = 0; i < Np; ++i)
            for (j = 0; j < N; ++j)
                * (candidates + i * N + j) = 0;

        // Get candidates
        * nutri = 0;
        for (i = 0; i < Np; ++i)
            for (j = 0; j < N; ++j) {

                // If object cell
                if ( * (O + i * N + j) == 1)
                    continue;

                // Sum
                int sum = 0;

                // cell (u, v): adjacent cell of cell (i, j)
                for (r = 0; r < 4; ++r) {
                    int u, v;
                    u = i + dx[r];
                    v = j + dy[r];
                    if (u >= 0 && u < Np && v >= 0 && v < N && * (O + u * N + v) == 1)
                        sum += 1;
                }

                // If last row, check object below
                if (i == Np - 1 && * (Obelow + j) == 1)
                    sum += 1;

                // If sum postive: there is at least 1 adjacent object
                if (sum > 0) {
                    * nutri += * (C + i * N + j);
                    *(candidates + i * N + j) = 1;
                }
            }

        // Gather nutrition
        MPI_Allgather(nutri, 1, MPI_FLOAT, nutris, 1, MPI_FLOAT, MPI_COMM_WORLD);

        // total nutrition
        float total_nutri = 0.0;

        for (i = 0; i < size; ++i)
            total_nutri += * (nutris + i);

        // Grow
        for (i = 0; i < Np; ++i)
            for (j = 0; j < N; ++j)
                if ( * (candidates + i * N + j) == 1 && r2() <= (*(C + i * N + j) / total_nutri)) {
                    *(O + i * N + j) = 1;
                    *(C + i * N + j) = 0;
                }
    }
    
    // Write piece of results (result of process) to output file
    printf("Process %d, exporting output to: %s\n", rank, fpath);
    for (i = 0; i < Np; ++i) {
        for (j = 0; j < N; ++j) {
            if (*(O + i * N + j) == 1)
                *(C + i * N + j) = 1;
            fprintf(f, "%lf\t", *(C + i * N + j));
        }
        fprintf(f, "\n");
    }

    MPI_Finalize();
    return 0;
}