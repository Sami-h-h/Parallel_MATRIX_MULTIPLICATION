

#include <stdlib.h>
#include <stdio.h>
#include <time.h>

#define TILE_SIZE 32

void printMatrix(int *a, int numRows, int numCols){
    for(int i = 0; i < numRows; i++){
        for(int j = 0; j < numCols; j++){
            printf("%d ", a[i*numCols + j]);
        }
        printf("\n");
    }
}

void initialiseMatrix(int *a, int numRows, int numCols){
    for(int i = 0; i < numRows; i++){
        for(int j = 0; j < numCols; j++){
            a[i * numCols + j] = rand() % 100;
        }
    }
}

int main(){
    int N = 1 << 10;
    int M = 1 << 8;

    int *A = (int *)malloc(N * M * sizeof(int));
    int *B = (int *)malloc(M * N * sizeof(int));
    int *C = (int *)malloc(N * N * sizeof(int));

    initialiseMatrix(A, N, M);
    initialiseMatrix(B, M, N);

    
    clock_t start_time = clock();

    #pragma acc kernels copyin(A[0:N*M], B[0:M*N]) copy(C[0:N*N])
    {
        #pragma acc loop independent collapse(2)
        for(int i = 0; i < N; i += TILE_SIZE){
            for(int j = 0; j < N; j += TILE_SIZE){
                #pragma acc loop independent collapse(2)
                for(int ii = i; ii < i + TILE_SIZE; ii++){
                    for(int jj = j; jj < j + TILE_SIZE; jj++){
                        float sumMatrix = 0;
                        #pragma acc loop independent reduction(+:sumMatrix)
                        for(int x = 0; x < M; x++){
                            sumMatrix += A[ii * M + x] * B[x * N + jj];
                        }
                        C[ii * N + jj] = sumMatrix;
                    }
                }
            }
        }
    }

    // Timer end
    clock_t end_time = clock();
    double elapsed_time = ((double)(end_time - start_time)) / CLOCKS_PER_SEC *1000;
    printf("Time taken: %f ms\n", elapsed_time);



    free(A);
    free(B);
    free(C);

    return 0;
}
