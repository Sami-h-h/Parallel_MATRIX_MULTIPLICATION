#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#define TILE_SIZE 32

__global__ void matrixMultiplicationTiled(int *matrixA, int *matrixB, int *matrixC, int numRows, int numCols){
    __shared__ int tileA[TILE_SIZE][TILE_SIZE];
    __shared__ int tileB[TILE_SIZE][TILE_SIZE];

    int blockX = blockIdx.x, blockY = blockIdx.y;
    int threadX = threadIdx.x, threadY = threadIdx.y;

    int row = blockY * TILE_SIZE + threadY;
    int col = blockX * TILE_SIZE + threadX;

    int productSum = 0;

    for(int phase = 0; phase < (numCols / TILE_SIZE); phase++){
        
        tileA[threadY][threadX] = matrixA[row * numCols + phase * TILE_SIZE + threadX];
        tileB[threadY][threadX] = matrixB[(phase * TILE_SIZE + threadY) * numRows + col];
        __syncthreads();

        
        for(int k = 0; k < TILE_SIZE; k++){
            productSum += tileA[threadY][k] * tileB[k][threadX];
        }
        __syncthreads();
    }

    y
    if(row < numRows && col < numCols)
        matrixC[row * numCols + col] = productSum;
}


void initializeMatrix(int *matrix, int rows, int cols){
    for(int i = 0; i < rows; i++){
        for(int j = 0; j < cols; j++){
            matrix[i * cols + j] = rand() % 100;
        }
    }
}

void displayMatrix(int *matrix, int rows, int cols){
    for(int i = 0; i < rows; i++){
        for(int j = 0; j < cols; j++){
            printf("%d ", matrix[i * cols + j]);
        }
        printf("\n");
    }
}

int main(){
    int numRows = 1000;
    int numCols = 2000;

    size_t bytes = numRows * numCols * sizeof(int);

    int *hostA, *hostB, *hostC;

    hostA = (int *) malloc(bytes);
    hostB = (int *) malloc(bytes);
    hostC = (int *) malloc(bytes);

    int *deviceA, *deviceB, *deviceC;

    cudaMalloc(&deviceA, bytes);
    cudaMalloc(&deviceB, bytes);
    cudaMalloc(&deviceC, bytes);

    initializeMatrix(hostA, numRows, numCols);
    initializeMatrix(hostB, numCols, numRows);

    cudaMemcpy(deviceA, hostA, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(deviceB, hostB, bytes, cudaMemcpyHostToDevice);

    dim3 grid((numCols + TILE_SIZE - 1) / TILE_SIZE, (numRows + TILE_SIZE - 1) / TILE_SIZE);
    dim3 block(TILE_SIZE, TILE_SIZE);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start, NULL);

    matrixMultiplicationTiled<<<grid, block>>>(deviceA, deviceB, deviceC, numRows, numCols);

    cudaEventRecord(stop, NULL);
    cudaEventSynchronize(stop);

    cudaMemcpy(hostC, deviceC, bytes, cudaMemcpyDeviceToHost);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Time in GPU was %.2f ms\n", milliseconds);

    printf("Matrix C:\n");
    displayMatrix(hostC, numRows, numCols);

    free(hostA);
    free(hostB);
    free(hostC);

    cudaFree(deviceA);
    cudaFree(deviceB);
    cudaFree(deviceC);

    return 0;
}
