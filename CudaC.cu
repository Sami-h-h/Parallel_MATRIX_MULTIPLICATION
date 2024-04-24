#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

void initializeMatrix(int *matrix, int rows, int cols){
    for(int i = 0; i < rows; i++){
        for(int j = 0; j < cols; j++){
            matrix[i*cols+j] = rand() % 100;
        }
    }
}

void printMatrix(int* matrix, int rows, int cols){
    for(int i = 0; i < rows; i++){
        for(int j = 0; j < cols; j++){
            printf("%d ", matrix[i*cols + j]);
        }
        printf("\n");
    }
}

__global__ void matrixMultiplicationKernel (int *A, int *B, int *C, int rowsA, int colsA, int colsB){
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    int sum = 0;

    if((row < rowsA) && (col < colsB)){
        for(int k = 0; k < colsA; k++){
            sum += A[row*colsA+k] * B[k*colsB+col];
        }
        C[row * colsB + col] = sum;
    }
}

int main(){
    int numRowsA = 1000;
    int numColsA = 2000;
    int numColsB = 1000;

    size_t bytesA = numRowsA * numColsA * sizeof(int);
    size_t bytesC = numRowsA * numColsB * sizeof(int);

    int *A, *B, *C;

    A = (int*) malloc(bytesA);
    B = (int*) malloc(bytesA);
    C = (int*) malloc(bytesC);

    int *deviceA, *deviceB, *deviceC;

    cudaMalloc(&deviceA, bytesA);
    cudaMalloc(&deviceB, bytesA);
    cudaMalloc(&deviceC, bytesC);

    initializeMatrix(A, numRowsA, numColsA);
    initializeMatrix(B, numColsA, numColsB);

    cudaMemcpy(deviceA, A, bytesA, cudaMemcpyHostToDevice);
    cudaMemcpy(deviceB, B, bytesA, cudaMemcpyHostToDevice);

    int blockSize = 16;
    int numBlocks = (int)ceil((float)numRowsA / blockSize);

    dim3 grid(numBlocks, numBlocks);
    dim3 threads(blockSize, blockSize);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start, NULL);

    matrixMultiplicationKernel <<<grid, threads>>> (deviceA, deviceB, deviceC, numRowsA, numColsA, numColsB);

    cudaMemcpy(C, deviceC, bytesC, cudaMemcpyDeviceToHost);

    cudaEventRecord(stop, NULL);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Time in GPU: %.2f ms\n", milliseconds);
    printf("Number of threads: %d\n", numBlocks * numBlocks * blockSize * blockSize);

    printMatrix(C, numRowsA, numColsB);

    cudaFree(deviceA);
    cudaFree(deviceB);
    cudaFree(deviceC);
    free(A);
    free(B);
    free(C);

    return 0;
}
