#include <stdlib.h>
#include <stdio.h>
#include <time.h>

void printMatrix(int *matrix, int rows, int columns){
    for(int i = 0; i < rows; i++){
        for(int j = 0; j < columns; j++){
            printf("%d ", matrix[i*columns + j]);
        }
        printf("\n");
    }
}

void initializeMatrix(int *matrix, int rows, int columns){
	for(int i = 0; i < rows; i++){
        for(int j = 0; j < columns; j++){
            matrix[i * columns + j] = rand() % 100;
        }
    }
}

int main(){
  int numRows = 1 << 10; 
  int numCols = 1 << 8;  

  int *matrixA = (int *)malloc(numRows * numCols * sizeof(int));
  int *matrixB = (int *)malloc(numCols * numRows * sizeof(int));
  int *resultMatrix = (int *)malloc(numRows * numRows * sizeof(int));
  
  initializeMatrix(matrixA, numRows, numCols);
  initializeMatrix(matrixB, numCols, numRows);

  clock_t startTime = clock(); 

  #pragma acc kernels copyin(matrixA[0:numRows*numCols], matrixB[0:numCols*numRows]) copy(resultMatrix[0:numRows*numRows])
  {
    #pragma acc loop independent
    for(int row = 0; row < numRows; row++){ 
      #pragma acc loop independent
      for(int col = 0; col < numRows; col++){
        float sum = 0; 
        #pragma acc loop independent reduction(+:sum)
        for(int k = 0; k < numCols; k++){  
          sum += matrixA[row * numCols + k] * matrixB[k * numRows + col];
        }
        resultMatrix[row * numRows + col] = sum;
      }
    }
  }
  
  clock_t endTime = clock(); 
  double elapsedTime = ((double)(endTime - startTime)) / CLOCKS_PER_SEC * 1000;
  printf("Time taken : %f ms\n", elapsedTime);


  free(matrixA);
  free(matrixB);
  free(resultMatrix);

  return 0;
}
