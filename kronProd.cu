#include <stdio.h>
#include <stdlib.h>
#include <math.h>

typedef struct Matrix{
    float *matrix;
    int rows;
    int *position;
    int columns;
    unsigned int sparseCount;
}Matrix;

void fillMatrix(Matrix*);
void COOTheMatrix(Matrix*);
void printMatrix(Matrix*);
void kronProd(Matrix*, Matrix*, Matrix*);
__global__ void multiply(Matrix*, Matrix*, Matrix*);

int main(){

    Matrix *a, *b, *c;

    a = (Matrix*)malloc(sizeof(Matrix));
    b = (Matrix*)malloc(sizeof(Matrix));
    c = (Matrix*)malloc(sizeof(Matrix));

    if(a == NULL || b == NULL || c == NULL){
        fprintf(stderr, "Failed to allocate memory on CPU");
    }

    /*
     * Initialilizing the matrix parameters    
     */

    a -> rows = b -> columns = 2;
    a -> columns = b -> rows = 3;

    a -> matrix = (float*)malloc(sizeof(float) * a -> rows * a -> columns);
    b -> matrix = (float*)malloc(sizeof(float) * b -> rows * b -> columns);

    fillMatrix(a);
    fillMatrix(b);

    printMatrix(a);
    printMatrix(b);

    kronProd(a, b, c);

    printMatrix(c);

    free(a);
    free(b);
    free(c);

    return(0);
}

void fillMatrix(Matrix *m){
    int i, elements;
    float a = 5.0;

    elements = m -> rows * m -> columns;
    srand((unsigned int)time(NULL));

    for(i = 0; i < elements;i++){
        m -> matrix[i] = ((float)rand()/(float)(RAND_MAX)) * a;
    }

    COOTheMatrix(m);

    return;
}

void COOTheMatrix(Matrix *m){
    int i, elements;
    unsigned int count = 0;
    float *holding;

    elements = m -> rows * m -> columns;

    for(i = 0;i < elements;i++){

        if(m -> matrix[i] != 0){
            count++;    
        }
    }
    
    m -> sparseCount = count;
    m -> position = (int*)malloc(sizeof(int) * count);
    holding = (float*)malloc(sizeof(float) * count);
    
    for(i = 0, count  = 0;i < elements;i++){
        if(m -> matrix[i] != 0){
            m -> position[count] = i;
            holding[count++] = m -> matrix[i];
        }
    }

    free(m -> matrix);
    m -> matrix = holding;

    return;
}

void printMatrix(Matrix *m){
    unsigned int i, elements, count;

    elements = m -> rows * m -> columns;
    float *holding = (float*)malloc(sizeof(float) * elements);
   
    for(i = 0, count = 0;i < elements;i++){
        if(m -> position[count] == i)
            holding[i] = m -> matrix[count++];
        else
            holding[i] = 0;

        printf("%f\t", holding[i]);

        if((i + 1) % m -> columns == 0)
            printf("\n");
    } 

    return;
}

void kronProd(Matrix *a, Matrix *b, Matrix *c){
    Matrix *a_gpu, *b_gpu, *c_gpu;

    cudaMalloc(&a_gpu, sizeof(a));
    cudaMalloc(&b_gpu, sizeof(b));

    cudaMemcpy(a_gpu, a, sizeof(a), cudaMemcpyHostToDevice);
    cudaMemcpy(b_gpu, b, sizeof(b), cudaMemcpyHostToDevice);

    //  Initializing C 
    c = (Matrix*)malloc(sizeof(Matrix));

    c -> sparseCount = a -> sparseCount * b -> sparseCount;
    c -> rows = a -> rows * b -> rows;
    c -> columns = a -> columns * b -> columns;
    //c -> positions = (unsigned int*)malloc(sizeof(unsigned int) * c -> sparseCount);

    cudaMalloc(&c_gpu, sizeof(c));
    cudaMemcpy(c_gpu, c, sizeof(c), cudaMemcpyHostToDevice);

    //  Allocating CUDA device memory to matrix array in 'c_gpu'
    cudaMalloc(&(c_gpu -> matrix), sizeof(int) * c -> sparseCount);

    //  Allocating CUDA device memory to position array in 'c_gpu'
    cudaMalloc(&(c_gpu -> position), sizeof(int) * c -> sparseCount);

    double blocks = ceil((double)c -> sparseCount / 1024);

    unsigned int numBlocks = (unsigned int)blocks;
    unsigned int threadsPerBlock = 1024;

    multiply<<<numBlocks, threadsPerBlock>>>(a_gpu, b_gpu, c_gpu);

    cudaMemcpy(c, c_gpu, sizeof(c_gpu), cudaMemcpyDeviceToHost);

    cudaFree(a_gpu);
    cudaFree(b_gpu);
    cudaFree(c_gpu);

    return;
}

__global__ void multiply(Matrix *a, Matrix *b, Matrix *c){

    unsigned int i, cDim;

    cDim = a -> sparseCount * b -> sparseCount;
    // i is the index of c -> matrix
    i = blockIdx.x * blockDim.x + threadIdx.x;

    if(i < cDim){
        unsigned int aIndex = (unsigned int)i / b -> sparseCount;
        unsigned int bIndex = (unsigned int)i % b -> sparseCount;
        unsigned int unit = b -> rows * b -> columns;
        unsigned int rowUnit = a -> columns * unit;
        unsigned int bColumnNum, bRowNum, aColumnNum, aRowNum;

        bColumnNum = (b -> position[bIndex] + 1) % b -> columns;
        bRowNum = (b -> position[bIndex] + 1) / b -> columns;

        aColumnNum = (a -> position[aIndex] + 1) % a -> columns;
        aRowNum = (a -> position[aIndex] + 1) / a -> columns;

        c -> matrix[i] = a -> matrix[aIndex] * b -> matrix[bIndex];
        c -> position[i] = (aColumnNum * b -> columns) + (aRowNum * rowUnit) + bColumnNum + (bRowNum * unit);
    }

    return;
}
