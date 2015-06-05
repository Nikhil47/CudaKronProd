#include <stdio.h>
#include <stdlib.h>
#include <math.h>

typedef struct{
    float *matrix;
    int rows;
    int *position;
    int columns;
    unsigned int sparseCount;
}Matrix;

void fillMatrix(Matrix*);
void COOTheMatrix(Matrix*);
void printMatrix(Matrix*);

void prepareGPUCopy(Matrix**);
void prepareHostGPUCopy(Matrix**, Matrix*);
void prepareResultHGPUCopy(Matrix**, Matrix**, Matrix**);
void retrieveResult(Matrix**);

void kronProd(Matrix*, Matrix*, Matrix**);
__global__ void multiply(Matrix*, Matrix*, Matrix*, float*);

#define checkError(call) { checkGPUError((call), __LINE__); }
inline void checkGPUError(cudaError_t errCode, int line){

    if(errCode != cudaSuccess){
        printf("\n");
        printf("Error at %d: %s\n", line, cudaGetErrorString(errCode));
    }

    return;
}

int main(){

    Matrix *a, *b, *c;
    Matrix *a_hgpu, *b_hgpu;

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

    prepareHostGPUCopy(&a_hgpu, a);
    prepareHostGPUCopy(&b_hgpu, b);
    
    kronProd(a_hgpu, b_hgpu, &c);
    
    retrieveResult(&c);
    
    int i;
    for(i = 0;i < 36;i++)
        printf("a[%d]: %f\n", i, c -> matrix[i]);

    cudaFree(a_hgpu -> matrix);
    cudaFree(a_hgpu -> position);
    cudaFree(b_hgpu -> matrix);
    cudaFree(b_hgpu -> position);
    cudaFree(a_hgpu);
    cudaFree(b_hgpu);

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

void prepareHostGPUCopy(Matrix **m_hgpu, Matrix *m){
    
    (*m_hgpu) = (Matrix*)malloc(sizeof(Matrix));
    (*m_hgpu) -> rows = m -> rows;
    (*m_hgpu) -> columns = m -> columns;
    (*m_hgpu) -> sparseCount = m -> sparseCount;
    
    checkError(cudaMalloc(&((*m_hgpu) -> matrix), m -> sparseCount * sizeof(float)));
    checkError(cudaMemcpy((*m_hgpu) -> matrix, m -> matrix, m -> sparseCount * sizeof(float), cudaMemcpyHostToDevice));

    checkError(cudaMalloc(&((*m_hgpu) -> position), m -> sparseCount * sizeof(int)));
    checkError(cudaMemcpy((*m_hgpu) -> position, m -> position, m -> sparseCount * sizeof(int), cudaMemcpyHostToDevice));

    return;
}

void prepareGPUCopy(Matrix **m_gpu, Matrix **m_hgpu){

    checkError(cudaMalloc(m_gpu, sizeof(Matrix)));
    checkError(cudaMemcpy(*m_gpu, *m_hgpu, sizeof(Matrix), cudaMemcpyHostToDevice));

    return;
}

void prepareResultHGPUCopy(Matrix **c, Matrix **m1, Matrix **m2){

    *c = (Matrix*)malloc(sizeof(Matrix));
    (*c) -> sparseCount = (*m1) -> sparseCount * (*m2) -> sparseCount;
    (*c) -> rows = (*m1) -> rows * (*m2) -> rows;
    (*c) -> columns = (*m1) -> columns * (*m2) -> columns;
    
    return;
}

void retrieveResult(Matrix **m){

    Matrix *retVal;

    retVal = (Matrix*)malloc(sizeof(Matrix));

    retVal -> columns = (*m) -> columns;
    retVal -> rows = (*m) -> rows;
    retVal -> sparseCount = (*m) -> sparseCount;
    retVal -> matrix = (float*)malloc(sizeof(float) * (*m) -> sparseCount);
    retVal -> position = (int*)malloc(sizeof(int) * (*m) -> sparseCount);
    
    checkError(cudaMemcpy(retVal -> matrix, (*m) -> matrix, (*m) -> sparseCount * sizeof(float), cudaMemcpyDeviceToHost));
    checkError(cudaMemcpy(retVal -> position, (*m) -> position, (*m) -> sparseCount * sizeof(int), cudaMemcpyDeviceToHost));

    cudaFree((*m) -> matrix);
    cudaFree((*m) -> position);
    free(*m);
    
    *m = retVal;
    
    return;
}

void kronProd(Matrix *a_hgpu, Matrix *b_hgpu, Matrix **c_hgpu){

    Matrix *c_gpu, *a_gpu, *b_gpu;

    prepareGPUCopy(&a_gpu, &a_hgpu);
    prepareGPUCopy(&b_gpu, &b_hgpu);
    //Initializing C_host 

    prepareResultHGPUCopy(c_hgpu, &a_hgpu, &b_hgpu);
   
    checkError(cudaMalloc(&((*c_hgpu) -> matrix), (*c_hgpu) -> sparseCount * sizeof(float)));
    checkError(cudaMalloc(&((*c_hgpu) -> position), (*c_hgpu) -> sparseCount * sizeof(int)));

    prepareGPUCopy(&c_gpu, c_hgpu);

    int cSparse = (*c_hgpu) -> sparseCount;
    float blocks = ceil((float)cSparse / 512);

    unsigned int numBlocks = ((unsigned int)blocks);
    unsigned int threadsPerBlock = 512;

    float *aH = (float*)malloc(sizeof(float) * cSparse);
    float *aG;
    checkError(cudaMalloc(&aG, sizeof(float) * cSparse));
    
    multiply<<<numBlocks, threadsPerBlock>>>(a_gpu, b_gpu, c_gpu, aG);

    checkError(cudaMemcpy(aH, aG, sizeof(float) * cSparse, cudaMemcpyDeviceToHost));
    
    cudaFree(c_gpu);
    cudaFree(a_gpu);
    cudaFree(b_gpu);

    return;
}

__global__ void multiply(Matrix *a, Matrix *b, Matrix *c, float *aG){

    unsigned int i, cDim;

    i = blockIdx.x * blockDim.x + threadIdx.x;
    cDim = a -> sparseCount * b -> sparseCount;

    if(i < cDim){
        unsigned int aIndex = (unsigned int)((double)i / b -> sparseCount);
        unsigned int bIndex = i % b -> sparseCount;
        unsigned int unit = b -> rows * b -> columns;
        unsigned int rowUnit = a -> columns * unit;
        unsigned int bColumnNum, bRowNum, aColumnNum, aRowNum;

        bColumnNum = (b -> position[bIndex] + 1) % b -> columns;
        bRowNum = (unsigned int)((double)(b -> position[bIndex] + 1) / b -> columns);
        aColumnNum = (a -> position[aIndex] + 1) % a -> columns;
        aRowNum = (a -> position[aIndex] + 1) / a -> columns;

        c -> matrix[i] = a -> matrix[aIndex] * b -> matrix[bIndex];
        aG[i] = c -> matrix[i];
        c -> position[i] = (aColumnNum * b -> columns) + (aRowNum * rowUnit) + bColumnNum + (bRowNum * unit);
    }

    return;
}
