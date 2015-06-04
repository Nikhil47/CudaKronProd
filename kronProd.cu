#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define DEBUG_TYPE float

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
void kronProd(Matrix*, Matrix*, Matrix**);
__global__ void multiply(Matrix*, Matrix*, Matrix*, DEBUG_TYPE*);
__global__ void kernel(DEBUG_TYPE*);

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

    kronProd(a, b, &c);
//    printf("C populated: %d\n", c -> sparseCount);
//    printMatrix(c);

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

void prepareGPUCopy(Matrix **m_hgpu, Matrix **m){
    
    (*m_hgpu) = (Matrix*)malloc(sizeof(Matrix));
    (*m_hgpu) -> rows = (*m) -> rows;
    (*m_hgpu) -> columns = (*m) -> columns;
    (*m_hgpu) -> sparseCount = (*m) -> sparseCount;
    
    checkError(cudaMalloc(&((*m_hgpu) -> matrix), (*m) -> sparseCount * sizeof(float)));
    checkError(cudaMemcpy((*m_hgpu) -> matrix, (*m) -> matrix, (*m) -> sparseCount * sizeof(float), cudaMemcpyHostToDevice));

    checkError(cudaMalloc(&((*m_hgpu) -> position), (*m) -> sparseCount * sizeof(int)));
    checkError(cudaMemcpy((*m_hgpu) -> position, (*m) -> position, (*m) -> sparseCount * sizeof(int), cudaMemcpyHostToDevice));

    return;
}

void kronProd(Matrix *a, Matrix *b, Matrix **c){

    Matrix *a_gpu, *b_gpu, *c_gpu;
    Matrix *a_hgpu, *b_hgpu, c_hgpu;
    Matrix *cRet;

    prepareGPUCopy(&a_hgpu, &a);
    checkError(cudaMalloc(&a_gpu, sizeof(Matrix)));
    checkError(cudaMemcpy(a_gpu, a_hgpu, sizeof(Matrix), cudaMemcpyHostToDevice));

    prepareGPUCopy(&b_hgpu, &b);
    checkError(cudaMalloc(&b_gpu, sizeof(Matrix)));
    checkError(cudaMemcpy(b_gpu, b_hgpu, sizeof(Matrix), cudaMemcpyHostToDevice));
    
    //  Initializing C_host 
    //c_hgpu = (Matrix*)malloc(sizeof(Matrix));
    int cSparse = c_hgpu.sparseCount = a -> sparseCount * b -> sparseCount;
    c_hgpu.rows = a -> rows * b -> rows;
    c_hgpu.columns = a -> columns * b -> columns;
   
    checkError(cudaMalloc(&(c_hgpu.matrix), cSparse * sizeof(float)));
    checkError(cudaMalloc(&(c_hgpu.position), cSparse * sizeof(int)));

    checkError(cudaMalloc(&c_gpu, sizeof(Matrix)));
    checkError(cudaMemcpy(c_gpu, &c_hgpu, sizeof(Matrix), cudaMemcpyHostToDevice));

    printf("ok here\n");
    float blocks = ceil((float)cSparse / 512);

    unsigned int numBlocks = ((unsigned int)blocks);
    unsigned int threadsPerBlock = 512;
    
    DEBUG_TYPE *aH = (DEBUG_TYPE*)malloc(sizeof(DEBUG_TYPE) * cSparse);
    DEBUG_TYPE *aG;
    checkError(cudaMalloc(&aG, sizeof(DEBUG_TYPE) * cSparse));

    multiply<<<numBlocks, threadsPerBlock>>>(a_gpu, b_gpu, c_gpu, aG);

    cRet = (Matrix*)malloc(sizeof(Matrix));
    cRet -> matrix = (float*)malloc(sizeof(float) * cSparse);
    cRet -> position = (int*)malloc(sizeof(int) * cSparse);

    checkError(cudaMemcpy(cRet, c_gpu, sizeof(Matrix), cudaMemcpyDeviceToHost));

    checkError(cudaMemcpy(cRet -> matrix, c_hgpu.matrix, cSparse * sizeof(float), cudaMemcpyDeviceToHost));
    checkError(cudaMemcpy(cRet -> position, c_hgpu.position, cSparse * sizeof(int), cudaMemcpyDeviceToHost));

    checkError(cudaMemcpy(aH, aG, cSparse * sizeof(DEBUG_TYPE), cudaMemcpyDeviceToHost));
printf("display\n");    
    int i;
    for(i = 0;i < 36;i++)
        printf("a[%d]: %f\n", i, cRet -> matrix[i]);

    cudaFree(a_gpu);
    cudaFree(b_gpu);
    cudaFree(c_gpu);
    cudaFree(a_hgpu -> matrix);
    cudaFree(a_hgpu -> position);
    cudaFree(b_hgpu -> matrix);
    cudaFree(b_hgpu -> position);
    cudaFree(c_hgpu.matrix);
    cudaFree(c_hgpu.position);
    cudaFree(aG);

    free(aH);
    free(a_hgpu);
    free(b_hgpu);
    //free(c_hgpu);
    
//    printf("C populated: %f\n", c_host.matrix[0]);
    //printMatrix(c);
    return;
}

__global__ void multiply(Matrix *a, Matrix *b, Matrix *c, DEBUG_TYPE *aG){

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

__global__ void kernel(DEBUG_TYPE *aG){

    int i =  blockIdx.x * blockDim.x + threadIdx.x;
    if(i < 36)
        aG[2] = 89;
    return;
}
