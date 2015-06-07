#include <stdio.h>
#include <stdlib.h>
#include <math.h>

typedef struct{
    float *matrix;
    unsigned long long int rows;
    unsigned long long int *position;
    unsigned long long int columns;
    unsigned long long int sparseCount;
    size_t pitch;
}Matrix;

void fillMatrix(Matrix*);
void COOTheMatrix(Matrix*);
void printMatrix(Matrix*);

void prepareGPUCopy(Matrix**);
void prepareHostGPUCopy(Matrix**, Matrix*);
void prepareResultHGPUCopy(Matrix**, Matrix**, Matrix**);
void retrieveResult(Matrix**);

void kronProd(Matrix*, Matrix*, Matrix**);
__global__ void multiply(Matrix*, Matrix*, Matrix*, unsigned long long int*);

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
    printMatrix(c);

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
    unsigned long long int i, elements, count = 0;
    float a = 5.0;

    elements = m -> rows * m -> columns;
    srand((unsigned int)time(NULL));

    for(i = 0; i < elements;i++){
        printf("%f\t", m -> matrix[i] = ((float)rand()/(float)(RAND_MAX)) * a);

        if(m -> matrix[i] != 0)
            count++;

        if((i + 1) % m -> columns == 0)
            printf("\n");
    }

    printf("\n");
    m -> sparseCount = count;
    COOTheMatrix(m);

    return;
}

void COOTheMatrix(Matrix *m){
    unsigned long long int i, elements, count;
    float *holding;

    count = m -> sparseCount;
    elements = m -> rows * m -> columns;

    m -> position = (unsigned long long int*)malloc(sizeof(unsigned long long int) * count * 2);
    m -> pitch = count * sizeof(unsigned long long int);
    holding = (float*)malloc(sizeof(float) * count);
    
    for(i = 0, count = 0;i < elements;i++){
        if(m -> matrix[i] != 0){
            m -> position[count] = i / (m -> columns);
            m -> position[m -> sparseCount + count] = i % (m -> columns); 
            holding[count++] = m -> matrix[i];
        }
    }

    free(m -> matrix);
    m -> matrix = holding;

    return;
}

void printMatrix(Matrix *m){
    unsigned long long int i, elements, count;
    unsigned long long int columns = m -> sparseCount;

    elements = m -> rows * m -> columns;
    printf("%llu\n", m -> columns);
    for(i = 0, count = 0;i < elements && count <= m -> sparseCount;i++){
      //  if(m -> position[count] == (i / m -> columns) && m -> position[columns + count] == (i % m -> columns)){
            printf("%f[%llu, %llu]\t", m -> matrix[count], m -> position[count], m -> position[columns + count]);
            count++;
      //  }
      //  else
      //      printf("0\t");

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

    checkError(cudaMallocPitch(&((*m_hgpu) -> position), &((*m_hgpu) -> pitch), m -> sparseCount * sizeof(unsigned long long int), 2));
    checkError(cudaMemcpy2D((*m_hgpu) -> position, (*m_hgpu) -> pitch, m -> position, m -> pitch, m -> sparseCount * sizeof(unsigned long long int), 2, cudaMemcpyHostToDevice));

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

    //Pitch doesn't have to be declared here as this method returns Host GPU copy
    //Which means that the pitch will be extracted from the cudaMallocPitch call

    return;
}

void retrieveResult(Matrix **m){

    Matrix *retVal;

    retVal = (Matrix*)malloc(sizeof(Matrix));

    retVal -> columns = (*m) -> columns;
    retVal -> rows = (*m) -> rows;
    retVal -> sparseCount = (*m) -> sparseCount;
    retVal -> pitch = (*m) -> sparseCount * sizeof(unsigned long long int);
    retVal -> matrix = (float*)malloc(sizeof(float) * (*m) -> sparseCount);
    retVal -> position = (unsigned long long int*)malloc(sizeof(unsigned long long int) * (*m) -> sparseCount * 2);

    checkError(cudaMemcpy(retVal -> matrix, (*m) -> matrix, (*m) -> sparseCount * sizeof(float), cudaMemcpyDeviceToHost));
    checkError(cudaMemcpy2D(retVal -> position, retVal -> pitch, (*m) -> position, (*m) -> pitch, (*m) -> sparseCount * sizeof(unsigned long long int), 2, cudaMemcpyDeviceToHost));

    //cudaFree((*m) -> matrix);
    //cudaFree((*m) -> position);
    //free(*m);
   printf("Jio %llu\n", (*m) -> sparseCount); 
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
    checkError(cudaMallocPitch(&((*c_hgpu) -> position), &((*c_hgpu) -> pitch), (*c_hgpu) -> sparseCount * sizeof(unsigned long long int), 2));

    prepareGPUCopy(&c_gpu, c_hgpu);

    int cSparse = (*c_hgpu) -> sparseCount;
    float blocks = ceil((float)cSparse / 512);

    unsigned int numBlocks = ((unsigned int)blocks);
    unsigned int threadsPerBlock = 512;

    unsigned long long int *aH = (unsigned long long int*)malloc(sizeof(unsigned long long int) * cSparse);
    unsigned long long int *aG;
    checkError(cudaMalloc(&aG, sizeof(unsigned long long int) * cSparse));
    
    multiply<<<numBlocks, threadsPerBlock>>>(a_gpu, b_gpu, c_gpu, aG);

    checkError(cudaMemcpy(aH, aG, sizeof(unsigned long long int) * cSparse, cudaMemcpyDeviceToHost));

    int i;
    for(i = 0;i < 36;i++)
        printf("aH[%d]: %llu\n", i, aH[i]);
    
    cudaFree(c_gpu);
    cudaFree(a_gpu);
    cudaFree(b_gpu);

    return;
}

__global__ void multiply(Matrix *a, Matrix *b, Matrix *c, unsigned long long int *aG){

    unsigned long long int i, cDim;

    i = blockIdx.x * blockDim.x + threadIdx.x;
    cDim = a -> sparseCount * b -> sparseCount;

    if(i < cDim){
        unsigned long long int aIndex = (unsigned int)((double)i / b -> sparseCount);
        unsigned long long int bIndex = i % b -> sparseCount;

        c -> matrix[i] = a -> matrix[aIndex] * b -> matrix[bIndex];

        unsigned long long int *cRow0 = (unsigned long long int*)((char*)c -> position); // + (0 * c -> pitch)
        unsigned long long int *cRow1 = (unsigned long long int*)((char*)c -> position + (1 * c -> pitch));

        unsigned long long int *aRow0 = (unsigned long long int*)((char*)a -> position); // + (0 * a -> pitch)
        unsigned long long int *aRow1 = (unsigned long long int*)((char*)a -> position + (1 * a -> pitch));
       
        unsigned long long int *bRow0 = (unsigned long long int*)((char*)b -> position); // + (0 * b -> pitch)
        unsigned long long int *bRow1 = (unsigned long long int*)((char*)b -> position + (1 * b -> pitch));

        cRow0[i] = aRow0[aIndex] * (b -> rows) + bRow0[bIndex];
        cRow1[i] = aRow1[aIndex] * (b -> columns) + bRow1[bIndex];

        aG[i] = aRow1[aIndex];
    }

    return;
}
