#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cusp/coo_matrix.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/device_ptr.h>

typedef cusp::host_memory host;
typedef cusp::device_memory device;

typedef cusp::coo_matrix<unsigned long long int, float, host> COOHost;
typedef cusp::coo_matrix<unsigned long long int, float, device> COODevice;

typedef struct{
    unsigned long long int rows;
    unsigned long long int columns;
    unsigned long long int sparseCount;
    COOHost *m_host;
    COODevice *m_device;
    size_t pitch;
}Matrix;

typedef struct{
    unsigned long long int *row;
    unsigned long long int rowSize;
    unsigned long long int *column;
    unsigned long long int colSize;
    float *values;
    unsigned long long int valSize;
}RawMatrix;

void fillMatrix(Matrix*);
void printMatrix(Matrix*);

void prepareGPUCopy(Matrix*);
void prepareResultMatrix(Matrix**, Matrix**, Matrix**);
void prepareRawMatrix(COODevice*, RawMatrix**);
void prepareResultRawMatrix(Matrix*, RawMatrix**);

void retrieveResult(Matrix**);
void kronProd(COODevice*, COODevice*, Matrix**);
__global__ void multiply(RawMatrix*, RawMatrix*, RawMatrix*, unsigned long long int*);

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

    a -> m_host = new COOHost(a -> rows, a -> columns, 6);
    b -> m_host = new COOHost(b -> rows, b -> columns, 6);

    fillMatrix(a);
    fillMatrix(b);

    printMatrix(a);
    printMatrix(b);

    prepareGPUCopy(a);
    prepareGPUCopy(b);
    prepareResultMatrix(&c, &a, &b);
    kronProd((a -> m_device), (b -> m_device), &c);
    
    retrieveResult(&c);
    printMatrix(c);

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
        printf("%f\t", m -> m_host -> values[i] = ((float)rand()/(float)(RAND_MAX)) * a);
        m -> m_host -> row_indices[count] = i / (m -> columns);
        m -> m_host -> column_indices[count] = i % (m -> columns); 

        if(m -> m_host -> values[i] != 0)
            count++;

        if((i + 1) % m -> columns == 0)
            printf("\n");
    }

    printf("\n");
    m -> sparseCount = count;
    return;
}

void printMatrix(Matrix *m){
    unsigned long long int i, elements, count;
    //unsigned long long int columns = m -> sparseCount;

    elements = m -> rows * m -> columns;
    printf("%llu\n", m -> columns);
    for(i = 0, count = 0;i < elements && count <= m -> sparseCount;i++){
        printf("%f[%llu, %llu]\t", 
                m -> m_host -> values[count], m -> m_host -> row_indices[count], 
                m -> m_host -> column_indices[count]);
        count++;

        if((i + 1) % m -> columns == 0)
            printf("\n");
    } 

    return;
}

void prepareGPUCopy(Matrix *m){

    m -> m_device = new COODevice(m -> rows, m -> columns, m -> sparseCount);
    thrust::copy(m -> m_host -> row_indices.begin(), m -> m_host -> row_indices.end(), m -> m_device -> row_indices.begin());
    thrust::copy(m -> m_host -> column_indices.begin(), m -> m_host -> column_indices.end(), m -> m_device -> column_indices.begin());
    thrust::copy(m -> m_host -> values.begin(), m -> m_host -> values.end(), m -> m_device -> values.begin());

    return;
}

void prepareResultMatrix(Matrix **c, Matrix **m1, Matrix **m2){

    *c = (Matrix*)malloc(sizeof(Matrix));
    (*c) -> sparseCount = (*m1) -> sparseCount * (*m2) -> sparseCount;
    (*c) -> rows = (*m1) -> rows * (*m2) -> rows;
    (*c) -> columns = (*m1) -> columns * (*m2) -> columns;

    return;
}

void retrieveResult(Matrix **m){

    (*m) -> m_host = new COOHost((*m) -> rows, (*m) -> columns, (*m) -> sparseCount);
    thrust::copy((*m) -> m_device -> row_indices.begin(), (*m) -> m_device -> row_indices.end(), (*m) -> m_host -> row_indices.begin());
    thrust::copy((*m) -> m_device -> column_indices.begin(), (*m) -> m_device -> column_indices.end(), (*m) -> m_host -> column_indices.begin());
    thrust::copy((*m) -> m_device -> values.begin(), (*m) -> m_device -> values.end(), (*m) -> m_host -> values.begin());

    return;
}

/*
 * This function accepts one COODevice matrix and converts it into individual
 * thrust vectors and copies them into another struct which can be passed to multiply
 */
void prepareRawMatrix(COODevice *m_device, RawMatrix **m){

    RawMatrix *rm = (RawMatrix*)malloc(sizeof(RawMatrix));

    rm -> row = thrust::raw_pointer_cast(&(m_device -> row_indices[0]));
    rm -> rowSize = m_device -> num_rows;

    rm -> column = thrust::raw_pointer_cast(&(m_device -> column_indices[0]));
    rm -> colSize = m_device -> num_cols;

    rm -> values = thrust::raw_pointer_cast(&(m_device -> values[0]));
    rm -> valSize = m_device -> values.size();

    checkError(cudaMalloc(m, sizeof(RawMatrix)));
    checkError(cudaMemcpy(*m, rm, sizeof(RawMatrix), cudaMemcpyHostToDevice));

    free(rm);

    return;
}

void prepareResultRawMatrix(Matrix *m, RawMatrix **rm){

    RawMatrix *host = (RawMatrix*)malloc(sizeof(RawMatrix));

    checkError(cudaMalloc(&(host -> row), sizeof(unsigned long long int) * m -> sparseCount));
    host -> rowSize = m -> rows;

    checkError(cudaMalloc(&(host -> column), sizeof(unsigned long long int) * m -> sparseCount));
    host -> colSize = m -> columns;

    checkError(cudaMalloc(&(host -> values), sizeof(float) * m -> sparseCount));
    host -> valSize = m -> sparseCount;

    checkError(cudaMalloc(rm, sizeof(RawMatrix)));
    checkError(cudaMemcpy(*rm, host, sizeof(RawMatrix), cudaMemcpyHostToDevice));

    return;
}

/*
 * This function accepts two GPU memory COO Matrices and a result matrix
 * initialized with only row and column count and generates
 * a Matrix struct with the result stored in GPU memory. The host COO matrix will be 
 * empty.
 */
void kronProd(COODevice *a, COODevice *b, Matrix **c){

    RawMatrix *aRaw, *bRaw, *cRaw, *cRsltRaw;
    //Initializing C_host 
    prepareRawMatrix(a, &aRaw);
    prepareRawMatrix(b, &bRaw);
    prepareResultRawMatrix(*c, &cRaw);
   
    int cSparse = (*c) -> sparseCount;
    float blocks = ceil((float)cSparse / 512);

    unsigned int numBlocks = ((unsigned int)blocks);
    unsigned int threadsPerBlock = 512;

    unsigned long long int *aH = (unsigned long long int*)malloc(sizeof(unsigned long long int) * cSparse);
    unsigned long long int *aG;
    checkError(cudaMalloc(&aG, sizeof(unsigned long long int) * cSparse));
    
    multiply<<<numBlocks, threadsPerBlock>>>(aRaw, bRaw, cRaw, aG);

    checkError(cudaMemcpy(aH, aG, sizeof(unsigned long long int) * cSparse, cudaMemcpyDeviceToHost));

    cRsltRaw = (RawMatrix*)malloc(sizeof(RawMatrix));
    checkError(cudaMemcpy(cRsltRaw, cRaw, sizeof(RawMatrix), cudaMemcpyDeviceToHost));

    COOHost c_host = COOHost((*c) -> rows, (*c) -> columns, (*c) -> sparseCount);
    //COODevice c_device(c_host);
    (*c) -> m_device = new COODevice(c_host);

    thrust::device_ptr<unsigned long long int> cRow(cRsltRaw -> row);
    thrust::device_ptr<unsigned long long int> cCol(cRsltRaw -> column);
    thrust::device_ptr<float> cVal(cRsltRaw -> values);

    thrust::device_vector<unsigned long long int> cVRow(cRow, cRow + cRsltRaw -> valSize);
    thrust::device_vector<unsigned long long int> cVCol(cCol, cCol + cRsltRaw -> valSize);
    thrust::device_vector<float> cVVal(cVal, cVal + cRsltRaw -> valSize);
    
    thrust::copy(cVRow.begin(), cVRow.end(), (*c) -> m_device -> row_indices.begin());
    thrust::copy(cVCol.begin(), cVCol.end(), (*c) -> m_device -> column_indices.begin());
    thrust::copy(cVVal.begin(), cVVal.end(), (*c) -> m_device -> values.begin());

    cudaFree(aG);
    cudaFree(aRaw);
    cudaFree(bRaw);
    cudaFree(cRaw);

    free(aH);

    return;
}

__global__ void multiply(RawMatrix *a, RawMatrix *b, RawMatrix *c, unsigned long long int *aG){

    unsigned long long int i, cDim;
    unsigned long long int aValSize = a -> valSize;
    unsigned long long int bValSize = b -> valSize;

    i = blockIdx.x * blockDim.x + threadIdx.x;
    cDim = aValSize * bValSize;

    if(i < cDim){

        unsigned long long int aIndex = (unsigned int)((double)i / bValSize);
        unsigned long long int bIndex = i % bValSize;

        unsigned long long int bRowSize = b -> rowSize;
        unsigned long long int bColSize = b -> colSize;

        unsigned long long int cRowIndex = a -> row[aIndex] * bRowSize + b -> row[bIndex];
        unsigned long long int  cColumnIndex = a -> column[aIndex] * bColSize + b -> column[bIndex];
        unsigned long long int index = cRowIndex * c -> colSize + cColumnIndex;

        c -> row[index] = cRowIndex;
        c -> column[index] = cColumnIndex;
        c -> values[index] = a -> values[aIndex] * b -> values[bIndex];

        aG[index] = bRowSize;
    }

    return;
}
