//Cuda method to add scalar values to the diagonal of a matrix

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cusp/coo_matrix.h>
#include <thrust/device_vector.h>
#include <thrust/device_ptr.h>
#include <thrust/device_free.h>
#include <thrust/host_vector.h>

typedef unsigned int ulli;

typedef cusp::host_memory host;
typedef cusp::device_memory device;

typedef cusp::coo_matrix<ulli, double, host> COOHost;
typedef cusp::coo_matrix<ulli, double, device> COODevice;

//Structures for Matrix and RawMatrix
typedef struct{
    ulli rows;
    ulli columns;
    ulli sparseCount;
    COOHost *m_host;
    COODevice *m_device;
}Matrix;

typedef struct{
    ulli *row;      //stores row indices - array
    ulli rowSize;   //stores number of rows - single int
    ulli *column;   //stores column indices - array
    ulli colSize;   //stores number of columns - single int
    double *values; //stores values - array
    ulli valSize;   //stores number of values - single int
}RawMatrix;

//Function prototypes
void fillMatrix(Matrix*);
void prepareGPUCopy(Matrix*);
void retrieveResult(Matrix**);
void printMatrix(Matrix*);

void diagSum(double, COODevice*);
__global__ void cudaDiagSum(double*, RawMatrix*);

#define checkError(call) { checkGPUError((call), __LINE__); }
inline void checkGPUError(cudaError_t errCode, int line){

    if(errCode != cudaSuccess){
        printf("\n");
        printf("Error at %d: %s\n", line, cudaGetErrorString(errCode));
    }

    return;
}


int main(){

    Matrix *m;
    ulli dim = 5;

    m = (Matrix*)malloc(sizeof(Matrix));

    if(m == NULL){
        fprintf(stderr, "Failed to allocate memory to Matrix");
    }

    m -> rows = m -> columns = dim;
    m -> m_host = new COOHost(m -> rows, m -> columns, dim * dim);

    fillMatrix(m);
    printMatrix(m);

    prepareGPUCopy(m);    
    diagSum(5.0, m -> m_device);

    printf("After diag sum: \n");
    retrieveResult(&m);
    printMatrix(m);

    return(0);
}

void fillMatrix(Matrix *m){
    ulli i;
    double a = 5.0;

    srand((unsigned int)time(NULL));
    for(i = 0;i < m -> m_host -> values.size();i++){
        m -> m_host -> values[i] = ((double)rand()/(double)(RAND_MAX)) * a;
        m -> m_host -> row_indices[i] = i / m -> columns;
        m -> m_host -> column_indices[i] = i % m -> columns;
    }

    m -> sparseCount = m -> m_host -> values.size();
    return;
}

void printMatrix(Matrix *m){
    unsigned long long int i, elements, count;
    //unsigned long long int columns = m -> sparseCount;

    elements = m -> rows * m -> columns;
    printf("%u\n", m -> columns);
    for(i = 0, count = 0;i < elements && count <= m -> sparseCount;i++){
        printf("%f[%u, %u]\t",
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

void retrieveResult(Matrix **m){

    (*m) -> m_host = new COOHost((*m) -> rows, (*m) -> columns, (*m) -> sparseCount);

    thrust::copy((*m) -> m_device -> row_indices.begin(), (*m) -> m_device -> row_indices.end(), (*m) -> m_host -> row_indices.begin());
    thrust::copy((*m) -> m_device -> column_indices.begin(), (*m) -> m_device -> column_indices.end(), (*m) -> m_host -> column_indices.begin());
    thrust::copy((*m) -> m_device -> values.begin(), (*m) -> m_device -> values.end(), (*m) -> m_host -> values.begin());

    return;
}

void diagSum(double num, COODevice *m_device){

    RawMatrix *m;
    double *num_gpu;

    prepareRawMatrix(m_device, &m);
    checkError(cudaMalloc(&num_gpu, sizeof(double)));
    checkError(cudaMemcpy(num_gpu, &num, sizeof(double), cudaMemcpyHostToDevice));

    //Calculating the blocks and grid dimensions
    int cSparse = m_device -> values.size();
    float blocks = ceil((float)cSparse / 512);

    unsigned int numBlocks = ((unsigned int)blocks);
    unsigned int threadsPerBlock = 512;

    cudaDiagSum<<<numBlocks, threadsPerBlock>>>(num_gpu, m);

    cudaFree(num_gpu);

    return;
}

__global__ void cudaDiagSum(double *num, RawMatrix *rm){

    ulli i;

    i = blockIdx.x * blockDim.x + threadIdx.x;

    if(rm -> row[i] == rm -> column[i]){
        rm -> values[i] += *num;
    }

    return;
}
