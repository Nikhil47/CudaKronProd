//Attempt to store the result in a 2D array to reduce page waste

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <iterator>
#include <array>
#include <cusp/coo_matrix.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/device_ptr.h>
#include <thrust/device_free.h>

typedef cusp::host_memory host;
typedef cusp::device_memory device;

typedef cusp::coo_matrix<unsigned int, float, host> COOHost;
typedef cusp::coo_matrix<unsigned int, float, device> COODevice;

typedef struct{
    unsigned int rows;
    unsigned int columns;
    unsigned int sparseCount;
    COOHost *m_host;
    COODevice *m_device;
}Matrix;

typedef struct{
    unsigned int *row;
    unsigned int rowSize;
    unsigned int *column;
    unsigned int colSize;
    float *values;
    unsigned int valSize;
}RawMatrix;

void fillMatrix(Matrix*, int);
void printMatrix(Matrix*);
void freeMatrix(Matrix**);

float prepareGPUCopy(Matrix*);
void prepareResultMatrix(Matrix**, COODevice*, COODevice*);
void prepareRawMatrix(COODevice*, RawMatrix**);
void prepareResultRawMatrix(Matrix*, RawMatrix**);

void retrieveResult(Matrix**);
float kronProd(COODevice*, COODevice*, Matrix**);

__global__ void multiply(RawMatrix*, RawMatrix*, RawMatrix*);
float cpuMultiply(COOHost*, COOHost*, COOHost**);

#define checkError(call) { checkGPUError((call), __LINE__); }
inline void checkGPUError(cudaError_t errCode, int line){

    if(errCode != cudaSuccess){
        printf("\n");
        printf("Error at %d: %s\n", line, cudaGetErrorString(errCode));
    }

    return;
}

int main(int argc, char *argv[]){

    unsigned int dim;
    Matrix *a, *b, *c_gpu, *c_cpu;
    float tTransfer = 0.0, tKronProd = 0.0;
    FILE *cpuReadings, *gpuReadings;

    cpuReadings = fopen("cpuTimes.dat", "a");
    if(cpuReadings == NULL){
        fprintf(stderr, "Failed to open cpu readings file");
    }

    gpuReadings = fopen("gpuTimes.dat", "a");
    if(gpuReadings == NULL){
        fprintf(stderr, "Failed to open gpu readings file");
    }

    a = (Matrix*)malloc(sizeof(Matrix));
    b = (Matrix*)malloc(sizeof(Matrix));
//    c_cpu = (Matrix*)malloc(sizeof(Matrix));

    if(a == NULL || b == NULL){
        fprintf(stderr, "Failed to allocate memory on CPU\n");
    }

    /*
     * Initialilizing the matrix parameters    
     */

    a -> rows = b -> rows = atoi(argv[1]);
    a -> columns = b -> columns = atoi(argv[2]);
    dim = a -> rows * a -> columns;

    fillMatrix(a, atoi(argv[3]));
    fillMatrix(b, atoi(argv[3]));

    //printMatrix(a);
    //printMatrix(b);

    //Kronecker Product being calculated on GPU
    tTransfer += prepareGPUCopy(a);
    tTransfer += prepareGPUCopy(b);

    tKronProd += kronProd((a -> m_device), (b -> m_device), &c_gpu);
    //delete result matrix, assuming it has been retrieved
    delete c_gpu -> m_device;

    //Write GPU's tTransfer + tKronProd to an external file
    fprintf(gpuReadings, "%u %f %f\n", dim, tTransfer, tKronProd);
/*
    //Kronecker Product being calculated on CPU
    tKronProd = tTransfer = 0.0;
    prepareResultMatrix(&c_cpu, &a, &b);
    tKronProd += cpuMultiply(a -> m_host, b -> m_host, &(c_cpu -> m_host));
    tTransfer += prepareGPUCopy(c_cpu);
    //delete result matrix, assuming it has served its purpose
    delete c_cpu -> m_device;

    //Write CPU's tTransfer + tKronProd to an external file
    fprintf(cpuReadings, "%llu %f %f\n", dim, tTransfer, tKronProd);
  
    retrieveResult(&c_gpu);
    printMatrix(c_gpu);
*/    
    freeMatrix(&a);
    freeMatrix(&b);
//    freeMatrix(&c_cpu);
    freeMatrix(&c_gpu);
    
    cudaDeviceSynchronize();
    cudaDeviceReset();
    size_t mem_total, mem_free;
    cudaMemGetInfo(&mem_free, &mem_total);
    printf("Memory: %lu / %lu\n", mem_free, mem_total);

    fclose(cpuReadings);
    fclose(gpuReadings);

    return(0);
}

void freeMatrix(Matrix **m){

    delete (*m) -> m_host;
    free(*m);

    return;
}

void fillMatrix(Matrix *m, int percent){
    unsigned int i, elements, count = 0;
    float a = 5.0;

    elements = (unsigned int)((percent / 100) * m -> rows * m -> columns);
    srand((unsigned int)time(NULL));

    m -> m_host = new COOHost(m -> rows, m -> columns, elements);

    while(count < elements){
        unsigned int index = (unsigned int) (elements * ((double) rand() / (RAND_MAX + 1.0)));

        if(!m -> m_host -> values[index]){
            m -> m_host -> values[count] = ((float)rand()/(float)(RAND_MAX)) * a;
            m -> m_host -> row_indices[count] = index / (m -> columns);
            m -> m_host -> column_indices[count] = index % (m -> columns); 
            count++;
        }
    }

    m -> sparseCount = elements;
    return;
}

void printMatrix(Matrix *m){
    unsigned int i, elements, count;
    //unsigned int columns = m -> sparseCount;

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

float prepareGPUCopy(Matrix *m){

    float time;
    cudaEvent_t start, stop;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start, 0);

    m -> m_device = new COODevice(m -> rows, m -> columns, m -> sparseCount);
    thrust::copy(m -> m_host -> row_indices.begin(), m -> m_host -> row_indices.end(), m -> m_device -> row_indices.begin());
    thrust::copy(m -> m_host -> column_indices.begin(), m -> m_host -> column_indices.end(), m -> m_device -> column_indices.begin());
    thrust::copy(m -> m_host -> values.begin(), m -> m_host -> values.end(), m -> m_device -> values.begin());

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaDeviceSynchronize();

    cudaEventElapsedTime(&time, start, stop);

    return time;
}

void prepareResultMatrix(Matrix **c, COODevice *m1, COODevice *m2){

    *c = (Matrix*)malloc(sizeof(Matrix));
    (*c) -> sparseCount = m1 -> values.size() * m2 -> values.size();
    (*c) -> rows = m1 -> num_rows * m2 -> num_rows;
    (*c) -> columns = m1 -> num_cols * m2 -> num_cols;

    (*c) -> m_device = new COODevice((*c) -> rows, (*c) -> columns, (*c) -> sparseCount);

    return;
}

void retrieveResult(Matrix **m){

    (*m) -> m_host = new COOHost((*m) -> rows, (*m) -> columns, (*m) -> sparseCount);
    thrust::copy((*m) -> m_device -> row_indices.begin(), (*m) -> m_device -> row_indices.end(), (*m) -> m_host -> row_indices.begin());
    thrust::copy((*m) -> m_device -> column_indices.begin(), (*m) -> m_device -> column_indices.end(), (*m) -> m_host -> column_indices.begin());
    thrust::copy((*m) -> m_device -> values.begin(), (*m) -> m_device -> values.end(), (*m) -> m_host -> values.begin());

    cudaDeviceSynchronize();

    return;
}

/*
 * This function accepts one COODevice matrix and converts it into individual
 * thrust vectors and copies them into another struct which can be passed to multiply function.
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

/*
 * Takes the matrix in which the result is to be stored and returns a 
 * RawMatrix pointer residing on GPU memory.
 */
void prepareResultRawMatrix(Matrix *m, RawMatrix **rm){

    RawMatrix *host = (RawMatrix*)malloc(sizeof(RawMatrix));

    checkError(cudaMalloc(&(host -> row), sizeof(unsigned int) * m -> sparseCount));
    host -> rowSize = m -> rows;

    checkError(cudaMalloc(&(host -> column), sizeof(unsigned int) * m -> sparseCount));
    host -> colSize = m -> columns;

    checkError(cudaMalloc(&(host -> values), sizeof(float) * m -> sparseCount));
    host -> valSize = m -> sparseCount;

    checkError(cudaMalloc(rm, sizeof(RawMatrix)));
    checkError(cudaMemcpy(*rm, host, sizeof(RawMatrix), cudaMemcpyHostToDevice));

    free(host);

    return;
}

/*
 * This function accepts two GPU memory COO Matrices and a result matrix
 * initialized with only row and column count and generates
 * a Matrix struct with the result stored in GPU memory. The host COO matrix will be 
 * empty.
 * Returns the time taken to complete the multiplication.
 */
float kronProd(COODevice *a, COODevice *b, Matrix **c){

    float time;
    cudaEvent_t start, stop;
    RawMatrix *aRaw, *bRaw, *cRaw;

    //Initializing events to record time taken to multiply the matrices
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    //Copying a and b to RawMatrix - tp pass it to multiply function
    prepareRawMatrix(a, &aRaw);
    prepareRawMatrix(b, &bRaw);

    //Initializing the matrix for GPU, where result will be stored
    prepareResultMatrix(c, a, b);
    prepareRawMatrix((*c) -> m_device, &cRaw);

    //Calculating the blocks and grid dimensions
    int cSparse = (*c) -> sparseCount;
    float blocks = ceil((float)cSparse / 512);

    unsigned int numBlocks = ((unsigned int)blocks);
    unsigned int threadsPerBlock = 512;
/*
    //Initializing the debug array used to check various values being generated in the GPU
    float *aH = (float*)malloc(sizeof(float) * cSparse);
    float *aG;
    checkError(cudaMalloc(&aG, sizeof(float) * cSparse));
*/
    //Start recording time
    cudaEventRecord(start, 0);
    
    multiply<<<numBlocks, threadsPerBlock>>>(aRaw, bRaw, cRaw);
    
    //Stop recording time
    cudaDeviceSynchronize();
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
   
    size_t mem_total, mem_free;
    cudaMemGetInfo(&mem_free, &mem_total);
    printf("Memory after multiply: %lu / %lu\n", mem_free, mem_total);

    delete a;
    delete b;
/*
    checkError(cudaMemcpy(aH, aG, sizeof(float) * cSparse, cudaMemcpyDeviceToHost));
*/
    cudaFree(aRaw);
    cudaFree(bRaw);

    cudaDeviceSynchronize();

//    free(aH);

    //calculating time for multiplication
    cudaEventElapsedTime(&time, start, stop);

    return time;
}

__global__ void multiply(RawMatrix *a, RawMatrix *b, RawMatrix *c){

    unsigned int i, cDim;
    unsigned int aValSize = a -> valSize;
    unsigned int bValSize = b -> valSize;

    i = blockIdx.x * blockDim.x + threadIdx.x;
    cDim = aValSize * bValSize;

    if(i < cDim){

        unsigned int aIndex = (unsigned int)((double)i / bValSize);
        unsigned int bIndex = i % bValSize;

        unsigned int bRowSize = b -> rowSize;
        unsigned int bColSize = b -> colSize;

        unsigned int cRowIndex = a -> row[aIndex] * bRowSize + b -> row[bIndex];
        unsigned int  cColumnIndex = a -> column[aIndex] * bColSize + b -> column[bIndex];
        unsigned int index = cRowIndex * c -> colSize + cColumnIndex;

        c -> row[index] = cRowIndex;
        c -> column[index] = cColumnIndex;
        c -> values[index] = a -> values[aIndex] * b -> values[bIndex];

        //aG[index] = c -> values[index];
    }

    return;
}

float cpuMultiply(COOHost *a, COOHost *b, COOHost **c){
    clock_t begin, end;
    float time_spent;
    unsigned int i, index, aIndex, bIndex, bRowSize, bColSize, cRowIndex,cColumnIndex;
    unsigned int sparseCount = a -> values.size() * b -> values.size();

    begin = clock();

    *c = new COOHost(a -> num_rows * b -> num_rows
            , a -> num_cols * b -> num_cols
            , sparseCount);

    for(i = 0;i < sparseCount;i++){
        aIndex = (unsigned int)((double)i / b -> values.size());
        bIndex = i % b -> values.size();

        bRowSize = b -> num_rows;
        bColSize = b -> num_cols;

        cRowIndex = a -> row_indices[aIndex] * bRowSize + b -> row_indices[bIndex];
        cColumnIndex = a -> column_indices[aIndex] * bColSize + b -> column_indices[bIndex];
        index = cRowIndex * (*c) -> num_cols + cColumnIndex;

        (*c) -> values[index] = a -> values[aIndex] * b -> values[bIndex];
        (*c) -> column_indices[index] = cColumnIndex;
        (*c) -> row_indices[index] = cRowIndex;
    }

    end = clock();
    time_spent = (float)(end - begin) / CLOCKS_PER_SEC;
    
    return time_spent;
}
