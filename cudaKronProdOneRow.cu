//Giving input as one row vectors and generating the matrices in device memory

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

/*
 * This struct holds matrix attributes in the Host memory,
 * 'rows' holds the number of rows in the original matrix,
 * 'columns' holds the number of columns in the original matrix,
 * 'sparseCount' holds the number of non-zero elements in the original matrix.
 * m_host is used to store the matrix in the sparse format in host memory
 * m_device is used to store the address of the matrix in the device memory.
 * Note: device memory sparse matrix can not be accessed using the m_device.
 * Will give segmentation fault. 
 */
typedef struct{
    unsigned int rows;
    unsigned int columns;
    unsigned int sparseCount;
    COOHost *m_host;

    COODevice *m_device;
}Matrix;

/* 
 * This struct holds the Matrix attributes in the Device memory,
 * the rowSize holds the number of entries in the array 'row',
 * the colSize holds the number of entries in the array 'column',
 * the valSize holds the number of entries in the array 'values',
 * This struct can not be used to recreate the whole matrix in the host device.
 * For that the Matrix struct is required,
 * This matrix is used to pass data into the kernel as cuda kernels do not 
 * recognize sparse matrix representations.
 */
typedef struct{
    unsigned int rows;
    unsigned int *row;
    unsigned int rowSize;

    unsigned int columns;
    unsigned int *column;
    unsigned int colSize;

    unsigned int sparseCount;
    float *values;
    unsigned int valSize;
}RawMatrix;

/*
 * The fillMatrix function is just a filler function, meaning it is used to 
 * generate test matrices to test this code.
 */  
void fillMatrix(Matrix*, int);

/*
 * The printMatrix function is used to print the sparse matrix representations
 * as a full matrix on console. 
 */
void printMatrix(Matrix*);

/*
 * The freeMatrix function is used to free the 'Matrix' struct.
 */
void freeMatrix(Matrix**);

/*
 * The prepareGPUCopy function is used to allocate memory for the sparse matrix representation
 * on the device memory and transfer the matrix from host to device memory. 
 */
float prepareGPUCopy(Matrix*);

/*
 * The prepareResultMatrix function is used to allocate memory for the resultant matrix on the
 * device memory.
 */
void prepareResultMatrix(Matrix**, COODevice*, COODevice*);

/*
 * The prepareRawMatrix function is used to convert the sparse matrix now residing on the device
 * memory to the raw format which can be passed to the Kernel. This is because the Sparse Matrix
 * representations are defined by third party libraries.
 */
void prepareRawMatrix(COODevice*, RawMatrix**);

/*
 * The prepareResultRawMatrix function does the same job as the prepareRawMatrix function, but now for
 * the resultant matrix.
 */
void prepareResultRawMatrix(Matrix*, RawMatrix**);

/*
 * the retrieveResult function is to get the resultant matrix back to the host memory.
 */
void retrieveResult(Matrix**);

/*
 * The kronProd function is the wrapper function for the kernel function. The requirement dictates that the 
 * this method accepts two Sparse Matrix representations residing in the Device memory and produce a result
 * matrix in the host memory.
 */
float kronProd(Matrix*, Matrix*, Matrix**);

/*
 * The multiply function is used to perform the actual Kronecker product on the Device.
 */
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
    MatrixAttr *aAttr, *bAttr;
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
    aAttr = (MatrixAttr)malloc(sizeof(MatrixAttr));

    b = (Matrix*)malloc(sizeof(Matrix));
    bAttr = (MatrixAttr)malloc(sizeof(MatrixAttr));
//    c_cpu = (Matrix*)malloc(sizeof(Matrix));

    if(a == NULL || b == NULL){
        fprintf(stderr, "Failed to allocate memory on CPU\n");
    }

    /*
     * Initialilizing the matrix parameters    
     */
    
    // Matrix as only one row.
    a -> rows = a -> columns = atoi(argv[1]);

    b -> rows = b -> columns = atoi(argv[2]);

    dim = a -> columns * b -> columns;

    fillMatrix(a, atoi(argv[3]));
    fillMatrix(b, atoi(argv[3]));

    //printMatrix(a);
    //printMatrix(b);

    //Kronecker Product being calculated on GPU
    tTransfer += prepareGPUCopy(a);
    tTransfer += prepareGPUCopy(b);

    tKronProd += kronProd((a -> m_device), aAttr, (b -> m_device), bAttr, &c_gpu);
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

    elements = 1 * m -> columns;
    srand((unsigned int)time(NULL));

    m -> m_host = new COOHost(1, m -> columns, elements);

    while(count < elements){
            
        m -> m_host -> values[count] = ((float)rand()/(float)(RAND_MAX)) * a;
        m -> m_host -> row_indices[count] = 1;
        m -> m_host -> column_indices[count] = count; 
        count++;
    }

    m -> sparseCount = elements * elements;
    return;
}

void printMatrix(Matrix *m){
    unsigned int i, elements, count;
    //unsigned int columns = m -> sparseCount;

    elements = 1 * m -> columns;
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
/*
    MatrixAttr *attr = (MatrixAttr)malloc(sizeof(MatrixAttr));
    attr -> rows_device = m -> rows;
    attr -> columns_device = m -> columns; 
    attr -> sparseCount_device = m -> sparseCount;

    checkError(cudaMalloc(mAttr, sizeof(MatrixAttr)));
    checkError(cudaMemcpy(*mAttr, attr, sizeof(MatrixAttr), cudaMemcpyHostToDevice));
*/
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaDeviceSynchronize();

    cudaEventElapsedTime(&time, start, stop);

    return time;
}

void prepareResultMatrix(Matrix **c, Matrix *m1, Matrix *m2){

    *c = (Matrix*)malloc(sizeof(Matrix));
    (*c) -> sparseCount = m1 -> sparseCount * m2 -> sparseCount;
    (*c) -> rows = m1 -> rows * m2 -> rows;
    (*c) -> columns = m1 -> columns * m2 -> columns;

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
void prepareRawMatrix(Matrix *m, RawMatrix **rmf){

    RawMatrix *rm = (RawMatrix*)malloc(sizeof(RawMatrix));

    rm -> row = thrust::raw_pointer_cast(&(m -> m_device -> row_indices[0]));
    rm -> rowSize = m -> m_device -> num_rows;
    rm -> rows = m -> rows;

    rm -> column = thrust::raw_pointer_cast(&(m -> m_device -> column_indices[0]));
    rm -> colSize = m -> m_device -> num_cols;
    rm -> columns = m -> columns;

    rm -> values = thrust::raw_pointer_cast(&(m -> m_device -> values[0]));
    rm -> valSize = m -> m_device -> values.size();

    rm -> sparseCount = m -> sparseCount;

    checkError(cudaMalloc(rmf, sizeof(RawMatrix)));
    checkError(cudaMemcpy(*rmf, rm, sizeof(RawMatrix), cudaMemcpyHostToDevice));
    
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
float kronProd(Matrix *a, Matrix *b, Matrix **c){

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
    prepareRawMatrix(*c, &cRaw);

    //Calculating the blocks and grid dimensions
    int cSparse = (*c) -> sparseCount;
    float blocks = ceil((float)((*c) -> rows *  (*c) -> columns)/ 512);

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
    unsigned int aValSize = a -> sparseCount;
    unsigned int bValSize = b -> sparseCount;

    i = blockIdx.x * blockDim.x + threadIdx.x;
    cDim = c -> sparseCount;

    if(i < cDim){

        unsigned int aCollapsedIndex, bCollapsedIndex;
        unsigned int aIndex = (unsigned int)((double)i / bValSize);
        unsigned int bIndex = i % bValSize;

        unsigned int aRow, aCol, bRow, bCol, aTrueRow, bTrueRow, aTrueCol, bTrueCol;
        aRow = (unsigned int)((double)aIndex / aValsize);
        aCol = aIndex % aValsize;
        bRow = (unsigned int)((double)bIndex / bValsize);
        bCOl = bIndex % bValsize;

        if(aCol != aRow){
            aCollapsedIndex = abs(aCol - aRow);
            
            if(aCol > aRow){
                aTrueCol = a -> column[aCollapsedIndex] + aRow;
                aTrueRow = a -> row[aCollapsedIndex] + aRow;
            }
            else{
                aTrueCol = a -> row[aCollapsedIndex] + aRow - 1;
                aTrueRow = a -> column[aCollapsedIndex] + aRow - 1;
            }
        }
        else{
            aCollapsedIndex = 0;

            aTrueCol = a -> column[aCollapsedIndex] + aRow;
            aTrueRow = a -> row[aCollapsedIndex] + aRow;
        }

        if(bCol != bRow){
            bCollapsedIndex = abs(bCol - bRow);

            if(bCol > bRow){
                bTrueCol = b -> column[bCollapsedIndex] + bRow;
                bTrueRow = b -> row[bCollapsedIndex] + bRow;
            }
            else{
                bTrueCol = b -> row[bCollapsedIndex] + bRow - 1;
                bTrueRow = b -> column[bCollapsedIndex] + bRow - 1;
            }
        }
        else{
            bCollapsedIndex = 0;

            bTrueCol = b -> column[bCollapsedIndex] + bRow;
            bTrueRow = b -> row[bCollapsedIndex] + bRow;
        }

        c -> row[i] = aTrueRow * b -> rows + bTrueRow;
        c -> column[i] = aTrueCol * b -> columns + bTrueCol;
        c -> values[i] = a -> values[aCollapsedIndex] * b -> values[bCollapsedIndex];
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
