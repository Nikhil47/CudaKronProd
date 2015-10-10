/* Header file for Kronecker Product calculation on GPU project 
 * Author: Nikhil Singhal
 */

#ifndef GPONGPU_H
#define GPONGPU_H

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <cusp/coo_matrix.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/device_ptr.h>
#include <thrust/device_free.h>

typedef cusp::host_memory host;
typedef cusp::device_memory device;

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

#define checkError(call) { checkGPUError((call), __LINE__); }
inline void checkGPUError(cudaError_t errCode, int line){

    if(errCode != cudaSuccess){
        printf("\n");
        printf("Error at %d: %s\n", line, cudaGetErrorString(errCode));
    }   

    return;
}

#endif
