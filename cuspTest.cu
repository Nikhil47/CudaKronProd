#include <stdio.h>
#include <cusp/coo_matrix.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

typedef cusp::host_memory host;
typedef cusp::device_memory device;

typedef cusp::coo_matrix<int, float, host> COOHost;
typedef cusp::coo_matrix<int, float, device> COODevice;

struct matrix{
    unsigned long int columns;
    unsigned long int rows;
    COOHost m_host;
    COODevice m_device;
}Matrix;

void initialize(COOHost *);
void kronProd();

int main(){

    Matrix a, b;

    initialize(&a_host);
    initialize(&b_host);

    return(0);
}

void initialize(Matrix *m){

    int i, j;

    printf("%lu %lu\n", m -> row_indices.size(), m -> column_indices.size());

    return;
}
