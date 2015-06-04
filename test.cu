#include <stdio.h>
#include <stdlib.h> 

typedef struct{
    int *pass;
    int element;
}Pass; 

__global__ void hello(int *a, int *b, Pass *p){

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < *b)
        p -> pass[i] = p -> pass[i] + p -> element;
}

int main(){

    int *a_host, b_host = 5;
    int *a_gpu, *b_gpu;
    Pass *p, *p_gpu, p_copy;

    a_host = (int*)malloc(sizeof(int) * 5);
    cudaMalloc(&a_gpu, 5 * sizeof(int));

    cudaMalloc(&b_gpu, sizeof(int));
    cudaMemcpy(b_gpu, &b_host, sizeof(int), cudaMemcpyHostToDevice);

    p = (Pass*)malloc(sizeof(Pass));
    p -> pass = (int*)malloc(5 * sizeof(int));
    
    for(int i = 0;i < 5;i++)
        p -> pass[i] = i;
    p -> element = 5;

    p_copy.element = 5;

    cudaMalloc(&p_gpu, sizeof(Pass));
    cudaMalloc(&(p_copy.pass), 5 * sizeof(int));
    cudaMemcpy((p_copy.pass), p -> pass, 5 * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(p_gpu, &p_copy, sizeof(p_copy), cudaMemcpyHostToDevice);
//    cudaMemcpy(p_gpu, p, sizeof(p), cudaMemcpyHostToDevice);

    int numBlocks = 1;
    int threadPerBlock = 512;

    hello<<<numBlocks, threadPerBlock>>>(a_gpu, b_gpu, p_gpu);

   // cudaMemcpy(a_host, a_gpu, 5 * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(p -> pass, (p_copy.pass), 5 * sizeof(int), cudaMemcpyDeviceToHost);

    int i;
    for(i = 0;i < 5;i++)
        printf("a[%d]: %d\n", i, p -> pass[i]);

    cudaFree(p_gpu);
    cudaFree(a_gpu);
    cudaFree(b_gpu);
    cudaFree(&p_copy.pass);

    free(p);
    free(a_host);

    return(0);    
}
