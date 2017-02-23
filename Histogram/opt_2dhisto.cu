#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

#include <cutil.h>
#include "util.h"
#include "ref_2dhisto.h"
#include "opt_2dhisto.h"

//----------------------------------------------------------------------------
//      Kernel Function_1_Main_computation
//----------------------------------------------------------------------------
__device__  void atomicADD(uint32_t *address, uint32_t val) {
    unsigned int *address_as_uint = (unsigned int *)address;
    unsigned int old = *address_as_uint, assumed;
    do {
        if(old==UINT8_MAX)    
            break;
        if(old+val>UINT8_MAX)
            val = 255-old;
        assumed = old;
        old = atomicCAS(address_as_uint, assumed, assumed+val);
    } while(old != assumed);
}

__global__ void opt_2dhisto_kernel(uint32_t *input_device, int input_size, uint32_t *device_bins)
{
    int tx = threadIdx.x;
    int bx = blockIdx.x;
    int i = tx + bx* blockDim.x;

    if (device_bins[input_device[i]] <255) 
        atomicAdd(&(device_bins[input_device[i]]), 1);

    //int n = (threadIdx.y*blockDim.x)+threadIdx.x;

    //__shared__ uint32_t s_hist[HISTO_WIDTH];
    //s_hist[n]=0;

    //__syncthreads();

    /*
    for(int i=0; i<(INPUT_WIDTH/BLOCK_SIZE); i++)
        for(int j=0; j<(INPUT_HEIGHT/BLOCK_SIZE); j++)
        {
            int x = threadIdx.x + i*BLOCK_SIZE;
            int y = threadIdx.y + j*BLOCK_SIZE;
            if(x<INPUT_WIDTH && y<INPUT_HEIGHT)
                atomicADD(&s_hist[*(input+(y*INPUT_WIDTH)+x)], 1);
        }
        */

    //__syncthreads();
    //atomicAdd(&hist[n], s_hist[n]);
   // __syncthreads();
}

//----------------------------------------------------------------------------
//     HISTO_function 
//----------------------------------------------------------------------------

void opt_2dhisto(uint32_t *input_device, uint32_t *device_bins) 
{
    /* This function should only contain a call to the GPU 
       histogramming kernel. Any memory allocations and
       transfers must be done outside this function */

    int size = INPUT_HEIGHT * INPUT_WIDTH;
    cudaError_t cuda_ret;
    cudaMemset(device_bins, 0, sizeof(uint32_t) * HISTO_WIDTH);

    dim3 dimBlock(BLOCK_SIZE, 1,1);
    dim3 dimGrid( ((INPUT_HEIGHT * INPUT_WIDTH +  dimBlock.x - 1) / dimBlock.x) ,1,1);

    opt_2dhisto_kernel<<<dimGrid, dimBlock>>>(input_device, size, device_bins);
    cuda_ret = cudaDeviceSynchronize(); 
    if(cuda_ret != cudaSuccess) printf("Unable to launch/execute kernel \n");
}

/* Include below the implementation of any other functions you need */
uint8_t *AllocateDeviceMemory(int histo_width, int histo_height, int size_of_element)
{
    uint8_t *t_memory;
    cudaMalloc((void **)&t_memory, histo_width * histo_height * size_of_element);
    return t_memory;
}

void free_cuda(uint32_t *ptr)
{
    cudaFree(ptr);
}

void CopyToDevice(uint32_t *device, uint32_t *host, uint32_t input_height, uint32_t input_width, int size_of_element)
{
    const size_t x_size_padded = (input_width + 128) & 0xFFFFFF80;
    size_t row_size = input_width * size_of_element;

    for(int i=0; i<input_height; i++)
    {
        cudaMemcpy(device, host, row_size, cudaMemcpyHostToDevice);
        device += input_width;
        host += (x_size_padded);
    }
}

void CopyToHost(uint32_t *host, uint32_t *device, int size)
{
    cudaMemcpy(host,device, size, cudaMemcpyDeviceToHost);
    for(int i = 0; i < HISTO_WIDTH * HISTO_HEIGHT; i++)
        host[i] = host[i]>UINT8_MAX?UINT8_MAX:host[i];
}

void cuda_memset(uint32_t *ptr, uint32_t value, uint32_t byte_count)
{
    cudaMemset((void *)ptr, value, (size_t)byte_count);
}
