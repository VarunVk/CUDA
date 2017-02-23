#ifndef _PRESCAN_CU_
#define _PRESCAN_CU_

// includes, kernels
#include <assert.h>


#define NUM_BANKS 16
#define LOG_NUM_BANKS 4
// MP4.2 - You can use any other block size you wish.
#define BLOCK_SIZE 512 
#define MAX_BLKS 16384
__constant__ float Mc[MAX_BLKS];

// MP4.2 - Host Helper Functions (allocate your own data structure...)
float *out1_H, *in1_H; 

// MP4.2 - Device Functions


// MP4.2 - Kernel Functions
__global__ void vecadd_kernel(float *outArray, int numElements, int offset)
{
    unsigned int i = offset + threadIdx.x + blockIdx.x*blockDim.x;
    //printf("Entering adding %.3f to out[%d] \n",Mc[blockIdx.x], i);
    if(i < numElements)
        outArray[i] = outArray[i] + Mc[blockIdx.x];
}

__global__ void scan_kernel(float *outArray, float *inArray, int numElements)
{
    __shared__ float scan_array[2*BLOCK_SIZE];
    
    unsigned int t = threadIdx.x;
    unsigned int start = 2*blockDim.x*blockIdx.x;
    unsigned int index;

    // loading 2 inputs in a coalesed way
    if(start + t < numElements)
        scan_array[t] = inArray[start + t];
    else
        scan_array[t] = 0;

    if((start + blockDim.x + t) < numElements)
        scan_array[blockDim.x + t] = inArray[start + blockDim.x + t];
    else
        scan_array[blockDim.x + t] = 0;

    __syncthreads();

    int stride = 1;
    while(stride <= BLOCK_SIZE)
    {
        index = (t+1)*(stride << 1) - 1;
        if(index < 2*BLOCK_SIZE){
            scan_array[index] += scan_array[index-stride];
        }
        stride = stride << 1;
        __syncthreads();
    }

    stride = BLOCK_SIZE/2;
    while(stride > 0)
    {
        index = (t+1)*(stride << 1) - 1;
        if((index+stride) < 2*BLOCK_SIZE)
        {
            scan_array[index+stride] += scan_array[index];
        }
        stride = stride >> 1;
        __syncthreads();
    }

    if(start + t < numElements)
        outArray[start + t] = scan_array[t];

    if((start + blockDim.x + t) < numElements)
        outArray[start + blockDim.x + t] = scan_array[blockDim.x + t];
}

// **===-------- MP4.2 - Modify the body of this function -----------===**
// You may need to make multiple kernel calls, make your own kernel
// function in this file, and then call them from here.
void prescanArray(float *outArray, float *inArray, int numElements)
{
    dim3 dim_grid, dim_block;
    unsigned int numBlks_1;

    numBlks_1 = ceil((float)numElements/(2*BLOCK_SIZE));
    //printf("%s %d num elements %d #blks %d #threads %d \n",__func__,__LINE__,
    //			 numElements, numBlks_1, BLOCK_SIZE);
    
    dim_block.x = BLOCK_SIZE;dim_block.y =1; dim_block.z =1;
    dim_grid.x = numBlks_1; dim_grid.y = 1; dim_grid.z = 1;
    

    scan_kernel<<<dim_grid, dim_block>>>(outArray, inArray, numElements);
    cudaDeviceSynchronize();
   
    cudaMemcpy(out1_H, outArray, numElements*sizeof(float), cudaMemcpyDeviceToHost);
  
    // get block end sums 
    in1_H[0] = 0.0;
    for(int i=1; i < numBlks_1; i++){
        in1_H[i] = out1_H[2*BLOCK_SIZE*(i) -1] + in1_H[i-1];
    }

#ifdef MULTICPY_V
    in1_H[0] = 0.0;
    for(int i=1; i < numBlks_1; i++){
        cudaMemcpy(&in1_H[i], &outArray[2*BLOCK_SIZE*(i) -1], sizeof(float), cudaMemcpyDeviceToHost);
        in1_H[i] += in1_H[i-1];  
    }
#endif

    dim_block.x = 2*BLOCK_SIZE;dim_block.y =1; dim_block.z =1;

    unsigned int num_blks_remaining = numBlks_1;
    int start = 0, num_cur_blks;

    while(num_blks_remaining){
        if(num_blks_remaining > MAX_BLKS){
            num_cur_blks = MAX_BLKS;
            num_blks_remaining -= MAX_BLKS;
        }
        else {
            num_cur_blks = num_blks_remaining;
            num_blks_remaining = 0;
        }
        
        dim_grid.x = num_cur_blks; dim_grid.y = 1; dim_grid.z = 1;
        cudaMemcpyToSymbol(Mc, &in1_H[start*MAX_BLKS], num_cur_blks*sizeof(float));

        vecadd_kernel<<<dim_grid, dim_block>>>(outArray, numElements, start*2*BLOCK_SIZE*MAX_BLKS);
        cudaDeviceSynchronize();
        start++;
    }
}
// **===-----------------------------------------------------------===**


#endif // _PRESCAN_CU_
