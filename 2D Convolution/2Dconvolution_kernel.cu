/*
 * Copyright 1993-2006 NVIDIA Corporation.  All rights reserved.
 *
 * NOTICE TO USER:   
 *
 * This source code is subject to NVIDIA ownership rights under U.S. and 
 * international Copyright laws.  
 *
 * This software and the information contained herein is PROPRIETARY and 
 * CONFIDENTIAL to NVIDIA and is being provided under the terms and 
 * conditions of a Non-Disclosure Agreement.  Any reproduction or 
 * disclosure to any third party without the express written consent of 
 * NVIDIA is prohibited.     
 *
 * NVIDIA MAKES NO REPRESENTATION ABOUT THE SUITABILITY OF THIS SOURCE 
 * CODE FOR ANY PURPOSE.  IT IS PROVIDED "AS IS" WITHOUT EXPRESS OR 
 * IMPLIED WARRANTY OF ANY KIND.  NVIDIA DISCLAIMS ALL WARRANTIES WITH 
 * REGARD TO THIS SOURCE CODE, INCLUDING ALL IMPLIED WARRANTIES OF 
 * MERCHANTABILITY, NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.   
 * IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY SPECIAL, INDIRECT, INCIDENTAL, 
 * OR CONSEQUENTIAL DAMAGES, OR ANY DAMAGES WHATSOEVER RESULTING FROM LOSS 
 * OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE 
 * OR OTHER TORTIOUS ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE 
 * OR PERFORMANCE OF THIS SOURCE CODE.  
 *
 * U.S. Government End Users.  This source code is a "commercial item" as 
 * that term is defined at 48 C.F.R. 2.101 (OCT 1995), consisting  of 
 * "commercial computer software" and "commercial computer software 
 * documentation" as such terms are used in 48 C.F.R. 12.212 (SEPT 1995) 
 * and is provided to the U.S. Government only as a commercial end item.  
 * Consistent with 48 C.F.R.12.212 and 48 C.F.R. 227.7202-1 through 
 * 227.7202-4 (JUNE 1995), all U.S. Government End Users acquire the 
 * source code with only those rights set forth herein.
 */

#ifndef _2DCONVOLUTION_KERNEL_H_
#define _2DCONVOLUTION_KERNEL_H_

#include <stdio.h>
#include "2Dconvolution.h"

// Matrix convolution kernel specification
__global__ void ConvolutionKernel(Matrix M, Matrix N, Matrix P)
{
    __shared__ float Ns[BLOCK_SIZE][BLOCK_SIZE];
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int row_o = blockIdx.y * TILE_SIZE + ty;
    int col_o = blockIdx.x * TILE_SIZE + tx;

    int  n = MASK_WIDTH/2;
    int row_i = row_o - n;
    int col_i = col_o - n;


    float output = 0.0f;
    // ghost element condition
    if((row_i >= 0) && (row_i < N.height) && 
            (col_i >= 0)  && (col_i < N.width) ) {
        Ns[ty][tx] = N.elements[row_i*N.width + col_i];
    }    else{
        Ns[ty][tx] = 0.0f;
    }
    __syncthreads();

    if(ty < TILE_SIZE && tx < TILE_SIZE){
        for(int i = 0; i < MASK_WIDTH; i++) {
            for(int j = 0; j < MASK_WIDTH; j++) {
                output += Mc[i][j] * Ns[i+ty][j+tx];
            }
        }
        __syncthreads();
        if(row_o < P.height && col_o < P.width)
            P.elements[row_o * P.width + col_o] = output;
    }

}

#if 0 
    if(tx<N.width && ty<N.height)
    {
        float value=0.0f;
        int i=(MID-tx)>0?(MID-tx):0;
        int i_end=((N.width-tx)<MID)?(MID+(N.width-tx)):KERNEL_SIZE;

        int j=(MID-ty)>0?(MID-ty):0;
        int j_end=((N.height-ty)<MID)?(MID+(N.height-ty)):KERNEL_SIZE;
        for(; i<i_end;i++)
            for(; j<j_end;j++)
                //if((tx-i-2)>=0 && (tx-i-2)<N.width && (ty-j-2>=0) &&
                 //       (ty-j-2)<N.height)
                    value+=
                        N.elements[(tx-i-MID)*N.width+(ty-j-MID)]*M.elements[i*KERNEL_SIZE+j];

        P.elements[tx*P.width+ty]=value;
    }

#endif
#endif // #ifndef _2DCONVOLUTION_KERNEL_H_
