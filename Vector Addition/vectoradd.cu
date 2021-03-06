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

/* Vector addition: C = A + B.
 * Host code.
 */

// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

// includes, project
#include <cutil.h>

// includes, kernels
#include <vectoradd_kernel.cu>

////////////////////////////////////////////////////////////////////////////////
// declarations, forward

extern "C"
void computeGold(float*, const float*, const float*, unsigned int);

Vector AllocateDeviceVector(const Vector V);
Vector AllocateVector(int size, int init);
void CopyToDeviceVector(Vector Vdevice, const Vector Vhost);
void CopyFromDeviceVector(Vector Vhost, const Vector Vdevice);
int ReadFile(Vector* V, char* file_name);
void WriteFile(Vector V, char* file_name);

void VectorAddOnDevice(const Vector A, const Vector B, Vector C);

void PrintVector(const float *A, unsigned int size)
{
	unsigned int i=0;
	for(;i<size;i++)
		printf("%f  %s", A[i], (((i+1)%8)==0 )?"\n":"");
}
void PrintAddition(const float *A, const float *B, const float *C,unsigned int size)
{
	unsigned int i=0;
	for(;i<size;i++)
		printf("%4.4f+%4.4f=%04.04f || %s", A[i], B[i], C[i], (((i+1)%8)==0 )?"\n":"");
}

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char** argv) {

	// Vectors for the program
	Vector A;
	Vector B;
	Vector C;
	// Number of elements in the vectors
	unsigned int size_elements = VSIZE;
	int errorA = 0, errorB = 0;
	
	srand(2012);
	
	// Check command line for input vector files
	if(argc != 3 && argc != 4) 
	{
		// No inputs provided
		// Allocate and initialize the vectors
		A  = AllocateVector(VSIZE, 1);
		B  = AllocateVector(VSIZE, 1);
		C  = AllocateVector(VSIZE, 0);
	}
	else
	{
		// Inputs provided
		// Allocate and read source vectors from disk
		A  = AllocateVector(VSIZE, 0);
		B  = AllocateVector(VSIZE, 0);		
		C  = AllocateVector(VSIZE, 0);
		errorA = ReadFile(&A, argv[1]);
		errorB = ReadFile(&B, argv[2]);
		// check for read errors
		if(errorA != size_elements || errorB != size_elements)
		{
			printf("Error reading input files %d, %d\n", errorA, errorB);
			return 1;
		}
	}

	// A + B on the device
    VectorAddOnDevice(A, B, C);
    
    // compute the vector addition on the CPU for comparison
    Vector reference = AllocateVector(VSIZE, 0);
    computeGold(reference.elements, A.elements, B.elements, VSIZE);
        
    // check if the device result is equivalent to the expected solution
    CUTBoolean res = cutComparefe(reference.elements, C.elements, 
									size_elements, 0.0001f);
    printf("Test %s\n", (1 == res) ? "PASSED" : "FAILED");
    
    // output result if output file is requested
    if(argc == 4)
    {
		WriteFile(C, argv[3]);
	}
	else if(argc == 2)
	{
	    WriteFile(C, argv[1]);
	}    

	// Free host matrices
    free(A.elements);
    A.elements = NULL;
    free(B.elements);
    B.elements = NULL;
    free(C.elements);
    C.elements = NULL;
	return 0;
}


////////////////////////////////////////////////////////////////////////////////
//! Run a simple test for CUDA
////////////////////////////////////////////////////////////////////////////////
void VectorAddOnDevice(const Vector A, const Vector B, Vector C)
{
	//Interface host call to the device kernel code and invoke the kernel
 	Vector A_d=AllocateDeviceVector(A);
 	Vector B_d=AllocateDeviceVector(B);
 	Vector C_d=AllocateDeviceVector(C);

	CopyToDeviceVector(A_d, A);
	CopyToDeviceVector(B_d, B);

	dim3 threads= dim3((VSIZE/256),1,1);
	dim3 blocks= dim3(256,1,1);


	VectorAddKernel<<<blocks,threads>>>(A_d,B_d,C_d);

	cudaDeviceSynchronize();

	CopyFromDeviceVector(C, C_d);
	PrintAddition(A.elements, B.elements, C.elements, VSIZE);

	cudaFree(A_d.elements);
	cudaFree(B_d.elements);
	cudaFree(C_d.elements);
	
}

// Allocate a device vector of same size as V.
Vector AllocateDeviceVector(const Vector V)
{
    Vector Vdevice = V;
    int size = V.length * sizeof(float);
    cudaMalloc((void**)&Vdevice.elements, size);
    return Vdevice;
}

// Allocate a vector of dimensions length
//	If init == 0, initialize to all zeroes.  
//	If init == 1, perform random initialization.
Vector AllocateVector(int length, int init)
{
    Vector V;
    V.length = length;
    V.elements = NULL;
		
	V.elements = (float*) malloc(length*sizeof(float));

	for(unsigned int i = 0; i < V.length; i++)
	{
		V.elements[i] = (init == 0) ? (0.0f) : (rand() / (float)RAND_MAX);
	}
    return V;
}	

// Copy a host vector to a device vector.
void CopyToDeviceVector(Vector Vdevice, const Vector Vhost)
{
    int size = Vhost.length * sizeof(float);
    Vdevice.length = Vhost.length;
    cudaMemcpy(Vdevice.elements, Vhost.elements, size, 
					cudaMemcpyHostToDevice);
}

// Copy a device vector to a host vector.
void CopyFromDeviceVector(Vector Vhost, const Vector Vdevice)
{
    int size = Vdevice.length * sizeof(float);
    cudaMemcpy(Vhost.elements, Vdevice.elements, size, 
					cudaMemcpyDeviceToHost);
}

// Read a floating point vector in from file
int ReadFile(Vector* V, char* file_name)
{
	unsigned int data_read = VSIZE;
	cutReadFilef(file_name, &(V->elements), &data_read, true);
	return data_read;
}

// Write a floating point vector to file
void WriteFile(Vector V, char* file_name)
{
    cutWriteFilef(file_name, V.elements, V.length,
                       0.0001f);
}
