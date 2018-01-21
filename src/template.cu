////////////////////////////////////////////////////////////////////////////
//
// Copyright 1993-2015 NVIDIA Corporation.  All rights reserved.
//
// Please refer to the NVIDIA end user license agreement (EULA) associated
// with this source code for terms and conditions that govern your use of
// this software. Any use, reproduction, disclosure, or distribution of
// this software and related documentation outside the terms of the EULA
// is strictly prohibited.
//
////////////////////////////////////////////////////////////////////////////

/* Template project which demonstrates the basics on how to setup a project
 * example application.
 * Host code.
 */

// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

// includes CUDA
#include <cuda_runtime.h>

// includes, project
#include <helper_cuda.h>
#include <helper_functions.h> // helper functions for SDK examples


#include "template_cpu.h"
////////////////////////////////////////////////////////////////////////////////
// declaration, forward
#define NUMSAMPLES 400 //
#define NUMVARS 16000// powers of two for now
#define NUMMI NUMVARS*NUMVARS
#define NUMBINS 25
#define TPBX 32 //threads per block dim 16*16
#define TOTAL NUMSAMPLES*NUMVARS*NUMBINS


////////////////////////////////////////////////////////////////////////////////
//
////////////////////////////////////////////////////////////////////////////////

__device__ float scale(int i, int n) {
	return ((float) i) / (n - 1);
}

__device__ float distance(float x1, float x2) {
	return sqrt((x2 - x1) * (x2 - x1));
}

//this just uses 1dim blocks
__global__ void histo2dGlobal(float *d_out, float *d_w, int numBins,
		int numSamples) {
	const int totalThreads = gridDim.x*blockDim.x;  // whic is actually numvars*numvars;

	const int curMiX = blockIdx.x * blockDim.x + threadIdx.x;
	const int curMiY = blockIdx.y * blockDim.y + threadIdx.y;
	const int curVarX  = curMiX;
	const int curVarY = curMiY;
	int temp = 0;
	int curVarXWeightStart = curVarX*NUMSAMPLES*NUMBINS;
	int curVarYWeightStart = curVarY*NUMSAMPLES*NUMBINS;

	for (int curBinX = 0; curBinX < numBins; ++curBinX) {
		for (int curBinY = 0; curBinY < numBins; ++curBinY) {
			for (int curSample = 0; curSample < numSamples; ++curSample) {
				temp += d_w[curVarXWeightStart+curBinY*numBins+curSample]+d_w[curVarYWeightStart+curBinY*numBins+curSample];
			}
		}
	}

	d_out[NUMVARS*curMiY+curMiX] = temp;

}

//////////// for benchmarking with CPU use template source
////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////



void genWeights(float *w, int numSamples, int numVars, int numBins){

	randomI(w, numSamples*numVars*numBins);

}

int main() {
	const float ref = 0.5f;



	// Declare a pointer for an array of floats
	float *h_out = 0;
	float *d_out = 0;
	float *h_w = 0;
	float *d_w = 0;

	// setup a time to calc the time
	StopWatchInterface *timer = 0;
	sdkCreateTimer(&timer);


	// Allocate device memory to store the output array with size number  samples
	// 1d for now
	cudaMalloc(&d_out, NUMMI * sizeof(float));
	cudaMalloc(&d_w, TOTAL * sizeof(float));

	h_out = (float*) calloc(NUMMI,sizeof(float));
	h_w  = (float *) calloc(TOTAL,sizeof(float)); // host mem for weights


	// gen random weights
	genWeights(h_w, NUMSAMPLES, NUMVARS, NUMBINS);

	//copy w to dev
	cudaMemcpy(d_w, h_w, TOTAL*sizeof(float), cudaMemcpyHostToDevice);

	//config kernel
	dim3 threadsPerBlock(TPBX, TPBX);
	dim3 blocksPerGrid(NUMVARS/TPBX, NUMVARS/TPBX);



	// Launch kernel to compute and store distance values
	printf("Start Runing \n %d Samples \n %d vars \n %d bins \n %dX%d blocksize \n\n",NUMSAMPLES , NUMVARS , NUMBINS , TPBX,TPBX);
	sdkStartTimer(&timer);
	histo2dGlobal<<<blocksPerGrid, threadsPerBlock>>>(d_out,d_w, NUMBINS, NUMSAMPLES);
	cudaDeviceSynchronize();
	sdkStopTimer(&timer);
	printf("Processing time GPU: %f (ms)\n", sdkGetTimerValue(&timer));

	cudaMemcpy(h_out, d_out, NUMMI*sizeof(float), cudaMemcpyDeviceToHost);



//	sdkResetTimer(&timer);
//	sdkStartTimer(&timer);
//	distanceCpu();
//	sdkStopTimer(&timer);
//	printf("Processing time CPU: %f (ms)\n", sdkGetTimerValue(&timer));

	cudaFree(d_out); // Free the memory
	return 0;
}
////////////////////////////////////////////////////////////////////////////////
//! Run a simple test for CUDA
////////////////////////////////////////////////////////////////////////////////
