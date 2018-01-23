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
//this change
// includes CUDA
#include <cuda_runtime.h>

// includes, project
// helper functions for SDK examples
#include <helper_functions.h>
#include <helper_cuda.h>

#include "template_cpu.h"
#include "InfoKit2.h"
////////////////////////////////////////////////////////////////////////////////
// declaration, forward
#define CNUMSAMPLES 400 //
#define CNUMVARS 128// powers of two for now
#define CNUMMI NUMVARS*NUMVARS
#define CNUMBINS 25
#define CBATCHSIZE 128 //128*128 batch size for hist mem management
#define CTPBX 16//threads per block dim 16*16
#define CTOTAL NUMSAMPLES*NUMVARS*NUMBINS

 int NUMSAMPLES;
 int NUMVARS;
 int NUMMI ;
 int NUMBINS;
 int BATCHSIZE;
 int TPBX ;
 int TOTAL ;
 int SPLINEORDER;

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
__global__ void histo2dGlobal(float *d_out, double *d_w, float *d_hist2d,
	dim3 curBatch ,int numBins, int numSamples, int BATCHSIZE, int NUMVARS) {
	
	const int totalThreads = gridDim.x*blockDim.x;  // whic is actually numvars*numvars;

	const int curMiX = blockIdx.x * blockDim.x + threadIdx.x;
	const int curMiY = blockIdx.y * blockDim.y + threadIdx.y;
	const int globalMiX  = BATCHSIZE*curBatch.x+curMiX; //global MI
	const int globalMiY = BATCHSIZE*curBatch.y+curMiY;;
	int histSize = numBins*numBins;
	

	float temp = 0;
	int curVarXWeightStart = globalMiX*numSamples*numBins;
	int curVarYWeightStart = globalMiY*numSamples*numBins;

	int curHistStart = (BATCHSIZE*curMiY+curMiX)*numBins*numBins;

	for (int curBinX = 0; curBinX < numBins; ++curBinX) {
		for (int curBinY = 0; curBinY < numBins; ++curBinY) {
			for (int curSample = 0; curSample < numSamples; ++curSample) {
				temp += (float) (d_w[curVarXWeightStart+curBinY*numBins+curSample]*d_w[curVarYWeightStart+curBinY*numBins+curSample])/numSamples;
			}
			d_hist2d[curHistStart+curBinX*numBins+curBinY]= temp; 
		}
	}

	//Calc entropy on h2d
	float incr = 0;
	float H2D = 0;
	for (int curBinX = 0; curBinX < numBins; ++curBinX) {
			for (int curBinY = 0; curBinY < numBins; ++curBinY) {
				incr = d_hist2d[curHistStart+curBinX*numBins+curBinY];
				if(incr > 0){
					H2D -= incr*log2f(incr); //calc entropy of current MI
				}
			}
	}


	

	
	// __syncthreads();
	d_out[NUMVARS*globalMiX+globalMiY] = H2D;

}

//////////// for benchmarking with CPU use template source
////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////



void genWeights(float *w, int numSamples, int numVars, int numBins){

	randomI(w, numSamples*numVars*numBins);

}

// TO RUN EACH BATCH 
void runBatch(int batchX, int batchY, float *d_w, float *d_out, float *d_hist2d, int numVars){
	int startVarX = batchX*BATCHSIZE; //for memory indexing
	int startVarY = batchY*BATCHSIZE; 
	int endVarX = startVarX+BATCHSIZE;
	int endVarY = startVarY+BATCHSIZE;

	StopWatchInterface *timer = 0;// This can be shared
	sdkCreateTimer(&timer);

	dim3 curBatch(batchX, batchY);

	dim3 threadsPerBlock(TPBX, TPBX);
	dim3 blocksPerGrid((BATCHSIZE+TPBX-1)/TPBX, (BATCHSIZE+TPBX-1)/TPBX);

	printf("Start Runing batch (%d,%d)  \n %d Samples \n %d vars \n %d bins \n %dX%d blocksize \n\n"
	,batchX, batchY, NUMSAMPLES , numVars , NUMBINS , TPBX,TPBX);
	sdkStartTimer(&timer);
	histo2dGlobal<<<blocksPerGrid, threadsPerBlock>>>(d_out, d_w, d_hist2d, curBatch,  NUMBINS, NUMSAMPLES, BATCHSIZE, NUMVARS);
	cudaDeviceSynchronize();
	sdkStopTimer(&timer);
	printf("Processing time GPU for batch: %f (ms)\n", sdkGetTimerValue(&timer));
	

	

}

void clac_numbins_entropies_wights(double data, float *entropies, double *w){


	double *knots = (double*) calloc(numBins + splineOrder, sizeof(double));
	double *hist1 = (double*) calloc(numBins, sizeof(double));

	////CALC KNOTS
	knotVector(knots, NUMBINS, SPLINEORDER);


	for(i=0; i<NUMVARS; i++){
		findWeights(data+(i*NUMSAMPLES), knots, w+i*NUMSAMPLES*NUMBINS, NUMSAMPLES, SPLINEORDER, NUMBINS, -1, -1);
	}

}

int main(int argc, char **argv) {
 NUMSAMPLES = atoi(argv[1]); //
  NUMVARS = atoi(argv[2]);// powers of two for now
 NUMMI = NUMVARS*NUMVARS;
 NUMBINS = atoi(argv[3]);
 BATCHSIZE = atoi(argv[4]); //128*128 batch size for hist mem management
 TPBX = atoi(argv[5]);//threads per block dim 16*16
 SPLINEORDER = 3;
 TOTAL = NUMSAMPLES*NUMVARS*NUMBINS;


	// Declare a pointer for an array of floats
	float *h_out = 0;
	float *d_out = 0;
	float *h_w = 0;
	float *d_w = 0;
	float *h_entrop1d=0;
	float *d_entrop1d=0;
	double  *h_data = 0;
	float *d_hist2d =0 ;

	// setup a time to calc the time
	StopWatchInterface *timer = 0;
	sdkCreateTimer(&timer);


	// Allocate device memory to store the output array with size number  samples
	// 1d for now
	cudaMalloc(&d_out, NUMMI * sizeof(float));
	cudaMalloc(&d_w, TOTAL * sizeof(float));
	cudaMalloc(&d_hist2d, NUMBINS*NUMBINS*BATCHSIZE*BATCHSIZE*sizeof(float));

	h_out = (float*) calloc(NUMMI,sizeof(float));
	h_w  = (double *) calloc(TOTAL,sizeof(double)); // host mem for weights /// why double ? ???
	h_data = (double *) calloc(NUMVARS*NUMSAMPLES, sizeof(double));
	h_entrop1d = (float *) calloc (NUMVARS, sizeof(float));

	// gen random weights
	genWeights(h_w, NUMSAMPLES, NUMVARS, NUMBINS);

	//copy w to dev
	cudaMemcpy(d_w, h_w, TOTAL*sizeof(float), cudaMemcpyHostToDevice);

	//config kernel
	

	// runing batches 

	int numBatches = (NUMVARS+BATCHSIZE-1)/BATCHSIZE; 
	sdkStartTimer(&timer);
	for(int curBatchX = 0; curBatchX <numBatches; ++curBatchX){

		for(int curBatchY = 0; curBatchY <numBatches; ++curBatchY){
			runBatch(curBatchX, curBatchY, d_w, d_out, d_hist2d, BATCHSIZE);
		}

	}
	sdkStopTimer(&timer);
	cudaMemcpy(h_out, d_out, NUMMI*sizeof(float), cudaMemcpyDeviceToHost);


	// Launch kernel to compute and store distance values
	// printf("Start Runing \n %d Samples \n %d vars \n %d bins \n %dX%d blocksize \n\n",NUMSAMPLES , NUMVARS , NUMBINS , TPBX,TPBX);
	// sdkStartTimer(&timer);
	// histo2dGlobal<<<blocksPerGrid, threadsPerBlock>>>(d_out, d_w, d_hist2d, NUMBINS, NUMSAMPLES);
	// cudaDeviceSynchronize();
	// sdkStopTimer(&timer);
	printf("Finished Runing  \n %d Samples \n %d vars \n %d bins \n %dX%d blocksize , %d batches \n\n",
			NUMSAMPLES , NUMVARS	, NUMBINS , TPBX,TPBX, BATCHSIZE);
	printf("Processing Total Time GPU: %f (ms)\n", sdkGetTimerValue(&timer));

	// cudaMemcpy(h_out, d_out, NUMMI*sizeof(float), cudaMemcpyDeviceToHost);



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
