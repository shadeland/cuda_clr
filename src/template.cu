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
#define CNUMVARS 128	// powers of two for now
#define CNUMMI NUMVARS *NUMVARS
#define CNUMBINS 25
#define CBATCHSIZE 128 //128*128 batch size for hist mem management
#define CTPBX 16	   //threads per block dim 16*16
#define CTOTAL NUMSAMPLES *NUMVARS *NUMBINS

int NUMSAMPLES;
int NUMVARS;
int NUMMI;
int NUMBINS;
int BATCHSIZE;
int TPBX;
int TOTAL;
int SPLINEORDER;
int V;
FILE *fp;

////////////////////////////////////////////////////////////////////////////////
//
////////////////////////////////////////////////////////////////////////////////

__device__ float scale(int i, int n)
{
	return ((float)i) / (n - 1);
}

__device__ float distance(float x1, float x2)
{
	return sqrt((x2 - x1) * (x2 - x1));
}

//this just uses 1dim blocks
__global__ void histo2dGlobal(float *d_out, double *d_w, float *d_hist2d,
							  dim3 curBatch, int numBins, int numSamples, int BATCHSIZE, int NUMVARS)
{

	const int totalThreads = gridDim.x * blockDim.x; // whic is actually numvars*numvars;

	const int curMiX = blockIdx.x * blockDim.x + threadIdx.x;
	const int curMiY = blockIdx.y * blockDim.y + threadIdx.y;
	const int globalMiX = BATCHSIZE * curBatch.x + curMiX; //global MI
	const int globalMiY = BATCHSIZE * curBatch.y + curMiY;
	printf("%d x %d y \n", globalMiX, globalMiY);
	printf("%d batch.x %d batch.y \n", curBatch.x, curBatch.y);
	int histSize = numBins * numBins;

	double temp = 0;
	int curVarXWeightStart = globalMiX * numSamples * numBins;
	int curVarYWeightStart = globalMiY * numSamples * numBins;

	int curHistStart = (BATCHSIZE * curMiY + curMiX) * numBins * numBins;

	for (int curBinX = 0; curBinX < numBins; ++curBinX)
	{
		for (int curBinY = 0; curBinY < numBins; ++curBinY)
		{
			for (int curSample = 0; curSample < numSamples; ++curSample)
			{
				temp += (d_w[curVarXWeightStart + (curBinX * numBins) + curSample] * d_w[curVarYWeightStart + (curBinY * numBins) + curSample]) / numSamples;
				printf("%d bx, %d by, %d s, %d mx, %d my, %0.2f wx, %0.2f wy, %0.2f temp \n",curBinX,curBinY,curSample,globalMiX,globalMiY, d_w[curVarXWeightStart + (curBinX * numBins) + curSample] ,
				 d_w[curVarYWeightStart + (curBinY * numBins) + curSample], temp);
			}
			d_hist2d[curHistStart + (curBinX * numBins) + curBinY] = temp;
			printf("%0.2f h2d \n",  d_hist2d[curHistStart + curBinX * numBins + curBinY]);
			temp = 0;
		}
	}

	//Calc entropy on h2d
	double incr = 0;
	double H2D = 0;
	for (int curBinX = 0; curBinX < numBins; ++curBinX)
	{
		for (int curBinY = 0; curBinY < numBins; ++curBinY)
		{
			incr = (double) d_hist2d[curHistStart + (curBinX * numBins) + curBinY];
			printf("%0.2f incr \n",  d_hist2d[curHistStart + (curBinX * numBins) + curBinY]);
			if (incr > 0)
			{
				H2D -= incr * log2(incr); //calc entropy of current MI
			}
		}
	}

	// __syncthreads();
	d_out[(NUMVARS * globalMiX) + globalMiY] = (float) H2D;
	printf("%d OUT %0.2f H2D",NUMVARS * globalMiX + globalMiY, H2D);
}

//////////// for benchmarking with CPU use template source
////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////

void genWeights(double *w, int numSamples, int numVars, int numBins)
{

	randomI(w, numSamples * numVars * numBins);
}

// TO RUN EACH BATCH
void runBatch(int batchX, int batchY, double *d_w, float *d_out, float *d_hist2d, int numVars)
{
	int startVarX = batchX * BATCHSIZE; //for memory indexing
	int startVarY = batchY * BATCHSIZE;
	int endVarX = startVarX + BATCHSIZE;
	int endVarY = startVarY + BATCHSIZE;

	StopWatchInterface *timer = 0; // This can be shared
	sdkCreateTimer(&timer);

	dim3 curBatch(batchX, batchY);

	dim3 threadsPerBlock(TPBX, TPBX);
	dim3 blocksPerGrid((BATCHSIZE + TPBX - 1) / TPBX, (BATCHSIZE + TPBX - 1) / TPBX);

	printf("Start Runing batch (%d,%d)  \n %d Samples \n %d vars \n %d bins \n %dX%d blocksize \n\n", batchX, batchY, NUMSAMPLES, numVars, NUMBINS, TPBX, TPBX);
	sdkStartTimer(&timer);
	histo2dGlobal<<<blocksPerGrid, threadsPerBlock>>>(d_out, d_w, d_hist2d, curBatch, NUMBINS, NUMSAMPLES, BATCHSIZE, NUMVARS);
	cudaDeviceSynchronize();
	sdkStopTimer(&timer);
	printf("Processing time GPU for batch: %f (ms)\n", sdkGetTimerValue(&timer));
}


void clac_numbins_entropies_wights(double *data, float *entropies, double *w)
{

	double *knots = (double *)calloc(NUMBINS + SPLINEORDER, sizeof(double));
	const double *hist1 = (double *)calloc(NUMBINS, sizeof(double));
	double *e2d = (double *)calloc(NUMVARS *NUMVARS, sizeof(double));
	////CALC KNOTS
	knotVector(knots, NUMBINS, SPLINEORDER);

	const double *knotsC = knots;

	for (int i = 0; i < NUMVARS; i++)
	{
		findWeights(data + (i * NUMSAMPLES), knotsC, w + i * NUMSAMPLES * NUMBINS, NUMSAMPLES, SPLINEORDER, NUMBINS, -1, -1);
		entropies[i] = entropy1d(data +i*NUMSAMPLES, knotsC, w + i * NUMSAMPLES * NUMBINS, NUMSAMPLES, SPLINEORDER, NUMBINS);

		for(int j=0; j<NUMVARS; j++){
			e2d[i*NUMVARS+j] = entropy2d(data + (i*NUMSAMPLES), data + (j*NUMSAMPLES), knotsC, w + i*NUMSAMPLES * NUMBINS, w + j*NUMSAMPLES * NUMBINS, NUMSAMPLES, SPLINEORDER, NUMBINS);
		}
	}
	fprintMat(fp,e2d, "ENTROPY 2D", NUMVARS, NUMVARS);
}




int main(int argc, char **argv)
{
	if (argc < 2)
	{
		printf("usage: template <numSamples> <numVars> <numBins> <batchSize> <threadperblock> <VEROBOSE=0||1>");
		return 1;
	}
	NUMSAMPLES = atoi(argv[1]); //
	NUMVARS = atoi(argv[2]);	// powers of two for now
	NUMMI = NUMVARS * NUMVARS;
	NUMBINS = atoi(argv[3]);
	BATCHSIZE = atoi(argv[4]); //128*128 batch size for hist mem management
	TPBX = atoi(argv[5]);	  //threads per block dim 16*16
	V = atoi(argv[6]);
	SPLINEORDER = 3;
	TOTAL = NUMSAMPLES * NUMVARS * NUMBINS;
	

	if (V == 1)
		fp = fopen("log", "w+");

	// Declare a pointer for an array of floats
	float *h_out = 0;
	float *d_out = 0;
	double *h_w = 0;
	double *d_w = 0;
	float *h_entrop1d = 0;
	float *d_entrop1d = 0;
	double *h_data = 0;
	float *d_hist2d = 0;

	// setup a time to calc the time
	StopWatchInterface *timer = 0;
	sdkCreateTimer(&timer);

	// Allocate device memory to store the output array with size number  samples
	// 1d for now
	cudaMalloc(&d_out, NUMMI * sizeof(float));
	cudaMalloc(&d_w, TOTAL * sizeof(double));
	cudaMalloc(&d_hist2d, NUMBINS * NUMBINS * BATCHSIZE * BATCHSIZE * sizeof(float));

	h_out = (float *)calloc(NUMMI, sizeof(float));
	h_w = (double *)calloc(TOTAL, sizeof(double)); // host mem for weights /// why double ? ???
	h_data = (double *)calloc(NUMVARS * NUMSAMPLES, sizeof(double));
	h_entrop1d = (float *)calloc(NUMVARS, sizeof(float));

	// gen random data
	genWeights(h_data, NUMSAMPLES, NUMVARS, 1);

	if (V == 1)
		fprintMat(fp, h_data, "DATA MAT", NUMVARS, NUMSAMPLES);

	clac_numbins_entropies_wights(h_data, h_entrop1d, h_w);

	if (V == 1)
		fprintMatf(fp, h_entrop1d, "ENTROPY1 MAT", NUMVARS, 1);

	if (V == 1)
		fprintMat(fp, h_w, "WEIGHT MAT", NUMVARS, NUMSAMPLES * NUMBINS);

	//copy w to dev
	cudaMemcpy(d_w, h_w, TOTAL * sizeof(double), cudaMemcpyHostToDevice);

	//config kernel

	// runing batches

	int numBatches = (NUMVARS + BATCHSIZE - 1) / BATCHSIZE;
	sdkStartTimer(&timer);
	for (int curBatchX = 0; curBatchX < numBatches; ++curBatchX)
	{

		for (int curBatchY = 0; curBatchY < numBatches; ++curBatchY)
		{
			runBatch(curBatchX, curBatchY, d_w, d_out, d_hist2d, BATCHSIZE);
		}
	}
	sdkStopTimer(&timer);
	cudaMemcpy(h_out, d_out, NUMMI * sizeof(float), cudaMemcpyDeviceToHost);
	if (V == 1)
		fprintMatf(fp, h_out, "ENTROPY2 MAT", NUMVARS, NUMVARS);

	// Launch kernel to compute and store distance values
	// printf("Start Runing \n %d Samples \n %d vars \n %d bins \n %dX%d blocksize \n\n",NUMSAMPLES , NUMVARS , NUMBINS , TPBX,TPBX);
	// sdkStartTimer(&timer);
	// histo2dGlobal<<<blocksPerGrid, threadsPerBlock>>>(d_out, d_w, d_hist2d, NUMBINS, NUMSAMPLES);
	// cudaDeviceSynchronize();
	// sdkStopTimer(&timer);
	printf("Finished Runing  \n %d Samples \n %d vars \n %d bins \n %dX%d blocksize , %d batches \n\n",
		   NUMSAMPLES, NUMVARS, NUMBINS, TPBX, TPBX, BATCHSIZE);
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
