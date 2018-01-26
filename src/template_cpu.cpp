/*
 * Copyright 1993-2015 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */
#include <stdlib.h>
#include <stdio.h>

#include "template_cpu.h"

////////////////////////////////////////////////////////////////////////////////
// export C interface

extern "C"
void computeGold(float *reference, float *idata, const unsigned int len);


////////////////////////////////////////////////////////////////////////////////
//! Compute reference data set
//! Each element is multiplied with the number of threads / array length
//! @param reference  reference data, computed but preallocated
//! @param idata      input data as provided to device
//! @param len        number of elements in reference / idata
////////////////////////////////////////////////////////////////////////////////
void
computeGold(float *reference, float *idata, const unsigned int len)
{
    const float f_len = static_cast<float>(len);

    for (unsigned int i = 0; i < len; ++i)
    {
        reference[i] = idata[i] * f_len;
    }
}


void randomI(float *data, int size)
{
    for (int i = 0; i < size; ++i)
        data[i] = rand()%100;
//        data[i] = 1;

}

void printMat(float *X, char *name, int numVars, int numSamples)
{
	printf("\n\n///////////// %s /////////////\n", name);

	for (int curSample = 0; curSample < numSamples; ++curSample)
	{
		for (int curVar = 0; curVar < numVars; ++curVar)
		{
			// printf("%d",(curVar*numSamples+curSample));
			printf(" %.2f", X[curVar * numSamples + curSample]);
		}
		printf("\n");
	}
}

void printMatf(float *X, char *name, int numVars, int numSamples)
{
	printf("\n\n///////////// %s /////////////\n", name);

	for (int curSample = 0; curSample < numSamples; ++curSample)
	{
		for (int curVar = 0; curVar < numVars; ++curVar)
		{
			// printf("%d",(curVar*numSamples+curSample));
			printf(" %.5f", X[curVar * numSamples + curSample]);
		}
		printf("\n");
	}
}

void fprintMat(FILE *fp, float *X, char *name, int numVars, int numSamples)
{
	fprintf(fp, "\n\n///////////// %s /////////////\n", name);

	for (int curSample = 0; curSample < numSamples; ++curSample)
	{
		for (int curVar = 0; curVar < numVars; ++curVar)
		{
			// printf("%d",(curVar*numSamples+curSample));
			fprintf(fp, " %.2f", X[curVar * numSamples + curSample]);
		}
		fprintf(fp, "\n");
	}
}

void fprintMatf(FILE *fp, float *X, char *name, int numVars, int numSamples)
{
	fprintf(fp, "\n\n///////////// %s /////////////\n", name);

	for (int curSample = 0; curSample < numSamples; ++curSample)
	{
		for (int curVar = 0; curVar < numVars; ++curVar)
		{
			// printf("%d",(curVar*numSamples+curSample));
			fprintf(fp, " %.5f", X[curVar * numSamples + curSample]);
		}
		fprintf(fp, "\n");
	}
}


