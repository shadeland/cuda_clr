#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <string.h>
#define EULER_MASCHERONI 0.577215664901532
#define INDEPENDENT 0
#define POSITIVE 1
#define NEGATIVE -1
#define NUM_INTEGRAL_PTS 100
#define PI 3.141592654


#ifdef __cplusplus
extern "C"
{
#endif

float simpson(float, float, int, float(*) (float, float*), float *);
float normCdf(float, float, float);
float normPdf (float, float *);
float zToP(float, int);
float fdr(const float *, int, float, int);
void clrGauss(float*, float*, int);
float log2f(float);
float log2d(float);
int compare_floats (const void *, const void *);
int compare_floats (const void *, const void *);
float iqr(float *, int);
void clrUnweightedStouffer(float*, float*, int);
float binWidth(float *, int);
int* calcNumBins(float *, int, int, float);
float maxd(const float*, int);
float mind(const float*, int);
int maxi(const int*, int);
int mini(const int*, int);
void xToZ(const float*, float*, int, int, int, float, float);
void miSubMatrix(const float*, float*, int, int, int, int, int, int);
float mediani(int *, int);
float *transpose(float *in, int numRows, int numCols);
float mean(float*, int);
float stdv(float*,int);
float meani(int*, int);
float stdi(int*,int);
/* void SplineKnots(int*,int,int); */
void knotVector(float*, int, int);
void kldSubMarix(const float*, const float*, float*, int, int, int, int, int, int, int);
void findWeights(const float *, const float *, float *, int, int, int, float, float);
float entropy1d(const float *, const float *, float *, int, int, int);
float entropy2d(const float *, const float *, const float *, const float *, const float*, int, int, int);

#ifdef __cplusplus
}
#endif