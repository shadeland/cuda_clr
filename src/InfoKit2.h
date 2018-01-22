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

float simpson(float, float, int, float(*) (float, float*), float *);
float normCdf(float, float, float);
float normPdf (float, float *);
float zToP(float, int);
float fdr(const float *, int, float, int);
void clrGauss(float*, float*, int);
float log2f(float);
double log2d(double);
int compare_floats (const float *, const float *);
int compare_doubles (const double *, const double *);
double iqr(double *, int);
void clrUnweightedStouffer(float*, float*, int);
double binWidth(double *, int);
int* calcNumBins(double *, int, int, double);
double max(const double*, int);
double min(const double*, int);
int maxi(const int*, int);
int mini(const int*, int);
void xToZ(const double*, double*, int, int, int, double, double);
void miSubMatrix(const double*, float*, int, int, int, int, int, int);
double mediani(int *, int);
double mean(double*, int);
// double std(double*,int);
double meani(int*, int);
double stdi(int*,int);
/* void SplineKnots(int*,int,int); */
void knotVector(double*, int, int);
void kldSubMarix(const double*, const double*, float*, int, int, int, int, int, int, int);
void findWeights(const double *, const double *, double *, int, int, int, double, double);
