/*
 * template_cpu.h
 *
 *  Created on: Jan 20, 2018
 *      Author: adel
 */

#ifndef TEMPLATE_CPU_H_
#define TEMPLATE_CPU_H_
// #define RAND_MAX 10

void randomI(float *data, int size);
void fprintMat(FILE *fp, float *X, char *name, int numVars, int numSamples);
void printMat(float *X, char *name, int numVars, int numSamples);
void fprintMatf(FILE *fp, float *X, char *name, int numVars, int numSamples);
void printMatf(float *X, char *name, int numVars, int numSamples);


#endif /* TEMPLATE_CPU_H_ */
