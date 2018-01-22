#include "InfoKit2.h"
#include "FileUtil.h"
#include <getopt.h>
#include <string.h>
#include <stdio.h>
#include <fcntl.h>

#include <time.h>

#define SEPARATOR "\t"

void printMat(double *X, int numVars, int numSamples)
{

    for (int curVar = 0; curVar < numVars; ++curVar)
    {

        for (int curSample = 0; curSample < numSamples; ++curSample)
        {
            // printf("%d",(curVar*numSamples+curSample));
            printf(" %.2f", X[curVar * numSamples + curSample]);
        }
        printf("\n");
    }
}

void printMatf(float *X, int numVars, int numSamples)
{

    for (int curVar = 0; curVar < numVars; ++curVar)
    {

        for (int curSample = 0; curSample < numSamples; ++curSample)
        {
            // printf("%d",(curVar*numSamples+curSample));
            printf(" %.5f", X[curVar * numSamples + curSample]);
        }
        printf("\n");
    }
}

int main()
{

    ////READ IN THE DATA

    int numVars = 5;
    int numSamples = 400;

    char *fileBuf, **lines, **linesTf, *headSep, *curSep, *curLine, *curLineTf, **rowNames;
    int i, j, f, l, placeholder, *binCount, numTfs, *tfIdx;
    long fLength, fLengthTf;
    gzFile *input, *inputTf;
    double *X, stdBinCount;
    float *C, *S;
    FILE *out;

    /************************************************************************************************
   read in expression file */

    /* returns the length of the file.  This is needed to initiate the memory for the filebuffer.
     Also returns the number of samples (conditions) and the number of variables (genes)*/
    fLength = sizeArray("foo.csv", &numSamples, &numVars);
    //   numSamples--; /* subtract 1 to account for the first row in the file which is the row of gene id */
    printf("There are %d genes and %d experimental conditions in file %s\n", numVars, numSamples, "foo.csv");

    /* initiate the fileBuffer and read the entire file in to the fileBuffer*/
    input = gzopen("foo.csv", "rb");
    fileBuf = (char *)calloc(fLength + 1, sizeof(char));
    /* BUG FIXED: X and lines are uninitialized */
    if (fileBuf == NULL) /* || X == NULL || lines == NULL) */
    {
        fprintf(stderr, "Cannot allocate a buffer of size %ld bytes for file reading.\n", fLength + 1);
        return -1;
    }
    gzread(input, fileBuf, fLength); /* Read the entire file into fileBuf */
    gzclose(input);

    /* the fileBuffer is a pointer, so to get at each line, you have to tokenize and create a pointer to a pointer with lines
     The + 1 is to take account for the first row. */
    lines = (char **)calloc((numSamples), sizeof(char *));

    /* I was trying the same idea as with lines to create a pointer to the pointer containing the gene names */
    rowNames = (char **)calloc(numVars, sizeof(char *));

    /* break the fileBuffer up by new lines so "lines" now contains each individual row in the file */
    curLine = strtok(fileBuf, NEW_LINE);
    f = 0;
    while (curLine != NULL)
    {
        lines[f] = curLine;
        curLine = strtok(NULL, NEW_LINE);
        ++f;
    }

    /* parse the values and assingn them to X, rember that samples are rows and genes are columns */
    X = (double *)calloc((numVars) * (numSamples), sizeof(double));
    for (l = 0; l < (numSamples); ++l)
    {
        f = l;
        // printf("%s\n",lines[l]);
        curSep = strtok(lines[l], ",");
        /* for each line, break apart the line on the tabs */
        while (curSep != NULL)
        {
            //   --f;
            //   printf("%d: %0.2f\n",f, atof(curSep));
            X[f] = atof(curSep); /* since one column is one gene, have to do a little math to get the indexes correct*/
            f += numSamples;
            curSep = strtok(NULL, ",");
        }
    }

    printf("CLR TEST %0.f\n", X[2]);
    // double *X;
    float *MI;

    // X = calloc(numVars*numSamples, sizeof(double));

    // for(int i=0; i<numVars*numSamples; ++i){

    //     X[i] = rand();

    // }
    // printf("%.2f", X[2]);
    printMat(X, numVars, numSamples);

    MI = (float *)calloc(numVars * numVars, sizeof(float));
    time_t start = clock();
    miSubMatrix(X, MI, 25, numVars, numSamples, 3, 0, 1);
    time_t end = clock();
    printMatf(MI, numVars, numVars);
    printf("\n time elapsed: %.04f", ((double)(end - start)) / CLOCKS_PER_SEC);

    return 0;
}