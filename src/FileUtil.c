#include "FileUtil.h"
#include <string.h>

long sizeArray(char *fname, int *numVars, int *numSamples) {
  /* Size up the array */
  long fLength;
  char *fileBuf;
  char *curLine, *firstLine, *curSep;
  int numLines, numFields;
  gzFile *gzInput;
  int bytes = 0;
  char *buffer;
	
  gzInput = gzopen(fname, "rb");
  if (gzInput == NULL) {
    fprintf(stderr, "Cannot open %s\n", fname);
    exit(0);
  }
  fLength = 0;
  buffer = (char*) calloc(BLOCK_SIZE, sizeof(char));
  while ((bytes = gzread(gzInput, buffer, BLOCK_SIZE))) {
    fLength += bytes;
  }
  free(buffer);
  gzrewind(gzInput);
	
  fileBuf = (char*) calloc(fLength + 1, sizeof(char));
	
  if(fileBuf == NULL )
    {
      fprintf(stderr, "Cannot allocate a buffer of size %ld bytes for file reading.\n", fLength + 1);
      exit(0);
    }
  gzread(gzInput, fileBuf, fLength);
  gzclose(gzInput);
	
  firstLine = fileBuf;
  curLine = strtok(fileBuf, NEW_LINE);
	
  numLines = 0;
  while (curLine != NULL) {
    curLine = strtok(NULL, NEW_LINE);
    ++numLines;
  }
	
  /* parse the SEPARATOR-delimited line */
  curSep = strtok(firstLine, SEPARATOR);
  numFields = 0;
  while (curSep != NULL) {
    curSep = strtok(NULL, SEPARATOR);
    ++numFields;
  }
	
  free(fileBuf);

  *numVars = numLines;
  *numSamples = numFields;
	
  return fLength;
}

void ReadFile(char *fname, double **Xptr, int *numVars, int *numSamples, char *colNames, char *rowNames) {
  long  fLength; /* File length */
  char *fileBuf; /* File buffer */
  char *curLine, *curSep, *tempNames; /* auxiliary variables */
  char **lines;
  int numLines;
  gzFile *input;
  int l, f;
  double *X;

  fLength = sizeArray(fname, numVars, numSamples);
  if (fLength == 0) {
    fprintf(stderr, "Couldn't allocate memory at some point!  Bailing out!\n");
    exit(0);
  }

  input = gzopen(fname, "rb");
	
  fileBuf = (char*) calloc(fLength + 1, sizeof(char));
  X = (double*) calloc((*numVars - 1)*(*numSamples - 1), sizeof(double));
  lines = (char**) calloc((*numVars), sizeof(char*));
  tempNames = (char*) calloc((*numVars - 1), sizeof(char*));

  if(fileBuf == NULL || X == NULL || lines == NULL)
    {
      fprintf(stderr, "Cannot allocate a buffer of size %ld bytes for file reading.\n", fLength + 1);
      return;
    }

  gzread(input, fileBuf, fLength); /* Read the entire file into fileBuf */
  gzclose(input);

  /* how many lines? */
  curLine = strtok(fileBuf, NEW_LINE);
  colNames = curLine;
  numLines = 0;
  while (curLine != NULL) {
    lines[numLines] = curLine;
    curLine = strtok(NULL, NEW_LINE);
    ++numLines;
  }

  /* assume the first line is the column header */

  f = 0;

  for (l = 1; l < numLines; ++l) {
    /* parse the SEPARATOR-delimited line */
    curSep = strtok(lines[l], SEPARATOR);
    tempNames[l] = *curSep;
    curSep = strtok(NULL, SEPARATOR);
    while (curSep != NULL) {
      X[f] = atof(curSep);
      /*      printf("%s\t%f\t%d\t%f\t%f\n", curSep, atof(curSep), f, X[f], X[100]);*/
      curSep = strtok(NULL, SEPARATOR);
      ++f;
    }
  }
  (*Xptr) = X;
  rowNames = tempNames;
  printf("%ld\n", sizeof(rowNames));
  free(lines);
  free(fileBuf);
}


