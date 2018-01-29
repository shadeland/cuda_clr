#include "InfoKit2.h"
#include "FileUtil.h"
#include <getopt.h>
#include <string.h>
#include <stdio.h>
#include <fcntl.h>

typedef struct
{
  char *dataFile;
  char *tfIdxFile;
  char *mapFile;
  int cut;
  int numBins;
  int splineDegree;
} ARGUMENTS;

ARGUMENTS parseOptions(int argc, char *argv[])
{
  int c;
  ARGUMENTS args;

  /* set defaults */
  args.dataFile = (char *)NULL;
  args.tfIdxFile = (char *)NULL;
  args.mapFile = (char *)NULL;
  args.cut = -1;
  args.numBins = -1;
  args.splineDegree = 3;

  while (1)
  {
    static struct option long_options[] =
        {
            /* These options don't set a flag.
	   We distinguish them by their indices. */
            {"data", required_argument, 0, 'd'},
            {"reg", required_argument, 0, 'r'},
            {"out", required_argument, 0, 'o'},
            {"cut", required_argument, 0, 'c'},
            {"bins", required_argument, 0, 'b'},
            {"spline", required_argument, 0, 's'}};
    /* getopt_long stores the option index here. */
    int option_index = 0;

    c = getopt_long_only(argc, argv, "d:r:o:c:b:s:",
                         long_options, &option_index);

    /* Detect the end of the options. */
    if (c == -1)
      break;

    switch (c)
    {
    case 'd':
      /* printf ("option -d with value `%s'\n", optarg); */
      args.dataFile = (char *)malloc((strlen(optarg) + 1) * sizeof(char));
      strcpy(args.dataFile, optarg);
      break;
    case 'r':
      /* printf ("option -r with value `%s'\n", optarg); */
      args.tfIdxFile = (char *)malloc((strlen(optarg) + 1) * sizeof(char));
      strcpy(args.tfIdxFile, optarg);
      break;
    case 'o':
      /* printf ("option -o with value `%s'\n", optarg); */
      args.mapFile = (char *)malloc((strlen(optarg) + 1) * sizeof(char));
      strcpy(args.mapFile, optarg);
      break;
    case 'c':
      /* printf ("option -c with value `%s'\n", optarg); */
      args.cut = atoi(optarg);
      break;
    case 'b':
      /* printf ("option -b with value `%s'\n", optarg); */
      args.numBins = atoi(optarg);
      break;
    case 's':
      /* printf ("option -s with value `%s'\n", optarg); */
      args.splineDegree = atoi(optarg);
      break;
    case '?':
      /*getopt_long already printed an error message. */
      printf("Usage: clr --data input_file_name [--reg regulator_file_name --cut cutoff --out output_file_name --bins num_bins --spline spline_degree]\n");
      exit(0);
      break;

    default:
      fprintf(stderr, "aborting!\n");
      exit(-1); /* Modules should return -1 if there was an error */
    }
  }
  /* Print any remaining command line arguments (not options). */
  if (optind < argc)
  {
    fprintf(stderr, "Unrecognized arguments: \n");
    while (optind < argc)
      fprintf(stderr, "%s ", argv[optind++]);
    putchar('\n');
  }

  if (args.dataFile == (char *)NULL)
  {
    printf("Usage: clr --data input_file_name [--reg regulator_file_name --cut cutoff --out output_file_name --bins num_bins --spline spline_degree]\n");
    exit(-1);
  }

  /* No, -1 should mean: print the complete list of edges */
  /*if (args.cut == -1)
    args.cut = 100000;*/

  /**
	 * Write output files to the working directory. Use the following convention for filenames.
	 *
	 * Use as output filename for the list of edge predictions:
	 *       <inputFilename>_<ModuleName>_predictions.txt
	 *
	 * If your module generates additional output files, use:
	 *       <inputFilename>_<ModuleName>_<label>.txt
	 *
	 * Example:
	 *       Input file:		./data/net1/net1_expression_data.tsv
	 *       Prediction file:	net1_expression_data_ExampleJava_predictions.txt
	 */

  /* Construct the output filename if it was not specified */
  if (args.mapFile == (char *)NULL)
  {
    /* The beginning of the filename (without the path) */
    char *start = strrchr(args.dataFile, '/'); /* strrchr returns the last occurence of the char in the string */
    if (start == NULL)
      start = strrchr(args.dataFile, '\\'); /* windows */

    if (start == NULL)
    {
      start = args.dataFile;
    }
    else
    {
      start = start + 1;
    }

    /* The end of the filename (without file extension) */
    char *end = strrchr(args.dataFile, '.');
    int filenameLength;
    int startIndex = start - args.dataFile;
    if (end == NULL || end <= start)
      filenameLength = strlen(args.dataFile + startIndex);
    else
      filenameLength = end - start;

    args.mapFile = malloc((filenameLength + strlen("_CLR_predictions.txt") + 1) * sizeof(char));
    strncpy(args.mapFile, start, filenameLength);
    strcat(args.mapFile, "_CLR_predictions.txt");
  }

  return args;
}

int main(int argc, char *argv[])
{
  ARGUMENTS args = parseOptions(argc, argv);
  char *fileBuf, **lines, **linesTf, *headSep, *curSep, *curLine, *curLineTf, **rowNames;
  int i, j, f, l, numVars, numSamples, placeholder, *binCount, numTfs, *tfIdx;
  long fLength, fLengthTf;
  gzFile *input, *inputTf;
  double *X, stdBinCount;
  float *MI, *C, *S;
  FILE *out;

  /************************************************************************************************
   read in expression file */

  /* returns the length of the file.  This is needed to initiate the memory for the filebuffer.
     Also returns the number of samples (conditions) and the number of variables (genes)*/
  fLength = sizeArray(args.dataFile, &numSamples, &numVars);
  numSamples--; /* subtract 1 to account for the first row in the file which is the row of gene id */
  printf("There are %d genes and %d experimental conditions in file %s\n", numVars, numSamples, args.dataFile);

  /* initiate the fileBuffer and read the entire file in to the fileBuffer*/
  input = gzopen(args.dataFile, "rb");
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
  lines = (char **)calloc((numSamples + 1), sizeof(char *));

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
  for (l = 1; l < (numSamples + 1); ++l)
  {
    f = l;
    curSep = strtok(lines[l], SEPARATOR); /* for each line, break apart the line on the tabs */
    while (curSep != NULL)
    {
      --f;
      X[f] = atof(curSep); /* since one column is one gene, have to do a little math to get the indexes correct*/
      f += numSamples + 1;
      curSep = strtok(NULL, SEPARATOR);
    }
  }

  /* in the end, X should contain a flattened matrix of gene expression values, where the values for gene 2 are placed sequentially
     after gene 1, and gene 3 after gene 2, etc. */

  /* tokenize the first row to get the gene ids into "rowNames" */
  headSep = strtok(lines[0], SEPARATOR);
  f = 0;
  while (headSep != NULL)
  {
    rowNames[f] = headSep;
    ++f;
    headSep = strtok(NULL, SEPARATOR);
  }

  free(lines);

  /************************************************************************************************
   read in TF file */

  if (args.tfIdxFile != (char *)NULL)
  {
    /* same as with the data file, get the length and number of TFs */
    fLengthTf = sizeArray(args.tfIdxFile, &numTfs, &placeholder);
    printf("There are %d TFs in file %s\n", numTfs, args.tfIdxFile);

    /* initiate the fileBuffer and read the entire file in to the fileBuffer*/
    inputTf = gzopen(args.tfIdxFile, "rb");
    char *fileBufTf = (char *)calloc(fLength + 1, sizeof(char));
    if (fileBufTf == NULL)
    {
      fprintf(stderr, "Cannot allocate a buffer of size %ld bytes for file reading.\n", fLengthTf + 1);
      return -1;
    }
    gzread(inputTf, fileBufTf, fLengthTf); /* Read the entire file into fileBuf */
    gzclose(inputTf);

    /* read in each TF which will be a separate line in the TF file */
    linesTf = (char **)calloc(numTfs, sizeof(char *));
    curLineTf = strtok(fileBufTf, NEW_LINE);
    f = 0;
    while (curLineTf != NULL)
    {
      linesTf[f] = curLineTf;
      curLineTf = strtok(NULL, NEW_LINE);
      ++f;
    }

    /* loop through all the genes and flag the indices that match TFs */
    tfIdx = calloc(numTfs, sizeof(int));
    f = 0;
    for (i = 0; i < numVars; ++i)
    {
      for (j = 0; j < numTfs; ++j)
      {
        if (strcmp(rowNames[i], linesTf[j]) == 0)
        {
          tfIdx[f] = i;
          ++f;
          break;
        }
      }
    }

    free(linesTf);
    free(fileBufTf);
  }
  else
  { /* default mode: every gene is a TF */
    numTfs = numVars;
    tfIdx = calloc(numTfs, sizeof(int));
    for (i = 0; i < numTfs; i++)
    {
      tfIdx[i] = i;
    }
  }

  /************************************************************************************************
   get bins, calculate MI and CLR */

  /* will estimate the number of bins if the user did not supply a bin size */
  binCount = calcNumBins(X, numVars, numSamples, 1);
  if (args.numBins == -1)
  {
    args.numBins = (int)floor(mediani(binCount, numVars));
    stdBinCount = stdi(binCount, numVars);
    printf("Bin count not supplied. Autodetected that dataset warrants %d bins (median); stddev == %f\n", args.numBins, stdBinCount);
    if (args.numBins > 15)
      fprintf(stdout, "Warning: this automatic bin count (%d) may be a bit slow on large datasets, and may warrant a spline degree above 3\n", args.numBins);
    if (args.numBins < 2)
    {
      fprintf(stderr, "Too few bins (%d)!\n", args.numBins);
      exit(-1);
    }
  }

  printf("Computing MI: spline degree %d, number of bins %d\n", args.splineDegree, args.numBins);

  /* calculate MI and transform using stouffers method */
  MI = (float *)calloc(numVars * numVars, sizeof(float));
  miSubMatrix(X, MI, args.numBins, numVars, numSamples, args.splineDegree, 0, numVars - 1);

  C = (float *)calloc(numVars * numVars, sizeof(float));
  clrUnweightedStouffer(MI, C, numVars);
  free(MI);

  /************************************************************************************************
   get and sort the output */

  /***** A more efficient way to do it would be as in the Java example "BasicCorrelation" *****/

  /* from the CLR matrix, select out all TF to target gene interactions and place them into S */
  int numPossibleEdges = (numTfs * numVars) - numTfs;
  S = (float *)calloc(numPossibleEdges, sizeof(float));
  f = 0;
  for (i = 0; i < numTfs; i++)
  {
    for (j = 0; j < numVars; j++)
    {
      if (tfIdx[i] != j)
      {
        S[f] = C[tfIdx[i] * numVars + j];
        ++f;
      }
    }
  }

  qsort(S, numPossibleEdges, sizeof(float), (void *)compare_floats);

  /* -1 means print all possible edges */
  if (args.cut == -1)
    args.cut = numPossibleEdges;

  /* loop through the ranked set of CLR values and print out any  TF to target prediction
     that matches the current CLR score.  Stop at 100,000 and print to the output file. */
  f = 0;
  out = fopen(args.mapFile, "w");
  float prev = 0;
  for (l = numPossibleEdges - 1; l > 0; --l)
  {
    if (S[l] != prev)
    { /* make sure we are not repeating predictions. this will avoid printing predictions more than once */
      prev = S[l];
      for (i = 0; i < numTfs; i++)
      {
        for (j = 0; j < numVars; j++)
        {
          if (tfIdx[i] != j)
          {
            if (C[tfIdx[i] * numVars + j] == S[l])
            {
              fprintf(out, "%s\t%s\t%f\n", rowNames[tfIdx[i]], rowNames[j], C[tfIdx[i] * numVars + j]);
              ++f;
              /* If the cutoff is reached, break the three loops */
              if (f >= args.cut)
              {
                j = numVars;
                i = numTfs;
                l = -1;
              }
            }
          }
        }
      }
    }
  }
  fclose(out);

  printf("Success, your output has been written to %s\n", args.mapFile);

  free(fileBuf);
  free(tfIdx);
  free(binCount);
  free(X);
  free(C);
  free(S);
  free(args.dataFile);
  if (args.tfIdxFile != (char *)NULL)
  {
    free(args.tfIdxFile);
  }
  if (args.mapFile != (char *)NULL)
    free(args.mapFile);
  free(rowNames);

  return 0;
}
