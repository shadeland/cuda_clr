# cuda_clr
Estimating mutual information using B-spline functions – an improved similarity measure for analysing gene expression data

# Compile
./src/make

#RUN
./src/template <NUMSAPLES> <NUMVARS> <NUMBINS> <BACHSIZE> <THREADPERBLOCK>

##TIPS
to run gdbserver on host : `gdb-server :port file`
to connect to remote gdb : `target remote addr:port` in your local gdb 
