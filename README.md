# cuda_clr
Estimating mutual information using B-spline functions – an improved similarity measure for analysing gene expression data

# Compile
./src/make
# Config Performance
It's very important to choose batch size and ThreadBlock size wisele. Too small, oe too large sizes would
affect the performance. since global memory access 
#TODO
- readfile
- clacbins
- distributed gpus
- 
# RUN
./src/template <NUMSAPLES> <NUMVARS> <NUMBINS> <BACHSIZE> <THREADPERBLOCK>

# TIPS
to run gdbserver on host : `gdb-server :port file`
to connect to remote gdb : `target remote addr:port` in your local gdb

###for large files in git
git filter-branch -f --index-filter 'git rm -r --cached --ignore-unmatch src/loggpu' HEAD
 
