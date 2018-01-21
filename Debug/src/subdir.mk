################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CU_SRCS += \
../src/template.cu 

CPP_SRCS += \
../src/template_cpu.cpp 

OBJS += \
./src/template.o \
./src/template_cpu.o 

CU_DEPS += \
./src/template.d 

CPP_DEPS += \
./src/template_cpu.d 


# Each subdirectory must supply rules for building sources it contributes
src/%.o: ../src/%.cu
	@echo 'Building file: $<'
	@echo 'Invoking: NVCC Compiler'
	/usr/local/cuda-9.1/bin/nvcc -I"/usr/local/cuda-9.1/samples/0_Simple" -I"/usr/local/cuda-9.1/samples/common/inc" -I"/home/adel/Projects/cuda/playground/first/mi1" -G -g -O0 -gencode arch=compute_30,code=sm_30  -odir "src" -M -o "$(@:%.o=%.d)" "$<"
	/usr/local/cuda-9.1/bin/nvcc -I"/usr/local/cuda-9.1/samples/0_Simple" -I"/usr/local/cuda-9.1/samples/common/inc" -I"/home/adel/Projects/cuda/playground/first/mi1" -G -g -O0 --compile --relocatable-device-code=false -gencode arch=compute_30,code=compute_30 -gencode arch=compute_30,code=sm_30  -x cu -o  "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '

src/%.o: ../src/%.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: NVCC Compiler'
	/usr/local/cuda-9.1/bin/nvcc -I"/usr/local/cuda-9.1/samples/0_Simple" -I"/usr/local/cuda-9.1/samples/common/inc" -I"/home/adel/Projects/cuda/playground/first/mi1" -G -g -O0 -gencode arch=compute_30,code=sm_30  -odir "src" -M -o "$(@:%.o=%.d)" "$<"
	/usr/local/cuda-9.1/bin/nvcc -I"/usr/local/cuda-9.1/samples/0_Simple" -I"/usr/local/cuda-9.1/samples/common/inc" -I"/home/adel/Projects/cuda/playground/first/mi1" -G -g -O0 --compile  -x c++ -o  "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


