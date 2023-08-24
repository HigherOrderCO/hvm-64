NVCC = /usr/local/cuda-12.2/bin/nvcc
NVCC_FLAGS = -arch=sm_70 -Wno-deprecated-gpu-targets -w -O3
SRC = main.cu runtime.cu debug.cu memory.cu tests.cu
OBJ = $(SRC:.cu=.o)
EXE = main

all: $(SRC) $(EXE)

$(EXE): $(OBJ)
	$(NVCC) $(NVCC_FLAGS) $(OBJ) -o $@

%.o: %.cu
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@

clean:
	rm -f $(OBJ) $(EXE)
