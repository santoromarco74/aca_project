# ===========================================================================
#  Makefile alternativo a CMake, utile su Colab e sulle macchine dove non si
#  vuole configurare nulla.
#
#    make            build CPU (seriale + OpenMP)
#    make cuda       build completa, richiede nvcc nel PATH
#    make test       compila ed esegue la suite di correttezza
#    make bench      esegue un benchmark di esempio
#    make clean
# ===========================================================================

CXX      ?= g++
NVCC     ?= nvcc
CXXFLAGS := -O3 -std=c++17 -Wall -Wextra -Iinclude -fopenmp
NVCCFLAGS := -O3 -std=c++17 -Iinclude -lineinfo -arch=sm_60 -DUSE_CUDA

BUILD := build
CPU_SRC := src/common.cpp src/kmeans_serial.cpp src/kmeans_omp.cpp
CUDA_SRC := src/kmeans_cuda.cu

.PHONY: all cuda test bench clean

all: $(BUILD)/kmeans_bench $(BUILD)/kmeans_test

$(BUILD):
	mkdir -p $(BUILD)

# --- build CPU ------------------------------------------------------------
$(BUILD)/kmeans_bench: $(CPU_SRC) src/main.cpp | $(BUILD)
	$(CXX) $(CXXFLAGS) $^ -o $@

$(BUILD)/kmeans_test: $(CPU_SRC) tests/test_correctness.cpp | $(BUILD)
	$(CXX) $(CXXFLAGS) $^ -o $@

# --- build con CUDA -------------------------------------------------------
# nvcc compila anche le unita' C++ e propaga -fopenmp all'host compiler con
# -Xcompiler, cosi' la versione OpenMP resta disponibile nel binario GPU.
cuda: | $(BUILD)
	$(NVCC) $(NVCCFLAGS) -Xcompiler "-O3 -fopenmp" \
		$(CPU_SRC) $(CUDA_SRC) src/main.cpp -o $(BUILD)/kmeans_bench_cuda
	$(NVCC) $(NVCCFLAGS) -Xcompiler "-O3 -fopenmp" \
		$(CPU_SRC) $(CUDA_SRC) tests/test_correctness.cpp -o $(BUILD)/kmeans_test_cuda

test: $(BUILD)/kmeans_test
	./$(BUILD)/kmeans_test

bench: $(BUILD)/kmeans_bench
	./$(BUILD)/kmeans_bench --n 200000 --d 32 --k 16 --reps 3

clean:
	rm -rf $(BUILD)
