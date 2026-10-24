# FP-Growth (CPU reference implementation) -- ESE 327 project
#
#   make            optimised build        -> ./fpgrowth
#   make debug      ASan + debug symbols   -> ./fpgrowth_debug   (needs libasan: Linux/clang)
#   make run-han    build and run the Han & Kamber example with --print-tree
#   make test       build both binaries and run the full test suite
#                   (edge cases, brute-force differential, mlxtend cross-check)
#   make data       download the UCI benchmark datasets and convert them to
#                   data/processed/<name>.dat (+ .map.tsv item names)
#   make bench      support sweep, scaling, itemset lengths, mlxtend timing
#                   -> results/*.csv  (resumable: finished rows are skipped;
#                   delete a CSV to re-measure it).  Takes ~20 min.
#   make figures    results/figures/*.pdf|png and results/tables.md
#   make omp        OpenMP build (parallel top-level mining) -> ./fpgrowth_omp
#   make cuda       CUDA + OpenMP build                       -> ./fpgrowth_cuda
#   make gpuinfo    CUDA smoke test: device report + one kernel launch -> ./gpuinfo
#   make test-par   edge cases + brute force on fpgrowth_omp (and fpgrowth_cuda if
#                   nvcc and a GPU are present), and diff against ./fpgrowth on
#                   every processed dataset at 2 support levels
#   make clean
#
# The host C++ code of all three binaries is compiled by the same $(CXX) with
# the same $(CXXFLAGS), so differences in speed come from parallelism, not
# from compilers; only src/cuda/*.cu goes through nvcc (with NVCC_CCBIN), and
# nvcc only drives the final link of fpgrowth_cuda.
#
# CUDA variables (override on the command line, e.g. on the Jetson Orin:
#   make gpuinfo ARCH=sm_87 NVCC_CCBIN=g++ ):
#   NVCC        nvcc to use                        (default: nvcc on PATH)
#   NVCC_CCBIN  host compiler nvcc drives (-ccbin); must be one the installed
#               CUDA version supports.  CUDA 12.4 rejects g++ 14+, hence the
#               g++-13 default for the dev machine.
#   ARCH        GPU architecture: sm_89 RTX 4070 (dev), sm_87 Jetson Orin;
#               lab server: check with ./gpuinfo or nvidia-smi --query-gpu=compute_cap
#
# Recipe lines must start with a TAB.

CXX      ?= g++
CXXFLAGS ?= -std=c++17 -O2 -DNDEBUG -Wall -Wextra -Wpedantic
DBGFLAGS := -std=c++17 -O1 -g -fsanitize=address -fno-omit-frame-pointer -Wall -Wextra -Wpedantic
LDLIBS   :=
EXE      :=

NVCC       ?= nvcc
NVCC_CCBIN ?= g++-13
ARCH       ?= sm_89
NVCCFLAGS  ?= -std=c++17 -O2 -DNDEBUG -lineinfo
NVCCALL    := $(NVCCFLAGS) -arch=$(ARCH) -ccbin $(NVCC_CCBIN) -Xcompiler -Wall,-Wextra
OMPFLAGS   ?= -fopenmp

ifeq ($(OS),Windows_NT)
  EXE    := .exe
  LDLIBS += -lpsapi          # GetProcessMemoryInfo (peak memory fallback on Windows)
endif

SRC  := src/fptree.cpp src/fpgrowth.cpp src/main.cpp
HDR  := src/fptree.h src/fpgrowth.h
OBJ  := $(patsubst src/%.cpp,build/release/%.o,$(SRC))
DOBJ := $(patsubst src/%.cpp,build/debug/%.o,$(SRC))

TARGET  := fpgrowth$(EXE)
DTARGET := fpgrowth_debug$(EXE)

.PHONY: all debug omp cuda run-han test test-par venv data bench figures clean

all: $(TARGET)
debug: $(DTARGET)

$(TARGET): $(OBJ)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDLIBS)

$(DTARGET): $(DOBJ)
	$(CXX) $(DBGFLAGS) -o $@ $^ $(LDLIBS)

build/release/%.o: src/%.cpp $(HDR)
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) -c $< -o $@

build/debug/%.o: src/%.cpp $(HDR)
	@mkdir -p $(dir $@)
	$(CXX) $(DBGFLAGS) -c $< -o $@

# OpenMP build: the serial sources plus src/parallel.cpp, with -DFPG_OMP.
PSRC := $(SRC) src/parallel.cpp
PHDR := $(HDR) src/parallel.h
OOBJ := $(patsubst src/%.cpp,build/omp/%.o,$(PSRC))
OTARGET := fpgrowth_omp$(EXE)

omp: $(OTARGET)

$(OTARGET): $(OOBJ)
	$(CXX) $(CXXFLAGS) $(OMPFLAGS) -o $@ $^ $(LDLIBS)

build/omp/%.o: src/%.cpp $(PHDR)
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) $(OMPFLAGS) -DFPG_OMP -c $< -o $@

# CUDA.  Not part of 'all', so the CPU build still works without nvcc.
# Everything CUDA is rebuilt when ARCH or NVCC_CCBIN change (stamp file).
CUDA_STAMP := build/cuda/.config-$(ARCH)-$(notdir $(NVCC_CCBIN))
COBJ := $(patsubst src/%.cpp,build/cuda/%.o,$(PSRC)) build/cuda/gpu.o
CTARGET := fpgrowth_cuda$(EXE)

cuda: $(CTARGET)

# nvcc links (it knows where libcudart is); its host compiler here is only
# the linker driver, so $(CXX) is fine even if nvcc would reject it for code.
$(CTARGET): $(COBJ)
	$(NVCC) -ccbin $(CXX) -arch=$(ARCH) -Xcompiler $(OMPFLAGS) -o $@ $^

build/cuda/%.o: src/%.cpp $(PHDR) src/gpu.h $(CUDA_STAMP)
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) $(OMPFLAGS) -DFPG_CUDA -c $< -o $@

build/cuda/gpu.o: src/cuda/gpu.cu src/gpu.h $(HDR) $(CUDA_STAMP)
	@mkdir -p $(dir $@)
	$(NVCC) $(NVCCALL) -c $< -o $@

$(CUDA_STAMP):
	@mkdir -p $(dir $@)
	@rm -f build/cuda/.config-*
	@touch $@

gpuinfo: src/cuda/gpuinfo.cu $(CUDA_STAMP)
	$(NVCC) $(NVCCALL) -o $@ $<

run-han: $(TARGET)
	./$(TARGET) data/examples/han.dat 2 --print-tree

# ---------------------------------------------------------------------------
# Tests.  System pip is locked (PEP 668), so Python dependencies live in a
# project-local virtualenv created on first use.
# ---------------------------------------------------------------------------
VENV    := .venv
VPY     := $(VENV)/bin/python
PYSTAMP := $(VENV)/.stamp

$(PYSTAMP): requirements.txt
	python3 -m venv $(VENV)
	$(VPY) -m pip install --quiet --upgrade pip
	$(VPY) -m pip install --quiet -r requirements.txt
	touch $@

venv: $(PYSTAMP)

test: $(TARGET) $(DTARGET) $(PYSTAMP)
	bash tests/run_edge_cases.sh ./$(TARGET)
	bash tests/run_edge_cases.sh ./$(DTARGET)
	$(VPY) tests/brute_force.py --binary ./$(TARGET) --trials 500 --seed 1
	$(VPY) tests/brute_force.py --binary ./$(DTARGET) --trials 200 --seed 2
	$(VPY) tests/compare_mlxtend.py data/examples/han.dat 2
	@echo "All tests passed."

# Parallel binaries: fpgrowth_cuda is built (and tested) only if nvcc exists;
# the script also skips it when no GPU is usable.  Needs only python3.
HAVE_NVCC := $(shell command -v $(NVCC) 2>/dev/null)

test-par: $(TARGET) $(OTARGET) $(if $(HAVE_NVCC),$(CTARGET))
	bash tests/run_parallel_tests.sh

# ---------------------------------------------------------------------------
# Benchmark datasets (UCI): raw downloads in data/raw/, FIMI files in
# data/processed/.  Both are gitignored; download skips existing files.
# ---------------------------------------------------------------------------
data: $(PYSTAMP)
	bash scripts/download_data.sh
	$(VPY) scripts/preprocess.py
	$(VPY) scripts/dataset_stats.py

# ---------------------------------------------------------------------------
# Benchmarks (run one job at a time; every run is time- and memory-capped,
# see scripts/benchlib.py) and the report figures / tables built from them.
# ---------------------------------------------------------------------------
bench: $(TARGET) $(PYSTAMP)
	$(VPY) scripts/run_bench.py
	$(VPY) scripts/run_scaling.py
	$(VPY) scripts/itemset_lengths.py
	$(VPY) scripts/vs_mlxtend.py

figures: $(PYSTAMP)
	$(VPY) scripts/plot.py
	$(VPY) scripts/make_tables.py

clean:
	rm -rf build $(TARGET) $(DTARGET) $(OTARGET) $(CTARGET) gpuinfo
