# FP-Growth (CPU reference implementation) -- ESE 327 project
#
#   make            optimised build        -> ./fpgrowth
#   make debug      ASan + debug symbols   -> ./fpgrowth_debug   (needs libasan: Linux/clang)
#   make run-han    build and run the Han & Kamber example with --print-tree
#   make test       build both binaries and run the full test suite
#                   (edge cases, brute-force differential, mlxtend cross-check)
#   make data       download the UCI benchmark datasets and convert them to
#                   data/processed/<name>.dat (+ .map.tsv item names)
#   make clean
#
# Recipe lines must start with a TAB.

CXX      ?= g++
CXXFLAGS ?= -std=c++17 -O2 -DNDEBUG -Wall -Wextra -Wpedantic
DBGFLAGS := -std=c++17 -O1 -g -fsanitize=address -fno-omit-frame-pointer -Wall -Wextra -Wpedantic
LDLIBS   :=
EXE      :=

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

.PHONY: all debug run-han test venv data clean

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

# ---------------------------------------------------------------------------
# Benchmark datasets (UCI): raw downloads in data/raw/, FIMI files in
# data/processed/.  Both are gitignored; download skips existing files.
# ---------------------------------------------------------------------------
data: $(PYSTAMP)
	bash scripts/download_data.sh
	$(VPY) scripts/preprocess.py
	$(VPY) scripts/dataset_stats.py

clean:
	rm -rf build $(TARGET) $(DTARGET)
