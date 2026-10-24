# FP-Growth (CPU reference implementation) -- ESE 327 project
#
#   make            optimised build        -> ./fpgrowth
#   make debug      ASan + debug symbols   -> ./fpgrowth_debug   (needs libasan: Linux/clang)
#   make run-han    build and run the Han & Kamber example with --print-tree
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

.PHONY: all debug run-han clean

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

clean:
	rm -rf build $(TARGET) $(DTARGET)
