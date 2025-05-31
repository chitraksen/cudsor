# Compiler and flags
CXX = g++
CXXFLAGS = -std=c++17 -O3
INCLUDE_DIR = -Isrc

# Directories
BIN_DIR = bin
SRC_DIR = src
EXAMPLES_DIR = examples
TESTS_DIR = tests

# Source files
TENSOR_SRC = $(SRC_DIR)/tensor.cpp

# Create bin directory if it doesn't exist
$(shell mkdir -p $(BIN_DIR))

# Default target
.PHONY: all
all: test example1 example2

# Test target
.PHONY: test
test: $(BIN_DIR)/test
	@echo "\n▶ Running tests:\n"
	./$(BIN_DIR)/test

$(BIN_DIR)/test: $(TESTS_DIR)/test.cpp $(TENSOR_SRC)
	@echo "\n▶ Compiling tests..."
	$(CXX) $(CXXFLAGS) $(INCLUDE_DIR) -o $@ $^

# Example targets
.PHONY: example1 xor
example1 xor: $(BIN_DIR)/xor
	@echo "\n▶ Running XOR example:\n"
	./$(BIN_DIR)/xor

$(BIN_DIR)/xor: $(EXAMPLES_DIR)/xor.cpp $(TENSOR_SRC)
	@echo "\n▶ Compiling XOR example..."
	$(CXX) $(CXXFLAGS) $(INCLUDE_DIR) -o $@ $^

.PHONY: example2 mnist
example2 mnist: $(BIN_DIR)/mnist
	@echo "\n▶ Running MNIST example:\n"
	./$(BIN_DIR)/mnist

$(BIN_DIR)/mnist: $(EXAMPLES_DIR)/mnist.cpp $(TENSOR_SRC)
	@echo "\n▶ Compiling MNIST example..."
	$(CXX) $(CXXFLAGS) $(INCLUDE_DIR) -o $@ $^

# Build-only targets (compile without running)
.PHONY: build-test build-example1 build-example2
build-test: $(BIN_DIR)/test
build-example1: $(BIN_DIR)/xor
build-example2: $(BIN_DIR)/mnist

# Clean target
.PHONY: clean
clean:
	rm -f $(BIN_DIR)/*

# Help target
.PHONY: help
help:
	@echo "Available targets:"
	@echo "  make test        - Compile and run tests"
	@echo "  make example1    - Compile and run XOR example"
	@echo "  make example2    - Compile and run MNIST example"
	@echo "  make xor         - Alias for example1"
	@echo "  make mnist       - Alias for example2"
	@echo "  make build-test  - Compile tests without running"
	@echo "  make build-example1 - Compile XOR example without running"
	@echo "  make build-example2 - Compile MNIST example without running"
	@echo "  make clean       - Remove compiled binaries"
	@echo "  make all         - Build all targets"
