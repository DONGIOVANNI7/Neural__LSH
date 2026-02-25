CXX      := g++
CXXFLAGS := -std=gnu++23 -O3 -Wall -Wextra -Wconversion -Wno-sign-conversion -Wpedantic -MMD -MP -march=native -mtune=native -pipe
LDFLAGS  := -ltbb
TARGET   := approx_knn

SRC_DIR   := src
BUILD_DIR := build
BIN_DIR   := bin

SRCS := $(wildcard $(SRC_DIR)/*.cpp)
OBJS := $(patsubst $(SRC_DIR)/%.cpp, $(BUILD_DIR)/%.o, $(SRCS))
DEPS := $(OBJS:.o=.d)
BIN  := $(BIN_DIR)/$(TARGET)

MNIST_DATA = data/mnist-data.dat
MNIST_QUERY = data/mnist-query.dat

SIFT_DATA = data/sift_base.fvecs
SIFT_QUERY = data/sift_query.fvecs

all: $(BIN)

sillouette:
	$(CXX) $(CXXFLAGS) sillouette/main.cpp src/kmeans.cpp src/mnist.cpp src/sift.cpp -o bin/sillouette
	parallel bin/sillouette data/mnist-data.dat data/sift_base.fvecs {} ::: {1..99}

$(BIN): $(OBJS)
	@mkdir -p $(BIN_DIR)
	$(CXX) $(OBJS) -o $@ $(LDFLAGS)

$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cpp
	@mkdir -p $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	$(RM) -rf $(BUILD_DIR) $(BIN_DIR)

.PHONY: all clean sillouette

-include $(DEPS)
