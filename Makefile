CUDA_PATH ?= /usr/local/cuda-11.1
NVCC := $(CUDA_PATH)/bin/nvcc
LIBSMCTRL_PATH := /apollo/cyber/examples/bachelor_thesis/libsmctrl
CXXFLAGS := -O2 -std=c++11 -Xcompiler -fPIC -I$(LIBSMCTRL_PATH) -cudart=shared
LDFLAGS := -lcuda -shared -L$(LIBSMCTRL_PATH) -lsmctrl -ldl -lpthread
TARGET := libgpu_intercept.so
SRCS := intercept.cpp

all: $(TARGET)

$(TARGET): $(SRCS)
	$(NVCC) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)

clean:
	rm -f $(TARGET)

.PHONY: all clean
