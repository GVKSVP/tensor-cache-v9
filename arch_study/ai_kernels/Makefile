# Model-derived AI kernels: RISC-V (gem5, with m5 ROI markers) and native builds.
#   make all     build bin/riscv/* and bin/native/*
#   make verify  check native binaries against the numpy reference
#   make clean   remove bin/

GEM5    ?= $(HOME)/gem5
RVCC    ?= riscv64-linux-gnu-gcc
CC      ?= gcc
PYTHON  ?= $(HOME)/gem5-project/.venv/bin/python

KERNELS := gemm conv2d_nchw attention pool embedding mlp

CFLAGS_COMMON := -O2 -ffp-contract=off -std=c11 -D_ISOC11_SOURCE -Wall -Wextra -Iinclude
RVFLAGS := -static -march=rv64gc -mabi=lp64d -DGEM5 -I$(GEM5)/include
M5OP    := $(GEM5)/util/m5/src/abi/riscv/m5op.S

RV_BINS     := $(addprefix bin/riscv/,$(KERNELS))
NATIVE_BINS := $(addprefix bin/native/,$(KERNELS))

.PHONY: all riscv native verify clean

all: riscv native

riscv: $(RV_BINS)
native: $(NATIVE_BINS)

bin/riscv/%: src/%.c include/ai_common.h | bin/riscv
	$(RVCC) $(CFLAGS_COMMON) $(RVFLAGS) $< $(M5OP) -lm -o $@

bin/native/%: src/%.c include/ai_common.h | bin/native
	$(CC) $(CFLAGS_COMMON) $< -lm -o $@

bin/riscv bin/native:
	mkdir -p $@

verify: native
	$(PYTHON) -I verify_reference.py

clean:
	rm -rf bin
