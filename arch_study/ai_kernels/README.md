# Model-derived AI kernels

**These are model-derived kernels, not full benchmark implementations.**
They reproduce documented layer shapes from the MLPerf Inference reference models (ResNet-50 v1.5, BERT-Large, DLRM-v2) and from Baidu DeepBench, whose GEMM micro-benchmark is also the basis of the Alibaba AI-Matrix micro-benchmark. They do not run MLPerf, DeepBench or AI-Matrix, and they produce no benchmark scores.

They exist to give controlled, repeatable tensor workloads for cache-behaviour studies in gem5 (RISC-V, SE mode).

## Kernels

| Binary | Operation | Arguments (before `[units] [mode]`) |
|---|---|---|
| `gemm` | C[M][N] = A[M][K]·B[K][N], row-major; `naive` (i-k-j) or `blocked` | `M N K [units] [naive\|blocked] [tile] [mode]` |
| `conv2d_nchw` | direct 2D convolution, NCHW, batch 1 | `C H W K R S stride pad [units] [mode]` |
| `attention` | per-head softmax(QKᵀ/√D)·V, unfused | `S D heads [units] [mode]` |
| `pool` | max or average pooling (square window) | `max\|avg C H W R stride pad [units] [mode]` |
| `embedding` | `gather` (BERT token lookup) or `bag` (DLRM-v2 sum-pooled multi-hot) | `gather V D L [units] [mode]` / `bag B R_cap [units] [mode]` |
| `mlp` | chained fully-connected layers with ReLU (DLRM MLPs) | `d0,d1,...,dn B [units] [mode]` |

- **`units`** is the number of independent work units computed (output rows, output channels, heads, lookups or samples). `0` means all of them.
- **`mode`**:
  - `cold` (default): a 64 MiB buffer is streamed before the region of interest, so no initialisation data stays in the caches.
  - `warm`: the kernel runs once before the region of interest.
  - `none`: neither.

Every binary prints:
- `META {...}`: a JSON line with the dimensions, work fraction, MACs, tensor bytes and the label `model-derived kernel`.
- `CHECKSUM <sum> <abs-sum> <count>`: computed over the output elements.

## Region of interest (ROI)
In the RISC-V build (`-DGEM5`), the kernel calls `m5_reset_stats` just before the measured region and `m5_dump_stats` just after it. The **first** statistics block in gem5's `stats.txt` therefore covers the kernel only, not initialisation. The native build compiles these calls out.

## Shape-faithful, work-sampled
- Every tensor is allocated with its real dimensions and layout. Footprint, strides and reuse distances are therefore those of the real layer.
- Only `units` work units are computed, so a gem5 run takes minutes. Each run reports `work_fraction = units_executed / units_total`.
- A full-layer extrapolation is an estimate and must be labelled as one.

## Layer registry
`layers.csv` lists all 27 layers. Each row has its kernel arguments, operand roles, source citations and notes.

Markers used in the registry:
- **[A]**: an assumption made for this study.
- **[TBV]**: a value still to be verified against its source.

Parameters marked [A] or [TBV]:
- **DLRM-v2 tables [A]:** tables larger than `R_cap` rows (default 65,536 rows = 32 MiB each) are capped. Capped tables remain much larger than the simulated last-level cache. The 26 row counts, the embedding dimension 128 and the multi-hot sizes are the real values.
- **DLRM-v2 indices and batch [A]:** indices are uniform (the MLPerf synthetic multi-hot distribution type is `uniform`); batch size 64.
- **DLRM-v2 dense inputs [TBV]:** 13 dense features.
- **DLRM-v2 top-MLP input [A]:** dimension 3456 = 27×128, the DCN interaction output.
- **DeepBench GEMM layout [A]:** shapes are column-major (cuBLAS) in DeepBench and are mapped to row-major C = A·B here.

## Precision and code generation
- float32 (the MLPerf reference models are fp32). DeepBench specifies int8 multiply with int32 accumulate for inference; this is not modelled.
- RISC-V build: `riscv64-linux-gnu-gcc -O2 -static -march=rv64gc -mabi=lp64d -ffp-contract=off`. This is scalar code with no RVV.

## Limitations
- The code is plain C: naive or blocked GEMM, direct convolution, unfused attention. Production frameworks use packed or im2col BLAS micro-kernels, operator fusion and vector units, so their cache behaviour differs.
- Batch size 1 except DLRM (batch B); DCN cross layers are not included.
- BERT token ids and DLRM indices are synthetic.

## Usage
```
make all       # bin/riscv/* (for gem5) and bin/native/* (for checks)
make verify    # numpy reference check of the native binaries on tiny shapes
```

## Sources
- He et al., "Deep Residual Learning for Image Recognition", 2016, Table 1: https://arxiv.org/abs/1512.03385
- MLPerf Inference: https://github.com/mlcommons/inference
  - BERT config: `language/bert/bert_config.json`; `max_seq_length` 384 in `language/bert/README.md`
  - DLRM-v2: `recommendation/dlrm_v2/pytorch/python/main.py`, `backend_pytorch_native.py`
- DLRM multi-hot sizes: https://github.com/facebookresearch/dlrm/tree/main/torchrec_dlrm
- DeepBench: https://github.com/baidu-research/DeepBench (`code/kernels/gemm_problems.h`, `conv_problems.h`)
- AI-Matrix: https://github.com/alibaba/ai-matrix (its micro-benchmark "comes from Baidu's DeepBench")
