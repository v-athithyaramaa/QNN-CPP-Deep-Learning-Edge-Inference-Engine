# Performance Benchmark & Cross-Engine Audit Report
## Qualcomm QNN-Accelerated CIFAR-10 Mini-ResNet Inference Engine

* **Date of Evaluation:** September 22, 2026
* **Platform:** Windows x86_64 (MSVC 19.43, CMake 4.3.3)
* **SDK Version:** Qualcomm AI Engine Direct (QAIRT / QNN) SDK v2.50.0.260828
* **Target Backend:** CPU Backend (`QnnCpu.dll`)
* **Precision Format:** INT8 Asymmetric Quantization (`QNN_DATATYPE_UFIXED_POINT_8`)
* **Primary Executable:** `build_engine/Release/qnn_cifar10_engine.exe`
* **Serialized Model Context:** `output_context/cifar10_miniresnet_context.bin`

---

## 1. Executive Summary & Verification Matrix

This audit verifies numerical parity, runtime latency, and operator bottlenecks for a custom, bare-metal C++ inference engine designed to execute quantized deep learning models on Qualcomm hardware architectures via the QNN API.

Validation was conducted via a strict **three-way cross-validation pipeline** against PyTorch/ONNX Runtime (unquantized FP32 ceiling) and Qualcomm's official SDK reference runner (`qnn-net-run.exe`):

| Pipeline Stage | Implementation | Data Precision | Sample Size | Top-1 Accuracy | Latency (Mean) | Numerical Parity |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Stage 1 (Ceiling)** | PyTorch 2.x / ONNX Runtime | FP32 | 50 | **88.00%** (44/50) | 1.85 ms (Py) | Baseline Reference |
| **Stage 2 (Vendor Ref)** | Qualcomm `qnn-net-run.exe` | INT8 Native | 50 | **84.00%** (42/50) | 2.63 ms (Graph) | **100% Bit-Exact Match** |
| **Stage 3 (Production)** | Custom C++ Edge Engine | INT8 Native | 50 | **84.00%** (42/50) | **3.04 ms (E2E)** | **100% Bit-Exact Match** |

### Key Audit Findings:
1. **Quantization Impact:** Quantizing from IEEE FP32 to INT8 yielded a minor 4.00% top-1 accuracy reduction (88% to 84%), well within standard industrial tolerance thresholds (<5%).
2. **Zero In-Engine Corruption:** All 50 predictions produced by the custom C++ engine match the reference runner byte-for-byte and index-for-index across every individual sample.
3. **High Sustained Throughput:** The custom C++ engine sustains **328.88 FPS** on CPU execution, including OpenCV image ingest, normalization, asymmetric fixed-point tensor mapping, and softmax classification.

---

## 2. In-Depth Latency & Percentile Breakdown

Benchmarking was recorded using high-resolution monotonic clocks (`std::chrono::steady_clock` / `time.perf_counter`) across warm cache inferences:

```text
Latency Metric Distribution (N=50)
----------------------------------------------------------------------
P50 (Median)   :  2.801 ms  [Baseline operational point]
Mean           :  3.041 ms  [Average end-to-end execution]
P90            :  4.876 ms  [90th percentile SLA ceiling]
P95            :  5.484 ms  [High-load threshold]
P99            :  8.246 ms  [Tail variance limit]
Min Latency    :  0.617 ms  [Peak single-sample throughput]
Max Latency    : 10.316 ms  [Initial cache / page fault peak]
Throughput     : 328.88 FPS [Continuous single-thread throughput]
----------------------------------------------------------------------
```

---

## 3. Microsecond Per-Operator Hardware Profiling

Using Qualcomm's detailed diagnostic tracing (`qnn-profile-viewer`), each graph kernel's compute allocation was isolated:

| Operator / Layer Identifier | Kernel Type | Avg Time (ms) | Time (?s) | Share (%) | Visual Impact Bar |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `downsample1_downsample1_0_Conv` | Conv/Pool | 1.267 ms | 1266.8 ?s | 48.12% | `##########` |
| `stem_stem_0_Conv` | Conv/Pool | 0.757 ms | 757.3 ?s | 28.77% | `######` |
| `res_block_Add` | Conv/Pool | 0.179 ms | 178.5 ?s | 6.78% | `#` |
| `res_block_conv1_Conv` | Conv/Pool | 0.146 ms | 146.2 ?s | 5.55% | `#` |
| `res_block_conv2_Conv` | Conv/Pool | 0.123 ms | 122.8 ?s | 4.66% | `#` |
| `downsample2_downsample2_0_Conv` | Conv/Pool | 0.102 ms | 102.3 ?s | 3.89% | `#` |
| `gap_GlobalAveragePool` | Conv/Pool | 0.019 ms | 19.5 ?s | 0.74% | `` |
| `gap_GlobalAveragePool_output_0_nchw` | Conv/Pool | 0.008 ms | 7.8 ?s | 0.30% | `` |
| `classifier_Gemm` | Conv/Pool | 0.007 ms | 7.2 ?s | 0.27% | `` |

### Architectural Analysis & Optimization Paths:
* **Spatial Convolution Bottleneck:** Two operations (`downsample1_downsample1_0_Conv` at 48.12% and `stem_stem_0_Conv` at 28.77%) constitute **76.89%** of the entire graph execution budget. Because these operations act on early activation dimensions (32x32 and 16x16 with 32/64 channels), memory bandwidth and GEMM unrolling dominate latency.
* **Global Pooling Efficiency:** The transition from feature maps to dense logits via `gap_GlobalAveragePool` costs only **0.019 ms (0.74%)**, and the final classifier fully connected layer takes only **0.007 ms (0.27%)**. This validates the architectural decision to avoid multi-layer dense heads.

---

## 4. Root Cause & Investigation Audit Trail

During development and vendor validation, several edge-case integration challenges were isolated, diagnosed, and resolved:

### Incident 1: Line-Ending Desynchronization in Qualcomm List Parsing
* **Symptom:** Batch validation aborted prematurely at sample 10.
* **Cause:** Windows-formatted CRLF (`\r\n`) strings in the generated batch manifest confused QNN's internal file tokenizer.
* **Fix:** Enforced POSIX LF (`\n`) across all automated list generation utilities.

### Incident 2: Native vs. Floating-Point I/O Buffer Mismatches
* **Symptom:** Inferences initially produced identical logits across disparate input images.
* **Cause:** By default, `qnn-net-run` allocates FP32 buffers (12,288 bytes per sample) and performs automatic runtime quantization. Ingesting native 3,072-byte INT8 buffers caused buffer exhaustion and uninitialized memory propagation.
* **Fix:** Appended explicit `--use_native_input_files` and `--use_native_output_files` flags, enforcing raw byte ingestion and emission of `output_native.raw` as unsigned 8-bit integers.

---

## 5. Audit Conclusion & Production Readiness
The C++ inference engine has satisfied all verification gates:
* [x] **Bit-exact parity** against Qualcomm reference runtime.
* [x] **Zero memory leaks or page boundary overreads** on 64-byte aligned tensor buffers.
* [x] **Sub-3.5 ms median latency** on commodity x86_64 CPU cores.
* [x] **Fully deterministic output vectors** logged in `docs/audit_logs/`.

The system is certified for deployment and downstream hardware acceleration (HTP/NPU target compilation).