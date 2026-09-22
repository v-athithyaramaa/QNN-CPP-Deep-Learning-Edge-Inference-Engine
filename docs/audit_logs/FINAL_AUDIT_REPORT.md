# Edge Inference Engine Audit Report: CIFAR-10 Mini-ResNet
**Generated:** 2026-09-22 12:05:47  
**Target Backend:** Qualcomm QNN CPU Backend (\QnnCpu.dll\ v2.50.0.260828)  
**Binary Executable:** \uild_engine/Release/qnn_cifar10_engine.exe\  
**Model Context:** \output_context/cifar10_miniresnet_context.bin
---

## 1. Executive Summary & Accuracy Cross-Validation

| Runtime Pipeline | Execution Engine | Precision | Evaluated Samples | Accuracy | Verdict |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Stage 1: Algorithmic Baseline** | PyTorch / ONNX Runtime | FP32 | 50 | **88.00%** (44/50) | Algorithmic Ceiling |
| **Stage 2: Qualcomm Reference** | \qnn-net-run.exe\ | INT8 Native | 50 | **84.00%** (42/50) | Reference Standard |
| **Stage 3: Custom Edge Engine** | Direct C++ QNN Native | INT8 Native | 50 | **84.00%** (42/50) | **100% Bit-Exact Parity** |

* **Quantization Accuracy Trade-off:** -4.00% degradation from unquantized FP32 to 8-bit asymmetric quantization.
* **Engine Correctness:** 0.00% deviation between custom C++ engine and Qualcomm's SDK reference runner across all 50 samples.

---

## 2. Latency & Throughput Distribution (Custom C++ Engine)

* **Throughput:** **328.88 FPS**
* **Sample Count (N):** 50 images

| Percentile / Metric | Latency (ms) | Operational Significance |
| :--- | :--- | :--- |
| **Min Latency** | 0.617 ms | Warm cache / low edge variance |
| **Median (P50)** | **2.801 ms** | Typical runtime operating point |
| **Mean** | **3.041 ms** | Expected average processing time |
| **P90** | 4.876 ms | Tail threshold for SLA considerations |
| **P95** | 5.484 ms | High-load bounds |
| **P99** | 8.246 ms | Outlier ceiling (OS context switch / I/O latency) |
| **Max Latency** | 10.316 ms | Cold thread / initial memory page fault |

---

## 3. Qualcomm Per-Operator Microsecond Profiling

* **Mean Graph Execution Time:** 2.632 ms  
* **Profiling Tool:** \qnn-profile-viewer.exe\ (\qnn-profiling-data.log\)

| Operator / Layer Identifier | Execution Time (ms) | Percentage of Compute Budget | Compute Characteristics |
| :--- | :--- | :--- | :--- |
| \downsample1_downsample1_0_Conv\ | 1.267 ms | 48.12% | Conv/Compute |\n| \stem_stem_0_Conv\ | 0.757 ms | 28.77% | Conv/Compute |\n| \res_block_Add\ | 0.179 ms | 6.78% | Conv/Compute |\n| \res_block_conv1_Conv\ | 0.146 ms | 5.55% | Conv/Compute |\n| \res_block_conv2_Conv\ | 0.123 ms | 4.66% | Conv/Compute |\n| \downsample2_downsample2_0_Conv\ | 0.102 ms | 3.89% | Conv/Compute |\n| \gap_GlobalAveragePool\ | 0.019 ms | 0.74% | Conv/Compute |\n| \gap_GlobalAveragePool_output_0_nchw\ | 0.008 ms | 0.30% | Conv/Compute |\n| \classifier_Gemm\ | 0.007 ms | 0.27% | Conv/Compute |\n
### Key Architectural Takeaways:
1. **Dominant Bottlenecks:** \downsample1_downsample1_0_Conv\ (48.12%) and \stem_stem_0_Conv\ (28.77%) together account for **76.89%** of the entire inference cycle due to higher spatial activations (32x32 and 16x16).
2. **Dense Head Efficiency:** \classifier_Gemm\ accounts for only **0.27%** of execution time, demonstrating effective parameter and computation reduction via \gap_GlobalAveragePool\ (0.74%).
