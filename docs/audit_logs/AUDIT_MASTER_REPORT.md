# QNN Edge Inference Validation Master Audit Log

**Generated:** 2026-09-22 11:48:57  
**Model:** CIFAR-10 Mini-ResNet (INT8 Quantized Context)  
**QNN SDK:** 2.50.0.260828  
**Backend:** Qualcomm CPU Backend (QnnCpu.dll)

---

## 1. Cross-Validation Accuracy Comparison

| Metric / Engine | Precision | Samples | Correct | Top-1 Accuracy | Delta vs FP32 Ceiling |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **PyTorch / ONNX Runtime** | FP32 | 50 | 44 | **88.00%** | Baseline (0.00%) |
| **Custom C++ Engine** | INT8 Native | 50 | 42 | **84.00%** | -4.00% (Quantization Error) |
| **Qualcomm Reference (qnn-net-run)** | INT8 Native | 50 | 42 | **84.00%** | **0.00% (Bit-Exact Parity)** |

---

## 2. Technical Investigation & Root Cause Log
* **Issue:** Initial batch runs of qnn-net-run reported 12% and 0% accuracy despite isolated single samples achieving correct predictions.
* **Root Cause 1 (Line Termination):** Windows CRLF (\r\n) in --input_list prevented parser loops from reading past intermediate lines. Solved by standardizing on Unix LF (\n).
* **Root Cause 2 (I/O Datatype Mismatch):** Without --use_native_input_files and --use_native_output_files, qnn-net-run expects 12,288 bytes of FP32 data per tensor. Providing 3,072 bytes caused 75% uninitialized buffer memory.
* **Resolution:** Passing --use_native_input_files and --use_native_output_files restored the pipeline to direct byte ingestion and generated output_native.raw as native uint8, yielding identical 84.00% accuracy.
