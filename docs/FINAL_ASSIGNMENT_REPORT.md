# DL Assignment 3 - Production Qualcomm QNN C++ Edge Inference Engine: Asymmetric INT8 Quantization, Microarchitectural Profiling, and Double-Buffered Concurrency

**Author:** V Athithya Ramaa
**Target Hardware:** AMD Ryzen Multi-Core Processor (x86_64 Architecture, MSVC C++17 Toolchain)
**Acceleration Stack:** Qualcomm AI Engine Direct SDK (QNN v2.50.0 CPU Backend: QnnCpu.dll, QnnSystem.dll)
**Primary Dependencies:** OpenCV 4.x (Zero-Copy Image Ingest), Qualcomm QNN Core/System APIs, Windows Native Multimedia Timers

---

## 1. Executive Summary & Objective

In Assignment 1, I trained an edge-optimized 7-layer Mini-ResNet (~168k parameters) to 84.13% validation accuracy on CIFAR-10 and analyzed graph lowering through ONNX. In Assignment 2, I stripped away third-party library abstractions to write a bare-metal C++17 operator runtime from scratch, establishing baseline FP32 mathematical parity (<4.17e-7 max diff).

The objective of Assignment 3 is to elevate this engine to an industrial SDE-2 standard: building a production-grade C++ model deployment pipeline utilizing the Qualcomm Neural Network (QNN) SDK. The application must ingest an offline-quantized serialized context binary on the QNN CPU backend, integrate OpenCV for Windows for raw image ingest, execute multi-sample evaluation loops controlled via CLI flags, collect microsecond-level layer profiling alongside hardware execution cycles, and guarantee numerical cross-validation across runtimes.

---

## 2. Model Compilation & Offline Quantization Pipeline

To execute neural networks efficiently on edge hardware, the continuous 32-bit floating-point parameters from Assignment 1 and 2 must be lowered into discrete 8-bit integer domain representations.

### The Quantization Mathematics
The QNN CPU backend targets asymmetric uniform affine quantization (QNN_DATATYPE_UFIXED_POINT_8):
* q = clamp(round(x / Scale) + ZeroPoint, 0, 255)
* x_hat = Scale * (q - ZeroPoint)

Where:
* Scale (S) is the positive real step size between consecutive discrete levels.
* Zero-Point (Z) is the integer offset mapping real zero to an exact integer.

### Calibration Dynamics & Toolchain Flow
A curated 500-sample calibration subset across all 10 CIFAR-10 classes captures dynamic activation ranges. 
1. **qnn-onnx-converter**: Folds constants, maps activations, and generates .cpp, .bin, and .json graph artifacts.
2. **qnn-model-lib-generator.py**: Compiles the graph source into a relocatable dynamic library (model.dll).
3. **qnn-context-binary-generator**: Resolves backend kernels against QnnCpu.dll and serializes the complete compiled context into cifar10_miniresnet_context.bin.

> **[DIAGRAM 1: Model Compilation & Offline Ingestion Pipeline]**
> ![Diagram 1](assets/diagram1_compilation.png)

---

## 3. High-Performance C++ Application Architecture

The C++ inference engine (qnn_cifar10_engine.exe) is constructed around an extensible, decoupled architecture adhering to strict SDE-2 standards:
* **QnnDynamicLoader**: Dynamically resolves symbols from QnnCpu.dll and QnnSystem.dll at runtime.
* **QnnModelContext**: Deserializes context binaries and introspects tensor scales and zero-points.
* **ImagePreprocessor**: Implements zero-copy OpenCV ingestion directly into pinned memory buffers.
* **QnnProfiler**: Captures microsecond-level execution timestamps via QnnProfile handles.
* **AsyncPipeline**: A double-buffered ping-pong concurrent execution engine overlapping I/O and compute.
* **TelemetryExporter**: Emits structured JSON metrics (qnn_telemetry.json) and dashboard feeds.

> **[DIAGRAM 2: Modular Subsystems & Dynamic Linkage Layout]**
> ![Diagram 2](assets/diagram2_architecture.png)

---

## 4. Hardware-Oriented Memory Mechanics & Lifecycle Management

Building on Assignment 2's flat-memory design, the engine eliminates runtime heap allocations:
* **64-Byte Cacheline Alignment**: Pinned host buffers are allocated via _aligned_malloc on 64-byte boundaries, matching CPU cache lines.
  * Input Tensor: 3,072 bytes (uint8, Scale: 0.0165, Offset: 120)
  * Output Tensor: 10 bytes (uint8, Scale: 0.0942593, Offset: -109)
* **Zero-Copy Ingestion**: OpenCV writes normalized and quantized pixels directly into the client memory buffer bound to QNN_TENSORMEMTYPE_RAW.

> **[DIAGRAM 5: 64-Byte Aligned Tensor Memory & Cache Layout]**
> ![Diagram 5](assets/diagram5_memory.png)

---

## 5. End-to-End Application Lifecycle & State Machine

The engine progresses through three discrete phases:
1. **Startup & Warmup**: Parses CLI flags, resolves paths dynamically, instantiates context, and executes 5 warmup runs to clear OS page faults.
2. **Execution Routing**: Branches into Single-Image (--image), In-Memory Benchmark (--benchmark), Synchronous Loop, or Asynchronous Double-Buffering (--async).
3. **Wrap-Up & Teardown**: Dequantizes logits, calculates numerically stable softmax, evaluates top-1 accuracy vs. CSV, exports JSON telemetry, and frees resources in reverse order (Profile -> Context -> Backend -> DLLs).

> **[DIAGRAM 3: Execution Lifecycle State Machine]**
> ![Diagram 3](assets/diagram3_lifecycle.png)

---

## 6. Advanced Concurrency: Asynchronous Double-Buffering

In synchronous mode, the accelerator thread remains idle while waiting for OpenCV disk reads (307.71 FPS).
Our double-buffered async engine overlaps tasks using two pinned 64-byte buffers:
* **Worker Thread**: Preprocesses image N+1 into Slot B.
* **QNN Thread**: Executes graph on image N from Slot A.
* **Atomic Ping-Pong Swap**: Pointers are swapped via condition variables with zero memory copying.

| Execution Mode | Average Latency | Throughput | Resource Utilization |
| :--- | :--- | :--- | :--- |
| Synchronous Sequential | 3.250 ms | 307.71 FPS | High idle time; compute blocked on disk reads |
| Double-Buffered Async | 1.903 ms | 525.43 FPS (+70.75%) | Overlapped; I/O hidden behind graph execution |

> **[DIAGRAM 4: Concurrency Architecture & Sequence Execution]**
> ![Diagram 4](assets/diagram4_concurrency.png)

---

## 7. Numerical Parity & 3-Way Cross-Validation

The engine was cross-validated across three environments on 50 standardized test samples:
* **PyTorch / ONNX FP32**: 88.00% (Algorithmic ceiling)
* **Qualcomm qnn-net-run (INT8)**: 84.00% (-4.00% quantization delta)
* **Custom C++ Engine (INT8)**: 84.00% (Bit-exact match with Qualcomm reference runner, 0.00% delta)

---

## 8. Microarchitectural Profiling & Hardware Cycle Analysis

Rather than estimating CPU cycles via clock frequency multiplication, hardware cycles are captured directly using fenced __rdtsc intrinsics:
* **Spatial Convolutions**: Account for 92.00% of graph execution time.
* **Global Average Pooling**: Executes in just 0.019 ms (1.0% of graph time), proving the efficiency of replacing dense fully connected layers with GAP.
* Total graph execution averages 1.903 ms across on-device nodes.

---

## 9. In-Memory Latency Benchmark Suite

Isolating graph execution from filesystem I/O across 100 iterations (--benchmark 100):
* Mean Latency: 1.124 ms
* P50 Median Latency: 0.639 ms
* P90 Latency: 2.735 ms
* P99 Latency: 4.343 ms
* Sustained Throughput: 889.8 FPS

---

## 10. Interactive Production Telemetry & Dynamic Web Dashboard

The engine exports structured metrics to qnn_telemetry.json, fed directly into docs/dashboard/index.html:
* Real-time KPI banner (Accuracy, FPS, Latency).
* Layer-wise runtime budget horizontal bar chart.
* SLA percentile staircase (Min to P99).
* Searchable 50-sample verification matrix table.

> **[DIAGRAM 6: Telemetry, Profiling & Browser Dashboard Ingestion]**
> ![Diagram 6](assets/diagram6_telemetry.png)

---

## 11. Standalone Deployment Packaging & Portability

The project bundles a self-contained release in dist/release_bundle/ containing the compiled engine, runtime DLLs (QnnCpu.dll, QnnSystem.dll, opencv_world4100.dll), model binaries, sample test images, and an automated run_demo.bat launcher.

---

## 12. Verification & Deliverables Checklist

* [x] ONNX Model Artifact (model/cifar10_miniresnet.onnx)
* [x] Compiled QNN Context Binary (output_context/cifar10_miniresnet_context.bin)
* [x] Modular C++17 Application Source Code (src/)
* [x] CMake Build Configuration (CMakeLists.txt)
* [x] Layer-Wise Profiling & Cycles Measurement (src/QnnProfiler.hpp)
* [x] OpenCV Windows C++ Integration (src/ImagePreprocessor.hpp)
* [x] Multi-Sample Execution Loop & CLI Flags (--image, --benchmark, --async, --eval)
* [x] Application Architecture & Flow Diagrams (docs/ARCHITECTURE.md)
* [x] Standalone Portable Distribution Bundle (dist/release_bundle/)
* [x] Interactive Browser Telemetry Dashboard (docs/dashboard/)
