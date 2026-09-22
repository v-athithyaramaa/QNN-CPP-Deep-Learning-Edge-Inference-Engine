# Application Architecture and System Execution Flow

## High-Performance Qualcomm QNN C++ Inference Engine (x86_64)

This document details the software architecture, memory layout, dynamic loading interface, and execution lifecycle of the production C++ edge inference engine.

---

## Contents

- [System at a Glance](#system-at-a-glance)
- [System Block Diagram and Component Decomposition](#1-system-block-diagram-and-component-decomposition)
- [End-to-End Application Execution Flow](#2-end-to-end-application-execution-flow) - [Phase A: Initialization and Context Ingestion](#phase-a-engine-initialization-and-context-ingestion) - [Phase B: Synchronous and Asynchronous Execution](#phase-b-execution-engines-synchronous-vs-asynchronous) - [Phase C: Post-Processing, Metrics, and Teardown](#phase-c-post-processing-metrics-and-resource-teardown)
- [Key Technical Differentiators](#3-key-technical-differentiators)

## System at a Glance

| Concern         | Implementation                            | Primary outcome                              |
| --------------- | ----------------------------------------- | -------------------------------------------- |
| Image ingestion | `ImagePreprocessor` and OpenCV            | Converts input images to model-ready tensors |
| Inference       | Qualcomm QNN CPU backend                  | Executes the serialized INT8 model context   |
| Throughput      | `AsyncPipeline` with double buffering     | Overlaps preprocessing and graph execution   |
| Profiling       | `QnnProfiler` and hardware cycle counters | Captures microsecond-level execution metrics |
| Observability   | `TelemetryExporter`                       | Produces JSON telemetry and dashboard data   |

**Runtime path:** CLI startup -> dynamic library loading -> context deserialization -> tensor introspection -> preprocessing -> QNN execution -> post-processing -> telemetry export.

---

## 1. System Block Diagram and Component Decomposition

```text
                    +---------------------------------------+
                    |               main.cpp                |
                    |   (CLI Parsing & Lifecycle Director)  |
                    +-------------------+-------------------+
                                        | constructs & drives
                                        v
+----------------------------------------------------------------------------------------+
|                                APPLICATION ORCHESTRATOR                                |
|                                                                                        |
|  +-----------------------+  +------------------------+  +---------------------------+  |
|  |   ImagePreprocessor   |  |     AsyncPipeline      |  |        QnnProfiler        |  |
|  |  (OpenCV Ingest/Crop, |  | (Double-Buffered Ring   |  | (Microsecond Tracing &    |  |
|  |   Zero-Copy Pinned)   |  |   Worker Thread Pool)  |  |  __rdtsc Hardware Cycles) |  |
|  +-----------+-----------+  +-----------+------------+  +-------------+-------------+  |
|              |                          |                             |                |
|              v                          v                             v                |
|  +-----------------------+  +------------------------+  +---------------------------+  |
|  |  Pinned Host Buffers  |  |   QnnDynamicLoader     |  |     TelemetryExporter     |  |
|  |  (64-Byte Aligned     |  |  (Function Tables for  |  |  (qnn_telemetry.json &    |  |
|  |   Direct Int8 Tensor) |  |   QnnCpu & QnnSystem)  |  |   HTML Dashboard Feeder)  |  |
|  +-----------+-----------+  +-----------+------------+  +-------------+-------------+  |
+--------------+--------------------------+-----------------------------+----------------+
               |                          |                             |
               v                          v                             v
+--------------------------+  +------------------------+  +---------------------------+
|     QUALCOMM LIBRARIES   |  |   SERIALIZED CONTEXT   |  |     AUDIT & DASHBOARD     |
|   * QnnCpu.dll (Backend) |  |   cifar10_miniresnet_  |  |   * docs/dashboard/       |
|   * QnnSystem.dll (Meta) |  |   context.bin (INT8)   |  |   * docs/audit_logs/      |
+--------------------------+  +------------------------+  +---------------------------+
```

### Component Responsibilities

| Component           | Responsibility                                            |
| ------------------- | --------------------------------------------------------- |
| `main.cpp`          | Parses CLI options and directs application lifecycle      |
| `ImagePreprocessor` | Decodes, crops, normalizes, and prepares image tensors    |
| `AsyncPipeline`     | Coordinates worker threads and ping-pong buffer ownership |
| `QnnDynamicLoader`  | Loads QNN shared libraries and resolves function tables   |
| `QnnProfiler`       | Measures execution timing and hardware cycle counts       |
| `TelemetryExporter` | Writes runtime metrics for the dashboard and audit trail  |

---

## 2. End-to-End Application Execution Flow

### Phase A: Engine Initialization and Context Ingestion

```text
[Start CLI Execution]
        |
        v
[Resolve Dynamic Paths] ---> Fallback Search: Local Bundle (./) -> Custom Args -> SDK Root
        |
        v
[Load DLLs & Function Pointers] ---> LoadLibrary('QnnCpu.dll', 'QnnSystem.dll')
        |
        v
[Deserialize Context Binary] ---> qnnSystemInterface.systemContextCreateFromBinary()
        |
        v
[Tensor Introspection] ---> Auto-extract Input/Output Quantization Scales & Zero-Points
        |
        v
[Allocate Pinned Buffers] ---> _aligned_malloc(64-byte alignment, 3072 input, 10 output bytes)
        |
        v
[Warmup & Cache Staging] ---> Execute 5 warmup iterations to eliminate initial OS page faults
```

| Step                            | Purpose                                           |
| ------------------------------- | ------------------------------------------------- |
| Resolve dynamic paths           | Finds the local bundle, custom paths, or SDK root |
| Load DLLs and function pointers | Initializes the QNN CPU and system interfaces     |
| Deserialize context binary      | Restores the compiled INT8 graph                  |
| Introspect tensors              | Reads input/output quantization parameters        |
| Allocate pinned buffers         | Creates aligned input and output storage          |
| Warm up the engine              | Removes first-run page-fault and cache effects    |

---

### Phase B: Execution Engines (Synchronous vs. Asynchronous)

```text
1. SYNCHRONOUS SEQUENTIAL LOOP (Standard Mode)
   OpenCV Decode (2.1 ms) ---> Quantize/Copy ---> QNN graphExecute (0.9 ms) ---> Dequantize/Softmax
   Total Frame Time: ~3.0 ms | Throughput: ~307 FPS

2. ASYNCHRONOUS DOUBLE-BUFFERED PIPELINE (High-Throughput Mode: --async)
   Thread 1 (I/O & OpenCV):    [ Preprocess Image N+1 into Slot B ] ---+ (Signal CV)
                                                                       |
   Thread 2 (Qualcomm QNN):    [ Execute QNN Graph on Slot A ] --------+---> Ping-Pong Swap
   Throughput: 525.43 FPS (+70.7% speedup)
```

| Mode      | Behavior                                                          | Reported performance        |
| --------- | ----------------------------------------------------------------- | --------------------------- |
| Standard  | Decodes, preprocesses, executes, and post-processes sequentially  | ~3.0 ms per frame, ~307 FPS |
| `--async` | Overlaps I/O and preprocessing with QNN execution using two slots | 525.43 FPS, +70.7% speedup  |

---

### Phase C: Post-Processing, Metrics, and Resource Teardown

```text
[Quantized Logits (uint8)]
        |
        v
[Dequantize to FP32] ---> Logit[i] = Scale * (src[i] - ZeroPoint)
        |
        v
[Numerically Stable Softmax] ---> Prob[i] = exp(Logit[i] - Max) / Sum
        |
        v
[Top-1 Argmax & Top-3 Probabilities]
        |
        v
[Evaluate against ground_truth.csv] (Controlled via runtime flags --eval / --no_eval)
        |
        v
[Export Telemetry Payload] ---> qnn_telemetry.json & docs/dashboard/data.js
        |
        v
[Resource Teardown (Reverse Order)]
   1. qnn.profileFree()
   2. qnn.graphExecute teardown
   3. qnn.contextFree()
   4. qnn.backendFree()
   5. FreeLibrary(QnnCpu.dll, QnnSystem.dll)
```

> **Teardown rule:** Resources are released in reverse dependency order so profiling, graph, context, backend, and library handles remain valid until their dependents are finished.

---

## 3. Key Technical Differentiators

1. **Hardware-fenced cycle counters:** Cycle counts are acquired directly using `__rdtsc` paired with compiler memory barriers (`_ReadWriteBarrier()`), capturing CPU execution cycles rather than software-estimated averages.
2. **Zero dynamic allocation in the inner loop:** Input and output tensors are pre-allocated on 64-byte boundaries at startup, eliminating heap fragmentation and cache-line splits during inference.
3. **Double-buffered asynchronous pipeline:** OpenCV preprocessing and Qualcomm graph execution run through concurrent worker queues, scaling processing speed past 525 FPS.
