<div align="center">

# QNN CIFAR-10 Edge Inference Engine

**A Windows C++17, Qualcomm QNN CPU inference pipeline for an offline-quantized Mini-ResNet**

[Architecture](docs/ARCHITECTURE.md) · [Benchmark report](docs/TECHNICAL_BENCHMARK_REPORT.md) · [Telemetry dashboard](docs/dashboard/index.html) · [Release bundle](dist/release_bundle/)

</div>

This repository contains a production-oriented C++17 inference engine that loads a serialized Qualcomm AI Engine Direct (QAIRT/QNN) context, preprocesses CIFAR-10 images with OpenCV, executes an asymmetric INT8 Mini-ResNet on the QNN CPU backend, and exports accuracy, latency, profiling, and telemetry data.

The project is designed for Windows x86_64 and MSVC. It includes the ONNX source models, calibration data, generated QNN artifacts, native C++ runtime, test data, architecture documentation, audit reports, a browser dashboard, and a portable release bundle.

## Visual Overview

The project diagrams are intentionally shown here near the beginning so the execution pipeline can be understood before reading the implementation details. The images are stored in [`diagrams/`](diagrams/). There is no separate top-level `artifacts/` directory in the current repository; these six PNGs are the visual artifacts currently included.

### 1. Model calibration and compilation

![QNN model calibration and compilation](diagrams/QNN%20Model%20Calibration-Diagram%201.png)

### 2. Runtime architecture and dynamic linkage

![QNN runtime architecture](diagrams/QNN%20Model%20Calibration-Diagram%202.png)

### 3. Application execution lifecycle

![QNN execution lifecycle](diagrams/QNN%20Model%20Calibration-Diagram%203.png)

### 4. Asynchronous double-buffered execution

![QNN asynchronous double-buffered pipeline](diagrams/QNN%20Model%20Calibration-Diagram%204.png)

### 5. Quantized tensor memory and cache layout

![QNN tensor memory and cache layout](diagrams/QNN%20Model%20Calibration-Diagram%205.png)

### 6. Profiling, telemetry, and dashboard flow

![QNN telemetry and dashboard flow](diagrams/QNN%20Model%20Calibration-Diagram%206.png)

## What This Project Does

The executable performs the following end-to-end workflow:

1. Resolves `QnnCpu.dll`, `QnnSystem.dll`, the serialized context, and the test-image directory.
2. Dynamically loads QNN backend and system providers at runtime.
3. Reads context metadata and retrieves the compiled graph.
4. Allocates 64-byte-aligned input and output buffers.
5. Decodes and resizes images to 32x32 RGB with OpenCV.
6. Applies CIFAR-10 normalization and asymmetric UINT8 quantization directly into the input buffer.
7. Executes the QNN graph synchronously or through the optional double-buffered `--async` pipeline.
8. Dequantizes output logits, applies numerically stable softmax, and reports the top predictions.
9. Optionally evaluates predictions against `test_data/ground_truth.csv`.
10. Exports structured telemetry to `qnn_telemetry.json` and can feed the browser dashboard.

## Results At A Glance

The checked-in reports record the following measurements. Exact values depend on CPU, build configuration, cache state, and input set.

| Measurement | Result | Context |
| --- | ---: | --- |
| FP32 PyTorch/ONNX reference | 88.00% | 50-image reference set |
| QNN native INT8 reference | 84.00% | 42/50 images |
| Custom C++ INT8 engine | 84.00% | Bit-exact match with QNN reference |
| Continuous end-to-end throughput | 328.88 FPS | Includes image ingest and preprocessing |
| In-memory benchmark throughput | 889.8 FPS | `--benchmark 100`, excludes file I/O |
| Async pipeline throughput | 525.43 FPS | Double-buffered execution report |
| Reported P50 latency | 2.801 ms | 50-image audit run |
| Reported P99 latency | 8.246 ms | 50-image audit run |

For methodology and the full operator breakdown, see [`docs/TECHNICAL_BENCHMARK_REPORT.md`](docs/TECHNICAL_BENCHMARK_REPORT.md), [`docs/FINAL_ASSIGNMENT_REPORT.md`](docs/FINAL_ASSIGNMENT_REPORT.md), and [`docs/audit_logs/`](docs/audit_logs/).

## Requirements

### Required runtime and build tools

- Windows x86_64
- Visual Studio/MSVC with C++17 support
- CMake 3.20 or newer
- OpenCV 4.x with development headers and libraries
- Qualcomm AI Engine Direct / QNN SDK 2.50.0.260828, or a compatible SDK installation
- `QnnCpu.dll` and `QnnSystem.dll` available at runtime

The checked-in CMake configuration currently uses these local paths:

```cmake
QNN_SDK_ROOT = D:/qairt/2.50.0.260828
OpenCV_DIR   = D:/opencv/build
```

Update those values in [`CMakeLists.txt`](CMakeLists.txt) for another machine before configuring. The context binary was generated for the QNN graph and backend used by this project; regenerating it may require the matching QNN SDK tools.

## Quick Start

### Option A: Run the portable release bundle

The easiest way to try the project is the prebuilt bundle in [`dist/release_bundle/`](dist/release_bundle/). It contains the executable, QNN runtime DLLs, model context, sample images, and `run_demo.bat`.

From a Developer Command Prompt or PowerShell:

```powershell
Set-Location dist/release_bundle
.\run_demo.bat
```

If the shell does not execute batch files with the direct form, use:

```powershell
cmd /c .\run_demo.bat
```

### Option B: Run the checked-in build

The existing repository layout includes a Release build under `build_engine/Release/` on machines where the project has already been compiled:

```powershell
.\build_engine\Release\qnn_cifar10_engine.exe --help
```

A typical evaluation command from the repository root is:

```powershell
.\build_engine\Release\qnn_cifar10_engine.exe `
	--model output_context/cifar10_miniresnet_context.bin `
	--backend_dll D:/qairt/2.50.0.260828/lib/x86_64-windows-msvc/QnnCpu.dll `
	--system_dll D:/qairt/2.50.0.260828/lib/x86_64-windows-msvc/QnnSystem.dll `
	--test_dir test_data/images `
	--num_inputs 50 `
	--export_json qnn_telemetry.json
```

The executable also performs fallback path resolution for the local directory, `./models/`, `output_context/`, and the configured QNN SDK path. Explicit arguments are recommended for reproducible runs.

## Build From Source

1. Install the prerequisites and update the SDK/OpenCV paths in [`CMakeLists.txt`](CMakeLists.txt).
2. Configure a Visual Studio build directory:

```powershell
cmake -S . -B build_engine -G "Visual Studio 17 2022" -A x64
```

3. Build the Release target:

```powershell
cmake --build build_engine --config Release --target qnn_cifar10_engine
```

4. Run the executable with the model context and QNN DLL paths supplied as shown above.

The CMake target links OpenCV and compiles with MSVC warning level 4, optimization, and AVX2 enabled. The install rules place the executable and context under `bin/` and attempt to package QNN and OpenCV runtime DLLs.

## Command-Line Reference

Run `qnn_cifar10_engine.exe --help` for the same list printed by the executable.

| Option | Meaning | Default |
| --- | --- | --- |
| `--image <path>` | Run one image and print class probabilities | Disabled |
| `--model <path>` | Serialized QNN context binary | Auto-resolved |
| `--backend_dll <path>` | Path to `QnnCpu.dll` | Auto-resolved |
| `--system_dll <path>` | Path to `QnnSystem.dll` | Auto-resolved |
| `--test_dir <dir>` | Directory of evaluation images | Auto-resolved |
| `--num_inputs <N>` | Maximum number of directory inputs | `50` |
| `--benchmark <N>` | Run `N` in-memory benchmark iterations | `0` |
| `--async` | Enable producer/consumer double buffering | Off |
| `--warmup <N>` | Warmup executions before measurement | `5` |
| `--export_json <path>` | Telemetry output path | `qnn_telemetry.json` |
| `--no_profile` | Disable QNN detailed layer profiling | Profiling on |
| `--no_eval` | Disable ground-truth accuracy evaluation | Evaluation on |
| `--help`, `-h` | Print usage information | - |

Examples:

```powershell
# Single image classification
.\qnn_cifar10_engine.exe --image test_data/images/test_0_class_3.png

# Evaluate 50 images with asynchronous preprocessing
.\qnn_cifar10_engine.exe --test_dir test_data/images --num_inputs 50 --async

# Measure graph execution without filesystem image loading
.\qnn_cifar10_engine.exe --benchmark 100 --no_profile

# Run evaluation without accuracy comparison
.\qnn_cifar10_engine.exe --test_dir test_data/images --no_eval
```

## Model And Quantization Pipeline

The source model is a CIFAR-10 Mini-ResNet exported as ONNX:

- [`cifar10_miniresnet.onnx`](cifar10_miniresnet.onnx): primary model artifact
- [`cifar10_miniresnet_norm.onnx`](cifar10_miniresnet_norm.onnx): normalized model variant
- [`quant_overrides.json`](quant_overrides.json): quantization overrides
- [`calibration_data/`](calibration_data/): NHWC calibration inputs and input list
- [`calibration_data_norm/`](calibration_data_norm/): normalized calibration data
- [`model_quantized/`](model_quantized/): generated quantized model sources and metadata
- [`model_libs/`](model_libs/): generated model library artifacts
- [`output_context/cifar10_miniresnet_context.bin`](output_context/cifar10_miniresnet_context.bin): serialized executable context

The intended QNN conversion sequence is:

```text
ONNX model
	-> calibration data and quantization configuration
	-> qnn-onnx-converter
	-> generated model source, binary, and JSON metadata
	-> qnn-model-lib-generator.py
	-> model library
	-> qnn-context-binary-generator with QnnCpu.dll
	-> serialized context binary
```

The engine uses asymmetric affine quantization:

$$q = clamp(round(x / S) + Z, 0, 255)$$

and reconstructs output values with:

$$x_hat = S(q - Z)$$

where $S$ is the tensor scale and $Z$ is the zero-point read from the QNN context metadata. The runtime does not hard-code tensor dimensions or quantization parameters; [`QnnModelContext.hpp`](src/QnnModelContext.hpp) inspects the serialized graph and extracts them.

## Runtime Architecture

The main lifecycle is:

```text
CLI parsing
	-> dynamic QNN DLL loading
	-> backend creation
	-> context metadata inspection
	-> graph deserialization and retrieval
	-> aligned buffer allocation
	-> warmup
	-> image preprocessing / tensor quantization
	-> QNN graph execution
	-> dequantization and softmax
	-> evaluation and telemetry
	-> profiling and reverse-order cleanup
```

| Source file | Responsibility |
| --- | --- |
| [`src/main.cpp`](src/main.cpp) | CLI parsing, graph execution, evaluation, and application lifecycle |
| [`src/QnnDynamicLoader.hpp`](src/QnnDynamicLoader.hpp) | Windows DLL loading and QNN provider resolution |
| [`src/QnnModelContext.hpp`](src/QnnModelContext.hpp) | Context deserialization, graph retrieval, metadata inspection, and tensor buffers |
| [`src/ImagePreprocessor.hpp`](src/ImagePreprocessor.hpp) | OpenCV decode, RGB conversion, resize, normalization, and UINT8 quantization |
| [`src/AsyncPipeline.hpp`](src/AsyncPipeline.hpp) | Producer/consumer pipeline with two staging buffers |
| [`src/QnnProfiler.hpp`](src/QnnProfiler.hpp) | Detailed QNN profile events and layer-wise timing/cycle reporting |
| [`src/TelemetryExporter.hpp`](src/TelemetryExporter.hpp) | JSON telemetry generation and latency statistics |

### Memory and data path

Input tensors are prepared as 32x32 RGB data, giving 3,072 UINT8 values per sample. The model context supplies the input scale and zero-point. The input and output host buffers are allocated with 64-byte alignment and bound to QNN raw tensor memory. The inner inference loop reuses these buffers to avoid repeated allocations.

### Synchronous mode

The normal evaluation path processes an image sequentially: read, preprocess, execute, post-process, and record. This mode is simpler and is useful for correctness checks and end-to-end latency measurements.

### Asynchronous mode

With `--async`, a producer thread preprocesses the next image while the calling thread executes the current prepared buffer. Two slots alternate ownership through condition variables. This overlaps image I/O/preprocessing with graph execution and is intended for throughput-oriented workloads.

## Data Generation And Evaluation

[`generate_datasets.py`](generate_datasets.py) downloads the CIFAR-10 Python archive if it is not already present, then creates:

- 500 calibration samples, balanced at 50 samples per class, under `calibration_data/raw/`.
- `calibration_data/input_list.txt` for QNN calibration tooling.
- 50 PNG test images, balanced at 5 samples per class, under `test_data/images/`.
- `test_data/ground_truth.csv` containing image names, numeric labels, and class names.

Run it with Python and the `Pillow` and `numpy` packages installed:

```powershell
python generate_datasets.py
```

The CIFAR-10 classes are:

```text
0 airplane    1 automobile    2 bird    3 cat    4 deer
5 dog         6 frog          7 horse   8 ship   9 truck
```

The repository also contains [`test_images/`](test_images/) and several diagnostic input/output folders under [`test_data/`](test_data/).

## Telemetry Dashboard

Open [`docs/dashboard/index.html`](docs/dashboard/index.html) in a browser to inspect the saved telemetry dashboard. It displays:

- Top-1 accuracy and correct/total sample counts.
- Throughput and median latency.
- Graph execution latency.
- Per-operator runtime budget charts.
- Latency percentile distribution.
- Layer-wise profiling data.
- A searchable 50-sample verification table.

The dashboard loads [`docs/dashboard/data.js`](docs/dashboard/data.js). The native executable writes the portable JSON record to `qnn_telemetry.json`; copy or transform the latest runtime data into the dashboard's data feed when refreshing a report.

## Deployment

[`dist/release_bundle/`](dist/release_bundle/) is the portable deployment layout:

```text
dist/release_bundle/
	qnn_cifar10_engine.exe
	QnnCpu.dll
	QnnSystem.dll
	run_demo.bat
	models/
		cifar10_miniresnet_context.bin
	sample_images/
```

Keep the QNN DLLs discoverable beside the executable or pass explicit `--backend_dll` and `--system_dll` paths. Keep the context binary compatible with the QNN backend and SDK version used to build it.

## Troubleshooting

### `Failed to load QNN Backend DLL`

Confirm that `QnnCpu.dll` exists, that its dependent runtime DLLs are discoverable, and that the DLL architecture matches the executable (`x64`). Pass an explicit path with `--backend_dll`.

### `Failed to open context binary`

Pass the exact context path with `--model`, for example:

```powershell
--model output_context/cifar10_miniresnet_context.bin
```

### Images are not evaluated

Check that `--test_dir` points to a directory containing readable image files and that `test_data/ground_truth.csv` contains matching file names. Use `--image <path>` to isolate preprocessing and single-image execution.

### CMake cannot find OpenCV or QNN headers

Update `OpenCV_DIR` and `QNN_SDK_ROOT` in [`CMakeLists.txt`](CMakeLists.txt) to point to the installed development packages. Reconfigure after changing paths.

### Results differ from the reference runner

Check the input layout, native UINT8 input/output mode, tensor scale and zero-point, image normalization constants, and line endings in generated input lists. The audit notes in [`docs/audit_logs/`](docs/audit_logs/) document the previously investigated native-buffer and manifest-format failure modes.

## Repository Guide

```text
.
|-- CMakeLists.txt                    CMake/MSVC build configuration
|-- cifar10_miniresnet*.onnx          ONNX model artifacts
|-- generate_datasets.py              Calibration/test data generator
|-- quant_overrides.json              Quantization configuration
|-- calibration_data*/                Calibration raw inputs and manifests
|-- model_quantized/                  Generated quantized graph artifacts
|-- model_libs/                       Generated model libraries
|-- output_context/                   Serialized QNN context binary
|-- src/                              C++ runtime implementation
|-- test_data/                        Ground truth and diagnostic test assets
|-- test_images/                      Class-organized image samples
|-- diagrams/                         Visual project diagrams
|-- docs/                             Architecture, reports, dashboard, and audits
|-- dist/release_bundle/              Portable executable distribution
|-- build_engine/                     Generated Visual Studio build tree
`-- qnn_telemetry.json                Example/runtime telemetry output
```

## Documentation And Audit Artifacts

- [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md): detailed component decomposition, lifecycle, memory behavior, and execution modes.
- [`docs/FINAL_ASSIGNMENT_REPORT.md`](docs/FINAL_ASSIGNMENT_REPORT.md): project objective, quantization pipeline, concurrency, profiling, and deliverables.
- [`docs/TECHNICAL_BENCHMARK_REPORT.md`](docs/TECHNICAL_BENCHMARK_REPORT.md): cross-engine accuracy, latency percentiles, operator profiling, and incident analysis.
- [`docs/audit_logs/`](docs/audit_logs/): benchmark JSON, investigation logs, and audit reports.
- [`docs/dashboard/index.html`](docs/dashboard/index.html): interactive browser dashboard.

## Limitations And Scope

- The current application targets the QNN CPU backend and Windows x86_64. HTP/NPU deployment would require a compatible backend, context generation flow, and target device.
- The checked-in context binary and release DLLs are version-sensitive to the QNN SDK and Windows architecture.
- Reported benchmarks are evidence from the documented test runs, not universal performance guarantees.
- The engine is specialized for the included CIFAR-10 Mini-ResNet graph and its tensor contract, even though the context metadata path is designed to reduce hard-coded graph assumptions.

## License And Dataset Notice

No license file is currently included in the repository. Add the appropriate project license before distributing the source. The CIFAR-10 dataset is downloaded from the official Toronto CIFAR-10 distribution by [`generate_datasets.py`](generate_datasets.py); review and comply with its dataset terms when redistributing generated data.
