#include <iostream>
#include <vector>
#include <string>
#include <cmath>
#include <numeric>
#include <algorithm>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <unordered_map>
#include <iomanip>
#include <cstring>
#include <intrin.h>

#include "QnnDynamicLoader.hpp"
#include "QnnModelContext.hpp"
#include "QnnProfiler.hpp"
#include "ImagePreprocessor.hpp"
#include "TelemetryExporter.hpp"
#include "AsyncPipeline.hpp"

namespace fs = std::filesystem;

static const char *CIFAR10_CLASSES[10] = {
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"};

void dequantizeOutput(const uint8_t *srcQuant, float *dstFloat, size_t count, float scale, int32_t offset)
{
    for (size_t i = 0; i < count; ++i)
    {
        dstFloat[i] = scale * (static_cast<float>(srcQuant[i]) - static_cast<float>(offset));
    }
}

void softmax(float *logits, size_t count)
{
    float maxVal = *std::max_element(logits, logits + count);
    float sum = 0.0f;
    for (size_t i = 0; i < count; ++i)
    {
        logits[i] = std::exp(logits[i] - maxVal);
        sum += logits[i];
    }
    float invSum = 1.0f / sum;
    for (size_t i = 0; i < count; ++i)
    {
        logits[i] *= invSum;
    }
}

std::unordered_map<std::string, int> loadGroundTruth(const std::string &csvPath)
{
    std::unordered_map<std::string, int> gtMap;
    std::ifstream file(csvPath);
    if (!file.is_open())
        return gtMap;

    std::string line;
    std::getline(file, line);
    while (std::getline(file, line))
    {
        std::stringstream ss(line);
        std::string imgName, labelStr, className;
        if (std::getline(ss, imgName, ',') && std::getline(ss, labelStr, ','))
        {
            try
            {
                gtMap[imgName] = std::stoi(labelStr);
            }
            catch (...)
            {
            }
        }
    }
    return gtMap;
}

std::string resolveExistingPath(const std::vector<std::string>& candidates)
{
    for (const auto& p : candidates)
    {
        if (fs::exists(p)) return p;
    }
    return candidates.empty() ? "" : candidates.front();
}

void printUsage(const char* binName)
{
    std::cout << "\nUsage: " << binName << " [OPTIONS]\n"
              << "Options:\n"
              << "  --image <path>        Execute single image inference and print class probabilities\n"
              << "  --model <path>        Path to serialized QNN context binary\n"
              << "  --backend_dll <path>  Path to QnnCpu.dll\n"
              << "  --system_dll <path>   Path to QnnSystem.dll\n"
              << "  --test_dir <dir>      Path to directory of test images\n"
              << "  --num_inputs <N>      Max number of directory inputs to evaluate (default: 50)\n"
              << "  --benchmark <N>       Run N continuous benchmark iterations on in-memory tensors\n"
              << "  --async               Enable asynchronous double-buffered producer-consumer pipeline\n"
              << "  --warmup <N>          Number of warmup executions (default: 5)\n"
              << "  --export_json <path>  Output path for structured telemetry JSON (default: qnn_telemetry.json)\n"
              << "  --no_profile          Disable QNN layer profiling\n"
              << "  --no_eval             Disable accuracy evaluation against ground truth\n"
              << "  --help, -h            Show this help message\n\n";
}

int main(int argc, char *argv[])
{
    std::string backendDll = "";
    std::string systemDll = "";
    std::string contextBin = "";
    std::string testDir = "";
    std::string singleImagePath = "";
    std::string gtCsvPath = "test_data/ground_truth.csv";
    std::string telemetryJsonPath = "qnn_telemetry.json";
    int numInputsToRun = 50;
    int benchmarkIterations = 0;
    int warmupIterations = 5;
    bool enableAccuracy = true;
    bool enableProfiling = true;
    bool enableAsyncPipeline = false;

    for (int i = 1; i < argc; ++i)
    {
        std::string arg = argv[i];
        if (arg == "--help" || arg == "-h")
        {
            printUsage(argv[0]);
            return 0;
        }
        else if (arg == "--image" && i + 1 < argc)
        {
            singleImagePath = argv[++i];
        }
        else if (arg == "--model" && i + 1 < argc)
        {
            contextBin = argv[++i];
        }
        else if (arg == "--backend_dll" && i + 1 < argc)
        {
            backendDll = argv[++i];
        }
        else if (arg == "--system_dll" && i + 1 < argc)
        {
            systemDll = argv[++i];
        }
        else if (arg == "--num_inputs" && i + 1 < argc)
        {
            numInputsToRun = std::stoi(argv[++i]);
        }
        else if (arg == "--benchmark" && i + 1 < argc)
        {
            benchmarkIterations = std::stoi(argv[++i]);
        }
        else if (arg == "--async")
        {
            enableAsyncPipeline = true;
        }
        else if (arg == "--eval")
        {
            enableAccuracy = true;
        }
        else if (arg == "--no_eval")
        {
            enableAccuracy = false;
        }
        else if (arg == "--test_dir" && i + 1 < argc)
        {
            testDir = argv[++i];
        }
        else if (arg == "--no_profile")
        {
            enableProfiling = false;
        }
        else if (arg == "--warmup" && i + 1 < argc)
        {
            warmupIterations = std::stoi(argv[++i]);
        }
        else if (arg == "--export_json" && i + 1 < argc)
        {
            telemetryJsonPath = argv[++i];
        }
    }

    // Dynamic path resolution for portable standalone execution
    if (backendDll.empty()) {
        backendDll = resolveExistingPath({
            "QnnCpu.dll",
            "./QnnCpu.dll",
            "D:/qairt/2.50.0.260828/lib/x86_64-windows-msvc/QnnCpu.dll"
        });
    }
    if (systemDll.empty()) {
        systemDll = resolveExistingPath({
            "QnnSystem.dll",
            "./QnnSystem.dll",
            "D:/qairt/2.50.0.260828/lib/x86_64-windows-msvc/QnnSystem.dll"
        });
    }
    if (contextBin.empty()) {
        contextBin = resolveExistingPath({
            "models/cifar10_miniresnet_context.bin",
            "./models/cifar10_miniresnet_context.bin",
            "output_context/cifar10_miniresnet_context.bin",
            "cifar10_miniresnet_context.bin"
        });
    }
    if (testDir.empty()) {
        testDir = resolveExistingPath({
            "sample_images",
            "test_data/images",
            "./sample_images"
        });
    }

    std::cout << "==============================================================\n";
    std::cout << "  QNN C++ Edge Inference Engine: CIFAR-10 Mini-ResNet (x86_64) \n";
    std::cout << "==============================================================\n";
    std::cout << "[Config] Model Context: " << contextBin << "\n";

    try
    {
        qnn_engine::QnnDynamicLoader loader;
        loader.load(backendDll, systemDll);

        const auto &qnn = loader.backend()->QNN_INTERFACE_VER_NAME;

        Qnn_BackendHandle_t backendHandle = nullptr;
        Qnn_ErrorHandle_t err = qnn.backendCreate(nullptr, nullptr, &backendHandle);
        if (err != QNN_SUCCESS || !backendHandle)
        {
            throw std::runtime_error("backendCreate failed with code: " + std::to_string(err));
        }
        std::cout << "[Engine] QnnCpu backend initialized.\n";

        qnn_engine::QnnModelContext model(loader);
        model.loadAndDeserialize(contextBin, backendHandle);

        auto &inBuffer = model.inputBuffer();
        auto &outBuffer = model.outputBuffer();

        std::unique_ptr<qnn_engine::QnnProfiler> profiler;
        if (enableProfiling)
        {
            profiler = std::make_unique<qnn_engine::QnnProfiler>(loader.backend(), backendHandle);
        }

        // Setup Tensors
        Qnn_Tensor_t inputTensor = QNN_TENSOR_INIT;
        inputTensor.version = QNN_TENSOR_VERSION_1;
        inputTensor.v1.id = inBuffer.id;
        inputTensor.v1.name = inBuffer.name.c_str();
        inputTensor.v1.type = QNN_TENSOR_TYPE_APP_WRITE;
        inputTensor.v1.dataFormat = QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER;
        inputTensor.v1.dataType = inBuffer.dataType;
        inputTensor.v1.quantizeParams.encodingDefinition = QNN_DEFINITION_DEFINED;
        inputTensor.v1.quantizeParams.quantizationEncoding = QNN_QUANTIZATION_ENCODING_SCALE_OFFSET;
        inputTensor.v1.quantizeParams.scaleOffsetEncoding.scale = inBuffer.scale;
        inputTensor.v1.quantizeParams.scaleOffsetEncoding.offset = inBuffer.zeroPoint;
        inputTensor.v1.rank = static_cast<uint32_t>(inBuffer.dimensions.size());
        inputTensor.v1.dimensions = inBuffer.dimensions.data();
        inputTensor.v1.memType = QNN_TENSORMEMTYPE_RAW;
        inputTensor.v1.clientBuf.data = inBuffer.hostBuffer;
        inputTensor.v1.clientBuf.dataSize = static_cast<uint32_t>(inBuffer.totalBytes);

        Qnn_Tensor_t outputTensor = QNN_TENSOR_INIT;
        outputTensor.version = QNN_TENSOR_VERSION_1;
        outputTensor.v1.id = outBuffer.id;
        outputTensor.v1.name = outBuffer.name.c_str();
        outputTensor.v1.type = QNN_TENSOR_TYPE_APP_READ;
        outputTensor.v1.dataFormat = QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER;
        outputTensor.v1.dataType = outBuffer.dataType;
        outputTensor.v1.quantizeParams.encodingDefinition = QNN_DEFINITION_DEFINED;
        outputTensor.v1.quantizeParams.quantizationEncoding = QNN_QUANTIZATION_ENCODING_SCALE_OFFSET;
        outputTensor.v1.quantizeParams.scaleOffsetEncoding.scale = outBuffer.scale;
        outputTensor.v1.quantizeParams.scaleOffsetEncoding.offset = outBuffer.zeroPoint;
        outputTensor.v1.rank = static_cast<uint32_t>(outBuffer.dimensions.size());
        outputTensor.v1.dimensions = outBuffer.dimensions.data();
        outputTensor.v1.memType = QNN_TENSORMEMTYPE_RAW;
        outputTensor.v1.clientBuf.data = outBuffer.hostBuffer;
        outputTensor.v1.clientBuf.dataSize = static_cast<uint32_t>(outBuffer.totalBytes);

        Qnn_Tensor_t inputs[] = {inputTensor};
        Qnn_Tensor_t outputs[] = {outputTensor};

        // MODE 1: Single Image Mode
        if (!singleImagePath.empty())
        {
            if (!fs::exists(singleImagePath))
            {
                throw std::runtime_error("Specified --image file not found: " + singleImagePath);
            }
            std::cout << "[Engine: Single Image Mode] Processing: " << singleImagePath << "\n";
            if (!qnn_engine::ImagePreprocessor::preprocessDirect(
                    singleImagePath,
                    static_cast<uint8_t *>(inBuffer.hostBuffer),
                    inBuffer.scale,
                    inBuffer.zeroPoint))
            {
                throw std::runtime_error("Preprocessing failed for " + singleImagePath);
            }

            auto start = std::chrono::high_resolution_clock::now();
            err = qnn.graphExecute(model.graphHandle(), inputs, 1, outputs, 1, nullptr, nullptr);
            auto end = std::chrono::high_resolution_clock::now();

            if (err != QNN_SUCCESS)
            {
                throw std::runtime_error("Execution failed with code: " + std::to_string(err));
            }

            double latencyMs = std::chrono::duration<double, std::milli>(end - start).count();

            std::vector<float> logits(10);
            dequantizeOutput(static_cast<const uint8_t *>(outBuffer.hostBuffer),
                             logits.data(), 10, outBuffer.scale, outBuffer.zeroPoint);
            softmax(logits.data(), 10);

            std::vector<int> sortedIndices(10);
            std::iota(sortedIndices.begin(), sortedIndices.end(), 0);
            std::sort(sortedIndices.begin(), sortedIndices.end(),
                      [&logits](int a, int b) { return logits[a] > logits[b]; });

            std::cout << "\n================ PREDICTION RESULTS ================\n";
            std::cout << "Top Prediction : " << CIFAR10_CLASSES[sortedIndices[0]]
                      << " (" << std::fixed << std::setprecision(2) << (logits[sortedIndices[0]] * 100.0f) << "%)\n";
            std::cout << "Latency        : " << latencyMs << " ms\n";
            std::cout << "----------------------------------------------------\n";
            std::cout << "Top 3 Classes:\n";
            for (int k = 0; k < 3; ++k)
            {
                int idx = sortedIndices[k];
                std::cout << "  [" << (k + 1) << "] " << std::left << std::setw(12)
                          << CIFAR10_CLASSES[idx] << " : "
                          << std::fixed << std::setprecision(2) << (logits[idx] * 100.0f) << "%\n";
            }
            std::cout << "====================================================\n";

            qnn.backendFree(backendHandle);
            return 0;
        }

        // MODE 2: Continuous In-Memory Benchmark Mode
        if (benchmarkIterations > 0)
        {
            std::cout << "[Engine: In-Memory Benchmark Mode] Target iterations: " << benchmarkIterations << "\n";
            std::fill_n(static_cast<uint8_t *>(inBuffer.hostBuffer), inBuffer.totalBytes, static_cast<uint8_t>(inBuffer.zeroPoint));

            std::cout << "Warming up caches (" << warmupIterations << " runs)...\n";
            for (int w = 0; w < warmupIterations; ++w)
            {
                qnn.graphExecute(model.graphHandle(), inputs, 1, outputs, 1, nullptr, nullptr);
            }

            std::vector<float> lats;
            lats.reserve(benchmarkIterations);
            uint64_t totalCycles = 0;

            std::cout << "Benchmarking graph execution...\n";
            for (int b = 0; b < benchmarkIterations; ++b)
            {
                _ReadWriteBarrier();
                auto cStart = __rdtsc();
                auto start = std::chrono::high_resolution_clock::now();

                qnn.graphExecute(model.graphHandle(), inputs, 1, outputs, 1, nullptr, nullptr);

                auto end = std::chrono::high_resolution_clock::now();
                auto cEnd = __rdtsc();
                _ReadWriteBarrier();

                float ms = static_cast<float>(std::chrono::duration<double, std::milli>(end - start).count());
                lats.push_back(ms);
                totalCycles += (cEnd - cStart);
            }

            std::sort(lats.begin(), lats.end());
            float minVal = lats.front();
            float maxVal = lats.back();
            float p50 = lats[lats.size() * 50 / 100];
            float p90 = lats[lats.size() * 90 / 100];
            float p99 = lats[lats.size() * 99 / 100];
            float sum = std::accumulate(lats.begin(), lats.end(), 0.0f);
            float mean = sum / lats.size();
            float fps = 1000.0f / mean;

            std::cout << "\n================ BENCHMARK SUMMARY ================\n";
            std::cout << "Completed Runs : " << benchmarkIterations << "\n";
            std::cout << "Mean Latency   : " << std::fixed << std::setprecision(3) << mean << " ms\n";
            std::cout << "P50 (Median)   : " << p50 << " ms\n";
            std::cout << "P90 Latency    : " << p90 << " ms\n";
            std::cout << "P99 Latency    : " << p99 << " ms\n";
            std::cout << "Min Latency    : " << minVal << " ms\n";
            std::cout << "Max Latency    : " << maxVal << " ms\n";
            std::cout << "Throughput     : " << std::setprecision(1) << fps << " FPS\n";
            std::cout << "Mean Cycles    : " << (totalCycles / benchmarkIterations) << "\n";
            std::cout << "===================================================\n";

            qnn.backendFree(backendHandle);
            return 0;
        }

        // MODE 3: Directory Evaluation Mode
        auto groundTruth = loadGroundTruth(gtCsvPath);
        std::vector<std::pair<std::string, int>> testImages;
        if (fs::exists(testDir))
        {
            for (const auto &entry : fs::directory_iterator(testDir))
            {
                if (entry.is_regular_file())
                {
                    std::string filename = entry.path().filename().string();
                    int gt = -1;
                    if (groundTruth.find(filename) != groundTruth.end())
                    {
                        gt = groundTruth[filename];
                    }
                    else
                    {
                        size_t pos = filename.find("class_");
                        if (pos != std::string::npos)
                        {
                            try
                            {
                                gt = std::stoi(filename.substr(pos + 6));
                            }
                            catch (...)
                            {
                            }
                        }
                    }
                    testImages.push_back({entry.path().string(), gt});
                }
            }
        }

        std::cout << "[Engine] Located " << testImages.size() << " test images in [" << testDir << "]\n";
        int totalToRun = std::min(numInputsToRun, static_cast<int>(testImages.size()));
        if (totalToRun == 0)
        {
            std::cout << "[Engine] No test images found. Exiting.\n";
            return 1;
        }

        if (warmupIterations > 0 && !testImages.empty())
        {
            std::cout << "[Engine] Warming up execution engine (" << warmupIterations << " runs)...\n";
            for (int w = 0; w < warmupIterations; ++w)
            {
                qnn_engine::ImagePreprocessor::preprocessDirect(
                    testImages[0].first,
                    static_cast<uint8_t *>(inBuffer.hostBuffer),
                    inBuffer.scale,
                    inBuffer.zeroPoint);
                qnn.graphExecute(model.graphHandle(), inputs, 1, outputs, 1, nullptr, nullptr);
            }
        }

        if (enableAsyncPipeline)
        {
            std::cout << "[Engine] Running ASYNCHRONOUS Double-Buffered Worker Pipeline for " << totalToRun << " inputs...\n\n";

            auto preprocFn = [](const std::string& path, uint8_t* dst, float scale, int32_t zp) {
                return qnn_engine::ImagePreprocessor::preprocessDirect(path, dst, scale, zp);
            };

            auto execFn = [&](const uint8_t* inData, qnn_engine::InferenceResult& res) {
                std::memcpy(inBuffer.hostBuffer, inData, inBuffer.totalBytes);

                _ReadWriteBarrier();
                auto cStart = __rdtsc();
                auto startWall = std::chrono::high_resolution_clock::now();

                qnn.graphExecute(model.graphHandle(), inputs, 1, outputs, 1, nullptr, nullptr);

                auto endWall = std::chrono::high_resolution_clock::now();
                auto cEnd = __rdtsc();
                _ReadWriteBarrier();

                res.graphLatencyMs = static_cast<float>(
                    std::chrono::duration<double, std::milli>(endWall - startWall).count());
                res.cycles = cEnd - cStart;

                std::vector<float> logits(10);
                dequantizeOutput(static_cast<const uint8_t *>(outBuffer.hostBuffer),
                                 logits.data(), 10, outBuffer.scale, outBuffer.zeroPoint);
                softmax(logits.data(), 10);

                int predClass = static_cast<int>(std::distance(logits.begin(), std::max_element(logits.begin(), logits.end())));
                res.predictedClass = predClass;
                res.confidence = logits[predClass];
                res.isMatch = (res.groundTruth != -1 && predClass == res.groundTruth);
                return true;
            };

            qnn_engine::AsyncInferencePipeline pipeline(
                inBuffer.totalBytes, inBuffer.scale, inBuffer.zeroPoint, preprocFn, execFn);

            pipeline.start();

            for (int i = 0; i < totalToRun; ++i)
            {
                qnn_engine::InferenceTask task;
                task.sampleIndex = i;
                task.imagePath = testImages[i].first;
                task.groundTruth = testImages[i].second;
                pipeline.pushTask(task);
            }

            auto wallStart = std::chrono::high_resolution_clock::now();

            std::thread consumerThread([&]() {
                pipeline.processInferenceQueue();
            });

            int correctPredictions = 0;
            int evaluatedCount = 0;
            std::vector<float> sampleLatenciesMs;
            uint64_t totalCycles = 0;

            qnn_engine::InferenceResult res;
            while (pipeline.popResult(res))
            {
                std::string filename = fs::path(res.imagePath).filename().string();
                if (res.groundTruth != -1)
                {
                    evaluatedCount++;
                    if (res.isMatch) correctPredictions++;
                }
                totalCycles += res.cycles;
                sampleLatenciesMs.push_back(res.graphLatencyMs);

                std::cout << "  [" << (res.sampleIndex + 1) << "/" << totalToRun << "] " << filename
                          << " -> Pred: " << CIFAR10_CLASSES[res.predictedClass]
                          << " (" << std::fixed << std::setprecision(1) << (res.confidence * 100.0f) << "%)"
                          << " | GT: " << (res.groundTruth >= 0 && res.groundTruth < 10 ? CIFAR10_CLASSES[res.groundTruth] : "N/A")
                          << " [" << (res.isMatch ? "CORRECT" : "MISMATCH") << "]"
                          << " | Latency: " << std::setprecision(3) << res.graphLatencyMs << " ms\n";
            }

            pipeline.stop();
            if (consumerThread.joinable()) consumerThread.join();

            auto wallEnd = std::chrono::high_resolution_clock::now();
            double totalWallMs = std::chrono::duration<double, std::milli>(wallEnd - wallStart).count();
            float asyncFps = static_cast<float>(totalToRun / (totalWallMs / 1000.0));

            std::cout << "\n===================================================\n";
            std::cout << "         ASYNC DOUBLE-BUFFERED RUN SUMMARY         \n";
            std::cout << "===================================================\n";
            std::cout << "Total Wall-Clock Time   : " << totalWallMs << " ms\n";
            std::cout << "Effective Async FPS     : " << asyncFps << " FPS\n";
            std::cout << "Top-1 Accuracy          : " << ((float)correctPredictions / evaluatedCount * 100.0f) << "%\n";
            std::cout << "===================================================\n";

            qnn.backendFree(backendHandle);
            return 0;
        }

        // Synchronous Execution
        int correctPredictions = 0;
        int evaluatedCount = 0;
        double totalInferenceTimeUs = 0.0;
        uint64_t totalCycles = 0;
        std::vector<float> sampleLatenciesMs;
        sampleLatenciesMs.reserve(totalToRun);

        std::cout << "[Engine] Running SYNCHRONOUS inference loop for " << totalToRun << " inputs...\n\n";

        for (int i = 0; i < totalToRun; ++i)
        {
            const auto &[imgPath, gt] = testImages[i];
            std::string filename = fs::path(imgPath).filename().string();

            if (!qnn_engine::ImagePreprocessor::preprocessDirect(
                    imgPath,
                    static_cast<uint8_t *>(inBuffer.hostBuffer),
                    inBuffer.scale,
                    inBuffer.zeroPoint))
            {
                continue;
            }

            Qnn_ProfileHandle_t activeProfile = (profiler && i == 0) ? profiler->handle() : nullptr;

            _ReadWriteBarrier();
            auto cStart = __rdtsc();
            auto startWall = std::chrono::high_resolution_clock::now();

            err = qnn.graphExecute(model.graphHandle(), inputs, 1, outputs, 1, activeProfile, nullptr);

            auto endWall = std::chrono::high_resolution_clock::now();
            auto cEnd = __rdtsc();
            _ReadWriteBarrier();

            if (err != QNN_SUCCESS)
            {
                std::cerr << "Graph execution failed on input " << i << " with error: " << err << "\n";
                continue;
            }

            double latencyUs = std::chrono::duration<double, std::micro>(endWall - startWall).count();
            float latencyMs = static_cast<float>(latencyUs / 1000.0);
            uint64_t cycles = cEnd - cStart;

            totalInferenceTimeUs += latencyUs;
            totalCycles += cycles;
            sampleLatenciesMs.push_back(latencyMs);

            std::vector<float> logits(10);
            dequantizeOutput(static_cast<const uint8_t *>(outBuffer.hostBuffer),
                             logits.data(), 10, outBuffer.scale, outBuffer.zeroPoint);
            softmax(logits.data(), 10);

            int predClass = static_cast<int>(std::distance(logits.begin(), std::max_element(logits.begin(), logits.end())));

            bool isMatch = (gt != -1 && predClass == gt);
            if (gt != -1)
            {
                evaluatedCount++;
                if (isMatch)
                    correctPredictions++;
            }

            std::cout << "  [" << (i + 1) << "/" << totalToRun << "] " << filename
                      << " -> Pred: " << CIFAR10_CLASSES[predClass]
                      << " (" << std::fixed << std::setprecision(1) << (logits[predClass] * 100.0f) << "%)"
                      << " | GT: " << (gt >= 0 && gt < 10 ? CIFAR10_CLASSES[gt] : "N/A")
                      << " [" << (isMatch ? "CORRECT" : "MISMATCH") << "]"
                      << " | Latency: " << std::setprecision(3) << latencyMs << " ms"
                      << " | Cycles: " << cycles << "\n";
        }

        double avgLatencyMs = (totalInferenceTimeUs / totalToRun) / 1000.0;
        uint64_t avgCycles = totalCycles / totalToRun;

        std::cout << "\n===================================================\n";
        std::cout << "               INFERENCE RUN SUMMARY               \n";
        std::cout << "===================================================\n";
        std::cout << "Total Samples Processed : " << totalToRun << "\n";
        std::cout << "Average Latency         : " << avgLatencyMs << " ms\n";
        std::cout << "Average Cycles          : " << avgCycles << " cycles\n";
        std::cout << "Throughput              : " << (1000.0 / avgLatencyMs) << " FPS\n";

        float finalAccuracy = 0.0f;
        if (enableAccuracy)
        {
            std::cout << "\n================ ACCURACY REPORT ==================\n";
            if (evaluatedCount > 0)
            {
                finalAccuracy = (static_cast<float>(correctPredictions) / evaluatedCount) * 100.0f;
                std::cout << "Evaluated Samples       : " << evaluatedCount << "\n";
                std::cout << "Correct Predictions     : " << correctPredictions << "\n";
                std::cout << "Top-1 Accuracy          : " << std::fixed << std::setprecision(2) << finalAccuracy << "%\n";
            }
            std::cout << "===================================================\n";
        }

        if (profiler)
        {
            profiler->dumpLayerWiseReport(avgLatencyMs, avgCycles);
        }

        qnn_engine::TelemetryData tData;
        tData.total_samples = totalToRun;
        tData.correct_samples = correctPredictions;
        tData.top1_accuracy = finalAccuracy;
        tData.avg_cycles = avgCycles;

        qnn_engine::TelemetryExporter::computeAndExport(telemetryJsonPath, tData, sampleLatenciesMs);

        qnn.backendFree(backendHandle);
    }
    catch (const std::exception &e)
    {
        std::cerr << "\n[Engine Fatal Error]: " << e.what() << "\n";
        return 1;
    }

    return 0;
}
