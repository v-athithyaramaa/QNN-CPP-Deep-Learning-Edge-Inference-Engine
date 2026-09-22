#pragma once

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <memory>
#include <cstring>
#include <stdexcept>
#include <cstdlib>

#include "QnnDynamicLoader.hpp"
#include "QnnTypes.h"
#include "QnnContext.h"
#include "QnnGraph.h"
#include "System/QnnSystemContext.h"

namespace qnn_engine
{

    struct TensorBuffer
    {
        std::string name;
        uint32_t id{0};
        Qnn_DataType_t dataType{QNN_DATATYPE_UNDEFINED};
        std::vector<uint32_t> dimensions;
        size_t totalBytes{0};
        float scale{1.0f};
        int32_t zeroPoint{0};
        void *hostBuffer{nullptr};

        ~TensorBuffer()
        {
            if (hostBuffer)
            {
                _aligned_free(hostBuffer);
                hostBuffer = nullptr;
            }
        }

        TensorBuffer() = default;

        TensorBuffer(const TensorBuffer &) = delete;
        TensorBuffer &operator=(const TensorBuffer &) = delete;

        TensorBuffer(TensorBuffer &&other) noexcept
        {
            moveFrom(std::move(other));
        }

        TensorBuffer &operator=(TensorBuffer &&other) noexcept
        {
            if (this != &other)
            {
                if (hostBuffer)
                {
                    _aligned_free(hostBuffer);
                }
                moveFrom(std::move(other));
            }
            return *this;
        }

        void allocateAligned(size_t bytes, size_t alignment = 64)
        {
            if (hostBuffer)
            {
                _aligned_free(hostBuffer);
            }
            totalBytes = bytes;
            hostBuffer = _aligned_malloc(bytes, alignment);
            if (!hostBuffer)
            {
                throw std::bad_alloc();
            }
            std::memset(hostBuffer, 0, bytes);
        }

    private:
        void moveFrom(TensorBuffer &&other) noexcept
        {
            name = std::move(other.name);
            id = other.id;
            dataType = other.dataType;
            dimensions = std::move(other.dimensions);
            totalBytes = other.totalBytes;
            scale = other.scale;
            zeroPoint = other.zeroPoint;
            hostBuffer = other.hostBuffer;
            other.hostBuffer = nullptr;
        }
    };

    class QnnModelContext
    {
    public:
        QnnModelContext(const QnnDynamicLoader &loader) : m_loader(loader) {}

        ~QnnModelContext()
        {
            destroy();
        }

        QnnModelContext(const QnnModelContext &) = delete;
        QnnModelContext &operator=(const QnnModelContext &) = delete;

        void loadAndDeserialize(const std::string &contextBinPath, Qnn_BackendHandle_t backendHandle)
        {
            m_backendHandle = backendHandle;

            std::ifstream file(contextBinPath, std::ios::binary | std::ios::ate);
            if (!file.is_open())
            {
                throw std::runtime_error("Failed to open context binary at: " + contextBinPath);
            }

            std::streamsize fileSize = file.tellg();
            file.seekg(0, std::ios::beg);

            m_binaryBuffer.resize(static_cast<size_t>(fileSize));
            if (!file.read(reinterpret_cast<char *>(m_binaryBuffer.data()), fileSize))
            {
                throw std::runtime_error("Failed to read context binary contents into memory.");
            }
            std::cout << "[QnnModelContext] Read context binary: " << fileSize << " bytes\n";

            inspectMetadata();

            const auto &qnn = m_loader.backend()->QNN_INTERFACE_VER_NAME;
            Qnn_ErrorHandle_t err = qnn.contextCreateFromBinary(
                m_backendHandle,
                nullptr,
                nullptr,
                m_binaryBuffer.data(),
                m_binaryBuffer.size(),
                &m_contextHandle,
                nullptr);

            if (err != QNN_SUCCESS || !m_contextHandle)
            {
                throw std::runtime_error("QnnContext_createFromBinary failed with error: " + std::to_string(err));
            }
            std::cout << "[QnnModelContext] QnnContext successfully instantiated from binary buffer.\n";

            err = qnn.graphRetrieve(m_contextHandle, m_graphName.c_str(), &m_graphHandle);
            if (err != QNN_SUCCESS || !m_graphHandle)
            {
                throw std::runtime_error("Failed to retrieve graph handle for: " + m_graphName);
            }
            std::cout << "[QnnModelContext] Graph [" << m_graphName << "] retrieved successfully.\n";

            allocateZeroCopyBuffers();
        }

        void destroy()
        {
            if (m_contextHandle && m_loader.backend())
            {
                m_loader.backend()->QNN_INTERFACE_VER_NAME.contextFree(m_contextHandle, nullptr);
                m_contextHandle = nullptr;
            }
            m_graphHandle = nullptr;
        }

        [[nodiscard]] Qnn_GraphHandle_t graphHandle() const { return m_graphHandle; }
        [[nodiscard]] TensorBuffer &inputBuffer() { return m_inputTensor; }
        [[nodiscard]] TensorBuffer &outputBuffer() { return m_outputTensor; }
        [[nodiscard]] const std::string &graphName() const { return m_graphName; }

    private:
        void inspectMetadata()
        {
            const auto &sys = m_loader.system()->QNN_SYSTEM_INTERFACE_VER_NAME;
            QnnSystemContext_Handle_t sysContext = nullptr;

            if (sys.systemContextCreate(&sysContext) != QNN_SUCCESS)
            {
                throw std::runtime_error("Failed to create QnnSystemContext handle.");
            }

            const QnnSystemContext_BinaryInfo_t *binaryInfo = nullptr;
            Qnn_ContextBinarySize_t binaryInfoSize = 0;

            Qnn_ErrorHandle_t err = sys.systemContextGetBinaryInfo(
                sysContext,
                reinterpret_cast<void *>(m_binaryBuffer.data()),
                m_binaryBuffer.size(),
                &binaryInfo,
                &binaryInfoSize);

            if (err != QNN_SUCCESS || !binaryInfo)
            {
                sys.systemContextFree(sysContext);
                throw std::runtime_error("Failed to extract binary info from context binary.");
            }

            std::cout << "[QnnModelContext] Metadata Binary Info Version: " << binaryInfo->version << "\n";

            uint32_t numGraphs = 0;
            QnnSystemContext_GraphInfo_t *graphs = nullptr;

            if (binaryInfo->version == QNN_SYSTEM_CONTEXT_BINARY_INFO_VERSION_1)
            {
                numGraphs = binaryInfo->contextBinaryInfoV1.numGraphs;
                graphs = binaryInfo->contextBinaryInfoV1.graphs;
            }
            else if (binaryInfo->version == QNN_SYSTEM_CONTEXT_BINARY_INFO_VERSION_2)
            {
                numGraphs = binaryInfo->contextBinaryInfoV2.numGraphs;
                graphs = binaryInfo->contextBinaryInfoV2.graphs;
            }
            else if (binaryInfo->version == QNN_SYSTEM_CONTEXT_BINARY_INFO_VERSION_3)
            {
                numGraphs = binaryInfo->contextBinaryInfoV3.numGraphs;
                graphs = binaryInfo->contextBinaryInfoV3.graphs;
            }
            else
            {
                sys.systemContextFree(sysContext);
                throw std::runtime_error("Unsupported BinaryInfo version: " + std::to_string(binaryInfo->version));
            }

            if (numGraphs == 0 || !graphs)
            {
                sys.systemContextFree(sysContext);
                throw std::runtime_error("Context binary reports 0 graphs or null graph pointer.");
            }

            // Extract graph details based on GraphInfo version
            const char *graphName = nullptr;
            uint32_t numInputs = 0;
            Qnn_Tensor_t *inputs = nullptr;
            uint32_t numOutputs = 0;
            Qnn_Tensor_t *outputs = nullptr;

            if (graphs[0].version == QNN_SYSTEM_CONTEXT_GRAPH_INFO_VERSION_1)
            {
                graphName = graphs[0].graphInfoV1.graphName;
                numInputs = graphs[0].graphInfoV1.numGraphInputs;
                inputs = graphs[0].graphInfoV1.graphInputs;
                numOutputs = graphs[0].graphInfoV1.numGraphOutputs;
                outputs = graphs[0].graphInfoV1.graphOutputs;
            }
            else if (graphs[0].version == QNN_SYSTEM_CONTEXT_GRAPH_INFO_VERSION_2)
            {
                graphName = graphs[0].graphInfoV2.graphName;
                numInputs = graphs[0].graphInfoV2.numGraphInputs;
                inputs = graphs[0].graphInfoV2.graphInputs;
                numOutputs = graphs[0].graphInfoV2.numGraphOutputs;
                outputs = graphs[0].graphInfoV2.graphOutputs;
            }
            else if (graphs[0].version == QNN_SYSTEM_CONTEXT_GRAPH_INFO_VERSION_3)
            {
                graphName = graphs[0].graphInfoV3.graphName;
                numInputs = graphs[0].graphInfoV3.numGraphInputs;
                inputs = graphs[0].graphInfoV3.graphInputs;
                numOutputs = graphs[0].graphInfoV3.numGraphOutputs;
                outputs = graphs[0].graphInfoV3.graphOutputs;
            }
            else
            {
                // Fallback for direct V1 access
                graphName = graphs[0].graphInfoV1.graphName;
                numInputs = graphs[0].graphInfoV1.numGraphInputs;
                inputs = graphs[0].graphInfoV1.graphInputs;
                numOutputs = graphs[0].graphInfoV1.numGraphOutputs;
                outputs = graphs[0].graphInfoV1.graphOutputs;
            }

            if (!graphName || numInputs == 0 || numOutputs == 0)
            {
                sys.systemContextFree(sysContext);
                throw std::runtime_error("Extracted graph metadata is incomplete or null.");
            }

            m_graphName = graphName;

            // Parse Input Tensor (V1 wrapper)
            const auto &inTensor = inputs[0];
            m_inputTensor.name = inTensor.v1.name;
            m_inputTensor.id = inTensor.v1.id;
            m_inputTensor.dataType = inTensor.v1.dataType;
            m_inputTensor.dimensions.clear();
            for (uint32_t i = 0; i < inTensor.v1.rank; ++i)
            {
                m_inputTensor.dimensions.push_back(inTensor.v1.dimensions[i]);
            }
            if (inTensor.v1.quantizeParams.quantizationEncoding == QNN_QUANTIZATION_ENCODING_SCALE_OFFSET)
            {
                m_inputTensor.scale = inTensor.v1.quantizeParams.scaleOffsetEncoding.scale;
                m_inputTensor.zeroPoint = inTensor.v1.quantizeParams.scaleOffsetEncoding.offset;
            }

            // Parse Output Tensor (V1 wrapper)
            const auto &outTensor = outputs[0];
            m_outputTensor.name = outTensor.v1.name;
            m_outputTensor.id = outTensor.v1.id;
            m_outputTensor.dataType = outTensor.v1.dataType;
            m_outputTensor.dimensions.clear();
            for (uint32_t i = 0; i < outTensor.v1.rank; ++i)
            {
                m_outputTensor.dimensions.push_back(outTensor.v1.dimensions[i]);
            }
            if (outTensor.v1.quantizeParams.quantizationEncoding == QNN_QUANTIZATION_ENCODING_SCALE_OFFSET)
            {
                m_outputTensor.scale = outTensor.v1.quantizeParams.scaleOffsetEncoding.scale;
                m_outputTensor.zeroPoint = outTensor.v1.quantizeParams.scaleOffsetEncoding.offset;
            }

            std::cout << "[QnnModelContext] Graph Name: " << m_graphName << "\n";
            std::cout << "  Input Tensor: " << m_inputTensor.name
                      << " [Scale: " << m_inputTensor.scale << ", Offset: " << m_inputTensor.zeroPoint << "]\n";
            std::cout << "  Output Tensor: " << m_outputTensor.name
                      << " [Scale: " << m_outputTensor.scale << ", Offset: " << m_outputTensor.zeroPoint << "]\n";

            sys.systemContextFree(sysContext);
        }

        void allocateZeroCopyBuffers()
        {
            size_t inputElements = 1;
            for (auto d : m_inputTensor.dimensions)
                inputElements *= d;
            m_inputTensor.allocateAligned(inputElements * sizeof(uint8_t), 64);

            size_t outputElements = 1;
            for (auto d : m_outputTensor.dimensions)
                outputElements *= d;
            m_outputTensor.allocateAligned(outputElements * sizeof(uint8_t), 64);

            std::cout << "[QnnModelContext] Allocated 64-byte aligned buffers: "
                      << m_inputTensor.totalBytes << " input bytes, "
                      << m_outputTensor.totalBytes << " output bytes.\n";
        }

        const QnnDynamicLoader &m_loader;
        std::vector<uint8_t> m_binaryBuffer;
        Qnn_BackendHandle_t m_backendHandle{nullptr};
        Qnn_ContextHandle_t m_contextHandle{nullptr};
        Qnn_GraphHandle_t m_graphHandle{nullptr};

        std::string m_graphName;
        TensorBuffer m_inputTensor;
        TensorBuffer m_outputTensor;
    };

}