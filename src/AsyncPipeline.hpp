#pragma once

#include <iostream>
#include <vector>
#include <string>
#include <queue>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <functional>
#include <chrono>

namespace qnn_engine {

struct InferenceTask {
    std::string imagePath;
    int groundTruth;
    int sampleIndex;
};

struct InferenceResult {
    int sampleIndex;
    std::string imagePath;
    int groundTruth;
    int predictedClass;
    float confidence;
    bool isMatch;
    float e2eLatencyMs;
    float graphLatencyMs;
    uint64_t cycles;
};

// Double-buffered frame slot for ping-pong memory staging
struct BufferSlot {
    std::vector<uint8_t> buffer;
    InferenceTask task;
    bool ready = false;
};

class AsyncInferencePipeline {
public:
    using PreprocessFunc = std::function<bool(const std::string&, uint8_t*, float, int32_t)>;
    using ExecuteFunc = std::function<bool(const uint8_t*, InferenceResult&)>;

    AsyncInferencePipeline(size_t bufferSizeBytes,
                           float scale,
                           int32_t zeroPoint,
                           PreprocessFunc preproc,
                           ExecuteFunc exec)
        : m_bufferSize(bufferSizeBytes),
          m_scale(scale),
          m_zeroPoint(zeroPoint),
          m_preprocessor(preproc),
          m_executor(exec),
          m_stop(false)
    {
        // Allocate two pinned buffers for double-buffering
        m_slots[0].buffer.resize(bufferSizeBytes);
        m_slots[1].buffer.resize(bufferSizeBytes);
    }

    ~AsyncInferencePipeline() {
        stop();
    }

    void start() {
        m_producerThread = std::thread(&AsyncInferencePipeline::producerLoop, this);
    }

    void pushTask(const InferenceTask& task) {
        {
            std::unique_lock<std::mutex> lock(m_taskMutex);
            m_tasks.push(task);
        }
        m_taskCv.notify_one();
    }

    void stop() {
        m_stop.store(true);
        m_taskCv.notify_all();
        m_queueCv.notify_all();
        if (m_producerThread.joinable()) {
            m_producerThread.join();
        }
    }

    bool popResult(InferenceResult& res) {
        std::unique_lock<std::mutex> lock(m_resultMutex);
        if (m_results.empty() && m_stop.load() && m_tasks.empty() && !m_hasStagedSlot) {
            return false;
        }
        m_resultCv.wait(lock, [this]() {
            return !m_results.empty() || (m_stop.load() && m_tasks.empty() && !m_hasStagedSlot);
        });

        if (m_results.empty()) return false;
        res = m_results.front();
        m_results.pop();
        return true;
    }

    // Run execution loop on the calling thread (Consumer / QNN Thread)
    void processInferenceQueue() {
        while (true) {
            BufferSlot* currentSlot = nullptr;
            {
                std::unique_lock<std::mutex> lock(m_queueMutex);
                m_queueCv.wait(lock, [this]() {
                    return m_hasStagedSlot || m_stop.load();
                });

                if (!m_hasStagedSlot && m_stop.load()) {
                    break;
                }

                currentSlot = &m_slots[m_consumerSlotIdx];
                m_hasStagedSlot = false;
            }

            // Execute inference directly on current slot
            InferenceResult res;
            res.sampleIndex = currentSlot->task.sampleIndex;
            res.imagePath = currentSlot->task.imagePath;
            res.groundTruth = currentSlot->task.groundTruth;

            m_executor(currentSlot->buffer.data(), res);

            // Signal producer that current slot has been consumed
            {
                std::unique_lock<std::mutex> lock(m_queueMutex);
                m_consumerSlotIdx = 1 - m_consumerSlotIdx;
            }
            m_queueCv.notify_one();

            // Push completed result
            {
                std::unique_lock<std::mutex> lock(m_resultMutex);
                m_results.push(res);
            }
            m_resultCv.notify_one();
        }

        m_resultCv.notify_all();
    }

private:
    void producerLoop() {
        while (!m_stop.load()) {
            InferenceTask task;
            {
                std::unique_lock<std::mutex> lock(m_taskMutex);
                m_taskCv.wait(lock, [this]() {
                    return !m_tasks.empty() || m_stop.load();
                });

                if (m_stop.load() && m_tasks.empty()) break;

                task = m_tasks.front();
                m_tasks.pop();
            }

            // Wait until the staging slot is free for writing
            {
                std::unique_lock<std::mutex> lock(m_queueMutex);
                m_queueCv.wait(lock, [this]() {
                    return !m_hasStagedSlot || m_stop.load();
                });
                if (m_stop.load()) break;
            }

            BufferSlot& slot = m_slots[m_producerSlotIdx];
            slot.task = task;

            // Preprocess directly into pinned memory
            m_preprocessor(task.imagePath, slot.buffer.data(), m_scale, m_zeroPoint);

            // Mark slot as ready and notify consumer
            {
                std::unique_lock<std::mutex> lock(m_queueMutex);
                m_hasStagedSlot = true;
                m_producerSlotIdx = 1 - m_producerSlotIdx;
            }
            m_queueCv.notify_one();
        }
    }

    size_t m_bufferSize;
    float m_scale;
    int32_t m_zeroPoint;
    PreprocessFunc m_preprocessor;
    ExecuteFunc m_executor;

    std::atomic<bool> m_stop;

    BufferSlot m_slots[2];
    int m_producerSlotIdx = 0;
    int m_consumerSlotIdx = 0;
    bool m_hasStagedSlot = false;

    std::queue<InferenceTask> m_tasks;
    std::mutex m_taskMutex;
    std::condition_variable m_taskCv;

    std::mutex m_queueMutex;
    std::condition_variable m_queueCv;

    std::queue<InferenceResult> m_results;
    std::mutex m_resultMutex;
    std::condition_variable m_resultCv;

    std::thread m_producerThread;
};

} // namespace qnn_engine
