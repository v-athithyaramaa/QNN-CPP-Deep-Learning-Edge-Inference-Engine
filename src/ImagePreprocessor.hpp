#pragma once

#include <string>
#include <vector>
#include <iostream>
#include <fstream>
#include <cmath>
#include <algorithm>
#include <filesystem>
#include <opencv2/opencv.hpp>
#include <immintrin.h>

namespace fs = std::filesystem;

namespace qnn_engine
{
    class ImagePreprocessor
    {
    public:
        // Direct in-place decode, normalize, and quantize into QNN-allocated memory
        static bool preprocessDirect(
            const std::string &imagePath,
            uint8_t *__restrict dstQuantBuffer,
            float scale,
            int32_t zeroPoint)
        {
            cv::Mat bgr = cv::imread(imagePath, cv::IMREAD_COLOR);
            if (bgr.empty())
                return false;

            cv::Mat rgb;
            cv::cvtColor(bgr, rgb, cv::COLOR_BGR2RGB);

            cv::Mat resized;
            if (rgb.rows != 32 || rgb.cols != 32)
            {
                cv::resize(rgb, resized, cv::Size(32, 32), 0, 0, cv::INTER_AREA);
            }
            else
            {
                resized = rgb;
            }

            // Raw pixel pointer (uint8_t interleaved RGB: R0, G0, B0, R1...)
            const uint8_t *src = resized.data;
            const float invScale = 1.0f / scale;
            const float inv255 = 1.0f / 255.0f;

            // Fused: (Pixel * (1/255) - Mean) / Std
            // x_norm = Pixel * alpha - beta
            // quant = round(x_norm / scale) + zeroPoint
            // Combined factor: Pixel * (alpha * invScale) - (beta * invScale) + zeroPoint
            const float mean[3] = {0.4914f, 0.4822f, 0.4465f};
            const float stdDev[3] = {0.2470f, 0.2435f, 0.2616f};

            float mul[3], sub[3];
            for (int c = 0; c < 3; ++c)
            {
                mul[c] = (inv255 / stdDev[c]) * invScale;
                sub[c] = (mean[c] / stdDev[c]) * invScale - static_cast<float>(zeroPoint);
            }

            // High-throughput fused pixel-quantization loop
            for (int i = 0; i < 1024; ++i)
            {
                int base = i * 3;
                float r = static_cast<float>(src[base + 0]) * mul[0] - sub[0];
                float g = static_cast<float>(src[base + 1]) * mul[1] - sub[1];
                float b = static_cast<float>(src[base + 2]) * mul[2] - sub[2];

                dstQuantBuffer[base + 0] = static_cast<uint8_t>(std::clamp(static_cast<int32_t>(std::round(r)), 0, 255));
                dstQuantBuffer[base + 1] = static_cast<uint8_t>(std::clamp(static_cast<int32_t>(std::round(g)), 0, 255));
                dstQuantBuffer[base + 2] = static_cast<uint8_t>(std::clamp(static_cast<int32_t>(std::round(b)), 0, 255));
            }

            return true;
        }
    };
}