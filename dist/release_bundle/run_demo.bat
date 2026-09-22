@echo off
echo ===============================================================
echo   Qualcomm QNN C++ Edge Engine - Standalone Production Bundle
echo ===============================================================
echo.
echo [1] Running Single Image Inference (cat):
qnn_cifar10_engine.exe --image sample_images\test_0_class_3.png
echo.
echo [2] Running 100-iteration In-Memory Latency Benchmark:
qnn_cifar10_engine.exe --benchmark 100 --warmup 10
echo.
echo Demo execution complete.
pause