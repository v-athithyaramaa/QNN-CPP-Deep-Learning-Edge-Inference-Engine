cmake_minimum_required(VERSION 3.20)
project(qnn_cifar10_engine LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# QNN SDK Path
set(QNN_SDK_ROOT "D:/qairt/2.50.0.260828")
include_directories(
    ${QNN_SDK_ROOT}/include/QNN
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

# OpenCV 4.14.0 configuration
set(OpenCV_DIR "D:/opencv/build")
find_package(OpenCV REQUIRED)
message(STATUS "Found OpenCV: ${OpenCV_VERSION}")
include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(qnn_cifar10_engine
    src/main.cpp
)

target_link_libraries(qnn_cifar10_engine
    PRIVATE
    ${OpenCV_LIBS}
)

if (MSVC)
    target_compile_options(qnn_cifar10_engine PRIVATE /W4 /O2 /arch:AVX2)
endif()

# --- Standalone Deployment Packaging ---
install(TARGETS qnn_cifar10_engine RUNTIME DESTINATION bin)
install(FILES "${CMAKE_CURRENT_SOURCE_DIR}/output_context/cifar10_miniresnet_context.bin" DESTINATION bin)

# Collect Qualcomm Runtime DLLs
file(GLOB QNN_DLLS "${QAIRT_SDK_ROOT}/lib/x86_64-windows-msvc/*.dll")
install(FILES ${QNN_DLLS} DESTINATION bin)

# Collect OpenCV Runtime DLLs
file(GLOB OPENCV_DLLS "${OpenCV_DIR}/bin/*.dll")
install(FILES ${OPENCV_DLLS} DESTINATION bin)