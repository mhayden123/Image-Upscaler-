cmake_minimum_required(VERSION 3.22.1)
project(upscaler LANGUAGES CXX)

# NCNN prebuilt for Android with Vulkan GPU support.
# Downloaded by scripts/fetch_ncnn.sh (CI does this automatically).
set(ncnn_DIR ${CMAKE_SOURCE_DIR}/ncnn-android-vulkan/${ANDROID_ABI}/lib/cmake/ncnn)
if(NOT EXISTS ${ncnn_DIR})
    message(FATAL_ERROR "NCNN not found at ${ncnn_DIR}. Run scripts/fetch_ncnn.sh first.")
endif()
find_package(ncnn REQUIRED)

add_library(upscaler SHARED upscaler_jni.cpp upscale_core.cpp)
target_compile_features(upscaler PRIVATE cxx_std_17)
target_compile_options(upscaler PRIVATE -O3 -Wall)
target_link_libraries(upscaler ncnn jnigraphics android log)
