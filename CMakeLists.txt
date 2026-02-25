cmake_minimum_required(VERSION 3.18 FATAL_ERROR)

# 启用 C++ 和原生 CUDA 语言支持
project(FlashBEV LANGUAGES CXX CUDA)

find_package(CUDAToolkit REQUIRED)

# 强制使用 C++ 17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# 生成最优机器码 (避免 PTX JIT 开销)
set(CMAKE_CUDA_ARCHITECTURES 120)

# 头文件搜索路径
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

# ==========================================
# 核心组件库 (Core Library)
# ==========================================
# 将所有 .cu 底层算子编译成一个静态库，避免多个可执行文件重复编译
add_library(flashbev_core STATIC
    cuda/data_mover.cu
    cuda/spatial_hash.cu
    cuda/ground_filter.cu
    cuda/occupancy_grid.cu
    # .cu 文件都加在这里
)

# 开启最高级别优化 (-O3) 和针对 CUDA 的极致优化 (-lineinfo 用于 Nsight 性能分析)
target_compile_options(flashbev_core PRIVATE 
    $<$<COMPILE_LANGUAGE:CXX>:-O3 -Wall>
    $<$<COMPILE_LANGUAGE:CUDA>:-O3 -use_fast_math -lineinfo>
)

# ==========================================
# 可执行文件 1：小工具 (AoS to SoA 独立验证)
# ==========================================
add_executable(data_SoA scripts/data_SoA.cpp)
target_link_libraries(data_SoA PRIVATE flashbev_core CUDA::cudart)

# ==========================================
# 可执行文件 2：空间索引化独立测试
# ==========================================
add_executable(spatial_hash scripts/spatial_hash.cpp)
target_link_libraries(spatial_hash PRIVATE flashbev_core CUDA::cudart)

# ==========================================
# 可执行文件 3：地面滤波独立测试
# ==========================================
add_executable(ground_filter scripts/ground_filter.cpp)
target_link_libraries(ground_filter PRIVATE flashbev_core CUDA::cudart)

# ==========================================
# 可执行文件 4：占据网格更新独立测试
# ==========================================
add_executable(occupancy scripts/occupancy.cpp)
target_link_libraries(occupancy PRIVATE flashbev_core CUDA::cudart)

# ==========================================
# 可执行文件 5：主线任务 (FlashBEV 实时感知系统)
# ==========================================
add_executable(flashbev src/main.cpp)
target_link_libraries(flashbev PRIVATE flashbev_core CUDA::cudart)