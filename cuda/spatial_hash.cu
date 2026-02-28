// cuda/spatial_hash.cu
#include "spatial_hash.h"
#include <cuda_runtime.h>
#include <math_functions.h>
#include <cstdio>
#include <iostream>
#include "FB_utils.h" // 统一的 CUDA 错误检查和 RAII 内存管理

namespace flashbev
{

    // __restrict__ 告诉编译器指针不别名，放心放进 L1 Cache
    __global__ void SpatialHashKernel(
        const float *__restrict__ soa_x,
        const float *__restrict__ soa_y,
        int *__restrict__ voxel_index,
        const uint32_t num_points,
        const float min_x,
        const float min_y,
        const float inv_resolution, // 物理优化：除法变乘法
        const int grid_w,
        const int grid_h)
    {
        int tid = blockIdx.x * blockDim.x + threadIdx.x;
        int stride = blockDim.x * gridDim.x;

        for (uint32_t idx = tid; idx < num_points; idx += stride)
        {
            // 1. 连续线程读取连续内存，最大化内存带宽利用率
            float x = soa_x[idx];
            float y = soa_y[idx];

            // 2. 物理坐标 -> 逻辑网格坐标转换 (使用底层的向下取整)
            // 预先在 Host 端算出 1/resolution (例如 1/0.2 = 5.0)，把高延迟的除法变成低延迟的乘法
            // floorf 是float32的版本，不加f的话默认是 double，算力消耗严重
            int grid_x = floorf((x - min_x) * inv_resolution);
            int grid_y = floorf((y - min_y) * inv_resolution);

            // 3. 越界检查掩码 (不包含任何 if 分支，计算布尔值)
            // 避免if分支导致的Warp Divergence，即使得同一Warp内的线程执行不同的指令路径，极大降低效率
            // 因为GPU是SIMD架构，所有线程必须执行相同的指令，如果有分支就会串行化执行不同分支的线程，性能大幅下降
            bool valid = (grid_x >= 0) && (grid_x < grid_w) &&
                         (grid_y >= 0) && (grid_y < grid_h);

            // 4. 空间展平：2D -> 1D
            // 此处就是我们所谓的“空间哈希”，也是为什么voxel_index / GRID_W 就是 grid_y，而 %GRID_W 就是 grid_x
            int flat_idx = grid_y * grid_w + grid_x;

            // 5. 无分支写回 (PTX指令：selp)
            // 如果 valid 为 true，写入 flat_idx；如果为 false，直接写 -1 扔进“垃圾桶”
            // 这种写法在 GPU 上效率极高，因为它避免了分支和条件跳转
            // 具体原因：执行三元运算符时，GPU 会生成一个条件选择汇编指令（selp），没有任何指令流分叉
            //         同一指令的所有线程（SIMT）在同一个时钟周期内，通过硬件电路的电平选择，完成了不同数据的写入
            voxel_index[idx] = valid ? flat_idx : -1;
        }
    }

    void LaunchSpatialHash(
        const float *d_soa_x,
        const float *d_soa_y,
        int *d_voxel_index,
        uint32_t num_points,
        float min_x,
        float min_y,
        float resolution,
        int grid_w,
        int grid_h,
        void *stream)
    {
        cudaStream_t cu_stream = static_cast<cudaStream_t>(stream);

        // 线程块配置：继续沿用 256 线程
        int threads = 256;
        int blocks = (num_points + threads - 1) / threads;

        // 物理优化技巧：除法在 GPU 中通常需要几十个时钟周期，而乘法只需要几个
        // 所以我们在 CPU 端先算好倒数传进去
        float inv_resolution = 1.0f / resolution;

        SpatialHashKernel<<<blocks, threads, 0, cu_stream>>>(
            d_soa_x, d_soa_y, d_voxel_index, num_points,
            min_x, min_y, inv_resolution, grid_w, grid_h);

        CHECK_CUDA(cudaGetLastError());
    }

} // namespace flashbev