// cuda/ground_filter.cu
#include "ground_filter.h"
#include <cuda_runtime.h>
#include <math_functions.h>
#include "FB_utils.h"

namespace flashbev {

// ==============================================================================
// 物理级黑客技巧：手写单精度浮点数的 atomicMin
// CUDA 硬件底层原生支持 int 的 CAS，我们通过寄存器位强转欺骗硬件，完成浮点数原子比较
// 现代 GPU (如 RTX 5080) 会在 L2 Cache 层级处理这个请求，极大降低了显存带宽压力
// ==============================================================================
__device__ __forceinline__ void atomicMinFloat(float* address, float val) {
    int* address_as_int = (int*)address;
    int old = *address_as_int;
    int assumed;
    
    do {
        assumed = old;
        // 如果 val 更小，我们就准备把更小的值的位模式写入
        // fminf 会在寄存器级极速比较出更小的浮点数
        // __float_as_int 和 __int_as_float 只是编译器层面的重解释转换，0 时钟周期开销
        old = atomicCAS(address_as_int, assumed,
                        __float_as_int(fminf(val, __int_as_float(assumed))));
        // 如果期间有其他线程改了这个地址，old 会不等于 assumed，循环重试（无锁化自旋）
    } while (assumed != old);
}

// ------------------------------------------------------------------
// Kernel 1: 极速显存重置 (代替耗时的 cudaMemset 同步调用)
// 将网格初始化为物理上的极高处 (例如 100.0 米)
// ------------------------------------------------------------------
__global__ void InitMinZGridKernel(
    float* __restrict__ min_z_grid, 
    const int total_grids) 
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int idx = tid; idx < total_grids; idx += stride) {
        min_z_grid[idx] = 100.0f; // 雷达扫不到 100 米高空的地板
    }
}

// ------------------------------------------------------------------
// Kernel 2: 原子化收集最低点
// ------------------------------------------------------------------
__global__ void ComputeMinZKernel(
    const float* __restrict__ soa_z,
    const int* __restrict__ voxel_index,
    float* __restrict__ min_z_grid,
    const uint32_t num_points) 
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (uint32_t idx = tid; idx < num_points; idx += stride) {
        int v_idx = voxel_index[idx];
        // 遇到越界点 (-1) 直接跳过。由于越界点是极少数，这里的 if 造成的 Warp Divergence 影响微乎其微
        if (v_idx != -1) {
            float z = soa_z[idx];
            atomicMinFloat(&min_z_grid[v_idx], z);
        }
    }
}

// ------------------------------------------------------------------
// Kernel 3: 高度差滤波（无分支暴力写出）
// ------------------------------------------------------------------
__global__ void FilterObstaclesKernel(
    const float* __restrict__ soa_z,
    const int* __restrict__ voxel_index,
    const float* __restrict__ min_z_grid,
    uint8_t* __restrict__ is_obstacle,
    const uint32_t num_points,
    const float threshold) 
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (uint32_t idx = tid; idx < num_points; idx += stride) {
        int v_idx = voxel_index[idx];
        
        // 核心优化：避免 if-else 带来的串行化惩罚
        // 如果是有效点，计算高度差是否大于阈值；如果是越界点，直接判为 0 (非障碍物)
        bool is_valid = (v_idx != -1);
        float min_z = is_valid ? min_z_grid[v_idx] : 0.0f;
        float z = soa_z[idx];
        
        // C++ 的比较运算符生成布尔值，直接静态转型为 uint8_t 写入 Global Memory
        // 没有任何指令分叉，完美压榨显存带宽
        is_obstacle[idx] = static_cast<uint8_t>(is_valid && (z - min_z > threshold));
    }
}

// ------------------------------------------------------------------
// 统一启动接口
// ------------------------------------------------------------------
void LaunchGroundFilter(
    const float* d_soa_z, 
    const int* d_voxel_index, 
    float* d_min_z_grid, 
    uint8_t* d_is_obstacle, 
    uint32_t num_points, 
    int grid_w, 
    int grid_h, 
    float height_threshold, 
    void* stream) 
{
    cudaStream_t cu_stream = static_cast<cudaStream_t>(stream);
    int threads = 256;

    // 1. 初始化 Grid (网格数量级别的并行)
    int total_grids = grid_w * grid_h;
    int blocks_grid = (total_grids + threads - 1) / threads;
    InitMinZGridKernel<<<blocks_grid, threads, 0, cu_stream>>>(d_min_z_grid, total_grids);

    // 2. 收集最低点 (点云数量级别的并行)
    int blocks_pts = (num_points + threads - 1) / threads;
    ComputeMinZKernel<<<blocks_pts, threads, 0, cu_stream>>>(
        d_soa_z, d_voxel_index, d_min_z_grid, num_points
    );

    // 3. 高度滤波 (点云数量级别的并行)
    FilterObstaclesKernel<<<blocks_pts, threads, 0, cu_stream>>>(
        d_soa_z, d_voxel_index, d_min_z_grid, d_is_obstacle, num_points, height_threshold
    );

    CHECK_CUDA(cudaGetLastError());
}

} // namespace flashbev