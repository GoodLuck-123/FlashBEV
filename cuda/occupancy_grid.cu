#include "occupancy_grid.h"
#include <device_launch_parameters.h>
#include <cstdio>

namespace flashbev {

// -----------------------------------------------------------------------------
// Kernel 1: 基于点的概率更新 (Atomic Update)
// -----------------------------------------------------------------------------
// 物理逻辑：
// 每个激光点根据其性质（障碍物 or 地面），向其归属的网格投出一票。
// 障碍物点投 +0.85 (Hit)，地面点投 -0.4 (Miss)。
// 多个点落入同一网格时，atomicAdd 自动处理累加竞争。
// -----------------------------------------------------------------------------
__global__ void UpdateOccupancyKernel(
    const int* __restrict__ voxel_index,     // [Input] D2 结果
    const uint8_t* __restrict__ is_obstacle, // [Input] D3 结果
    float* __restrict__ occupancy_grid,      // [Output] 概率地图
    uint32_t num_points)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= num_points) return;

    // 1. 获取该点归属的网格 ID
    int v_idx = voxel_index[idx];

    // 2. 如果点越界（-1），直接丢弃，不参与投票
    if (v_idx < 0) return;

    // 3. 查表决定更新值
    // 避免 if-else 分支指令，使用三目运算符，编译器可优化为选择指令 (CSEL)
    float update_val = (is_obstacle[idx] == 1) ? LO_HIT : LO_MISS;

    // 4. 原子累加
    // 5080 的 L2 Cache 原子单元吞吐量很高，
    // 但如果有大量地面点集中在同一个网格（如脚下），会有 Serialization（串行化）压力。
    // (D5阶段可以使用 Shared Memory 聚合来优化这里)
    atomicAdd(&occupancy_grid[v_idx], update_val);
}

// -----------------------------------------------------------------------------
// Kernel 2: 网格数值截断 (Clamping)
// -----------------------------------------------------------------------------
// 物理逻辑：
// 遍历所有网格，将 Log-Odds 值限制在 [-3.5, 3.5] 之间。
// -----------------------------------------------------------------------------
__global__ void ClampOccupancyKernel(
    float* __restrict__ occupancy_grid,
    int total_grids)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= total_grids) return;

    float val = occupancy_grid[idx];

    // 使用 CUDA 内置 fminf/fmaxf 指令，比 if 判断快
    val = fmaxf(val, LO_MIN); // 下限截断
    val = fminf(val, LO_MAX); // 上限截断

    occupancy_grid[idx] = val;
}

// -----------------------------------------------------------------------------
// Launcher Implementation
// -----------------------------------------------------------------------------
void LaunchOccupancyUpdate(
    const int* d_voxel_index,
    const uint8_t* d_is_obstacle,
    float* d_occupancy_grid,
    uint32_t num_points,
    int total_grids,
    cudaStream_t stream)
{
    // 1. 启动 Update Kernel (按点并行)
    int blockSize = 256;
    int numBlocksPoints = (num_points + blockSize - 1) / blockSize;
    
    UpdateOccupancyKernel<<<numBlocksPoints, blockSize, 0, stream>>>(
        d_voxel_index,
        d_is_obstacle,
        d_occupancy_grid,
        num_points
    );

    // 2. 启动 Clamp Kernel (按网格并行)
    // 典型的 BEV 网格数量为 250*250 = 62,500，远小于点数
    int numBlocksGrids = (total_grids + blockSize - 1) / blockSize;

    ClampOccupancyKernel<<<numBlocksGrids, blockSize, 0, stream>>>(
        d_occupancy_grid,
        total_grids
    );

    // 注意：不需要在此处 Synchronize，保持流的异步性
}

} // namespace flashbev