#ifndef FLASHBEV_OCCUPANCY_GRID_H
#define FLASHBEV_OCCUPANCY_GRID_H

#include <cuda_runtime.h>
#include <cstdint>

namespace flashbev
{

    // -----------------------------------------------------------------------------
    // D4: 占用栅格核心参数 (Log-Odds Policy)
    // -----------------------------------------------------------------------------
    // 物理意义：
    // Log-Odds L = log(p / (1 - p))
    // 初始状态 p=0.5 -> L=0
    //
    // 设定策略：
    // 1. 击中(Hit): 观测到障碍物，置信度增加。p_hit ≈ 0.7 => log(2.33) ≈ 0.85
    // 2. 错过(Miss): 观测到地面，置信度减少。p_miss ≈ 0.4 => log(0.66) ≈ -0.4
    // 3. 饱和截断(Clamp): 防止数值无限累加溢出，限制在 [MIN, MAX] 区间。
    //    3.5  => p ≈ 0.97 (确信有障碍)
    //    -3.5 => p ≈ 0.03 (确信是空地)
    // -----------------------------------------------------------------------------

    constexpr float LO_HIT = 0.85f;
    constexpr float LO_MISS = -0.4f;
    constexpr float LO_MAX = 3.5f;
    constexpr float LO_MIN = -3.5f;

    // -----------------------------------------------------------------------------
    // CUDA Kernel Launcher
    // -----------------------------------------------------------------------------

    /**
     * @brief 启动占用栅格更新流水线
     * * 逻辑流：
     * 1. Kernel 1: 根据 D2(索引) 和 D3(障碍物标记)，原子更新 d_occupancy_grid
     * 2. Kernel 2: 对全图进行 Clamp 截断，防止数值爆炸
     * * @param d_voxel_index    [输入] (Device) 每个点对应的网格 ID (来自 D2)
     * @param d_is_obstacle      [输入] (Device) 每个点的障碍物掩码 1=障碍, 0=地面 (来自 D3)
     * @param d_occupancy_grid   [输出] (Device) 需更新的全局概率地图 (Log-Odds值)
     * @param num_points         [参数] 当前帧点云数量
     * @param total_grids        [参数] 网格总数 (GridW * GridH)
     * @param stream             [参数] CUDA 流
     */
    void LaunchOccupancyUpdate(
        const int *d_voxel_index,
        const uint8_t *d_is_obstacle,
        float *d_occupancy_grid,
        uint32_t num_points,
        int total_grids,
        cudaStream_t stream = 0);

} // namespace flashbev

#endif // FLASHBEV_OCCUPANCY_GRID_H