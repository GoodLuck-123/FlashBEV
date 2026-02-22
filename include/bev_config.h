// include/bev_config.h
#pragma once
#include <cstdint>

namespace flashbev {
namespace config {

    // 1. BEV 物理边界参数 (雷达原点为中心，单位：米)
    inline constexpr float BEV_MIN_X = -25.0f;
    inline constexpr float BEV_MAX_X =  25.0f;
    inline constexpr float BEV_MIN_Y = -25.0f;
    inline constexpr float BEV_MAX_Y =  25.0f;
    inline constexpr float BEV_RES   =   0.2f;

    // 2. 编译期自动计算的网格维度 (零运行时开销)
    inline constexpr int GRID_W = static_cast<int>((BEV_MAX_X - BEV_MIN_X) / BEV_RES); // 250
    inline constexpr int GRID_H = static_cast<int>((BEV_MAX_Y - BEV_MIN_Y) / BEV_RES); // 250
    inline constexpr int TOTAL_GRIDS = GRID_W * GRID_H; // 62500

    // 3. 算法物理阈值
    inline constexpr float HEIGHT_THRESHOLD = 0.2f; // 障碍物高度判定阈值 (米)

    // 4. 系统级内存分配上限
    inline constexpr uint32_t MAX_POINTS = 300000;  // 单帧点云极限容量

} // namespace config
} // namespace flashbev