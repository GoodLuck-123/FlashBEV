// include/ground_filter.h
#pragma once
#include <cstdint>

namespace flashbev
{

    // ------------------------------------------------------------------
    // D3: 基于物理网格的地面高度滤波 (Ground Height Filtering)
    // 物理逻辑：在每个 2D BEV 网格中寻找最低的 Z 坐标作为局部参考地面，
    //           高度高于 (局部地面 + height_threshold) 的点被判定为障碍物。
    // 此逻辑在实际运用中会有歧义，因为网格太小了，但是没关系，重点在于演示如何在 GPU 上实现这个流程
    // 并且真实应用中激光雷达会有一个大概的物理先验安装高度，所以这个方法在实际中是有一定意义的
    // ------------------------------------------------------------------
    void LaunchGroundFilter(
        const float *d_soa_z,     // [输入] 点云 Z 坐标阵列 (Device Ptr)
        const int *d_voxel_index, // [输入] 空间哈希生成的网格索引 (Device Ptr)
        float *d_min_z_grid,      // [内部/输出] BEV 网格最低高度图，大小为 grid_w * grid_h (Device Ptr)
        uint8_t *d_is_obstacle,   // [输出] 点云障碍物掩码阵列，大小为 num_points (Device Ptr)
        uint32_t num_points,      // [参数] 当前帧点云总数
        int grid_w,               // [参数] BEV 物理网格宽度
        int grid_h,               // [参数] BEV 物理网格高度
        float height_threshold,   // [参数] 障碍物判定高度差阈值 (物理单位：米，例如 0.2f)
        void *stream = nullptr    // [参数] 异步执行流，默认使用缺省流
    );

} // namespace flashbev