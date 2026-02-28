// include/spatial_hash.h
#pragma once
#include <cstdint>

namespace flashbev
{

    // 启动空间哈希 Kernel
    // 纯解耦设计：通过 void* 传入 stream，避免头文件被 cuda_runtime 污染
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
        void *stream);

} // namespace flashbev