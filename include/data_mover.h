#pragma once
#include <cstdint>

namespace flashbev
{

    // 启动 AoS 到 SoA 的内存重排 Kernel
    // stream: 传入 cudaStream_t 的指针（用 void* 隐藏 CUDA 依赖，保持头文件纯净）
    void LaunchAosToSoA(
        const float *d_aos_input,
        float *d_soa_x,
        float *d_soa_y,
        float *d_soa_z,
        float *d_soa_i,
        uint32_t num_points,
        void *stream = nullptr);

} // namespace flashbev