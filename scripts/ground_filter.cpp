// scripts/test_ground_filter.cpp
#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include "data_mover.h"    // D1
#include "spatial_hash.h"  // D2
#include "ground_filter.h" // D3
#include "bev_config.h"   // BEV 参数配置
#include "FB_utils.h"

using namespace flashbev::utils;
using namespace flashbev::config;

int main(int argc, char **argv) {
    if (argc != 2) {
        std::cerr << "用法: " << argv[0] << " <AoS_bin文件路径>\n";
        return 1;
    }

    // 1. BEV 物理边界参数: bev_config.h

    try {
        // 2. 预热显存池
        std::cout << "[*] 加载显存...\n";
        
        // D1 显存
        DeviceBuffer<float> d_aos_input(MAX_POINTS * 4);
        DeviceBuffer<float> d_soa_x(MAX_POINTS);
        DeviceBuffer<float> d_soa_y(MAX_POINTS);
        DeviceBuffer<float> d_soa_z(MAX_POINTS);
        DeviceBuffer<float> d_soa_i(MAX_POINTS);
        
        // D2 显存
        DeviceBuffer<int> d_voxel_index(MAX_POINTS);

        // D3 显存
        DeviceBuffer<float> d_min_z_grid(TOTAL_GRIDS);
        DeviceBuffer<uint8_t> d_is_obstacle(MAX_POINTS);

        CudaStream stream;

        // 3. 加载二进制点云
        PinnedMappedFile bin_file(argv[1]); // 申请锁页内存并加载文件
        uint32_t num_points = *reinterpret_cast<const uint32_t *>(bin_file.data);
        if (num_points > MAX_POINTS) throw std::runtime_error("点数超限!");
        const float *host_aos_data = reinterpret_cast<const float *>(bin_file.data + 4);
        size_t aos_bytes = num_points * 4 * sizeof(float);

        // ==============================================================================
        // 核心流水线 (Pipeline)
        // CPU 以极低开销连续发射指令，GPU 内部形成无缝数据流转 (DMA -> D1 -> D2 -> D3)
        // ==============================================================================
        
        // 步骤 A: 异步 DMA 灌入
        CHECK_CUDA(cudaMemcpyAsync(d_aos_input.ptr, host_aos_data, aos_bytes, cudaMemcpyHostToDevice, stream));

        // 步骤 B (D1): 结构重排 AoS -> SoA
        flashbev::LaunchAosToSoA(d_aos_input.ptr, d_soa_x.ptr, d_soa_y.ptr, d_soa_z.ptr, d_soa_i.ptr, num_points, stream);

        // 步骤 C (D2): 空间索引化 (计算归属网格)
        flashbev::LaunchSpatialHash(
            d_soa_x.ptr, d_soa_y.ptr, d_voxel_index.ptr, num_points, 
            BEV_MIN_X, BEV_MIN_Y, BEV_RES, GRID_W, GRID_H, stream
        );

        // 步骤 D (D3): 提取地面，过滤障碍物
        flashbev::LaunchGroundFilter(
            d_soa_z.ptr, d_voxel_index.ptr, d_min_z_grid.ptr, d_is_obstacle.ptr, 
            num_points, GRID_W, GRID_H, HEIGHT_THRESHOLD, stream
        );

        // CPU 等待整个流水线清空
        CHECK_CUDA(cudaStreamSynchronize(stream));
        std::cout << "[+] Pipeline D3 走通！地面滤波与障碍物提取完成。\n";

        // ==============================================================================
        // 验证阶段
        // ==============================================================================
        std::cout << "\n>>> [验证阶段] 抽取前 10 个点核对地面提取物理逻辑...\n";
        
        const int CHECK_NUM = 10;
        float host_z[CHECK_NUM];
        int host_voxel[CHECK_NUM];
        uint8_t host_is_obs[CHECK_NUM];
        std::vector<float> host_min_z_grid(TOTAL_GRIDS); // 250x250的网格才 250KB，全拉回来检查没压力

        CHECK_CUDA(cudaMemcpy(host_z, d_soa_z.ptr, CHECK_NUM * sizeof(float), cudaMemcpyDeviceToHost));
        CHECK_CUDA(cudaMemcpy(host_voxel, d_voxel_index.ptr, CHECK_NUM * sizeof(int), cudaMemcpyDeviceToHost));
        CHECK_CUDA(cudaMemcpy(host_is_obs, d_is_obstacle.ptr, CHECK_NUM * sizeof(uint8_t), cudaMemcpyDeviceToHost));
        CHECK_CUDA(cudaMemcpy(host_min_z_grid.data(), d_min_z_grid.ptr, TOTAL_GRIDS * sizeof(float), cudaMemcpyDeviceToHost));

        for (int k = 0; k < CHECK_NUM; ++k) {
            float z = host_z[k];
            int v_idx = host_voxel[k];

            std::cout << "Point[" << k << "]: Z=" << z << "m | ";

            if (v_idx == -1) {
                std::cout << "状态: 越界 [丢弃]\n";
            } else {
                float local_min_z = host_min_z_grid[v_idx];
                bool is_obs = (host_is_obs[k] == 1);
                
                std::cout << "归属网格: (" << v_idx / GRID_W << "," << v_idx % GRID_W << ")" 
                          << " | 网格最低点(地面): " << local_min_z << "m"
                          << " | 高度差: " << (z - local_min_z) << "m"
                          << " -> 判定: " << (is_obs ? "[🔴 障碍物]" : "[🟢 地面点]") << "\n";
            }
        }
        std::cout << "==========================================================\n\n";

    } catch (const std::exception &e) {
        std::cerr << "异常: " << e.what() << "\n";
    }

    return 0;
}