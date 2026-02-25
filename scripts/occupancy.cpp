#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include <algorithm> // for std::max_element

// 引入所有模块的头文件
#include "data_mover.h"    // D1
#include "spatial_hash.h"  // D2
#include "ground_filter.h" // D3
#include "occupancy_grid.h"// D4 (New)
#include "bev_config.h"
#include "FB_utils.h"

using namespace flashbev::utils;
using namespace flashbev::config;

int main(int argc, char **argv) {
    if (argc != 2) {
        std::cerr << "用法: " << argv[0] << " <AoS_bin文件路径>\n";
        return 1;
    }

    try {
        std::cout << "[*] FlashBEV Pipeline...\n";

        // =====================================================================
        // 1. 资源分配
        // =====================================================================
        
        // 加载数据 (Pinned Memory)
        PinnedMappedFile bin_file(argv[1]);
        uint32_t num_points = *reinterpret_cast<const uint32_t *>(bin_file.data);
        const float *host_aos_data = reinterpret_cast<const float *>(bin_file.data + 4);

        if (num_points > MAX_POINTS) throw std::runtime_error("点数超限");

        // D1 显存: 输入缓冲 & SoA 缓冲
        DeviceBuffer<float> d_aos_input(MAX_POINTS * 4);
        DeviceBuffer<float> d_soa_x(MAX_POINTS);
        DeviceBuffer<float> d_soa_y(MAX_POINTS);
        DeviceBuffer<float> d_soa_z(MAX_POINTS);
        DeviceBuffer<float> d_soa_i(MAX_POINTS);

        // D2 显存: 空间索引
        DeviceBuffer<int> d_voxel_index(MAX_POINTS);

        // D3 显存: 地面高度图 & 障碍物掩码
        DeviceBuffer<float> d_min_z_grid(TOTAL_GRIDS);
        DeviceBuffer<uint8_t> d_is_obstacle(MAX_POINTS);

        // D4 显存: 占用栅格地图 (Log-Odds Map)
        // 注意：这是地图本身，通常在多帧之间复用。
        DeviceBuffer<float> d_occupancy_grid(TOTAL_GRIDS);

        // 初始化: 在首帧开始前，必须将地图清零 (Log-Odds = 0 表示 50% 概率，未知)
        CHECK_CUDA(cudaMemset(d_occupancy_grid.ptr, 0, TOTAL_GRIDS * sizeof(float)));

        CudaStream stream;

        // =====================================================================
        // 2. 执行流水线 (The Loop)
        // =====================================================================
        
        // Step A: H2D DMA
        CHECK_CUDA(cudaMemcpyAsync(d_aos_input.ptr, host_aos_data, num_points * 4 * sizeof(float), cudaMemcpyHostToDevice, stream));

        // Step B (D1): AoS -> SoA
        flashbev::LaunchAosToSoA(d_aos_input.ptr, d_soa_x.ptr, d_soa_y.ptr, d_soa_z.ptr, d_soa_i.ptr, num_points, stream);

        // Step C (D2): Spatial Hash
        flashbev::LaunchSpatialHash(
            d_soa_x.ptr, d_soa_y.ptr, d_voxel_index.ptr, num_points, 
            BEV_MIN_X, BEV_MIN_Y, BEV_RES, GRID_W, GRID_H, stream
        );

        // Step D (D3): Ground Filter
        // 注意：这里会复用 D2 算出的 voxel_index，并输出 is_obstacle 掩码
        flashbev::LaunchGroundFilter(
            d_soa_z.ptr, d_voxel_index.ptr, d_min_z_grid.ptr, d_is_obstacle.ptr, 
            num_points, GRID_W, GRID_H, HEIGHT_THRESHOLD, stream
        );

        // Step E (D4): Occupancy Update [NEW]
        // 核心逻辑：根据 is_obstacle 掩码，对 d_occupancy_grid 进行加减分
        flashbev::LaunchOccupancyUpdate(
            d_voxel_index.ptr, d_is_obstacle.ptr, d_occupancy_grid.ptr, 
            num_points, TOTAL_GRIDS, stream
        );

        // 等待 GPU 完成
        CHECK_CUDA(cudaStreamSynchronize(stream));
        std::cout << "[+] Pipeline 完成。占用栅格已更新。\n";

        // =====================================================================
        // 3. 验证结果 (Host 检查)
        // =====================================================================
        
        std::vector<float> host_grid(TOTAL_GRIDS);
        CHECK_CUDA(cudaMemcpy(host_grid.data(), d_occupancy_grid.ptr, TOTAL_GRIDS * sizeof(float), cudaMemcpyDeviceToHost));

        std::cout << "\n>>> 数据验证：占用栅格数据分析 (Log-Odds)\n";
        
        int occupied_count = 0;
        int free_count = 0;
        int unknown_count = 0;
        float max_val = -100.0f;
        float min_val = 100.0f;
        int max_idx = -1;

        for (int i = 0; i < TOTAL_GRIDS; ++i) {
            float val = host_grid[i];
            if (val > 0.1f) occupied_count++;       // 倾向于障碍物
            else if (val < -0.1f) free_count++;     // 倾向于自由区域
            else unknown_count++;

            if (val > max_val) { max_val = val; max_idx = i; }
            if (val < min_val) min_val = val;
        }

        std::cout << "----------------------------------------------------------\n";
        std::cout << "网格统计 (总数 " << TOTAL_GRIDS << "):\n";
        std::cout << " - 被占用网格 (Log-Odds > 0.1): " << occupied_count << "\n";
        std::cout << " - 自由区网格 (Log-Odds < -0.1): " << free_count << "\n";
        std::cout << " - 未知/静止 (Log-Odds ≈ 0): " << unknown_count << "\n";
        std::cout << "----------------------------------------------------------\n";
        std::cout << "最确信的障碍物 (Grid ID " << max_idx << "): Val = " << max_val << "\n";
        std::cout << "最确信的自由区: Val = " << min_val << "\n";
        
        if (max_val > 0.0f) {
            std::cout << "✅ 成功检测到高置信度障碍物！Log-Odds 逻辑生效。\n";
        } else {
            std::cout << "⚠️ 警告：全图未发现显著障碍物，请检查输入数据或 LO_HIT 参数。\n";
        }

        // 打印几个具体点看看
        std::cout << "\n[抽样检查] 前 5 个非零网格:\n";
        int printed = 0;
        for(int i=0; i<TOTAL_GRIDS && printed<5; ++i) {
            if(abs(host_grid[i]) > 0.01f) {
                std::cout << "Grid[" << i << "] = " << host_grid[i] << "\n";
                printed++;
            }
        }
        std::cout << "==========================================================\n";

    } catch (const std::exception &e) {
        std::cerr << "异常: " << e.what() << "\n";
        return 1;
    }

    return 0;
}