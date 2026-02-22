// scripts/test_spatial_hash.cpp
#include <iostream>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <cuda_runtime.h>
#include "data_mover.h"    // 复用 D1 的 AoS->SoA
#include "spatial_hash.h" // D2 即将实现的头文件
#include "FB_utils.h"     // 统一的 CUDA 错误检查和 RAII 内存管理

using namespace flashbev::utils;

int main(int argc, char **argv) {
    if (argc != 2) {
        std::cerr << "用法: " << argv[0] << " <AoS_bin文件路径>\n";
        return 1;
    }

    // 1. 定义 BEV 物理边界参数 (长宽 50m，分辨率 0.2m)
    // BEV坐标系直接使用激光雷达的物理坐标系，原点在雷达位置 (0,0)，X轴前向，Y轴左侧为正
    // 物理坐标系假设雷达为原点 (0,0): X 轴前向 [-25, 25]，Y 轴左侧为正 [-25, 25]
    const float BEV_MIN_X = -25.0f;
    const float BEV_MAX_X =  25.0f;
    const float BEV_MIN_Y = -25.0f;
    const float BEV_MAX_Y =  25.0f;
    const float BEV_RES   =   0.2f;
    const int GRID_W = static_cast<int>((BEV_MAX_X - BEV_MIN_X) / BEV_RES); // 250
    const int GRID_H = static_cast<int>((BEV_MAX_Y - BEV_MIN_Y) / BEV_RES); // 250

    try {
        // 2. 预分配 GPU 显存池
        // 假设一帧点云不会超过 30 万个点
        const uint32_t MAX_POINTS = 300000;
        std::cout << "[*] 正在预热显存池 (容量: " << MAX_POINTS << " 个点)...\n";
        
        // 分配 SoA 阵列空间
        DeviceBuffer<float> d_aos_input(MAX_POINTS * 4);
        DeviceBuffer<float> d_soa_x(MAX_POINTS);
        DeviceBuffer<float> d_soa_y(MAX_POINTS);
        DeviceBuffer<float> d_soa_z(MAX_POINTS);
        DeviceBuffer<float> d_soa_i(MAX_POINTS);
        
        // 每个点对应的 1D 空间哈希索引（网格 ID）
        // 设为 int，因为如果是越界点（不在 50x50 范围内），我们将赋值为 -1 丢入垃圾桶
        DeviceBuffer<int> d_voxel_index(MAX_POINTS);

        flashbev::utils::CudaStream stream;

        // 3. 模拟进入主循环获取新一帧数据
        PinnedMappedFile bin_file(argv[1]);
        // 头4字节是点云数量
        uint32_t num_points = *reinterpret_cast<const uint32_t *>(bin_file.data);
        
        if (num_points > MAX_POINTS) {
            throw std::runtime_error("点云数量超出显存池最大容量!");
        }

        const float *host_aos_data = reinterpret_cast<const float *>(bin_file.data + 4);
        size_t aos_bytes = num_points * 4 * sizeof(float);

        // --- 开始 Pipeline ---
        // 步骤 A: 异步 DMA 灌入显存
        CHECK_CUDA(cudaMemcpyAsync(d_aos_input.ptr, host_aos_data, aos_bytes, cudaMemcpyHostToDevice, stream));

        // 步骤 B: AoS -> SoA 转换
        flashbev::LaunchAosToSoA(d_aos_input.ptr, d_soa_x.ptr, d_soa_y.ptr, d_soa_z.ptr, d_soa_i.ptr, num_points, stream);

        // 步骤 C: 空间哈希 (待实现的内核)
        // 物理逻辑：传入 x 和 y 的坐标阵列，结合边界参数，输出离散的 Grid ID 给 d_voxel_index
        flashbev::LaunchSpatialHash(
            d_soa_x.ptr, d_soa_y.ptr, d_voxel_index.ptr, num_points, 
            BEV_MIN_X, BEV_MIN_Y, BEV_RES, GRID_W, GRID_H, stream
        );

        CHECK_CUDA(cudaStreamSynchronize(stream));
        std::cout << "[+] Pipeline 走通！D2 (Mock) 空间哈希完成。\n";

        // 验证：将 GPU 计算结果抽回 CPU 进行检查
        // 在真实的自动驾驶部署中不可在主循环里做 DeviceToHost 拷贝
        std::cout << "==========================================================";
        std::cout << "\n>>> [验证阶段] 抽取前 10 个点进行哈希映射核对...\n";
        
        const int CHECK_NUM = 10;
        float host_x[CHECK_NUM], host_y[CHECK_NUM];
        int host_voxel[CHECK_NUM];

        // 把显存里的前 CHECK_NUM 个结果拉回来
        CHECK_CUDA(cudaMemcpy(host_x, d_soa_x.ptr, CHECK_NUM * sizeof(float), cudaMemcpyDeviceToHost));
        CHECK_CUDA(cudaMemcpy(host_y, d_soa_y.ptr, CHECK_NUM * sizeof(float), cudaMemcpyDeviceToHost));
        CHECK_CUDA(cudaMemcpy(host_voxel, d_voxel_index.ptr, CHECK_NUM * sizeof(int), cudaMemcpyDeviceToHost));

        for (int k = 0; k < CHECK_NUM; ++k) {
            float x = host_x[k];
            float y = host_y[k];
            int v_idx = host_voxel[k];

            std::cout << "Point[" << k << "]: 物理坐标 (X=" << x << "m, Y=" << y << "m) -> ";

            if (v_idx == -1) {
                std::cout << "越界！[丢弃]\n";
            } else {
                // 逆向推导其 2D 网格坐标
                int grid_y = v_idx / GRID_W;
                int grid_x = v_idx % GRID_W;
                std::cout << "GridID: " << v_idx 
                          << " (2D网格: 行 " << grid_y << ", 列 " << grid_x << ")\n";
            }
        }
        std::cout << "==========================================================\n\n";

    } catch (const std::exception &e) {
        std::cerr << "异常: " << e.what() << "\n";
    }

    return 0;
}