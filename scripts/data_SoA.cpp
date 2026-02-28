#include <iostream>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <cuda_runtime.h>
#include "data_mover.h"
#include "FB_utils.h" // 统一的 CUDA 错误检查和 RAII 内存管理

using namespace flashbev::utils;

int main(int argc, char **argv)
{
    if (argc != 2)
    {
        std::cerr << "用法: " << argv[0] << " <AoS_bin文件路径>\n";
        return 1;
    }

    try
    {
        // 1. 零拷贝加载，少了一次SSD到内核态内存的复制，直接映射文件到用户空间，并钉住以供 CUDA 访问
        PinnedMappedFile bin_file(argv[1]);
        // 获取点云数量，py代码中以 uint32_t 的形式写在文件开头
        uint32_t num_points = *reinterpret_cast<const uint32_t *>(bin_file.data);
        // 跳过包头获取 AoS 数据指针
        const float *host_aos_data = reinterpret_cast<const float *>(bin_file.data + 4);
        // 计算激光雷达扫描点的总字节数
        size_t aos_bytes = num_points * 4 * sizeof(float);
        // 每个 SoA 数组的字节数
        size_t single_soa_bytes = num_points * sizeof(float);

        std::cout << "[*] 成功锁定物理内存页，准备泵入 GPU。点数: " << num_points << "\n";

        // 2. 分配显存 (Device Memory)
        // float *d_aos_input, *d_soa_x, *d_soa_y, *d_soa_z, *d_soa_i;
        // CHECK_CUDA(cudaMalloc(&d_aos_input, aos_bytes));
        // CHECK_CUDA(cudaMalloc(&d_soa_x, single_soa_bytes));
        // CHECK_CUDA(cudaMalloc(&d_soa_y, single_soa_bytes));
        // CHECK_CUDA(cudaMalloc(&d_soa_z, single_soa_bytes));
        // CHECK_CUDA(cudaMalloc(&d_soa_i, single_soa_bytes));
        DeviceBuffer<float> d_aos_input(aos_bytes);
        DeviceBuffer<float> d_soa_x(single_soa_bytes);
        DeviceBuffer<float> d_soa_y(single_soa_bytes);
        DeviceBuffer<float> d_soa_z(single_soa_bytes);
        DeviceBuffer<float> d_soa_i(single_soa_bytes);
        // 3. 创建 CUDA 流，实现异步流水线
        flashbev::utils::CudaStream stream;

        // 4. 从主机 Pinned 内存通过 PCIe 异步 DMA 到显存
        CHECK_CUDA(cudaMemcpyAsync(d_aos_input, host_aos_data, aos_bytes, cudaMemcpyHostToDevice, stream));

        // 5. Kernel，执行 AoS -> SoA 撕裂
        flashbev::LaunchAosToSoA(d_aos_input, d_soa_x, d_soa_y, d_soa_z, d_soa_i, num_points, stream);

        // 6. 强制同步，等待 GPU 完工
        CHECK_CUDA(cudaStreamSynchronize(stream));
        std::cout << "[+] D1 任务达成！数据已在显存中完美转换为 SoA 布局。\n";

        std::cout << "\n==========================================================\n";

        // 7. 打印一下看看有没有问题
        std::cout << ">>> [验证阶段] 正在提取前 2 个点的数据对齐情况...\n";

        // 打印 CPU 端的原始 AoS 数据 (交织的 1D 数组)
        std::cout << "--- 转换前 (AoS, CPU 内存直读) ---\n";
        for (int k = 0; k < 2; ++k)
        {
            std::cout << "Point[" << k << "]: "
                      << "X=" << host_aos_data[k * 4 + 0] << ", "
                      << "Y=" << host_aos_data[k * 4 + 1] << ", "
                      << "Z=" << host_aos_data[k * 4 + 2] << ", "
                      << "I=" << host_aos_data[k * 4 + 3] << "\n";
        }

        // 把显卡里的 SoA 前两个数据抽回来 (Device To Host)
        float check_x[2], check_y[2], check_z[2], check_i[2];
        CHECK_CUDA(cudaMemcpy(check_x, d_soa_x, 2 * sizeof(float), cudaMemcpyDeviceToHost));
        CHECK_CUDA(cudaMemcpy(check_y, d_soa_y, 2 * sizeof(float), cudaMemcpyDeviceToHost));
        CHECK_CUDA(cudaMemcpy(check_z, d_soa_z, 2 * sizeof(float), cudaMemcpyDeviceToHost));
        CHECK_CUDA(cudaMemcpy(check_i, d_soa_i, 2 * sizeof(float), cudaMemcpyDeviceToHost));

        // 打印 GPU 传回的 SoA 数据 (分离的 1D 数组)
        std::cout << "--- 转换后 (SoA, GPU 显存回抽) ---\n";
        for (int k = 0; k < 2; ++k)
        {
            std::cout << "Point[" << k << "]: "
                      << "X=" << check_x[k] << ", "
                      << "Y=" << check_y[k] << ", "
                      << "Z=" << check_z[k] << ", "
                      << "I=" << check_i[k] << "\n";
        }
        std::cout << "==========================================================\n\n";

        // 清理，如果其中一个cudaFree失败就会造成显存泄漏
        // 上面的 DevicePtr 类会自动管理显存生命周期，不需要手动 cudaFree
        // cudaFree(d_aos_input);
        // cudaFree(d_soa_x);
        // cudaFree(d_soa_y);
        // cudaFree(d_soa_z);
        // cudaFree(d_soa_i);
    }
    catch (const std::exception &e)
    {
        std::cerr << e.what() << "\n";
    }

    return 0;
}