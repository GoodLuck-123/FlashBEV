#include <iostream>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <cuda_runtime.h>
#include "data_mover.h"

// 数据处理任务1.2
// cpp文件任务：从 SSD 直接加载 AoS 格式的点云数据到 GPU 显存，并在 GPU 上转换为 SoA 格式

// 极简检查宏
// 对每一次的 CUDA API 调用进行错误检查，确保任何失败都能被捕获并报告
#define CHECK_CUDA(call)                                                                               \
    do                                                                                                 \
    {                                                                                                  \
        cudaError_t err = call;                                                                        \
        if (err != cudaSuccess)                                                                        \
        {                                                                                              \
            fprintf(stderr, "[CUDA Error] %s:%d - %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE);                                                                        \
        }                                                                                              \
    } while (0)

// RAII: 零拷贝的极限 DataLoader
// class PinnedMappedFile
// {
// public:
//     const uint8_t *data = nullptr;
//     size_t size = 0;
//     int fd = -1;

//     // 构造函数：打开文件，映射到内存，并钉住（Pin）以供 CUDA 直接访问
//     PinnedMappedFile(const char *filepath)
//     {
//         fd = open(filepath, O_RDONLY);
//         if (fd < 0)
//             throw std::runtime_error("无法打开文件");

//         struct stat sb;
//         fstat(fd, &sb);
//         size = sb.st_size; // 文件总的字节数

//         // 1. 映射到虚拟内存
//         data = static_cast<const uint8_t *>(mmap(nullptr, size, PROT_READ, MAP_PRIVATE, fd, 0));
//         if (data == MAP_FAILED) {
//             throw std::runtime_error("mmap 失败：可能文件为空或权限不足");
//         }

//         // 2. 钉住（Pin）这块映射内存，开启直接内存访问 (DMA)
//         CHECK_CUDA(cudaHostRegister(const_cast<uint8_t *>(data), size, cudaHostRegisterReadOnly));
//     }

//     // 析构函数：解除映射和解绑
//     ~PinnedMappedFile()
//     {
//         if (data && data != MAP_FAILED)
//         {
//             cudaHostUnregister(const_cast<uint8_t *>(data)); // 先解绑
//             munmap(const_cast<uint8_t *>(data), size);       // 再解除映射
//         }
//         if (fd >= 0)
//             close(fd);
//     }
// };

// RAII: 真正的极致物理内存加载器 (零缺页中断)
class PinnedMappedFile {
public:
    uint8_t* data = nullptr;
    size_t size = 0;
    int fd = -1;

    PinnedMappedFile(const char* filepath) {
        // 1. 打开文件并获取大小
        fd = open(filepath, O_RDONLY);
        if (fd < 0) throw std::runtime_error("无法打开文件，请检查路径");

        struct stat sb;
        fstat(fd, &sb);
        size = sb.st_size;

        // 2. 直接向 OS 申请一块天生就被“钉死”的物理内存 (Page-Locked)
        // 这块内存对 DMA 极其友好，且不归内核的 Page Cache 管辖
        CHECK_CUDA(cudaHostAlloc((void**)&data, size, cudaHostAllocDefault));

        // 3. 极速 IO：使用 POSIX read，将 SSD 里的数据一次性暴力灌入这块锁页内存
        // 没有任何缺页中断的性能损耗！
        size_t bytes_read = 0;
        while (bytes_read < size) {
            ssize_t result = read(fd, data + bytes_read, size - bytes_read);
            if (result < 0) throw std::runtime_error("读取文件失败");
            if (result == 0) break; // EOF
            bytes_read += result;
        }
    }

    ~PinnedMappedFile() {
        // 释放这块珍贵的锁页内存
        if (data) cudaFreeHost(data);
        if (fd >= 0) close(fd);
    }
};

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
        float *d_aos_input, *d_soa_x, *d_soa_y, *d_soa_z, *d_soa_i;
        CHECK_CUDA(cudaMalloc(&d_aos_input, aos_bytes));
        CHECK_CUDA(cudaMalloc(&d_soa_x, single_soa_bytes));
        CHECK_CUDA(cudaMalloc(&d_soa_y, single_soa_bytes));
        CHECK_CUDA(cudaMalloc(&d_soa_z, single_soa_bytes));
        CHECK_CUDA(cudaMalloc(&d_soa_i, single_soa_bytes));

        // 3. 创建 CUDA 流，实现异步流水线
        cudaStream_t stream;
        CHECK_CUDA(cudaStreamCreate(&stream));

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
        for (int k = 0; k < 2; ++k) {
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
        for (int k = 0; k < 2; ++k) {
            std::cout << "Point[" << k << "]: "
                      << "X=" << check_x[k] << ", "
                      << "Y=" << check_y[k] << ", "
                      << "Z=" << check_z[k] << ", "
                      << "I=" << check_i[k] << "\n";
        }
        std::cout << "==========================================================\n\n";

        // 清理，但是如果其中一个cudaFree失败就会造成显存泄漏 (实际工程中可封装为 RAII Device_Ptr)
        // // 极简的显存智能指针
        // template <typename T>
        // class DevicePtr {
        // public:
        //     T* ptr = nullptr;

        //     // 构造函数：出生时分配显存
        //     DevicePtr(size_t num_elements) {
        //         cudaMalloc(&ptr, num_elements * sizeof(T));
        //     }

        //     // 析构函数：死亡时释放显存 (绝不泄漏)
        //     ~DevicePtr() {
        //         if (ptr) cudaFree(ptr);
        //     }

        //     // 禁用拷贝，防止多个对象 free 同一块内存 (C++ 核心素养)
        //     DevicePtr(const DevicePtr&) = delete;
        //     DevicePtr& operator=(const DevicePtr&) = delete;
        // };
        cudaFree(d_aos_input);
        cudaFree(d_soa_x);
        cudaFree(d_soa_y);
        cudaFree(d_soa_z);
        cudaFree(d_soa_i);
        cudaStreamDestroy(stream);
    }
    catch (const std::exception &e)
    {
        std::cerr << e.what() << "\n";
    }

    return 0;
}