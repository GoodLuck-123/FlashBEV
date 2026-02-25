#pragma once

#include <cuda_runtime.h>
#include <iostream>
#include <stdexcept>
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>
#include <string>
#include <cstdio>

// ------------------------------------------------------------------
// 统一的 CUDA 错误捕获宏
// 必须包裹在 do-while(0) 中，防止在 if-else 语句中展开时出现语法错误
// ------------------------------------------------------------------
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

namespace flashbev
{
    namespace utils
    {

        // RAII: 零拷贝，注释的实现在Ubuntu下不可行，因为device不可抢夺内核态内存
        // 除非使用专门的驱动接口（如NVMe Direct），但这超出了本项目的范围
        // 我们改用 cudaHostAlloc 来申请锁页内存，并手动读取文件内容到这块内存中
        // 保证数据可被 CUDA 直接访问

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

        // ------------------------------------------------------------------
        // RAII: 零缺页物理内存加载器 (Page-Locked Memory)
        // 绕过操作系统 Page Cache，实现 SSD -> 内存 -> GPU 的极限 DMA 传输
        // ------------------------------------------------------------------
        class PinnedMappedFile
        {
        public:
            uint8_t *data = nullptr;
            size_t size = 0;
            int fd = -1;

            PinnedMappedFile(const char *filepath)
            {
                fd = open(filepath, O_RDONLY);
                if (fd < 0)
                    throw std::runtime_error("无法打开文件，请检查路径");

                struct stat sb;
                fstat(fd, &sb);
                size = sb.st_size;

                // 申请锁页内存
                CHECK_CUDA(cudaHostAlloc((void **)&data, size, cudaHostAllocDefault));

                // 暴力灌入
                size_t bytes_read = 0;
                while (bytes_read < size)
                {
                    ssize_t result = read(fd, data + bytes_read, size - bytes_read);
                    if (result < 0)
                        throw std::runtime_error("读取文件失败");
                    if (result == 0)
                        break; // EOF
                    bytes_read += result;
                }
            }

            ~PinnedMappedFile()
            {
                if (data)
                    cudaFreeHost(data);
                if (fd >= 0)
                    close(fd);
            }

            // 严格禁用拷贝语义 (Rule of Delete)
            PinnedMappedFile(const PinnedMappedFile &) = delete;
            PinnedMappedFile &operator=(const PinnedMappedFile &) = delete;
        };

        // ------------------------------------------------------------------
        // RAII: 显存池管理 (Device Memory)
        // 保证生命周期与作用域绑定，绝对不产生 Double Free 或显存泄漏
        // ------------------------------------------------------------------
        template <typename T>
        class DeviceBuffer
        {
        public:
            T *ptr = nullptr;
            size_t capacity = 0;

            DeviceBuffer(size_t max_elements) : capacity(max_elements)
            {
                if (max_elements > 0)
                    CHECK_CUDA(cudaMalloc(&ptr, max_elements * sizeof(T)));
            }

            ~DeviceBuffer()
            {
                if (ptr)
                    cudaFree(ptr);
            }

            // 严格禁用拷贝语义 (Rule of Delete)
            DeviceBuffer(const DeviceBuffer &) = delete;
            DeviceBuffer &operator=(const DeviceBuffer &) = delete;

            // --- 核心手感升级 ---
            // 重载类型转换操作符，允许对象像普通指针一样被直接使用
            operator T *() const { return ptr; }

            // 获取底层指针显式调用的接口
            T *get() const { return ptr; }
        };

        // ------------------------------------------------------------------
        // RAII: CUDA 流管理
        // ------------------------------------------------------------------
        class CudaStream
        {
        public:
            cudaStream_t stream;

            CudaStream()
            {
                // 创建非阻塞流 (cudaStreamNonBlocking)
                // 这让感知算子不会被其他默认流上的任务卡住
                CHECK_CUDA(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
            }

            ~CudaStream()
            {
                // 即使程序崩溃，栈展开时也会自动销毁流，防止驱动层产生僵尸流
                if (stream)
                    CHECK_CUDA(cudaStreamDestroy(stream));
            }

            // 严格禁用拷贝语义 (Rule of Delete)
            CudaStream(const CudaStream &) = delete;
            CudaStream &operator=(const CudaStream &) = delete;

            // 类型转换重载：在使用时可以直接把 CudaStream 对象传给需要 cudaStream_t 或 void* 的函数
            operator cudaStream_t() const { return stream; }
            operator void *() const { return static_cast<void *>(stream); }
        };

        // -----------------------------------------------------------------------------
        // 高精度 GPU 计时器 (基于 cudaEvent)
        // -----------------------------------------------------------------------------
        class GpuTimer
        {
        public:
            cudaEvent_t start_ev, stop_ev;
            std::string name;

            GpuTimer(const std::string &task_name = "Task") : name(task_name)
            {
                cudaEventCreate(&start_ev);
                cudaEventCreate(&stop_ev);
            }

            ~GpuTimer()
            {
                cudaEventDestroy(start_ev);
                cudaEventDestroy(stop_ev);
            }

            // 开始计时
            void Tick(cudaStream_t stream = 0)
            {
                cudaEventRecord(start_ev, stream);
            }

            // 结束计时并打印
            void Tock(cudaStream_t stream = 0)
            {
                cudaEventRecord(stop_ev, stream);
                cudaEventSynchronize(stop_ev); // 必须同步等待 GPU 走到这里

                float milliseconds = 0;
                cudaEventElapsedTime(&milliseconds, start_ev, stop_ev);

                std::printf(">> [%-15s] 耗时: \033[1;32m%.3f ms\033[0m\n", name.c_str(), milliseconds);
            }
        };

    } // namespace utils
} // namespace flashbev