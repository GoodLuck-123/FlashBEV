// 数据处理任务1.2
// CUDA实现AoS->SoA转换：极致加载 + 离散写出
#include "data_mover.h"
#include <cuda_runtime.h>
#include <iostream>
#include <stdexcept>

// 极简宏：捕获 Kernel 启动和执行时的任何异步报错
// 将来可以将其包含在一个公共的 CUDA 工具头文件中，供整个项目使用 cuda_utils.h
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

    // __restrict__ 承诺指针绝对不重叠，让编译器放心大胆地使用 L1/L2 Cache
    // 告诉编译器不会存在input_aos, out_x ... 等几个指针不会指向同一内存区域（指针别名）
    // 如果不使用 __restrict__，编译器必须假设这些指针可能指向同一内存区域，然后去 Global Menmory 转一圈
    // 这会限制编译器的优化能力，导致性能下降
    __global__ void ConvertAosToSoAKernel(
        const float4 *__restrict__ input_aos,
        float *__restrict__ out_x,
        float *__restrict__ out_y,
        float *__restrict__ out_z,
        float *__restrict__ out_i,
        const uint32_t num_points)
    {
        // 这里的 block 是一维的，线程也是一维的，适合处理大规模点云数据
        int tid = blockIdx.x * blockDim.x + threadIdx.x;
        int stride = blockDim.x * gridDim.x;

        for (uint32_t idx = tid; idx < num_points; idx += stride)
        {
            // 1. 物理层极限操作：一个时钟周期，将连续的 16 字节吞入 4 个 32位 寄存器
            float4 pt = input_aos[idx];

            // 2. 向分离的 Global Memory 连续地址写入
            // 当一个 Warp (32线程) 执行下面任意一行时，触发 32 * 4 = 128 字节的合并写入
            out_x[idx] = pt.x;
            out_y[idx] = pt.y;
            out_z[idx] = pt.z;
            out_i[idx] = pt.w; // w 代表第4个元素，即 intensity
        }
    }

    // void* stream 实现了在头文件中对 CUDA Runtime API 的完全解耦
    // 用户可以传入任何类型的流指针（如 cudaStream_t 或者其他封装的流对象）
    // 同时可以避免在头文件中包含庞大的 cuda_runtime.h，减少编译依赖和编译时间
    void LaunchAosToSoA(
        const float *d_aos_input,
        float *d_soa_x,
        float *d_soa_y,
        float *d_soa_z,
        float *d_soa_i,
        uint32_t num_points,
        void *stream)
    {
        // 强制类型转换：把传入的普通 float 数组当作 float4 数组来访问
        const float4 *d_aos_float4 = reinterpret_cast<const float4 *>(d_aos_input);
        cudaStream_t cu_stream = static_cast<cudaStream_t>(stream);

        // 线程配置：256 线程/Block
        int threads = 256; // blockDim，256是常见最佳选择
        //                      能充分利用 GPU 的并行能力，同时避免资源过度分配（挤兑 shared memory）
        int blocks = (num_points + threads - 1) / threads; // gridDim，确保覆盖所有点

        ConvertAosToSoAKernel<<<blocks, threads, 0, cu_stream>>>(
            d_aos_float4, d_soa_x, d_soa_y, d_soa_z, d_soa_i, num_points);

        // 捕获 Kernel 异步启动配置错误，如非法的块/线程数量，或者无效的流指针
        // 避免在后续的 CUDA API 调用中才发现 Kernel 启动失败，导致错误信息不明确
        CHECK_CUDA(cudaGetLastError());
    }

} // namespace flashbev