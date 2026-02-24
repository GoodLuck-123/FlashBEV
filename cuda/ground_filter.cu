// cuda/ground_filter.cu
#include "ground_filter.h"
#include <cuda_runtime.h>
#include <math_functions.h>
#include "FB_utils.h"

namespace flashbev {

// ==============================================================================
// 手写单精度浮点数的 atomicMin
// CUDA 硬件底层原生支持 int 的 CAS，我们通过寄存器位强转欺骗硬件，完成浮点数原子比较
// 现代 GPU (如 RTX 5080) 会在 L2 Cache 层级处理这个请求，极大降低了显存带宽压力
// ==============================================================================
// __forceinline__ 强制内联，即将这个函数代码逻辑整个复制到调用他的地方，避免函数调用开销
__device__ __forceinline__ void atomicMinFloat(float* address, float val) {
    int* address_as_int = (int*)address;
    int old = *address_as_int;
    int assumed;
    
    do {
        assumed = old;
        // 如果 val 更小，我们就准备把更小的值的位模式写入 address_as_int
        // assumed 和 old 是用于检测在我们准备写入之前，这个地址的值有没有被其他线程改动过的变量
        // fminf 会在寄存器级极速比较出更小的浮点数
        // __float_as_int 和 __int_as_float 只是编译器层面的重解释转换，0 时钟周期开销
        // atomicCAS 只认识int32
        old = atomicCAS(address_as_int, assumed,
                        __float_as_int(fminf(val, __int_as_float(assumed))));
        // 如果期间有其他线程改了这个地址，old 会不等于 assumed，循环重试（无锁化自旋）
        // 由于是无锁编程，所以会出现多进程竞争同一个地址的情况，但最终总能保证 min 的正确性
    } while (assumed != old);
}

// ------------------------------------------------------------------
// Kernel 1: 极速显存重置 (代替耗时的 cudaMemset 同步调用)
// 将网格初始化为物理上的极高处 (例如 100.0 米)
// ------------------------------------------------------------------
__global__ void InitMinZGridKernel(
    float* __restrict__ min_z_grid, 
    const int total_grids) 
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    // 这个核函数中 idx 关联 BEV 网格的索引，不会特别大；但是后面处理点云时数量非常大，所以用 uint32_t
    for (int idx = tid; idx < total_grids; idx += stride) {
        min_z_grid[idx] = 100.0f; // 雷达扫不到 100 米高空的地板
    }
    // cudaMemset 虽然也能设置浮点数，但它是字节级的，效率极低
    // 而且只能设置为 0 或 -1（全 0 或全 1 的位模式），无法满足我们初始化为 100.0f 的需求
}

// ------------------------------------------------------------------
// Kernel 2: 原子化收集最低点
// ------------------------------------------------------------------
__global__ void ComputeMinZKernel(
    const float* __restrict__ soa_z,
    const int* __restrict__ voxel_index,
    float* __restrict__ min_z_grid,
    const uint32_t num_points) 
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (uint32_t idx = tid; idx < num_points; idx += stride) {
        int v_idx = voxel_index[idx];
        // 遇到越界点 (-1) 直接跳过。由于越界点是极少数，这里的 if 造成的 Warp Divergence 影响微乎其微
        if (v_idx != -1) {
            float z = soa_z[idx];
            // 同一个BEV网格的v_idx是不变的，其最低点高度可能被多个线程同时更新
            // 但 atomicMinFloat 能保证最终结果是正确的最低点高度
            atomicMinFloat(&min_z_grid[v_idx], z);
        }
    }
}

// ------------------------------------------------------------------
// Kernel 3: 高度差滤波（无分支暴力写出）
// ------------------------------------------------------------------
__global__ void FilterObstaclesKernel(
    const float* __restrict__ soa_z,
    const int* __restrict__ voxel_index,
    const float* __restrict__ min_z_grid,
    uint8_t* __restrict__ is_obstacle,
    const uint32_t num_points,
    const float threshold) 
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (uint32_t idx = tid; idx < num_points; idx += stride) {
        int v_idx = voxel_index[idx];
        
        // 核心优化：避免 if-else 带来的串行化惩罚
        // 如果是有效点，计算高度差是否大于阈值；如果是越界点，直接判为 0 (非障碍物)
        bool is_valid = (v_idx != -1);
        float min_z = is_valid ? min_z_grid[v_idx] : 0.0f;
        float z = soa_z[idx];
        
        // C++ 的比较运算符生成布尔值，直接静态转型为 uint8_t 写入 Global Memory
        // 没有任何指令分叉，完美压榨显存带宽
        is_obstacle[idx] = static_cast<uint8_t>(is_valid && (z - min_z > threshold));
    }
}

// ------------------------------------------------------------------
// 统一启动接口
// ------------------------------------------------------------------
void LaunchGroundFilter(
    const float* d_soa_z, 
    const int* d_voxel_index, 
    float* d_min_z_grid, 
    uint8_t* d_is_obstacle, 
    uint32_t num_points, 
    int grid_w, 
    int grid_h, 
    float height_threshold, 
    void* stream) 
{
    cudaStream_t cu_stream = static_cast<cudaStream_t>(stream);
    int threads = 256;

    // 1. 初始化 Grid (网格数量级别的并行)
    int total_grids = grid_w * grid_h;
    // 下面这个算法是为了防止直接除线程数导致向下取整导致的线程不足，同时也避免了线程过多导致的资源浪费
    // 多出来的一个跑不满的 block 会被越界处理
    int blocks_grid = (total_grids + threads - 1) / threads;
    InitMinZGridKernel<<<blocks_grid, threads, 0, cu_stream>>>(d_min_z_grid, total_grids);

    // 2. 收集最低点 (点云数量级别的并行)
    // 之所以是点云数量级别的并行，是因为每个点都要处理，数量级远大于网格数量级，所以需要更多线程来压榨并行度
    // 如果用网格数量级别，一个网格要处理天文数字的点数，会导致单线程压力过大，反而效率低下
    // 每个线程都会知道自己负责点属于哪个网格，直接原子更新那个网格的最低点高度，完美利用 GPU 的并行计算能力和原子操作支持
    int blocks_pts = (num_points + threads - 1) / threads;
    ComputeMinZKernel<<<blocks_pts, threads, 0, cu_stream>>>(
        d_soa_z, d_voxel_index, d_min_z_grid, num_points
    );

    // 3. 高度滤波 (点云数量级别的并行)
    FilterObstaclesKernel<<<blocks_pts, threads, 0, cu_stream>>>(
        d_soa_z, d_voxel_index, d_min_z_grid, d_is_obstacle, num_points, height_threshold
    );

    CHECK_CUDA(cudaGetLastError());
}

} // namespace flashbev