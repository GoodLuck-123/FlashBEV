#include <iostream>
#include <cuda_runtime.h>

// 编译命令: nvcc query_gpu.cpp -o query_gpu -run
int main() {
    int deviceCount = 0;
    cudaError_t error_id = cudaGetDeviceCount(&deviceCount);

    if (error_id != cudaSuccess) {
        std::cerr << "cudaGetDeviceCount 返回错误: " << static_cast<int>(error_id) << "\n";
        std::cerr << cudaGetErrorString(error_id) << "\n";
        return 1;
    }

    if (deviceCount == 0) {
        std::cout << "未发现支持 CUDA 的设备。\n";
        return 0;
    }

    int dev = 0; // 默认取第一张卡
    cudaSetDevice(dev);
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, dev);

    std::cout << "\n========== NVIDIA GPU Info ==========\n";
    std::cout << "显卡型号: " << deviceProp.name << "\n";
    std::cout << "计算能力: " << deviceProp.major << "." << deviceProp.minor << "\n";
    std::cout << "流处理器(SM)数量: " << deviceProp.multiProcessorCount << "\n";
    
    // 显存信息
    double totalMemGB = static_cast<double>(deviceProp.totalGlobalMem) / (1024.0 * 1024.0 * 1024.0);
    std::cout << "显存总量 (Global Memory): " << totalMemGB << " GB\n";
    
    // 带宽计算 (Memory Clock * Bus Width * 2 / 8) -> GB/s
    // memoryClockRate 单位是 kHz
    double memClock = deviceProp.memoryClockRate * 1e3; 
    double busWidth = deviceProp.memoryBusWidth; 
    double bandwidth = 2.0 * memClock * (busWidth / 8.0) / 1.0e9;
    
    std::cout << "显存位宽: " << deviceProp.memoryBusWidth << "-bit\n";
    std::cout << "显存频率: " << deviceProp.memoryClockRate / 1000.0 << " MHz\n";
    std::cout << "理论显存带宽: " << bandwidth << " GB/s\n";
    
    // 线程限制
    std::cout << "每个 Block 最大线程数: " << deviceProp.maxThreadsPerBlock << "\n";
    std::cout << "每个 SM 最大常驻线程数: " << deviceProp.maxThreadsPerMultiProcessor << "\n";
    std::cout << "Warp Size: " << deviceProp.warpSize << "\n";
    std::cout << "=====================================\n";

    return 0;
}