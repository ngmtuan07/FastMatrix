#include "fastmatrix.h"
#include <iostream>
#include <vector>
#include <fstream>
#include <iomanip>
#include <cstdlib>

int main(int argc, char** argv) {
    // List of sizes to benchmark. You can modify or pass sizes via args.
    std::vector<int> sizes = {128, 256, 512, 1024}; // adjust as needed
    if (argc > 1) {
        sizes.clear();
        for (int i = 1; i < argc; ++i) sizes.push_back(std::atoi(argv[i]));
    }

    // Ensure data/ exists before running (benchmark writes to data/results.csv)
    std::ofstream test("data/.touch");
    test.close();

    std::ofstream out("data/results.csv");
    if (!out.is_open()) {
        std::cerr << "Failed to open data/results.csv for writing. Create data/ directory first.\n";
        return EXIT_FAILURE;
    }

    out << "N,CPU_Add_ms,GPU_Add_ms,GPU_Add_FP16_ms,CPU_Mul_ms,GPU_Mul_ms,GPU_Mul_FP16_ms\n";
    std::cout << "Benchmark sizes: ";
    for (int s : sizes) std::cout << s << " ";
    std::cout << "\n\n";

    for (int N : sizes) {
        std::cout << "Running N=" << N << " ...\n";
        // CPU add
        double cpuAdd = cpu_add_bench(N);
        std::cout << "  CPU Add: " << std::fixed << std::setprecision(3) << cpuAdd << " ms\n";

        // GPU add (FP32)
        double gpuAdd = -1;
        try {
            gpuAdd = gpu_add_bench(N);
            std::cout << "  GPU Add: " << gpuAdd << " ms\n";
        } catch (...) {
            std::cerr << "  GPU Add failed (no CUDA device / error). Setting GPU_Add_ms = -1\n";
            gpuAdd = -1;
        }

        // GPU add FP16 (with correctness check printed by the function)
        double gpuAddFP16 = -1;
        try {
            gpuAddFP16 = gpu_add_bench_fp16(N);
        } catch (...) {
            std::cerr << "  GPU Add FP16 failed (no CUDA device / error). Setting GPU_Add_FP16_ms = -1\n";
            gpuAddFP16 = -1;
        }

        // CPU mul
        double cpuMul = cpu_mul_bench(N);
        std::cout << "  CPU Mul: " << cpuMul << " ms\n";

        // GPU mul (FP32)
        double gpuMul = -1;
        try {
            gpuMul = gpu_mul_bench(N);
            std::cout << "  GPU Mul: " << gpuMul << " ms\n";
        } catch (...) {
            std::cerr << "  GPU Mul failed (no CUDA device / error). Setting GPU_Mul_ms = -1\n";
            gpuMul = -1;
        }

        // GPU mul FP16 (with correctness check printed by the function)
        double gpuMulFP16 = -1;
        try {
            gpuMulFP16 = gpu_mul_bench_fp16(N);
        } catch (...) {
            std::cerr << "  GPU Mul FP16 failed (no CUDA device / error). Setting GPU_Mul_FP16_ms = -1\n";
            gpuMulFP16 = -1;
        }

        out << N << "," << cpuAdd << "," << gpuAdd << "," << gpuAddFP16 << ","
            << cpuMul << "," << gpuMul << "," << gpuMulFP16 << "\n";
        out.flush();
        std::cout << "Wrote results for N=" << N << " to data/results.csv\n\n";
    }

    out.close();
    std::cout << "Benchmark completed. Plot results with: python3 scripts/plot_results.py\n";
    return 0;
}
