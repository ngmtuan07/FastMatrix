#include "fastmatrix.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <iostream>
#include <cstdlib>
#include <cmath>
#include <chrono>
#include <random>

#define TILE 16  // tile size for shared memory multiplication

// CUDA error checking helper
#define CUDA_CALL(call) do {                                  \
    cudaError_t err = (call);                                 \
    if (err != cudaSuccess) {                                 \
        fprintf(stderr, "CUDA error %s:%d: %s\n",             \
                __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE);                                   \
    }                                                         \
} while (0)

// ---------- FP32 kernels ----------

// Kernel: matrix addition (C = A + B)
__global__ void matAddKernel(const float* A, const float* B, float* C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < N && col < N) {
        int idx = row * N + col;
        C[idx] = A[idx] + B[idx];
    }
}

// Kernel: tiled/shared-memory matrix multiplication (FP32)
__global__ void matMulTiledKernel(const float* A, const float* B, float* C, int N) {
    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int row = blockIdx.y * TILE + ty;
    int col = blockIdx.x * TILE + tx;

    float sum = 0.0f;
    int numTiles = (N + TILE - 1) / TILE;

    for (int t = 0; t < numTiles; ++t) {
        int aRow = row;
        int aCol = t * TILE + tx;
        int bRow = t * TILE + ty;
        int bCol = col;

        As[ty][tx] = (aRow < N && aCol < N) ? A[aRow * N + aCol] : 0.0f;
        Bs[ty][tx] = (bRow < N && bCol < N) ? B[bRow * N + bCol] : 0.0f;

        __syncthreads();

        for (int k = 0; k < TILE; ++k) sum += As[ty][k] * Bs[k][tx];

        __syncthreads();
    }

    if (row < N && col < N) C[row * N + col] = sum;
}

// ---------- FP16 / mixed-precision kernels ----------
// NOTE: kernels accept float* inputs and float* outputs for easy host-device memcpy.
// Internally they do FP16 arithmetic (convert inputs to half, perform operations in
// half or mixed precision) and round results to FP16-equivalent before storing to output
// (stored as float values of the rounded FP16 results). This makes it easy to compare
// FP16-result-to-FP32-reference on host.

__global__ void matAddFP16Kernel_fromFloatInputs(const float* A, const float* B, float* C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < N && col < N) {
        int idx = row * N + col;
        // convert to half
        __half ha = __float2half(A[idx]);
        __half hb = __float2half(B[idx]);
        // half addition
        __half hr = __hadd(ha, hb);
        // store as float (rounded to FP16 then promoted)
        C[idx] = __half2float(hr);
    }
}

__global__ void matMulTiledFP16Kernel_fromFloatInputs(const float* A, const float* B, float* C, int N) {
    // shared arrays of half values
    __shared__ __half As[TILE][TILE];
    __shared__ __half Bs[TILE][TILE];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int row = blockIdx.y * TILE + ty;
    int col = blockIdx.x * TILE + tx;

    float sum = 0.0f;
    int numTiles = (N + TILE - 1) / TILE;

    for (int t = 0; t < numTiles; ++t) {
        int aRow = row;
        int aCol = t * TILE + tx;
        int bRow = t * TILE + ty;
        int bCol = col;

        // load and convert to half in shared memory
        float aVal = (aRow < N && aCol < N) ? A[aRow * N + aCol] : 0.0f;
        float bVal = (bRow < N && bCol < N) ? B[bRow * N + bCol] : 0.0f;
        As[ty][tx] = __float2half(aVal);
        Bs[ty][tx] = __float2half(bVal);

        __syncthreads();

        // accumulate in float (mixed precision), but inputs are quantized to half
        for (int k = 0; k < TILE; ++k) {
            float af = __half2float(As[ty][k]);
            float bf = __half2float(Bs[k][tx]);
            sum += af * bf;
        }

        __syncthreads();
    }

    if (row < N && col < N) {
        // round final sum to FP16-equivalent by converting to half then back to float
        __half hsum = __float2half(sum);
        C[row * N + col] = __half2float(hsum);
    }
}

// ---------- host helpers ----------
static void initHostMatrix(float* h, int N) {
    std::mt19937 rng(12345);
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    for (int i = 0; i < N * N; ++i) h[i] = dist(rng);
}

// ---------- wrappers ----------

// FP32 GPU add benchmark
extern "C" double gpu_add_bench(int N) {
    size_t bytes = size_t(N) * size_t(N) * sizeof(float);
    float *hA = (float*)malloc(bytes);
    float *hB = (float*)malloc(bytes);
    float *hC = (float*)malloc(bytes);

    initHostMatrix(hA, N);
    initHostMatrix(hB, N);

    float *dA = nullptr, *dB = nullptr, *dC = nullptr;
    CUDA_CALL(cudaMalloc(&dA, bytes));
    CUDA_CALL(cudaMalloc(&dB, bytes));
    CUDA_CALL(cudaMalloc(&dC, bytes));

    CUDA_CALL(cudaMemcpy(dA, hA, bytes, cudaMemcpyHostToDevice));
    CUDA_CALL(cudaMemcpy(dB, hB, bytes, cudaMemcpyHostToDevice));

    dim3 threads(16, 16);
    dim3 blocks((N + threads.x - 1) / threads.x, (N + threads.y - 1) / threads.y);

    cudaEvent_t start, stop;
    CUDA_CALL(cudaEventCreate(&start));
    CUDA_CALL(cudaEventCreate(&stop));
    CUDA_CALL(cudaEventRecord(start));

    matAddKernel<<<blocks, threads>>>(dA, dB, dC, N);
    CUDA_CALL(cudaGetLastError());
    CUDA_CALL(cudaEventRecord(stop));
    CUDA_CALL(cudaEventSynchronize(stop));

    float ms;
    CUDA_CALL(cudaEventElapsedTime(&ms, start, stop));

    CUDA_CALL(cudaMemcpy(hC, dC, bytes, cudaMemcpyDeviceToHost));

    CUDA_CALL(cudaFree(dA));
    CUDA_CALL(cudaFree(dB));
    CUDA_CALL(cudaFree(dC));
    free(hA); free(hB); free(hC);
    CUDA_CALL(cudaEventDestroy(start));
    CUDA_CALL(cudaEventDestroy(stop));
    return double(ms);
}

// FP32 GPU mul benchmark
extern "C" double gpu_mul_bench(int N) {
    size_t bytes = size_t(N) * size_t(N) * sizeof(float);
    float *hA = (float*)malloc(bytes);
    float *hB = (float*)malloc(bytes);
    float *hC = (float*)malloc(bytes);

    initHostMatrix(hA, N);
    initHostMatrix(hB, N);

    float *dA = nullptr, *dB = nullptr, *dC = nullptr;
    CUDA_CALL(cudaMalloc(&dA, bytes));
    CUDA_CALL(cudaMalloc(&dB, bytes));
    CUDA_CALL(cudaMalloc(&dC, bytes));

    CUDA_CALL(cudaMemcpy(dA, hA, bytes, cudaMemcpyHostToDevice));
    CUDA_CALL(cudaMemcpy(dB, hB, bytes, cudaMemcpyHostToDevice));

    dim3 threads(TILE, TILE);
    dim3 blocks((N + TILE - 1) / TILE, (N + TILE - 1) / TILE);

    cudaEvent_t start, stop;
    CUDA_CALL(cudaEventCreate(&start));
    CUDA_CALL(cudaEventCreate(&stop));
    CUDA_CALL(cudaEventRecord(start));

    matMulTiledKernel<<<blocks, threads>>>(dA, dB, dC, N);
    CUDA_CALL(cudaGetLastError());
    CUDA_CALL(cudaEventRecord(stop));
    CUDA_CALL(cudaEventSynchronize(stop));

    float ms;
    CUDA_CALL(cudaEventElapsedTime(&ms, start, stop));

    CUDA_CALL(cudaFree(dA));
    CUDA_CALL(cudaFree(dB));
    CUDA_CALL(cudaFree(dC));
    free(hA); free(hB); free(hC);
    CUDA_CALL(cudaEventDestroy(start));
    CUDA_CALL(cudaEventDestroy(stop));
    return double(ms);
}

// ---------- FP16 wrappers with correctness checks ----------

// Returns ms and prints FP16 error stats compared to FP32 reference
extern "C" double gpu_add_bench_fp16(int N) {
    size_t bytes = size_t(N) * size_t(N) * sizeof(float);
    float *hA = (float*)malloc(bytes);
    float *hB = (float*)malloc(bytes);
    float *hC_fp16 = (float*)malloc(bytes); // device result stored as float (rounded from fp16)
    float *hC_ref = (float*)malloc(bytes);

    initHostMatrix(hA, N);
    initHostMatrix(hB, N);

    // compute FP32 reference on host
    for (int i = 0; i < N * N; ++i) hC_ref[i] = hA[i] + hB[i];

    float *dA = nullptr, *dB = nullptr, *dC = nullptr;
    CUDA_CALL(cudaMalloc(&dA, bytes));
    CUDA_CALL(cudaMalloc(&dB, bytes));
    CUDA_CALL(cudaMalloc(&dC, bytes));

    CUDA_CALL(cudaMemcpy(dA, hA, bytes, cudaMemcpyHostToDevice));
    CUDA_CALL(cudaMemcpy(dB, hB, bytes, cudaMemcpyHostToDevice));

    dim3 threads(16, 16);
    dim3 blocks((N + threads.x - 1) / threads.x, (N + threads.y - 1) / threads.y);

    cudaEvent_t start, stop;
    CUDA_CALL(cudaEventCreate(&start));
    CUDA_CALL(cudaEventCreate(&stop));
    CUDA_CALL(cudaEventRecord(start));

    matAddFP16Kernel_fromFloatInputs<<<blocks, threads>>>(dA, dB, dC, N);
    CUDA_CALL(cudaGetLastError());

    CUDA_CALL(cudaEventRecord(stop));
    CUDA_CALL(cudaEventSynchronize(stop));

    float ms;
    CUDA_CALL(cudaEventElapsedTime(&ms, start, stop));

    CUDA_CALL(cudaMemcpy(hC_fp16, dC, bytes, cudaMemcpyDeviceToHost));

    // correctness check: compute max abs error and mean abs error
    double max_abs = 0.0;
    double sum_abs = 0.0;
    for (int i = 0; i < N * N; ++i) {
        double err = fabs((double)hC_fp16[i] - (double)hC_ref[i]);
        sum_abs += err;
        if (err > max_abs) max_abs = err;
    }
    double mean_abs = sum_abs / double(N * N);

    std::cout << "[FP16 Add] N=" << N << "  ms=" << ms
              << "  max_abs=" << max_abs << "  mean_abs=" << mean_abs << "\n";

    CUDA_CALL(cudaFree(dA));
    CUDA_CALL(cudaFree(dB));
    CUDA_CALL(cudaFree(dC));
    free(hA); free(hB); free(hC_fp16); free(hC_ref);
    CUDA_CALL(cudaEventDestroy(start));
    CUDA_CALL(cudaEventDestroy(stop));
    return double(ms);
}

// Returns ms and prints FP16 error stats compared to FP32 reference
extern "C" double gpu_mul_bench_fp16(int N) {
    size_t bytes = size_t(N) * size_t(N) * sizeof(float);
    float *hA = (float*)malloc(bytes);
    float *hB = (float*)malloc(bytes);
    float *hC_fp16 = (float*)malloc(bytes); // device result stored as float (rounded from fp16)
    float *hC_ref = (float*)malloc(bytes);

    initHostMatrix(hA, N);
    initHostMatrix(hB, N);

    // compute FP32 reference on host (naive) — caution: expensive for big N
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            double sum = 0.0;
            for (int k = 0; k < N; ++k) sum += double(hA[i * N + k]) * double(hB[k * N + j]);
            hC_ref[i * N + j] = float(sum);
        }
    }

    float *dA = nullptr, *dB = nullptr, *dC = nullptr;
    CUDA_CALL(cudaMalloc(&dA, bytes));
    CUDA_CALL(cudaMalloc(&dB, bytes));
    CUDA_CALL(cudaMalloc(&dC, bytes));

    CUDA_CALL(cudaMemcpy(dA, hA, bytes, cudaMemcpyHostToDevice));
    CUDA_CALL(cudaMemcpy(dB, hB, bytes, cudaMemcpyHostToDevice));

    dim3 threads(TILE, TILE);
    dim3 blocks((N + TILE - 1) / TILE, (N + TILE - 1) / TILE);

    cudaEvent_t start, stop;
    CUDA_CALL(cudaEventCreate(&start));
    CUDA_CALL(cudaEventCreate(&stop));
    CUDA_CALL(cudaEventRecord(start));

    matMulTiledFP16Kernel_fromFloatInputs<<<blocks, threads>>>(dA, dB, dC, N);
    CUDA_CALL(cudaGetLastError());

    CUDA_CALL(cudaEventRecord(stop));
    CUDA_CALL(cudaEventSynchronize(stop));

    float ms;
    CUDA_CALL(cudaEventElapsedTime(&ms, start, stop));

    CUDA_CALL(cudaMemcpy(hC_fp16, dC, bytes, cudaMemcpyDeviceToHost));

    // correctness check: compute max abs error and mean abs error
    double max_abs = 0.0;
    double sum_abs = 0.0;
    for (int i = 0; i < N * N; ++i) {
        double err = fabs((double)hC_fp16[i] - (double)hC_ref[i]);
        sum_abs += err;
        if (err > max_abs) max_abs = err;
    }
    double mean_abs = sum_abs / double(N * N);

    std::cout << "[FP16 Mul] N=" << N << "  ms=" << ms
              << "  max_abs=" << max_abs << "  mean_abs=" << mean_abs << "\n";

    CUDA_CALL(cudaFree(dA));
    CUDA_CALL(cudaFree(dB));
    CUDA_CALL(cudaFree(dC));
    free(hA); free(hB); free(hC_fp16); free(hC_ref);
    CUDA_CALL(cudaEventDestroy(start));
    CUDA_CALL(cudaEventDestroy(stop));
    return double(ms);
}
