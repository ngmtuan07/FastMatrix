#include "fastmatrix.h"
#include <vector>
#include <chrono>
#include <random>
#include <iostream>

// Initialize matrix with random floats in [0,1)
static void initMatrix(std::vector<float> &A, int N) {
    std::mt19937 rng(12345);
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    A.resize(N * N);
    for (int i = 0; i < N * N; ++i) A[i] = dist(rng);
}

extern "C" double cpu_add_bench(int N) {
    std::vector<float> A, B, C;
    initMatrix(A, N);
    initMatrix(B, N);
    C.resize(N * N);

    auto t1 = std::chrono::high_resolution_clock::now();
    // CPU matrix addition
    for (int i = 0; i < N * N; ++i) C[i] = A[i] + B[i];
    auto t2 = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t2 - t1).count();
    // protect against unused optimization in some builds: print 1 element (commented)
    // std::cout << "C[0]=" << C[0] << "\n";
    return ms;
}

extern "C" double cpu_mul_bench(int N) {
    std::vector<float> A, B, C;
    initMatrix(A, N);
    initMatrix(B, N);
    C.assign(N * N, 0.0f);

    auto t1 = std::chrono::high_resolution_clock::now();
    // Naive CPU matrix multiplication
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < N; ++k) sum += A[i * N + k] * B[k * N + j];
            C[i * N + j] = sum;
        }
    }
    auto t2 = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t2 - t1).count();
    return ms;
}
