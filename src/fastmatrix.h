#pragma once

#ifdef __cplusplus
extern "C" {
#endif

// Benchmarks return elapsed milliseconds as double
double gpu_add_bench(int N);
double gpu_mul_bench(int N);

double gpu_add_bench_fp16(int N);  // FP16 / mixed-precision variant (returns ms and prints errors)
double gpu_mul_bench_fp16(int N);  // FP16 / mixed-precision variant (returns ms and prints errors)

double cpu_add_bench(int N);
double cpu_mul_bench(int N);

#ifdef __cplusplus
}
#endif
