import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('build/data/results.csv')

# Plot add performance
plt.figure()
plt.plot(df['N'], df['CPU_Add_ms'], label='CPU Add')
plt.plot(df['N'], df['GPU_Add_ms'], label='GPU Add (FP32)')
plt.plot(df['N'], df['GPU_Add_FP16_ms'], label='GPU Add (FP16)')
plt.xlabel('Matrix Size (N)')
plt.ylabel('Time (ms)')
plt.title('Matrix Addition: CPU vs GPU')
plt.legend()
plt.grid(True)
plt.savefig('runtime_add_comparison.png')

# Plot multiply performance
plt.figure()
plt.plot(df['N'], df['CPU_Mul_ms'], label='CPU Mul')
plt.plot(df['N'], df['GPU_Mul_ms'], label='GPU Mul (FP32)')
plt.plot(df['N'], df['GPU_Mul_FP16_ms'], label='GPU Mul (FP16)')
plt.xlabel('Matrix Size (N)')
plt.ylabel('Time (ms)')
plt.title('Matrix Multiplication: CPU vs GPU')
plt.legend()
plt.grid(True)
plt.savefig('runtime_mul_comparison.png')

plt.show()
