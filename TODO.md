# GPU 上机流程

租到 GPU 后按这个顺序做。预计 2-3 小时搞定全部。

---

## Step 0: 环境确认 (5 min)

```bash
# 确认 GPU 和 CUDA
nvidia-smi
nvcc --version
cmake --version

# clone 项目
git clone https://github.com/你的用户名/cuda-attention-engine.git
cd cuda-attention-engine
```

如果用的是 Deep Learning AMI (AWS/Lambda)，CUDA 和 CMake 应该已经装好了。

---

## Step 1: 编译 + 正确性测试 (10 min)

```bash
mkdir build && cd build
cmake ..
make -j$(nproc)
cd ..

# 跑 correctness test
./build/test_correctness
```

期望输出：
```
[CHECK] Naive FP32 vs CPU reference:
  PASSED
[CHECK] Tiled FP32 vs CPU reference:
  PASSED
[CHECK] FP16 vs CPU reference:
  PASSED
[CHECK] Tensor Core vs CPU reference:
  PASSED

=== ALL TESTS PASSED ===
```

如果某个 FAILED → 记录 max error，可能是精度问题需要调 tolerance。

---

## Step 2: Kernel Benchmark (10 min)

```bash
./build/benchmark
```

期望输出：对比表。**截图保存这个表，面试要用。**

```
╔═══════════════════════════╦══════════╦══════════╦══════════╗
║ Kernel                    ║ Time(ms) ║  TFLOPS  ║ Speedup  ║
╠═══════════════════════════╬══════════╬══════════╬══════════╣
║ Naive FP32                ║   x.xxx  ║  x.xxxx  ║  1.00x   ║
║ Tiled FP32 (Shared Mem)   ║   x.xxx  ║  x.xxxx  ║  x.xxx   ║
║ FP16 Mixed Precision      ║   x.xxx  ║  x.xxxx  ║  x.xxx   ║
║ Tensor Core (WMMA)        ║   x.xxx  ║  x.xxxx  ║  x.xxx   ║
╚═══════════════════════════╩══════════╩══════════╩══════════╝
```

记录数字，填到 README 和 docs/profiling_analysis.md。

---

## Step 3: RL Workload Benchmark (15 min)

```bash
pip install torch flash-attn
python bench/workload_bench.py
```

**截图保存输出。** 重点关注：
- Workload 2 的 padding waste 百分比
- Workload 3 的 prefix redundancy 百分比
- Workload 4 的 scaling curve（是否呈 O(N²) 增长）
- FA2 vs SDPA 在哪个 seq_len 开始拉开差距

---

## Step 4: Nsight Systems (系统级 profiling) (20 min)

```bash
# 看 CUDA kernel 的 benchmark timeline
nsys profile --stats=true -o reports/bench_timeline ./build/benchmark

# 看 Python workload 的 timeline
nsys profile --trace=cuda -o reports/workload_timeline python bench/workload_bench.py
```

**看什么：**
- kernel 和 kernel 之间有没有 idle gap（= launch overhead）
- 3-kernel attention (QK→softmax→SV) 之间的 gap 有多大
- cudaMemcpy 占了多少时间

如果有 nsys-ui（本地 GUI）：
```bash
# 把 .nsys-rep 文件下载到本地用 nsys-ui 打开
scp gpu-server:~/cuda-attention-engine/reports/*.nsys-rep .
```

---

## Step 5: Nsight Compute (kernel 级 profiling) (30 min)

```bash
mkdir -p reports

# Profile 自己的 kernel（重点）
ncu --set full --export reports/all_kernels ./build/benchmark

# 只看 QK kernel 对比（naive vs tiled vs fp16 vs tc）
ncu --kernel-name regex:"qk" --set full --export reports/qk_comparison ./build/benchmark

# Profile PyTorch SDPA attention kernel
ncu --kernel-name regex:"attention|flash|sdpa" --set full \
    python -c "
import torch, torch.nn.functional as F
Q=torch.randn(8,32,2048,128,device='cuda',dtype=torch.float16)
K=torch.randn(8,32,2048,128,device='cuda',dtype=torch.float16)
V=torch.randn(8,32,2048,128,device='cuda',dtype=torch.float16)
for _ in range(10): F.scaled_dot_product_attention(Q,K,V,is_causal=True)
torch.cuda.synchronize()
"
```

**看什么（对每个 kernel 记录）：**
- Compute Throughput: ___%
- Memory Throughput: ___%
- Achieved Occupancy: ___%
- L2 Cache Hit Rate: ___%
- 诊断: memory bound / compute bound / latency bound

---

## Step 6: 填分析报告 (30 min)

打开 `docs/profiling_analysis.md`，把上面拿到的数据填进去。

重点写：
1. 每步优化为什么有效（用 ncu 数据支撑）
2. Tiled 比 Naive 快是因为 memory throughput 降了还是 compute 升了？
3. 在哪个 seq_len 下 FA2 明显优于 SDPA？为什么？
4. 3-kernel 之间的 launch gap 占总时间多少？

---

## Step 7: 更新 README (10 min)

把实际数字填回 README.md：
- 对比表（从 Step 2）
- 关键 insight（从 Step 5/6）
- 硬件信息（GPU 型号、CUDA 版本）

```bash
git add -A
git commit -m "Add profiling results and analysis"
git push
```

---

## 如果时间有剩余

优先级从高到低：

1. **多跑几个 seq_len 的 ncu**（512, 2048, 4096, 8192），看趋势
2. **尝试 CUDA Graph**：看有无 graph 时 launch overhead 差异
3. **对比不同 batch size** 下的 scaling 行为
4. **截几张 nsys timeline 图**放到 docs/ 里（面试展示用）

---

## 常见问题

**Q: nvcc 报错 "unsupported GPU architecture"**
→ 在 CMakeLists.txt 加: `set(CMAKE_CUDA_ARCHITECTURES 80)` (A100=80, H100=90, V100=70)

**Q: Tensor Core kernel 结果不对**
→ 检查 seq_len 和 head_dim 是否是 16 的倍数（WMMA 要求）

**Q: flash-attn 装不上**
→ 需要 CUDA 11.6+ 和对应的 PyTorch 版本。试 `pip install flash-attn --no-build-isolation`

**Q: ncu 报 "ERR_NVGPUCTRPERM"**
→ 需要 root 或设置: `echo 1 > /proc/sys/kernel/perf_event_paranoid` (需 sudo)
