# PyTorch 双基线测试：A/B → C/D

只需要两个分支。A/B 共用原始代码，C/D 共用改进代码；通过 `--compile` 切换编译。

| 组别 | 代码分支 | compile |
|---|---|---|
| A | 当前原始分支 | 关闭 |
| B | 当前原始分支 | max-autotune |
| C | improve-yitao | 关闭 |
| D | improve-yitao | max-autotune |

以下命令在 **Linux GPU 服务器的 openpi-main 根目录、Bash 中**执行，使用已配置好的项目环境。

## 1. 准备

```bash
export CUDA_VISIBLE_DEVICES=0
export JAX_PLATFORMS=cpu
mkdir -p results/pytorch traces/pytorch
git branch --show-current
```

默认测试 Pi0.5、batch=1、10 步去噪、固定 seed=0。每组先跑一次首调用，再预热 5 次，正式测量 100 次。四组使用同一张空闲 GPU，依次运行。

默认随机权重和假观测，只用于性能测试。如使用真实权重，下面所有测试命令都加上同一个 `--checkpoint /path/to/model.safetensors`。改进若改变了随机初始化顺序，必须使用相同 checkpoint 才能保证权重一致。

## 2. 当前分支运行 A/B

保持原始模型代码不变，依次执行：

```bash
# A：原始代码，不启用 compile
uv run python benchmark_pytorch.py --compile off \
  --warmup 5 --runs 100 --output results/pytorch/A

# B：原始代码，启用 compile
uv run python benchmark_pytorch.py --compile max-autotune \
  --warmup 5 --runs 100 --output results/pytorch/B
```

## 3. 创建改进分支，运行 C/D

先在当前分支提交测试脚本、执行说明及需要保留的原始代码。只暂存相关文件，不必提交测试结果。例如：

```bash
git add benchmark_pytorch.py compare_pytorch_benchmarks.py docs/pytorch_benchmark.md
git commit -m "Add PyTorch eager and compile benchmark baselines"
git switch -c improve-yitao
```

若这些文件已经提交，跳过 `git add` 和 `git commit`。若分支已存在，改用 `git switch improve-yitao`。

**在 improve-yitao 修改算法并保存后**，在同一版改进代码上依次执行：

```bash
# C：改进代码，不启用 compile
uv run python benchmark_pytorch.py --compile off \
  --warmup 5 --runs 100 --output results/pytorch/C

# D：改进代码，启用 compile
uv run python benchmark_pytorch.py --compile max-autotune \
  --warmup 5 --runs 100 --output results/pytorch/D
```

保留 A/B 结果文件。普通 `git switch` 不会自动备份实验结果；若使用另一份仓库运行 C/D，先复制 A/B 的 `.json` 和 `.output.npz` 到相同结果目录。

## 4. 对比结果

```bash
# 原始代码的 compile 收益：A / B
uv run python compare_pytorch_benchmarks.py results/pytorch/A results/pytorch/B

# 重点：编译后，算法改进的额外收益：B / D
uv run python compare_pytorch_benchmarks.py results/pytorch/B results/pytorch/D

# 不编译时，算法改进的收益：A / C
uv run python compare_pytorch_benchmarks.py results/pytorch/A results/pytorch/C
```

`p50_speedup > 1` 表示后一组更快。例如 B/D=1.2 表示改进后加速 1.2 倍。同时查看 P95、输出误差和环境差异。输出误差检查不能代替任务成功率评测。

结果 `.json` 包含延迟、吞吐、显存和环境信息，`.output.npz` 保存动作输出。测量范围是输入已在 GPU 上的模型推理，不包含外部预处理、数据传输和网络。首调用含编译与执行，单独记录。

建议最终重复 3 轮，输出分别命名 `A_r1/B_r1/C_r1/D_r1` 等；脚本拒绝覆盖旧结果。复测 A/B 时切回原始分支，复测 C/D 时切回改进分支，并先提交好各自的代码。

## 5. 可选：分析为什么变快

下面 A/B 命令在原始分支执行。分析 C/D 时，在改进分支执行相同命令，将输出名称中的 A/B 换成 C/D。Profiler 数据只用于定位原因，不用于正式测速对比。

### PyTorch Profiler：看阶段和编译区域

```bash
uv run python benchmark_pytorch.py --compile off \
  --profile torch --stage-labels --warmup 5 --runs 5 \
  --output traces/pytorch/A_torch

uv run python benchmark_pytorch.py --compile max-autotune \
  --profile torch --warmup 5 --runs 5 \
  --output traces/pytorch/B_torch
```

用 [Perfetto](https://ui.perfetto.dev) 打开生成的 `.trace.json`，查看前缀、去噪阶段和 Torch-Compiled Region。内部阶段标记只用于 eager，避免影响 compiled 的编译结构；这些 CPU 标记不等于 GPU 阶段耗时。

### Nsight Systems：看 kernel 和同步

服务器需安装 `nsys`。以下命令仅捕获预热后的 5 次推理，并展开 CUDA Graph 内部 kernel：

```bash
uv run nsys profile --trace=cuda,nvtx --sample=none \
  --capture-range=cudaProfilerApi --capture-range-end=stop \
  --cuda-graph-trace=node -o traces/pytorch/A_nodes \
  python benchmark_pytorch.py --compile off --profile nsys --stage-labels \
  --warmup 5 --runs 5 --output traces/pytorch/A_nsys

uv run nsys profile --trace=cuda,nvtx --sample=none \
  --capture-range=cudaProfilerApi --capture-range-end=stop \
  --cuda-graph-trace=node -o traces/pytorch/B_nodes \
  python benchmark_pytorch.py --compile max-autotune --profile nsys \
  --warmup 5 --runs 5 --output traces/pytorch/B_nsys
```

用 Nsight Systems 打开 `.nsys-rep`，重点比较耗时最多的 kernel、调用次数、去噪步间同步和 GPU 空隙。同步等待时间不能直接当成可节省时间。Compile 参数表示请求编译，实际编译区域需要结合 trace 确认。

脚本已做本地静态检查，仍需在服务器验证 CUDA 执行。初次运行可先用独立输出前缀，设置 `--warmup 1 --runs 1` 验证环境。
