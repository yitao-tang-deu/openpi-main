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

首次运行前，安装项目定制的 Transformers 文件。若初始化报 `transformers_replace is not installed correctly`，在仓库根目录执行以下命令；目标路径从实际 Python 环境获取，不写死 Python 3.11：

```bash
uv sync
uv run --no-sync python - <<'PY'
from pathlib import Path
import shutil
import transformers

assert transformers.__version__ == "4.53.2", transformers.__version__
source = Path("src/openpi/models_pytorch/transformers_replace")
target = Path(transformers.__file__).resolve().parent
assert source.is_dir(), source
for path in source.rglob("*.py"):
    destination = target / path.relative_to(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(path, destination)
print(f"Installed project Transformers files into {target}")
PY

uv run --no-sync python - <<'PY'
import transformers
from transformers.models.siglip import check
from transformers.models.gemma import modeling_gemma

assert check.check_whether_transformers_replace_is_installed_correctly()
assert "cond" in __import__("inspect").signature(modeling_gemma.GemmaRMSNorm.forward).parameters
print("Transformers replacement OK:", transformers.__version__, transformers.__file__)
PY
```

若之后重装或同步时重新安装了 Transformers，需再次复制并验证。该操作只修改当前虚拟环境中的 Transformers 文件。

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

## 6. 当前分支：相机合批消融

PyTorch eval 默认启用相机合批：将同形状、同 dtype、同 device 的相机沿 batch 维拼接，视觉编码一次后拆回原相机顺序。相机 mask 和语言 token 的拼接保持原逻辑。训练模式和不兼容输入保留逐相机执行。可通过 `Pi0Config(pytorch_camera_batching=False)` 关闭；JAX 路径不受该配置影响。

在当前分支、同一 checkpoint 下直接比较，无需切换分支。以下命令在 Linux GPU 服务器运行，将 checkpoint 路径替换为实际文件：

```bash
for mode in off max-autotune; do
  for cameras in serial batched; do
    uv run python benchmark_pytorch.py --compile "$mode" \
      --image-batching "$cameras" --checkpoint /path/to/model.safetensors \
      --warmup 5 --runs 100 --output "results/pytorch/cameras_${mode}_${cameras}"
  done
  uv run python compare_pytorch_benchmarks.py \
    "results/pytorch/cameras_${mode}_serial" "results/pytorch/cameras_${mode}_batched"
done
```

检查输出误差、P50/P95 和 JSON 中的峰值显存。合批扩大视觉编码的同时存活激活，可能增加峰值显存；不同 GEMM batch 形状也可能引入浮点误差，不能要求逐位一致。继续用 `--batch-size 2` 验证多样本顺序，并使用新输出前缀。正式结论应按上文重复 3 轮。

需要定位时，在第 5 节 nsys 命令中分别加 `--image-batching serial` / `--image-batching batched`。比较视觉阶段 kernel 形状、耗时与调用数量；合批不要求出现多个 CUDA stream。尚未提供 CUDA 速度或显存收益实测。

本地验证：在隔离的 Windows / Python 3.14 / PyTorch 2.14 CPU 环境，合批单元测试 12 项通过（batch=1/2/4、两种图像布局、串行回退、`torch.compile` eager backend 的 fullgraph 捕获）。完整 prefix/mask/训练路径集成测试因缺少 JAX 项目依赖而跳过；项目固定版本 PyTorch 2.7.1、CUDA/Inductor 与真实模型数值仍需服务器验证：

```bash
uv run pytest -q src/openpi/models_pytorch/image_batching_test.py
```
