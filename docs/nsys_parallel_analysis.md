# 当前分支的并行加速可行性分析

分析代码：`improve-yitao`，HEAD `094bf08`。本次只分析已有 SQLite 导出与代码，没有运行 GPU 优化实验，也没有修改模型。

## 1. 报告适用范围

已有报告运行的是 **JAX/XLA 的 `benchmark_infer_speed.py`**，不是 `benchmark_pytorch.py`。当前 JAX 脚本默认 Pi0，PyTorch benchmark 默认 Pi0.5；不能把这些报告当作 PyTorch A/B/C/D 的实测结果。历史报告未提供代码 commit 和完整模型配置，当前源码用于依赖分析，不代表已确认历史代码完全一致。

存在两份不同的 `pi0_nodes.sqlite`：

| 文件 | 5 次稳态 NVTX 推理耗时（ms） | kernel 数/次 |
|---|---|---:|
| `../nsys报告/pi0_nodes.sqlite` | 78.018、78.019、146.331、106.424、77.985 | 6,930 |
| `./pi0_nodes.sqlite` | 77.378、77.432、77.629、82.978、77.582 | 6,750 |

两份都是 node 模式，命令包含 `--nvtx --warmup 3 --runs 5`。不能将其合并为同一次采样。前者中位数 78.019 ms，两次较慢样本必须保留；缺少 GPU 时钟、其他进程和系统状态证据，不能确定抖动原因。

`pi0_inference_2024.sqlite` 为 graph 模式，含初始化和 autotune，没有上述 measurement 标记。其全局 kernel 排名中的 `redzone_checker`、`DelayKernel` 等不代表稳态模型热点；不能用它与 node 模式 kernel 总时长直接对比。报告 GPU 名称为 NVIDIA GeForce RTX 4090，128 SM；没有据此推断占用率或带宽饱和程度。

## 2. 稳态证据

统计方法：用 NVTX `infer_0000` 到 `infer_0004` 各自的起止时间，筛选完全位于区间内的 CUDA kernel/API/memcpy。kernel 均在 stream 13；记录存在每次合计约 11–12 微秒的区间重叠，活跃时间使用区间并集去重，不根据这些微小重叠推断有效计算并发。活跃时间不是 SM 利用率；下文各 kernel 名称组仍按原始时长求和。

| 指标（每次推理） | `nsys报告` 全部 5 次 | 仓库根目录全部 5 次 |
|---|---:|---:|
| 平均 NVTX 推理时长 | 97.356 ms | 78.600 ms |
| 平均 kernel 活跃时长（区间并集） | 88.472 ms | 74.923 ms |
| CUDA kernel stream 数 | 1 | 1 |
| `cuGraphLaunch` 次数 | 308 | 308 |
| `cuStreamSynchronize` 次数 | 11 | 11 |
| D2H 次数 | 11，每次 1 byte | 11，每次 1 byte |
| GPU D2H 时长合计 | 0.0135 ms | 0.0133 ms |
| GPU D2D 时长合计 | 0.3217 ms | 0.3199 ms |

`nsys报告` 中第 0、1、4 次较稳定：平均 78.008 ms，其中 kernel 活跃 75.438 ms，占 96.71%。这里只用这三次描述稳定状态，同时保留上表全部样本。

由此得到两个结论：

- **只填补没有 kernel 的时间，空间较小。** 对这三个稳定样本，即使把所有非 kernel 时间全部消除，仍保留原 kernel 活跃时长，也仅约 `78.008 / 75.438 = 1.034×`。这是该受限假设的上界，不是所有优化的上界，也不是无 profiler 的速度预测。
- **同步 API 的时长不是可直接删除的开销。** `nsys报告` 中 CPU `cuStreamSynchronize` 累计平均 82.820 ms/次，与 GPU 工作重叠；不能从 97.356 ms 中直接减掉它。1-byte D2H 和每次 11 次同步与循环条件回读相符，但还需用 HLO/调用关联确认具体来源。

稳定样本中，按 kernel 名称前缀聚合：

| kernel 名称组 | `nsys报告` 第 0、1、4 次平均 | 仓库根目录 5 次平均 |
|---|---:|---:|
| `loop_slice*` | 14.807 ms | 14.819 ms |
| `loop_convert*` | 18.220 ms | 18.240 ms |
| `loop_transpose*` | 2.127 ms | 2.126 ms |

这三组约 35 ms，是值得追查的数据移动/布局/转换线索。**名称含 fusion，不能将整个 kernel 当成纯拷贝，也不能承诺全部消除。** `loop_slice_fusion_1` 每次 180 次、`loop_slice_fusion` 每次 18 次，与代码层循环和去噪重复结构相符；精确映射需要 HLO 或 kernel 源码。CUDA memcpy 表中的 D2D 仅约 0.32 ms，不能代表 kernel 内部所有数据搬运。

## 3. 可并行的计算与串行依赖

```text
相机 1 编码 ─┐
相机 2 编码 ─┼─ 拼接图像/文本 token → prefix Transformer / KV cache
相机 3 编码 ─┘                              ↓
                               去噪 1 → 去噪 2 → … → 去噪 10
```

相机编码独立。去噪更新 `x_next = x_t + dt * v(x_t, t, KV)`，下一步依赖本步结果；同一步内动作 token 已通过矩阵计算并行处理，但 attention 使它们相互耦合，不能任意拆成独立动作请求。两套实现都已在一次推理内复用 prefix KV cache。

## 4. 优化候选与优先级

### P1：三路相机合批，优先于三路 stream

代码位置：PyTorch `pi0_pytorch.py::embed_prefix`，JAX `pi0.py::embed_prefix`。当前分别调用同一视觉编码器。

把同尺寸图像沿 batch 维合成 `[3B,H,W,C]`（按实际输入布局调整），一次编码后拆回三组，恢复原相机 token 顺序及 mask。这是最明确的独立分支并行候选，可能减少重复调度并提高矩阵计算效率。合批不能沿 token 维直接混合相机，否则会改变视觉编码器内部 attention 语义。

备选是三个 CUDA streams 配合事件汇合，但共享 GPU 算力、显存带宽和缓存可能使收益很小；输入依赖、输出汇合与张量生命周期也需显式管理。优先比较串行、合批，再考虑 streams。

报告没有可靠的视觉 GPU 阶段分解，暂不能给端到端收益数字。若视觉阶段占比为 `f`、实测该阶段加速 `s`，端到端加速为 `1 / (1-f+f/s)`。例如 **假设** `f=20%`、`s=2`，整体只有 1.11×；不是三相机就能整体 3×。

### P1：合并层内独立投影

PyTorch 替换版 `modeling_gemma.py::GemmaAttention.forward` 分别执行 Q/K/V 三个 Linear；`GemmaMLP.forward` 的 gate/up 两个 Linear 也共享输入、彼此独立。

候选：QKV packed Linear、gate/up packed Linear，再按原维度拆分输出。利用更大的 GEMM 执行这些独立计算，通常比每层增加多个 stream 更容易控制调度成本；实际速度仍需测试。需维护 GQA 的 Q/K/V 不同输出宽度、bias、checkpoint 加载和权重更新语义。

JAX 已存在部分 QKV/KV 合并，不能照搬 PyTorch 的收益假设。XLA 是否已实现等效融合，要看编译产物。

### P1：JAX 切片/转换热点；PyTorch 先核实对应热点

JAX `gemma.py` 使用 `nn.scan`，FFN 中有权重 `astype(dtype)`；结合约 35 ms 的上述 kernel 组，值得检查层权重动态切片、重复 dtype 转换和布局物化。

可实验：推理期预转换不变权重、预排布布局，以及有限展开层循环以减少运行时切片。不要一次性完全展开所有层和去噪步骤：需同时观察编译时间、代码规模和显存。规范化的 FP32 计算不能为了消除转换而随意降低精度。

这是减少工作量的优化，不是额外 streams。收益可能比填空隙大，但现有 trace 尚不足以证明具体改法。PyTorch 的 ModuleList 参数布局不同，必须用 PyTorch trace 判断是否存在同样问题。

### P1/P2：融合 attention，改善 kernel 内并行和访存

PyTorch `sample_actions`、`denoise_step` 显式设置 eager attention；替换版 Gemma 中可见 QK matmul、FP32 softmax、PV matmul。候选是适配 SDPA/fused attention。

必须保留 prefix/suffix 的块状 mask、GQA、缩放和数值行为，不能直接改成普通 causal attention。后端可能因为 mask、dtype、head dimension 等条件回退到 math 实现，需在 trace 中验证实际 kernel。该方案不是模型分支并行，但可能改善比多 stream 更主要的内部计算/访存成本。

### P2：固定步数循环与不变量外提

PyTorch `sample_actions` 用 CUDA tensor 控制 Python `while`，是同步和编译断图的候选点。固定 `num_steps` 时可试整数 `for` 循环，保留原时间步数值更新；比较 eager 和 compile 两种情况。

mask、position、时间频率可外提；Pi0.5 的时间 MLP 输入仅依赖时间，可尝试一次合批计算所有时间步条件；Pi0 的 action-time MLP 依赖 `x_t`，不能整体提前计算。JAX `lax.while_loop` 是否需要静态 scan/有限展开，应结合 1-byte 回读与 HLO 验证，不能直接套用 PyTorch 的机制解释。

这些操作不会消除相邻去噪步骤的数据依赖，也不能把 10 步变成 10 路独立执行。

### P2：多请求 batch 或流水，面向吞吐

多个独立机器人/轨迹请求可尝试 batch=2/4，或将请求 A 的去噪与请求 B 的视觉编码重叠。可以提高吞吐，但会增加显存、排队时间，可能恶化单请求 P95。

JAX benchmark 的 throughput/queue-depth 仅表示异步提交，不保证 GPU 并发。当前单 stream 报告也没有证明多请求可以有效重叠。在线闭环中若下一帧依赖本轮动作执行，跨帧流水还可能改变观测新鲜度，应单独评估。

### P3：多 GPU 与更细粒度流水

单请求 batch=1 下，多卡 tensor parallel 要增加逐层通信，现有单卡报告不支持其收益判断。独立请求的数据并行更直接。

prefix 第 `l` 层 KV 就绪后，理论上可与 suffix 对应层构建更细粒度流水；但当前接口一次生成完整 KV，改动和同步复杂，且同卡资源竞争仍在。暂不优先于合批、融合和移除重复工作。

## 5. 建议验证顺序

1. 按 `pytorch_benchmark.md` 采集真正的 PyTorch eager/compiled 基线；各版本使用同一 checkpoint、模型类型、输入、步数与硬件。
2. 补充视觉编码、prefix 和逐步去噪的 GPU 阶段耗时。NVTX CPU 区间只用于关联异步提交，不能直接替代 GPU 阶段时间。
3. 先做相机合批，再独立做 QKV/gate-up 合并、fused attention；固定循环和不变量外提单独做消融。每个候选都对比 eager 与 compiled，避免把 compile 已有收益重复计算。
4. 正式速度在关闭 profiler 后测，记录 P50/P95、吞吐、显存、首调用和编译时间；如原测试规范，重复 3 轮。已有 5 次 node trace 仅用于定位。
5. 检查固定 checkpoint/输入下输出误差，再用任务成功率判断可接受性。多请求方案另测排队与端到端 P95。

NVIDIA 文档说明 node 级 CUDA Graph tracing 可能带来显著额外开销，因此这里的毫秒数不能作为生产延迟承诺：[CUDA Graph profiling](https://docs.nvidia.com/dl-cuda-graph/troubleshooting/performance-issues.html)。多 stream 的同步与内存生命周期要求见 [PyTorch CUDA semantics](https://docs.pytorch.org/docs/main/notes/cuda.html)；融合 attention 的后端约束见 [PyTorch SDPA](https://docs.pytorch.org/docs/main/generated/torch.nn.functional.scaled_dot_product_attention.html)。

结论：**当前证据支持优先验证相机合批与层内投影合并；较大的潜在空间还来自切片、类型转换和 attention 的工作量削减。单纯增加 stream 或删除末尾等待，没有证据支持大幅单请求加速。**
