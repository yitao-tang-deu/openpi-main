# Pi0 推理性能分析

本次仅在 Windows 读取已有报告、检查和修改代码，没有连接 SSH，也没有运行模型或 GPU 测试。

## 数据来源与测量边界

- 原始报告：`../nsys报告/pi0_inference_2024.nsys-rep`，使用同目录 SQLite 导出进行只读查询。
- 两处 `.nsys-rep` 的 SHA256 相同：`4d4926bc3a853333ce87aacccf7426d29779115a27b6e09182088c88e4e16c2c`。
- 两处 SQLite 文件大小不同，但 Graph 数量、Graph 总时长、最后五次 XLA 推理均值一致。
- 捕获命令参数为 `uv run python benchmark_infer_speed.py --warmup 3 --runs 5`。结合当前代码，推断 batch=1、num_steps=10、Pi0 默认配置；报告不包含源代码快照，无法单靠它确认远程代码版本。
- 目标设备记录为 NVIDIA GeForce RTX 4090，128 SM；执行系统为 Unix。Windows 是分析主机，不能归因于 Windows CUDA 调度。
- 捕获约 58.112 秒，真正的八次 `jit_fun` 推理发生在 54.164–54.798 秒。前段含初始化、编译、autotuning 等，不能用全程平均利用率评估稳态。
- 捕获设置是 `CUDA_GRAPH_TRACE_OPTIONS:MODE=Graph`：Graph 内部 kernel 未逐节点展开。kernel 表中的 10,031 次、344.18 ms 不是完整推理 GPU 工作量；其中大量 `redzone_checker` 等属于 autotuning。

## 稳态结果

通过 `NVTX_EVENTS` 的 `XlaModule:#hlo_module=jit_fun,program_id=561#` 找到八次调用，前三次对应预热，最后五次对应测试。下面是 XLA 主机区间，不是原脚本的 Python perf_counter 输出，也不是纯 kernel 时间。

| 测试调用 | XLA 区间 ms |
|---|---:|
| 1 | 76.445097 |
| 2 | 76.415887 |
| 3 | 76.420576 |
| 4 | 76.442087 |
| 5 | 76.455477 |
| 均值 | 76.435825 |

最后一次调用内：

| 活动 | 数量 | GPU 区间累计 ms |
|---|---:|---:|
| CUDA Graph | 308 | 74.732350 |
| Graph 外可见 kernel | 51 | 0.159392 |
| 显式 memcpy | 231 | 0.353892 |

三类 GPU 区间合并去重后为 75.245634 ms，XLA 区间内剩余约 1.209843 ms。Graph 区间可能包含内部空隙，所以这不是 SM 利用率，也不能证明算力或带宽已饱和。Graph 内部优化空间仍然未知。

### 阶段定位

将最后一次推理的 Graph `correlationId` 关联 CUDA API，再查找包含该 API 的最内层 NVTX 范围，结合模型层数推断如下。它们是 Graph 执行时间汇总，不能替代新增阶段标记后的完整端到端阶段测量。

| 阶段（代码与调用次数推断） | graphId | 调用次数 | 合计 ms | 占全部 Graph 时间 |
|---|---|---:|---:|---:|
| 三路图像编码，27 层/路 | 20、27、34 | 81 | 13.591938 | 18.2% |
| 前缀 Transformer / KV 构建，18 层 | 48、51 | 36 | 35.476173 | 47.5% |
| 去噪主体，10 × 18 层及步前处理 | 54、57 | 190 | 25.607631 | 34.3% |
| 其他 | 41 | 1 | 0.056608 | 0.1% |

## 并行加速瓶颈与优先级

1. **当前是单请求、逐次同步测试，没有测并发扩展性。** 可见 kernel 都在 stream 13，Graph 也应按各自 stream 检查。不能从串行测试推导多请求并行加速比。先对比 batch=1/2/4 的 samples/s、batch 延迟和显存，再比较有界异步提交；增大 batch 可能改善吞吐，同时增加请求延迟。异步提交不保证多 CUDA stream。
2. **前缀成本大于去噪成本。** 默认输入是三路 224×224 图像，视觉 patch 与文本共同进入前缀计算；KV 已在每次请求内缓存，不会在每个去噪步重算。优先研究三相机合批编码（把 camera 合入 batch，之后恢复 token 顺序）、前缀 attention/GEMM、合理输入长度。相机 mask=false 不会跳过当前 `embed_prefix` 的图像编码；仅把 mask 设 false 不能减少这部分计算。删相机、降分辨率、缩短提示会改变工作负载，必须匹配训练与真实部署。
3. **去噪步骤有真实依赖，不能把十步直接并行。** `x_(t+dt)` 依赖本步模型输出。最后一次推理出现 11 次 1-byte DtoH 与 11 次 `cuStreamSynchronize`，符合动态 `lax.while_loop` 的条件回传模式。建议单独做固定步数 `lax.scan` / 静态 `fori_loop` A/B，并保证步数以静态方式传入编译；也可研究循环展开。它们可能改变编译规模、舍入路径和 Graph 划分，需要固定 noise 做输出误差比较。仅加 `static_argnames` 而保留动态时间条件，并不保证消除回传。
4. **同步耗时不能直接扣除。** 最后一次 `cuStreamSynchronize` 主机累计约 69.63 ms，大部分在等先前 GPU 工作完成，与 Graph 时间重叠。不能宣称删除同步就能省 69 ms。308 次 `cuGraphLaunch` 主机累计约 2.01 ms 也会与 GPU 重叠。已经使用 CUDA Graph，不能把“开启 CUDA Graph”当作尚未实施的优化。
5. **此报告无法区分内部算力、显存带宽和小 kernel 开销。** 需要 node 级短捕获揭示内部算子，再对主要 kernel 做 Nsight Compute 检查。现有证据不足以宣称 attention 必然是最大瓶颈，或承诺 FlashAttention、多 stream、量化能带来特定倍数提升。

理想化上限：只把约 25.61 ms 去噪 Graph 时间减半，其余完全不变，总时长约从 76.46 降至 63.65 ms，即约 1.20×；即使该部分归零，也只有约 1.50×。这是依据当前阶段归属估算的上限示意，不是实测收益。

## benchmark 是否合理，以及本次修改

原来的 `module_jit`、预热和 `block_until_ready(output)` 是正确方向，适合设备驻留输入的稳态模型延迟。JAX 异步派发要求显式等待结果：[JAX benchmarking](https://docs.jax.dev/en/latest/benchmarking.html)。

局限与已修改内容：

- 原脚本 `config.create` 是随机权重，无 checkpoint、tokenizer、策略输入输出变换、图像采集、网络及动作 DtoH，因此不能代表完整机器人服务延迟。修改后在结果中明确工作负载名称，仍保留这个合成基准范围。
- 随机 key 原先在计时前 split，但 split 本身也是异步工作，其尾部可能进入计时。现在预先生成并等待每个 key 和 observation，模型/输入/采样使用独立子 key。
- 输入 float 范围改为 [-1,1]，图像与 prompt mask 改为全有效，避免随机缺相机及不连续 prompt mask。仍是人工合成满长输入；新旧输入分布不同，严格对比优化收益应先重建基线。
- 首调用单列 `first_call_ms`，包含编译/autotuning/执行，不称为纯编译时间；然后再执行 `--warmup` 次预热。即现在总次数为 `1 + warmup + runs`，不同于旧脚本。
- 默认测试次数从 5 增至 50，支持参数合法性检查，输出 mean/p50/p95/stddev、全部原始延迟、batch/s、sample/s、配置、JAX/jaxlib、设备与 XLA_FLAGS，支持 JSON 保存。p95 在少量样本下仍不稳健，性能结论应多进程重复采样。
- 逐次打印移到测量结束后。`latency` 每次同步；`throughput` 按 queue-depth 分组提交并等待组内全部输出，统计总墙钟吞吐，不伪装成单请求延迟。
- 可选 NVTX 标记 setup、first_call、warmup、measurement 和每次请求/提交组，需远程环境安装 `nvtx`。普通计时不需要该依赖。
- action_horizon=50 表示每个样本输出动作块，samples/s 不是机器人控制频率，也不是把动作块长度乘进去后的真实闭环速率。

本次未更改模型算法：循环改写、相机合批及 attention 替换需要远程数值与性能 A/B 后再决定。

## 后续远程执行示例（本次未执行）

在远程仓库目录运行，无 profiler 的结果作为计时基线：

```bash
uv run python benchmark_infer_speed.py --warmup 5 --runs 100 --json latency_b1.json
uv run python benchmark_infer_speed.py --batch-size 2 --warmup 5 --runs 100 --json latency_b2.json
uv run python benchmark_infer_speed.py --batch-size 4 --warmup 5 --runs 100 --json latency_b4.json
uv run python benchmark_infer_speed.py --mode throughput --queue-depth 4 --runs 100 --json throughput_q4.json
```

对比 batch 吞吐增益 `samples/s(B) / samples/s(1)`，并同时查看 batch 延迟。模型/设备可用显存决定可测的最大 batch。队列模式可比较 depth=1/2/4，单 GPU 无须预设一定获益。

安装可选 NVTX 包后，采集有测量标记的短报告；在 GUI 中选择 measurement，排除首调用：

```bash
uv run --with nvtx nsys profile --trace=cuda,nvtx --sample=none --cuda-graph-trace=node -o pi0_nodes python benchmark_infer_speed.py --nvtx --warmup 3 --runs 5
```

node 追踪可能显著增大报告与开销；它用于定位算子，不能替代无 profiler 计时。CLI 选项应以远程安装版本 `nsys profile --help` 为准。Graph 与 CUDA API 解释参考 [Nsight Systems User Guide](https://docs.nvidia.com/nsight-systems/UserGuide/)。

## 查询复核

以下 SQLite SQL 可定位本报告的八次推理；时间单位为 ns：

```sql
SELECT n.start, n.end, (n.end-n.start)/1e6 AS xla_ms
FROM NVTX_EVENTS n LEFT JOIN StringIds s ON n.textId=s.id
WHERE COALESCE(n.text,s.value) = 'XlaModule:#hlo_module=jit_fun,program_id=561#'
ORDER BY n.start;

SELECT graphId, COUNT(*), SUM(end-start)/1e6 AS graph_ms
FROM CUPTI_ACTIVITY_KIND_GRAPH_TRACE
WHERE start >= 54721984792 AND end <= 54798440269
GROUP BY graphId ORDER BY graph_ms DESC;
```

本地验证仅为 Python AST 语法解析及 diff 检查；未验证远程 JAX 执行、NVTX 捕获或优化后的实际性能。
