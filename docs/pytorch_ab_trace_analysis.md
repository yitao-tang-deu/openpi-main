# PyTorch A/B：nsys 时间线归因

## 数据与口径

输入为 `traces/pytorch/A_nodes.nsys-rep`、`B_nodes.nsys-rep`，以及 `A_nsys.json`、`B_nsys.json`；正式性能来自 `results/pytorch/A.json`、`B.json`。当前目录没有单独的 PyTorch Profiler `.trace.json`，本文分析的是 nsys 的 CUDA/NVTX 时间线。

两组为 Pi0.5、batch=1、10 步去噪、随机权重、相同输入。采样提交为 `4cdb0418`，正式测速提交为 `49f8fb2`；记录的源码哈希与输入哈希逐项相同。A 为 eager，B 请求 `max-autotune`。两组仍为串行相机编码，不包含后续相机合批优化。

使用与报告匹配的 Nsight Systems 2024.6.2 离线导出 SQLite，通过 NVTX `infer_0000` 至 `infer_0004` 筛选事件；GPU 时间取区间并集，不把 CPU 同步等待与 GPU 执行相加。A 的阶段归因通过 NVTX 内 CPU launch 的 correlationId 关联 GPU kernel，而非把 CPU 标记时长当成 GPU 时长。只有一个被记录的 GPU kernel stream（7）。

复算文件：`traces/pytorch/analyze_ab_sqlite.py`、`AB_nsys_analysis.json`。在仓库根目录运行：

```bash
nsys export --type sqlite --output traces/pytorch/A_nodes.sqlite traces/pytorch/A_nodes.nsys-rep
nsys export --type sqlite --output traces/pytorch/B_nodes.sqlite traces/pytorch/B_nodes.nsys-rep
python traces/pytorch/analyze_ab_sqlite.py
```

## 1. 主要加速来源：减少调度空隙，其次是减少 GPU 工作

下表为每次推理平均值，全部来自采样，不能替代正式测速。B 的第 2/3/4 号样本单列用于观察较稳定状态，全部五次仍保留。

| 指标 | A 全部 5 次 | B 全部 5 次 | B 第 2/3/4 号样本 |
|---|---:|---:|---:|
| NVTX 推理时间 | 429.640 ms | 78.235 ms | 67.369 ms |
| kernel 数量 | 14,546 | 5,054 | 5,054 |
| kernel 活跃时间，并集 | 78.353 ms | 63.281 ms | 55.139 ms |
| GPU kernel/memcpy/memset 活跃时间，并集 | 78.481 ms | 63.309 ms | 55.164 ms |
| 推理区间减去上述活动 | 351.159 ms | 14.927 ms | 12.205 ms |
| 直接 kernel launch API 数量 | 14,546 | 1 | 1 |
| CUDA Graph launch 次数 | 0 | 11 | 11 |
| CPU stream synchronize 次数 | 24 | 11 | 11 |
| 1-byte D2H 次数 | 11 | 11 | 11 |

“推理区间减去活动”只代表本进程已记录活动以外的时间，不代表全卡空闲或 SM 利用率；报告还存在事件可能未收全的警告。

A 的 GPU 工作被大量短 kernel 和 CPU 提交间隔分散。A 中仅 `elementwise_kernel` 和 `vectorized_elementwise_kernel` 就有 8,215 次/推理，kernel 时间合计约 17.30 ms；还有 reduction、concat、split-K reduction 等调用。直接 launch API 自身累计约 97.63 ms/推理，此外仍有 Python/framework 执行及其他调度间隔，不能把全部 351 ms 都解释成 launch API 自身耗时。

B 出现 `triton_tem_fused_gelu_mm_mul_22`、`triton_red_fused__to_copy_add_mean_mul_pow_rsqrt_*`、融合 softmax 等 kernel，数量下降 **65.26%**。25265 个 graph 内 kernel 可关联到 55 次 graph launch，另有 5 次普通 kernel launch。由此可以确认：**本次采样确实执行了生成的 Triton kernel 和 CUDA Graph replay**，并非只有命令行请求了 compile。

不过，这不能证明整个 `sample_actions` 是一个无断图的编译区域。B 每次实际执行 **一个 1,503-kernel graph，加上十个 355-kernel graph，再加一个普通 kernel**。结合源码的 prefix 和 10 步循环，前者对应去噪前的前缀准备/计算，后者对应各步去噪；B 未开启内部阶段标记，因此更细的算子归属仍需 Inductor 生成代码确认。

正式测速 P50 为 A **316.37 ms**、B **58.64 ms**，约 **5.39×**。采样 P50 则为 429.31/68.30 ms，A/B 受到的扰动不同（A 还额外开启阶段标记），不能将采样表中的时间差精确当成正式测速的收益分摊。

## 2. 同步并未消失，但含义发生了变化

A 的 `cudaStreamSynchronize` CPU 时间平均约 9.41 ms，B 则约 65.00 ms。**B 的等待时间更长不代表它更低效**：CPU 一次提交 graph 后，会等待 GPU 完成较长的一段连续工作；A 的 CPU 则花更多时间逐个提交，GPU 经常赶上 CPU。

显式最后的 `sync_wait` NVTX 平均只有 A 0.029 ms、B 0.060 ms。主要同步发生在模型内部，不是 benchmark 最后一行 synchronize 造成了全部瓶颈。

两组每次都出现 11 次 1-byte D2H。源码 `while time >= -dt / 2` 用 CUDA tensor 控制 Python 循环，需要将标量条件提供给主机，与 10 次迭代加退出判定的次数吻合。B 保留 11 次 stream synchronize 和 11 次 graph replay，说明编译没有消除这条循环控制边界。固定整数步数循环值得单独实验，但不能删除相邻去噪步骤对 `x_t` 的依赖。

不能用“B 同步等待约 65 ms”推导“删掉同步可节省 65 ms”；该时间覆盖了大量必要的 GPU 执行。

## 3. 阶段瓶颈与相机合批的预期边界

A 的 GPU kernel 归因，按每次推理求平均：

| 阶段 | kernel 数量 | GPU kernel 活跃时间 |
|---|---:|---:|
| embed_prefix：三路视觉编码及语言 embedding | 1,576 | 11.234 ms |
| vlm_prefix：语言模型前缀/KV | 884 | 32.006 ms |
| 10 次 denoise_step 合计 | 12,010 | 35.018 ms |

剩余少量 kernel 属于阶段外的 mask、循环条件、Euler 更新等。

B 较稳定的三次样本，首 graph 的 kernel 活跃时间平均 **34.589 ms**，十个去噪 graph 合计 **20.548 ms**。首 graph 包含视觉和 VLM 前缀，不能把 34.589 ms 全算成视觉编码。

因此，三相机合批仍是独立分支的合理实验，但它只优化视觉部分；**不能期待将整个前缀或整个推理直接加速三倍**。判断后续合批增益应比较 B/D，不能将 A/B 的编译收益再次计入合批。

B 另有一个每次去噪出现一次的 `triton_per_fused_addmm_3`，正常样本累计约 **4.90 ms/推理**（约占去噪 kernel 时间的 24%），值得检查生成代码与输入形状；仅凭后缀编号不能认定它就是某个具体 Linear 或时间 MLP。

## 4. B 的 112 ms 慢调用：主要发生在首 graph 的 GPU 时间内

B 五次推理依次约 76.90、112.17、66.87、66.93、68.30 ms，所有调用的 kernel 数量均为 5,054，graph 数量均为 11，没有新增工作量的证据。

| 指标 | 较稳定三次平均 | 慢调用 infer_0001 |
|---|---:|---:|
| 推理区间 | 67.369 ms | 112.171 ms |
| 全部 kernel 活跃时间 | 55.139 ms | 93.477 ms |
| 首 graph kernel 活跃时间 | 34.589 ms | 73.002 ms |
| 10 个去噪 graph kernel 时间合计 | 20.548 ms | 20.473 ms |
| 区间内无已记录 GPU 活动的时间 | 12.205 ms | 18.668 ms |

增加的约 44.8 ms 中，约 38.3 ms 表现在 kernel 记录时长增加，并主要集中于首 graph。

| kernel 名称组 | 较稳定三次平均 | 慢调用 |
|---|---:|---:|
| `triton_tem_fused_mm_25`，34 次 | 13.405 ms | 32.891 ms |
| `triton_tem_fused_mm_27`，17 次 | 7.072 ms | 16.348 ms |
| `Kernel2`，684 次 | 3.902 ms | 10.741 ms |
| `triton_tem_fused_bmm_19`，17 次 | 0.528 ms | 3.267 ms |

观察支持“首 graph 的 GPU 执行记录被拉长”，不支持“只因 CPU 没及时 launch”或“重新编译增加了工作量”作为唯一解释。kernel 墙钟时长仍可能含 GPU 调度/抢占、资源竞争等影响；没有时钟、功耗、温度、系统级 GPU context-switch 指标，不能进一步断言是其他进程、降频或特定硬件问题。

正式 B 的 100 次样本在第 17–18、37–38、56–57、76–77、95–96 次出现慢样本，约每 19–20 次一组。当前 nsys 只采 5 次，不能证明它捕获的慢调用与正式测速的周期性现象是同一个原因。

## 5. 报告本身有兼容性限制

两组诊断事件均包含：

- `Installed CUDA driver version (13.2) is not supported by this build of Nsight Systems. CUDA trace will be collected using libraries for driver version 12.8`
- `Not all CUDA events might have been collected.`
- `Not all NVTX events might have been collected.`

B 还对另一个被捕获的进程给出没有 CUDA/NVTX 事件的提示，不能据此说 B 主推理进程没有 GPU 工作。两组每次 kernel 数量保持一致，但这不能完全排除系统性漏采。

这不会推翻无 profiler 下 A/B 的速度结果，也不否定已观察到的 graph replay 与融合；但 GPU 无活动区间的完整性和异常原因需要兼容当前 driver 的 Nsight 版本复核。NVIDIA 也说明 node 级 graph tracing 可能带来显著开销，见 [Nsight Systems User Guide](https://docs.nvidia.com/nsight-systems/UserGuide/)。

建议先用兼容 driver 的 nsys 在独占 GPU 条件下采集至少 100 次 B。用较低开销的 graph 模式判断首 graph 是否仍周期性变慢，再用 node 模式定位 kernel；另行记录 GPU 时钟、功耗和其他进程活动。若环境支持 GPU metrics/context-switch 采样，用它区分真正计算变慢与调度/抢占时间。

后续优化优先以 B 为起点：相机合批、固定步数循环消除主机条件回读、检查前缀 GEMM/attention，以及映射上述 addmm 热点。不要再将“删除末尾 synchronize”作为主要加速方向。
