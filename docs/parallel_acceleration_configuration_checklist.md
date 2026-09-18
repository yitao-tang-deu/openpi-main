# 基于 Pi0.5 的 VLA 模型：并行加速软硬件参数清单

本文共整理 **70 项因素**，覆盖：

- **硬件与系统**：算力、显存、带宽、互连、NUMA、功耗、资源共享。
- **GPU 并发**：stream、同步、异步拷贝、缓冲、合批、CUDA Graph。
- **软件执行**：编译、精度、attention 后端、算子形状、内存分配。
- **CPU 与部署扩展**：CPU 并行、数据流水线、多 GPU/多机通信及在线服务并发。

每类均列出配置入口、影响机制、适用条件与可能的反效果，并附 NVIDIA/PyTorch 官方来源；另提供当前 Pi0.5 VLA 项目的配置核对、实验优先级和最小对照矩阵。

整理日期：2026-09-18。本项目为**基于 Pi0.5 的 VLA（Vision-Language-Action，视觉—语言—动作）模型**。本文结合该模型的 PyTorch A/B 证据，梳理影响并行加速的软硬件因素，并扩展覆盖 CPU 并行、单 GPU 并发、多 GPU/多机、训练及推理服务。本文是参数调查与实验设计，没有执行新的性能实验，也不把候选优化写成已验证收益。

项目分析以多路相机编码、视觉/语言前缀计算与 KV cache、迭代去噪生成动作序列为主线。通用清单中的分布式训练、通信和在线服务参数用于拓展分析，不表示当前 Pi0.5 VLA 推理实现已经启用这些机制。

**并行加速由计算依赖、资源余量、任务粒度、调度与通信开销共同决定。stream 数、线程数、GPU 数都不是越大越好。** 需要分别衡量单请求延迟、整体吞吐和尾延迟。

## 1. 先明确比较的是什么

| 目标 | 建议指标 | 必须固定的条件 |
|---|---|---|
| 单请求变快 | 端到端 P50/P95/P99、各阶段耗时 | 输入规模、输出质量、请求并发、计时边界 |
| 单卡吞吐提高 | 完成样本数/实际墙钟时间 | 延迟约束、batch 定义、到达率 |
| 多卡扩展 | `S(N)=T(1)/T(N)`、`E(N)=S(N)/N` | 强扩展固定总工作量；弱扩展固定每卡工作量，两者分开 |
| 训练更快 | step 时间、达到目标质量所需时间 | 有效全局 batch、优化器及收敛目标 |
| 服务更稳定 | 排队时间、P99、超时率 | 请求长度分布、突发流量、资源共享条件 |

阶段占比为 `f`、阶段加速为 `s` 时，假设其余耗时不变，端到端加速为 `1/(1-f+f/s)`。阶段并发的收益还要扣除拆分、汇合、资源竞争与额外搬运；这是基于依赖图的分析框架，不是收益承诺。[CUDA Best Practices：评估与并行化](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/)

必须区分三件事：同一个 kernel 内的线程并行、不同 kernel 在多 stream 上重叠、多个请求或进程同时在途。一个 stream 不等于 GPU 串行计算；API 异步返回也不等于 GPU 实际发生重叠。

## 2. 硬件与系统：决定资源上限和运行稳定性

“选型”表示无法用普通运行参数改变；“可调”表示需要平台、设备及权限支持；“记录”表示实验控制变量。以下建议值是实验起点，不是厂商默认值。

| ID | 参数或因素 | 属性/配置入口 | 对并行效果的影响与检查方法 |
|---|---|---|---|
| H01 | GPU 架构、SM 数、计算能力、Tensor Core 支持 | 选型；CUDA device properties | 决定执行资源与支持的数值格式；相同 stream 配置不能直接跨卡外推 |
| H02 | 显存容量、实际可用显存 | 选型+记录；设备属性、峰值 allocated/reserved | 限制 batch、并发副本、KV cache 和 workspace；容量增加不等于计算变快 |
| H03 | 显存带宽、L2 容量 | 选型；硬件规格与实测 | 带宽受限任务并发时可能相互争抢；检查 DRAM/L2 吞吐与缓存行为 |
| H04 | 每 SM 寄存器、共享内存、驻留 block/warp 上限 | 选型；设备属性 | 决定 kernel 能否共同驻留；要结合每个 kernel 的资源需求判断 |
| H05 | 并发 kernel 能力、copy engine 能力 | 选型；`concurrentKernels`、`asyncEngineCount` | 是重叠执行的能力条件，不保证任意任务同时执行；用时间线验证 |
| H06 | PCIe 代际、实际链路宽度、插槽位置 | 选型+部署；拓扑与链路状态 | 影响 H2D/D2H、多卡交换；检查运行时链路，不能只读插槽标称规格 |
| H07 | GPU–GPU 互连、P2P 可达性 | 选型+部署；NVLink/NVSwitch/PCIe、拓扑 | 多卡任务可能受通信而非算力限制；型号名称不能代替 P2P 实测 |
| H08 | CPU 核数、单核性能、缓存、内存带宽 | 选型+记录 | kernel 提交和预处理可能受 CPU 限制；增加 GPU 并发不能补偿供给不足 |
| H09 | NUMA 拓扑、CPU/内存/GPU/NIC 位置 | 部署；CPU affinity、`numactl` 等 | 远端内存或跨 socket 通信增加开销；进程和数据尽量靠近所用设备 |
| H10 | 功耗上限、SM/显存频率、温度、降频原因 | 可调+记录；`nvidia-smi -q` 等 | 控制可持续速度和抖动；锁频/功耗设置并非所有显卡或系统都支持 |
| H11 | GPU 独占程度、显示任务、其他计算进程 | 部署+记录 | 共享资源可能延长 kernel 墙钟时间；记录其他进程与 GPU 调度活动 |
| H12 | OS/驱动模型、容器/虚拟化资源配额 | 部署+记录；Linux/Windows/WSL、WDDM/TCC、cpuset | 改变调度与可用工具；不能假定消费卡支持 TCC，也不能把编辑代码的主机当采样主机 |
| H13 | MIG 分区与实例规格 | 可调，仅支持的 GPU | 提供计算和内存隔离，也缩小单实例可用资源；适合多租户，不保证单任务加速 |
| H14 | MPS、多进程资源分配 | 可调，仅支持的平台/版本 | 提高部分多进程负载的资源利用率；不是单进程多 stream 的必需开关 |
| H15 | NIC 带宽、RDMA/GDR 路径、交换网络与拥塞 | 选型+部署 | 多机扩展的关键约束；应测真实消息规模的有效带宽和延迟 |
| H16 | 存储吞吐、数据解码、页缓存状态 | 选型+流水线配置 | 影响数据供给；若输入已在 GPU 上，不应把存储作为该次模型计时的直接瓶颈 |

硬件资源与 kernel 驻留关系见 [CUDA Best Practices](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/)，吞吐、缓存和资源占用诊断见 [Nsight Compute Profiling Guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html)。频率、功耗、驱动模式与字段支持范围见 [nvidia-smi 文档](https://docs.nvidia.com/deploy/nvidia-smi/index.html)。NUMA 与数据供给见 [PyTorch Performance Tuning Guide](https://docs.pytorch.org/tutorials/recipes/recipes/tuning_guide.html)。MIG/MPS 的用途分别见 [MIG User Guide](https://docs.nvidia.com/datacenter/tesla/mig-user-guide/) 和 [MPS 文档](https://docs.nvidia.com/deploy/mps/latest/index.html)。网络选择见 [NCCL 环境变量](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html)。

## 3. 单 GPU 并发：最直接相关的运行参数

| ID | 参数/代码入口 | 建议对照 | 影响、适用条件与反效果 |
|---|---|---|---|
| G01 | `torch.cuda.Stream()` 的数量 | 1/2/3/4；按独立分支数缩小范围 | 分支独立且资源有余量时才可能降低延迟；过多会竞争缓存、带宽和执行资源 |
| G02 | stream 优先级 | 默认与必要的优先级区分 | 是调度提示，不能作为任意抢占或时延保证 |
| G03 | 默认 stream 语义 | 检查 legacy/per-thread default stream 及扩展库行为 | legacy default stream 可能引入隐式同步；不能一概说所有默认 stream 都让全设备串行 |
| G04 | `wait_event` / `wait_stream` 的位置 | 生产完成→消费开始、分支结束→汇合 | 等待过早会破坏重叠；遗漏依赖会产生错误结果 |
| G05 | `record_stream`、跨 stream buffer 生命周期 | 为旁路 stream 使用的张量管理生命周期 | 防止 allocator 提前复用；不能以删除必要依赖换取表面速度 |
| G06 | `.item()`、`.cpu()`、tensor 布尔条件、显式 synchronize | 检查每个主机回读与同步点 | 可能迫使 CPU 等 GPU；必须区分“等待必要计算”和“额外调度停顿” |
| G07 | `pin_memory` + `non_blocking=True` | pageable/pinned；同步/异步拷贝 | 真正 copy/compute 重叠还需合适 stream、硬件支持和独立任务；`non_blocking` 单独设置不足以保证 |
| G08 | 分块大小、双/三缓冲、在途拷贝数 | 2/3 个 buffer，逐步调整块大小 | 块太小放大提交开销，太大降低流水重叠；固定总数据量比较 |
| G09 | 同批独立分支合批 | 逐分支、沿 batch 维合并、多 stream 三者比较 | 合批改变 GEMM 形状并减少调用；可能比多 stream 更有效，也可能增加显存峰值 |
| G10 | CUDA Graph 的捕获范围与 replay 粒度 | eager、分段 graph、可捕获的更大 graph | 减少重复 CPU 提交；不消除数据依赖，也不保证自动产生多 stream 并发 |
| G11 | 每进程模型副本、CUDA context 数 | 单实例与多实例 | 增加独立任务，同时重复占用权重、缓存和 workspace；多进程不保证 kernel 并发 |
| G12 | `CUDA_DEVICE_MAX_CONNECTIONS` | 默认优先；有队列竞争证据再试 | 控制主机到设备工作队列数量，不是 SM 数，也不是“最大 CUDA stream 数” |
| G13 | `CUDA_LAUNCH_BLOCKING` | 正式性能测试保持关闭/未设置 | 调试用同步提交会改变异步执行与重叠，不能混入基准 |
| G14 | `CUDA_VISIBLE_DEVICES` 与 UUID 映射 | 固定同一物理设备 | 是可见性/编号设置，不提供 GPU 独占或资源隔离 |

stream 同步、生命周期和异步拷贝要求见 [PyTorch 2.7 CUDA semantics](https://docs.pytorch.org/docs/2.7/notes/cuda.html)；队列、调试和可见性参数见 [CUDA Environment Variables](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/environment-variables.html)。多 stream 的资源竞争和执行模式见 [TensorRT 性能优化](https://docs.nvidia.com/deeplearning/tensorrt/latest/performance/optimization.html)。这些运行原则可用于分析 PyTorch，但 TensorRT 的配置接口不能直接用于 PyTorch。

## 4. 编译、算子与负载形状：决定每项工作占多少资源

| ID | 参数/配置入口 | 可比较的设置 | 影响与限制 |
|---|---|---|---|
| C01 | PyTorch/CUDA runtime/driver/Triton/cuBLAS/cuDNN 版本 | 固定版本后做独立升级对照 | 会改变 kernel、编译与调度；不能把库升级收益算成纯并行策略收益 |
| C02 | `torch.compile(mode=...)` | eager/default/reduce-overhead/max-autotune/max-autotune-no-cudagraphs | 分别影响编译、搜索与 graph 策略；CLI 请求成功不等于实际完整编译 |
| C03 | `fullgraph`、`dynamic`、输入 shape 稳定性 | 静态桶、动态形状；查看 graph break/recompile | `fullgraph=True` 用于要求完整捕获，失败会报错；不是自动消除断图 |
| C04 | autotune、融合范围、`shape_padding` | 默认与有针对性的选项 | 减少调用或寻找更优 kernel；可能增加启动时间、寄存器和 workspace |
| C05 | dtype、autocast、TF32 matmul 精度 | FP32/BF16/FP16；相关硬件上其他格式 | 改变算力与带宽需求，也改变误差；需模型输出/任务质量验证 |
| C06 | INT8/INT4/FP8 等量化策略 | 仅在硬件、后端和模型均支持时 | 权重存储减小不代表实际计算加速；量化/反量化与小矩阵成本可能抵消收益 |
| C07 | attention 后端 | SDPA 自动、支持的 Flash/efficient/math 实现 | 由 dtype、mask、head dimension、布局和硬件决定可用性；强制后端可能失败 |
| C08 | GEMM 的 M/N/K、batch 与对齐 | 保持语义的 batching/padding | 改变 tile 利用率和算术强度；不存在跨架构通用的“必须 8/16 倍数”最优配置 |
| C09 | layout、stride、连续性、格式转换 | 消除重复 transpose/contiguous/cast | 减少中间搬运；channels-last 主要按受益算子评估，不应盲目用于全部张量 |
| C10 | 自定义 kernel block/grid、tile、`num_warps`、`num_stages` | 编译器搜索或热点定向实验 | 影响驻留、流水、共享内存和寄存器；更高 occupancy 不一定更快 |
| C11 | allocator、预分配、workspace、显存碎片 | buffer 复用；必要时研究 allocator 后端 | 影响并发容量与分配停顿；不要在稳态循环里反复 `empty_cache()` |
| C12 | `model.eval()`、`no_grad`/`inference_mode` | 正确推理模式 | 避免无用 autograd 工作；`eval()` 与禁用梯度是不同机制 |
| C13 | `cudnn.benchmark`、确定性算法策略 | 固定 shape 下独立比较 | 限于相关算子；算法搜索、复现要求与性能存在取舍 |
| C14 | KV cache/前缀缓存与有效范围 | 同次推理复用；跨请求仅在前缀等条件兼容时 | 减少重复计算，也增加显存；不能复用已失效或语义不同的缓存 |
| C15 | batch、token 长度、图像分辨率/数量、action horizon | 固定工作量测试并行；规模变化另列 | 影响计算量和并行粒度；减少输入或输出规模属于负载/质量调整 |
| C16 | 迭代次数、循环表达与依赖 | 同步数相同的固定步数循环对照 | 改循环表达可减少主机交互；减少去噪步数改变算法，不能当等工作量并行加速 |

编译模式及选项见 [torch.compile 2.7](https://docs.pytorch.org/docs/2.7/generated/torch.compile.html)；精度、allocator 和 graph 使用约束见 [CUDA semantics 2.7](https://docs.pytorch.org/docs/2.7/notes/cuda.html)；attention 支持条件见 [SDPA 2.7](https://docs.pytorch.org/docs/2.7/generated/torch.nn.functional.scaled_dot_product_attention.html)。GEMM 形状与 tile 效率见 [NVIDIA Matrix Multiplication Guide](https://docs.nvidia.com/deeplearning/performance/dl-performance-matrix-multiplication/index.html)；格式、量化和执行优化见 [TensorRT 优化指南](https://docs.nvidia.com/deeplearning/tensorrt/latest/performance/optimization.html)。kernel 资源分析见 [Nsight Compute](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html)，通用推理/布局设置见 [PyTorch Tuning Guide](https://docs.pytorch.org/tutorials/recipes/recipes/tuning_guide.html)。缓存有效范围和工作量等价性是本文的正确性约束。

## 5. CPU 与数据流水线

| ID | 参数/入口 | 建议对照与判断 |
|---|---|---|
| P01 | `torch.set_num_threads`、`OMP_NUM_THREADS`、`MKL_NUM_THREADS` | 扫描 1/2/4/8 等不超过资源预算的值；多实例时按全部线程预算计算，避免超额订阅 |
| P02 | `torch.set_num_interop_threads` | 区分算子内部与算子间线程；进程初始化早期设置，不假定适用于所有执行后端 |
| P03 | OpenMP affinity、CPU affinity、NUMA 内存绑定 | 将工作线程与数据放在合适节点；单机多 socket 尤其需要检查 |
| P04 | DataLoader `num_workers` | 0/2/4/8 起步；受磁盘、解码和 IPC 影响，GPU 常驻合成输入不受其直接影响 |
| P05 | `prefetch_factor`、`persistent_workers`、`pin_memory` | 让数据及时就绪，检查 RAM/pinned memory 增长；worker 相关选项须满足 API 条件 |
| P06 | 预处理粒度、批处理、生产消费队列深度 | 区分读取、解码、变换、H2D；增加队列深度可能增加数据陈旧程度与尾延迟 |
| P07 | Python 线程/进程、原生算子并行 | Python 解释器计算和释放 GIL 的原生算子行为不同；多进程还要计入 IPC 和数据复制 |

线程与 NUMA 调优依据见 [PyTorch Performance Tuning Guide](https://docs.pytorch.org/tutorials/recipes/recipes/tuning_guide.html)；worker、预取、pinned memory 与多进程约束见 [DataLoader 2.7](https://docs.pytorch.org/docs/2.7/data.html)；算子间线程设置限制见 [set_num_interop_threads 2.7](https://docs.pytorch.org/docs/2.7/generated/torch.set_num_interop_threads.html)。实际起始扫描值及队列时延取舍为本文实验建议。API 细节应以项目安装版本为准。

## 6. 多 GPU / 多机：计算与通信一起配置

| ID | 参数/入口 | 对并行加速的影响 |
|---|---|---|
| D01 | 数据并行副本数、world size、rank 到设备映射 | 增加训练吞吐或推理服务容量；不自动缩短一个请求的执行路径 |
| D02 | tensor/pipeline/expert parallel 度与分区 | 将单模型分到多设备，也引入通信和负载不均；优先验证每阶段实际成本 |
| D03 | microbatch 数与大小、流水调度 | 平衡流水空泡、通信频率、显存和延迟；多 microbatch 不等于单请求更快 |
| D04 | 全局 batch、梯度累积、同步频率 | 改变每次同步可摊销的计算；训练比较必须保留收敛与有效 batch 口径 |
| D05 | DDP `bucket_cap_mb`、梯度就绪顺序 | 小 bucket 可更早开始通信，但增加调用；大 bucket 提高聚合效率却可能推迟重叠 |
| D06 | DDP `find_unused_parameters`、`static_graph`、`gradient_as_bucket_view` | 影响图遍历和内存；必须满足模型图/梯度语义条件 |
| D07 | FSDP/ZeRO 分片、prefetch、reshard、CPU offload | 在显存节省、通信次数和主机传输间取舍；不是单向的加速开关 |
| D08 | `NCCL_SOCKET_IFNAME`、`NCCL_IB_HCA`、rank/NIC 亲和 | 选择正确网络路径；先排除误用慢网卡和跨 NUMA 路径 |
| D09 | `NCCL_ALGO`、`NCCL_PROTO` | 算法/协议效果依赖拓扑、消息量和版本；先使用自动选择作为基线 |
| D10 | `NCCL_MIN_CTAS` / `NCCL_MAX_CTAS` | 通信 kernel 也占 GPU 资源；更多通信资源可能挤压计算 |
| D11 | P2P、GDR、跨 NIC 路由设置 | 如 `NCCL_P2P_LEVEL`、`NCCL_NET_GDR_LEVEL`、`NCCL_CROSS_NIC`；按实际拓扑诊断，避免照抄环境变量 |

DDP 参数依据见 [PyTorch DDP 2.7](https://docs.pytorch.org/docs/2.7/generated/torch.nn.parallel.DistributedDataParallel.html)；分片、预取和 offload 见 [FSDP 2.7](https://docs.pytorch.org/docs/2.7/fsdp.html)；microbatch 与流水调度见 [Pipeline Parallelism 教程](https://docs.pytorch.org/tutorials/intermediate/pipelining_tutorial.html)；通信参数与版本要求见 [NCCL Environment Variables](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html)。D01–D04、D07 是需要配合对应框架实现的策略层因素，不能直接套用到当前单卡推理 CLI。NCCL 中一些旧 channel 变量已被 CTA 配置取代或弃用，应检查安装版本。

## 7. 在线服务：吞吐、排队与单次执行要拆开

| ID | 参数/入口 | 影响与建议 |
|---|---|---|
| S01 | 客户端并发、在途请求上限、到达率 | 并发不足可能喂不满 GPU；过高则放大排队，必须一起报告吞吐和 P99 |
| S02 | 动态 batching：`max_batch_size`、`preferred_batch_size` | 提高合批机会；实际 batch 受流量和 shape 兼容性影响 |
| S03 | `max_queue_delay_microseconds` | 允许等待以形成更大 batch；不能只报告 GPU 执行时间而漏掉等待 |
| S04 | 模型实例数：Triton `instance_group` | 提高同时处理能力，但增加资源竞争和模型显存 |
| S05 | TensorRT `max_aux_streams`、跨请求 execution context 数 | 前者用于单推理独立层，后者用于请求间并发；分别做消融 |
| S06 | 长度/shape 分桶、队列优先级、背压 | 减少不兼容请求混批与长任务阻塞；需要真实到达分布验证 |

动态 batching、排队参数和实例数见 [Triton Dynamic Batching & Concurrent Model Execution](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/tutorials/Conceptual_Guide/Part_2-improving_resource_utilization/README.html)；两种 multi-streaming 见 [TensorRT Optimizing Performance](https://docs.nvidia.com/deeplearning/tensorrt/latest/performance/optimization.html)。S01、S06 的实验设计是根据排队与资源竞争作出的工程建议。Triton Inference Server 与生成 GPU kernel 的 Triton 编译器是不同项目。

## 8. 基于 Pi0.5 的 VLA 模型：已知设置、证据缺口与优先级

本节针对本项目基于 Pi0.5 的 VLA 模型，依据本地 [A/B trace 分析](pytorch_ab_trace_analysis.md)、[A/B 原始汇总](../traces/pytorch/AB_nsys_analysis.txt)、[B 正式结果](../results/pytorch/B.json) 和当前源码；历史采样与当前源码配置分开记录。

| 项目 | 已知情况 | 对实验的含义 |
|---|---|---|
| GPU | 结果记录为 NVIDIA GeForce RTX 4090 | 不能仅用名称填入所有硬件参数 |
| 显存 | JSON 为 `50866487296` bytes，约 47.37 GiB | 容量记录需与采样机设备属性、UUID/PCI ID 核对；不据此断言改装、虚拟化或记录错误 |
| 软件 | Python 3.11.16、PyTorch 2.7.1+cu126、CUDA runtime 12.6、Transformers 4.53.2 | 查 PyTorch 接口优先参考 2.7 文档；driver 与 runtime 版本是不同字段 |
| 工作量 | Pi0.5、随机权重、batch=1、10 步、action horizon=50、max token len=200、BF16 | 反映此合成负载；真实 checkpoint 的任务质量仍需另测 |
| 输入位置 | `synthetic_device_resident_sample_actions` | 当前计时不能用于推断相机采集、CPU 解码、H2D 或服务端到端性能 |
| 精度及库设置 | float32 matmul precision=`high`，cuDNN benchmark=false | `high` 的影响针对相应 FP32 运算，不表示所有 BF16 运算都被该开关改变 |
| A/B | A eager；B 请求 max-autotune；旧采样均串行相机 | A/B 主要比较编译，不能当相机并行实验 |
| 时间线 | 仅记录一个 GPU kernel stream；B 有融合 kernel 与 graph replay | 足以支持编译/graph 已实际发生，不支持多 stream 已加速的说法 |
| 性能 | 正式 P50：A 316.37 ms，B 58.64 ms，约 5.39× | 采样有额外扰动，不能拿采样分段时间精确分摊正式收益 |
| B 内部边界 | 每推理 11 个 graph launch、11 次 1-byte D2H/stream 同步 | 与 tensor 控制 Python 循环一致；固定步数表达值得独立验证 |
| 当前相机代码 | `pytorch_camera_batching=True`；仅推理时合批；shape/dtype/device 不兼容则回退 | 默认配置变化后，复现历史 B 必须显式指定 serial |
| 当前 benchmark | `--compile` 只接收 off/max-autotune；`--image-batching` 接收 serial/batched | 其他编译模式是通用候选，需要扩展 CLI 才能测 |
| 采样工具 | 旧报告提示 driver 13.2 与 nsys 2024.6.2 不兼容，事件可能漏采 | 新归因实验先用与采样机 driver 兼容的 profiler |
| 尾延迟 | B 存在慢样本；已有证据不足以确定原因 | 记录频率、功耗、温度、其他进程；不能直接归因降频或竞争 |

相关源码：[配置](../src/openpi/models/pi0_config.py)、[相机合批](../src/openpi/models_pytorch/image_batching.py)、[模型与去噪循环](../src/openpi/models_pytorch/pi0_pytorch.py)、[测速 CLI](../benchmark_pytorch.py)。

推荐实验顺序如下；优先级是本文根据现有瓶颈证据作出的判断。

| 优先级 | 实验 | 对照与验收 |
|---|---|---|
| P0 | 环境与基线复核 | 显式 serial + max-autotune；独占条件、兼容 profiler；保留全部慢样本 |
| P1 | 相机合批 | 同版本、同输入下 serial/batched；比较端到端、视觉阶段、显存和数值误差 |
| P1 | 固定整数步数循环 | 保持步数、时间序列、更新和随机输入等价；看主机回读、graph 边界和端到端是否改善 |
| P2 | GEMM/attention 热点 | 将前缀与 addmm kernel 映射到算子、形状和生成代码，再选后端/布局/融合实验 |
| P2 | 三路相机多 stream | 与最佳合批方案比较；检查真实重叠、分支单独/并发耗时和资源竞争 |
| P3 | 多请求流水 | 独立吞吐实验，增加并发/batch；报告排队、P99 和显存，不混入单请求加速结论 |
| P3 | 硬件频率/多 GPU/通信 | 只有证据指出相应瓶颈或部署目标需要时开展 |

三相机独立编码是候选并行区域；prefix 之后的去噪步骤依赖前一步状态，不能将 10 步简单分配到 10 个 stream。旧 A 中视觉与语言 embedding 的 GPU 活跃时间约 11.234 ms，但不能用它直接估计 B 的视觉占比，更不能推出整体三倍加速。

## 9. 最小实验矩阵与结果记录

先做小范围消融，再测交互，避免直接穷举全部参数。

| 组别 | 编译 | 相机执行方式 | 用途 |
|---|---|---|---|
| A′ | off | serial | 当前源码上的 eager 对照，不复用旧 A 时间充当新结果 |
| B′ | max-autotune | serial | 当前源码上的编译基线 |
| C′ | off | batched | 单独观察相机合批效应 |
| D′ | max-autotune | batched | 检查编译与合批的交互，核心比较 B′/D′ |
| E′ | 同 B′ | 2/3 路 stream | 需另行实现；不能当作现有 CLI 已支持 |

固定同一个 checkpoint/随机权重、输入哈希、seed、batch、shape、去噪步数和精度。先独立测 loop 改写、attention 后端等，再组合有收益的方案。合批与多 stream 会改变 kernel 可用资源，某配置下选出的最优 kernel 不保证在另一配置下仍最优。

每个配置建议至少 100 次稳态测量并做多轮交错对照；这是针对当前波动的起点，精确估计 P99 需要更多样本。保留冷启动、编译、autotune 与稳态时间的独立记录。旧 B 首调用约 329.9 秒，短生命周期部署不能忽略这项成本。

| 记录组 | 字段 |
|---|---|
| 身份 | git commit/dirty 状态、源码哈希、输入/checkpoint 哈希、seed |
| 硬件 | GPU UUID/PCI ID/型号/SM/显存；CPU、RAM、NUMA、PCIe/NVLink/NIC 拓扑 |
| 运行状态 | OS、驱动模型、driver/runtime/library 版本、共享进程、资源配额 |
| 配置 | 本次修改的参数与有效值、编译模式、graph break、shape 桶、线程/stream/实例数 |
| 时间 | 冷启动、P50/P95/P99、均值、原始逐次延迟、测量次数、真实吞吐 |
| 资源 | 峰值 allocated/reserved、SM/DRAM 指标、频率/功耗/温度、拷贝量 |
| 归因 | kernel 数、提交数、graph 数、同步点、阶段耗时、并发区间 |
| 正确性 | 输出误差容限、NaN/Inf、真实任务质量或训练收敛 |

### 测量时最容易混淆的四项

1. **时间线有 kernel 不等于所有 SM 已满载。** 占用率、SM 吞吐、显存带宽与 kernel 活跃时间是不同指标。
2. **CPU 同步等待与 GPU 执行不能相加。** 同步可能正好在等必要计算，删除同步调用不等于删掉等长工作。
3. **并发 kernel 的时长和不等于端到端时间。** 用时间区间并集与关键路径分析重叠；只看“重叠比例”也不能判断是否加速，因为 kernel 可能同时被拖慢。
4. **异步调用提交耗时不等于完成耗时。** 端到端计时包含目标工作完成；多 stream CUDA Event 计时需建立覆盖所有相关 stream 的依赖。不要在每个分支后立刻全局 synchronize，从而破坏被测并发。

先在不挂 profiler 的条件下测性能，再使用 Nsight Systems 解释提交/等待/重叠，使用 Nsight Compute 针对热点检查资源瓶颈；后者的 replay/采样可能改变原并发行为，不能把采样时间直接当正式时间。[Nsight Systems User Guide](https://docs.nvidia.com/nsight-systems/UserGuide/index.html)、[Nsight Compute Profiling Guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html)

## 10. 来源使用范围

文中来源均为已联网检索的 NVIDIA/PyTorch 官方文档，链接紧邻对应清单。PyTorch 核心 API 尽量使用项目匹配的 2.7 文档；其他在线文档可能随版本变化，变量是否存在、默认值、设备支持范围均须与实际部署版本核对。扫描值、优先级、实验矩阵和项目瓶颈推断属于本文建议，不是官方对本项目收益的保证。

当前清单覆盖 **70 个因素项**（H 16、G 14、C 16、P 7、D 11、S 6），其中既有可调旋钮，也有必须记录的硬件条件。实践上先找关键路径和限制资源，再从对应类别选少量参数实验。
