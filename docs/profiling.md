# 抓取与分析

打点之后的一切都是 NVIDIA 原版 IKET 工具：`run-iket` 负责注入、预扫、插桩、解码、出图。
本篇是用它时的要点；IKET 本身的参考是 NVIDIA 文档
[IKET Profiling](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/guides/iket_profiling.html)。

## run-iket 基本用法

```bash
bin/run-iket -o OUT --clobber [--log-level info] profile --postprocess perfetto --keep -- <命令及参数>
```

| 选项 | 说明 |
|---|---|
| `-o OUT` | 输出目录。已存在时会**交互式询问**是否删除；非交互环境（pod、脚本）里没人回答会卡住或失败，所以先删掉或加 `--clobber` |
| `--postprocess perfetto` | 外发版默认 `none`，不写就只有原始记录、没有 `.pftrace`；`all` 同时出 JSON 与 HTML 查看器 |
| `--keep` | 外发版默认 `false`，不写会删掉 `tracker/`、`gen-config/` 等中间件（`instrument.config.json` 就在里面） |
| `--max-ts-cnt-per-warp N` | 每个 warp 的事件容量提示；默认由预扫自动估计 |
| `--use-config FILE` | 用裁剪过的插桩配置（例如只保留一个 kernel），同时跳过预扫 |
| `--warp-trace` | 只记 warp 生命周期，不需要 marker |

`run-iket` 会把命令**至少跑两遍**：一遍 tracker 预扫统计事件量，一遍真正插桩。所以：

- 被测程序必须能重复运行、每次行为大致一样（事件数稳定，缓冲才够）；
- 程序自己写的结果文件会被写两次，按"最后一条"取；
- 耗时约是正常运行的两倍多，加上解码和出图。

`bin/run-iket` 与 NVIDIA 的 `run-iket` 是同一个入口，用于只装了 `iket` 包、没有这个命令的环境
（例如 `lmsysorg/sglang:*-runtime`）。`run-iket` 与 nsys、ncu 等基于 CUPTI 的工具互斥，不能同时用。

## 被测程序的形式

任何用 CUDA 的程序都可以：原生可执行文件、Python、PyTorch、torchrun 多进程都行，
因为注入是通过 CUDA 的注入机制完成的，子进程会继承。

**JIT 编译的 kernel**（NVRTC、`torch.utils.cpp_extension`、运行时调 nvcc 的库等）要保证编译时 `IKET_ENABLE` 生效：
编译参数里加 `-DIKET_ENABLE`，或在生成的源码开头 `#define IKET_ENABLE`。注意 JIT 缓存：打点版和原版要能区分开
（按源码或头文件内容做缓存键的库会自动区分），否则可能加载到旧的未打点版本。

**多进程 / 多卡**：`run-iket ... -- python3 -m torch.distributed.run --nproc-per-node 8 ...`。
每个进程出一个 `iket_pid_<pid>.pftrace`；torchrun 按 rank 顺序启动子进程，pid 升序大致就是 rank 顺序。
不跑被测 kernel 的进程/上下文会产生几十字节的空 trace。

**CUDA graph**：IKET 的解码格式和出图支持 graph 里的 launch（解码结果里单列为 `graphLaunches`，
Perfetto 里显示为 `GraphExec… Launch…`），但我们还没有实测过 graph 下的抓取；稳妥起见先用 eager 调用。

## trace 的大小

大 kernel 的 trace 很大：一个多卡的持久 kernel，每个进程的 `.pftrace` 可达一两百 MB，多卡乘以卡数；
解码 JSON 比 `.pftrace` 还大几倍。输出目录放在本地盘（如 `/tmp`），只拷走需要的产物。

## 产物

```
OUT/
  iket_pid_<pid>.pftrace                    每个进程的 Perfetto trace
  iket/pid_<pid>/iket.decoded_results.json  解码后的全部记录（汇总工具读它）
  gen-config/instrument.config.json         实际插桩的 kernel 与点（--keep）
  tracker/                                  预扫结果（--keep）
```

先看 `instrument.config.json`：确认要看的 kernel 在 `configs` 里、每个 marker 都有对应的插桩点。
缺了说明编译时 `IKET_ENABLE` 没生效，或加载的不是打点版（例如 JIT 用了旧缓存）。

## 读时间线

Perfetto 里的层级：

```
Grid<n>（每次 launch）→ GPC → TPC|VSM → CTA → Warp
    ├─ Warp Life Time          warp 从启动到退出
    ├─ StackedRanges           push/pop 区间，嵌套画成层叠
    ├─ <start/end 区间名>       每个 start/end 区间一条轨
    └─ Marker                  mark 时间点
```

常用读法：

- **谁在等谁**：把"等待"区间（barrier、mbarrier、spin）和相邻的"干活"区间对照。
  例如 MMA warp 上 `mma_wait_ab` 很宽、`mma_issue` 很窄，说明 tensor core 在等数据，而不是算不过来；
  反过来，`mma_issue` 占满、生产者的 `*_wait_empty` 很宽，才是计算受限。
- **跨 warp 对齐**：同一时刻各角色在做什么，用 Perfetto 的竖线（点一个时间点）对齐看。
- **尾部与不均衡**：看各 CTA 的 Warp Life Time 结束时间，以及最后一个 task 落在哪些 SM 上。

## 数字汇总

```bash
PYTHONPATH=<仓库> python3 -m iket_cuda.summary OUT/iket/pid_<pid>/iket.decoded_results.json --kernel <名字片段>
```

- 取匹配 kernel 的**最后一次** launch（预热和正式调用都会被记录）；
- 每个区间名一行：所在 warp、次数、p50 / p99、总时长、`share` = 总时长 / 所在 warp 的生命周期之和；
- 嵌套区间与父区间重叠，只和父或同一 warp 上的兄弟比；
- 对 CuTe DSL 和 C++ kernel 的输出完全通用，可以直接横比两个实现。

## 和 ncu 配合

iket 回答"哪个 warp 角色在关键路径上、在等谁"，ncu 回答"硬件单元有多忙、停在哪条指令"。几个要点：

- ncu 与 `run-iket` 不能同时用，要分开跑；ncu 测的是**不插桩**的 kernel，正好用来校准插桩带来的扰动；
- 跨卡同步的持久 kernel（例如经 NVLink 做 all-to-all 的 MoE kernel）ncu 无法 replay，用单卡代理（每卡计算负载相同）测计算部分；
- `--clock-control none` 让时长和计时运行可比；`--set full` 含 PM sampling，能看到不插桩 kernel 里随时间变化的单元利用率；
- warp 专业化 kernel 的停顿统计大多来自在等的 warp，用源码级采样（`--page source --print-source sass`）定位是哪个等待，再对回 iket 的角色。

## 开销与可信度

- 每个事件是几条指令加一次 shared 写，再被注入库补丁改成真正的落盘，**会改变时序**。
  只用 trace 做归因（先后、比例、谁在等谁）；耗时用不打点的版本单独测，并对照算出被拉长了多少。
- **热循环里的打点会拖慢被观察的 warp**：例如在 MMA warp 的 K 循环里每次迭代打多次点，会让"MMA 发指令"看起来成了瓶颈，
  其实是打点本身的开销。用"插桩 kernel 跨度 vs 不插桩时间"衡量扰动；可疑时去掉热循环里的点、只留阶段级的点复核。
- **多卡时一个 rank 的 host 停顿会传给所有 rank**：它晚发起 launch，其它 rank 就在第一个跨卡屏障上等它，看起来像 kernel 内的
  长时间等待（可达毫秒级）。整层计时被单次离群拉偏时，逐 launch、逐 rank 看 trace；跨卡屏障的时长要先排除这种错位再解读。
- 时间戳是各 GPU 自己的 `%globaltimer`（32 ns 粒度），跨 GPU 不对齐。
