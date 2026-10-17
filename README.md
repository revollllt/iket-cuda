# iket-cuda：给 CUDA C++ kernel 用的 IKET

让任何 CUDA C++ kernel（纯 CUDA、CuTe/CUTLASS C++ ……）像 CuTe DSL kernel 一样，被 NVIDIA 的 **`run-iket` 原样**
抓出 warp 级时间线：CUPTI 注入、SASS 补丁、预扫、解码、Perfetto 出图全是 NVIDIA 自己的工具。
本仓库只做一件事：**从 C++ 发出 CuTe DSL 发出的那套 IKET ABI**——核心是一个头文件 `include/iket/iket.cuh`。

```cpp
#include <iket/iket.cuh>            // -DIKET_ENABLE 打开；不加时所有宏展开为空、零开销

IKET_DECLARE_RANGE(load_tile, 1, "load_tile", ::iket::Payload::U32);

__global__ void kernel(...) {
    IKET_RANGE_PUSH(load_tile, tile_idx);
    /* ... */
    IKET_RANGE_POP();
}
```

```bash
nvcc -Iinclude -DIKET_ENABLE ... kernel.cu -o app
bin/run-iket -o out --clobber profile --postprocess perfetto --keep -- ./app     # out/*.pftrace 拖进 ui.perfetto.dev
python3 -m iket_cuda.summary out/iket/pid_<pid>/iket.decoded_results.json       # 每个区间的次数、分位数、占比
```

## 文档

| 文档 | 内容 |
|---|---|
| [快速上手](docs/getting-started.md) | 用 `examples/01_quickstart` 走完打点 → 编译 → 抓取 → 看图 → 汇总 → 关闭 |
| [API 参考](docs/api.md) | 宏、事件声明与 id、payload、语义、编译要求、打点位置建议、常见编译错误 |
| [抓取与分析](docs/profiling.md) | `run-iket` 选项与坑、多进程 / JIT、产物、怎么读时间线、开销与可信度 |
| [原理](docs/abi.md) | IKET ABI 细节、本仓库如何复现与验证、升级 cutlass-dsl 时怎么办 |

## 仓库结构

```
include/iket/iket.cuh       C++ 接口（header-only）：IKET ABI 的发出端
iket_cuda/summary.py        run-iket 结果的阶段汇总（CuTe DSL 与 C++ 通用）
bin/run-iket                run-iket 入口（给只装了 iket 包、没带 console script 的环境）
examples/01_quickstart/     上手示例：quickstart.cu + run.sh
tests/                      CPU 测试
docs/                       文档
```

Python 侧不需要安装：把仓库根目录放进 `PYTHONPATH`（`python3 -m iket_cuda.summary`）；C++ 侧 `-I<仓库>/include`。

## 兼容性

| 组件 | 已验证版本 | 说明 |
|---|---|---|
| IKET / CuTe DSL | nvidia-cutlass-dsl 4.6.2（IKET 0.5，magic 0xBABEF19D） | ABI 来源；升级见 [docs/abi.md](docs/abi.md) |
| CUDA | 13.0（nvcc 13.0.88） | 需 sm_90+；已测 sm_100a / sm_100f（Blackwell） |
| 运行环境 | `lmsysorg/sglang:v0.5.19-cu130-runtime` | 带齐 nvcc、iket、orjson；任何装了 CUDA 13 与 nvidia-cutlass-dsl 4.6.2 的环境都可以 |

**验证状态**：ABI 与 CuTe DSL 一致、关闭时零开销、上手示例的编译，均由 CPU 测试保证；
GPU 上 `run-iket` 对 nvcc 编出的 C++ kernel 的插桩、落盘、解码已实测（Blackwell，一个多卡的 warp 专业化持久 kernel，
extended 事件 id）。

## 测试（CPU，约 10 s）

```bash
python3 -m pytest -q          # 在上面的运行环境里（docker / apptainer 均可）
```

| 测试 | 证明什么 |
|---|---|
| `test_iket_abi.py::test_cpp_matches_cute_dsl[native/extended]` | 同一组事件（push/pop、start/end、mark；payload 无/i32/i64/f32），C++ 与 CuTe DSL 编出的 cubin 在 IKET 解析器看来完全一致 |
| `test_iket_abi.py::test_disabled_is_free` | 不定义 `IKET_ENABLE` 时 SASS 与删掉打点的 kernel 逐行相同，cubin 里无 `__iket` 符号 |
| `test_examples.py` | 上手示例按文档命令编译通过（host+device 程序），打点都能被 IKET 看到，关闭版无元数据 |

## 限制

- 依赖 IKET 0.5 未公开的 ABI；换 cutlass-dsl 版本后先跑测试（见 [docs/abi.md](docs/abi.md)）。NVIDIA 若改格式或官方支持
  C++ 打点，这个头文件就要跟着改或不再需要。
- 事件 id 需手工分配；一个模块超过 30 个名字要整体切到 extended id。
- 打点会改变时序（IKET 本身的代价，多卡时还会让各 rank 失步），只用于归因，耗时用不打点的版本测。
