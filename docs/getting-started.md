# 快速上手：给一个 CUDA kernel 抓 warp 级时间线

本篇用 `examples/01_quickstart` 走完整流程：在 kernel 里打点 → 编译 → 用 `run-iket` 抓取 →
在 Perfetto 里看时间线 → 看数字汇总 → 关掉打点。全程约 10 分钟。

## 0. 环境

需要三样东西，SGLang 的镜像里都有：

| 需要 | 用途 | 检查 |
|---|---|---|
| CUDA 13 的 `nvcc` | 编译 kernel | `nvcc --version` |
| Blackwell GPU（sm_100 / sm_103） | 运行；marker 用到 sm_90+ 的 `%cluster_ctarank` | `nvidia-smi` |
| Python 里的 `iket` 包（随 `nvidia-cutlass-dsl` 安装） | `run-iket`、解码、出图 | `python3 -c "import iket"` |

直接用镜像（`--nv` 才能看到 GPU；只编译不运行时可以不加）：

```bash
apptainer shell --nv $SIF        # SIF：lmsysorg/sglang:v0.5.19-cu130-runtime 转成的镜像
export PATH=/usr/local/cuda/bin:$PATH
cd iket-cuda                     # 本仓库
```

不要在别人正在用的 GPU 上跑：先 `nvidia-smi` 看显存和进程。

## 1. 在 kernel 里打点

`examples/01_quickstart/quickstart.cu` 是一个分块求和的 kernel：每个 block 依次把自己的 tile
从显存搬进 shared memory 再求和。打点只需要三步：

**① 引入头文件。** 编译时加 `-DIKET_ENABLE` 才生效，不加时所有宏都展开为空。

```cpp
#include <iket/iket.cuh>
```

**② 在全局命名空间声明事件。** 每个事件一行：标识符、模块内唯一的 id（1..30）、显示名、payload 类型。

```cpp
IKET_DECLARE_RANGE(load_tile, 1, "load_tile", ::iket::Payload::U32);      // push/pop 区间，带 tile 编号
IKET_DECLARE_RANGE(reduce_tile, 2, "reduce_tile", ::iket::Payload::None); // push/pop 区间
IKET_DECLARE_START_END(block_life, 3, "block_life", ::iket::Payload::None); // start/end 区间
IKET_DECLARE_MARK(tile_done, 4, "tile_done", ::iket::Payload::U32);       // 时间点
```

**③ 在 kernel 里标出阶段。**

```cpp
__global__ void tile_sum(const float* input, float* output) {
    IKET_RANGE_START(block_life);
    for (int t = 0; t < kTilesPerBlock; ++ t) {
        const unsigned tile_idx = blockIdx.x * kTilesPerBlock + t;

        IKET_RANGE_PUSH(load_tile, tile_idx);     // 带 payload：记录这是哪个 tile
        /* 显存 -> shared memory */
        IKET_RANGE_POP();

        IKET_RANGE_PUSH(reduce_tile);
        /* 求和 */
        IKET_RANGE_POP();

        IKET_MARK(tile_done, tile_idx);
    }
    IKET_RANGE_END(block_life);
}
```

几条规则（完整说明见 [api.md](api.md)）：

- 事件是 **warp 级**的：每个 warp 各自记录一条时间线；带 payload 时记录第一个活跃 lane 的值。
- `PUSH`/`POP` 在每个 warp 内必须一一配对、后进先出；每条执行路径上都要配平。
- payload 的类型必须和声明一致，否则编译报错。

## 2. 编译

```bash
examples/01_quickstart/run.sh --build-only
```

脚本做了两次编译（目标架构默认 `sm_100f`，sm_100 与 sm_103 通用；可用 `ARCH=103a` 等覆盖）：

```bash
nvcc -std=c++17 -O3 -gencode=arch=compute_100f,code=sm_100f -Iinclude -DIKET_ENABLE quickstart.cu -o out/quickstart_iket
nvcc -std=c++17 -O3 -gencode=arch=compute_100f,code=sm_100f -Iinclude              quickstart.cu -o out/quickstart
```

只要编译通过就说明声明、payload 类型、id 范围都合法（这些都在编译期检查）。
想确认 marker 都进了二进制，可以用 IKET 自己的解析器看（CPU 上就能跑）：

```bash
cd examples/01_quickstart/out && cuobjdump -xelf all quickstart_iket >/dev/null
python3 -c "
import glob; from iket import Context
cubin = [open(p,'rb').read() for p in glob.glob('*.cubin') if b'tile_sum' in open(p,'rb').read()][0]
for k in Context(cubin).get_kernels_instrument_info():
    print(k.kernel_name, [p.event_attr.event_name for p in k.instrument_points])"
```

应当列出 `block_life`、`load_tile`、`pop_range`、`reduce_tile`、`pop_range`、`tile_done`、`block_life`。

## 3. 抓取

```bash
examples/01_quickstart/run.sh
```

它先跑一遍不打点的版本确认结果正确，再用 `run-iket` 包住打点版：

```bash
bin/run-iket -o out/iket --clobber profile --postprocess perfetto --keep -- out/quickstart_iket
```

- `run-iket` 用 CUPTI 注入：先跑一遍 tracker 预扫，统计每个 warp 会记多少事件，据此分配缓冲，
  再正式跑一遍并解码，所以程序会被**执行两次**。
- NVIDIA 外发版里 `--postprocess` 默认是 `none`、`--keep` 默认是 `false`，必须显式写，否则什么都不产出。
- `bin/run-iket` 就是 NVIDIA 的 `run-iket`；有的镜像只装了 `iket` 包而没有这个命令，所以仓库里带一个同样的入口。

产物在 `out/iket/`：

| 文件 | 内容 |
|---|---|
| `iket_pid_<pid>.pftrace` | Perfetto trace，每个进程一个；没跑打点 kernel 的上下文会得到很小的空文件，忽略即可 |
| `iket/pid_<pid>/iket.decoded_results.json` | 解码后的全部记录 |
| `gen-config/instrument.config.json` | 实际被插桩的点：确认你的 marker 都在 |
| `tracker/` | 预扫的中间结果（`--keep` 才保留） |

## 4. 看时间线

把 `.pftrace` 拖进 <https://ui.perfetto.dev>。层级是：

```
Grid<n>（每次 launch 一个）→ GPC → TPC|VSM → CTA → Warp → StackedRanges（push/pop 区间）
                                                        → block_life（start/end 区间单独一条轨）
                                                        → Marker（tile_done 等时间点）
```

这个例子里每个 warp 应当看到 16 组交替的 `load_tile` → `reduce_tile`，每组后面一个 `tile_done`，
外面套一条 `block_life`。点开某个区间，payload（tile 编号）显示在名字和参数里。
示例 launch 了两次（预热 + 正式），所以有两个 Grid。

## 5. 看数字

```bash
PYTHONPATH=. python3 -m iket_cuda.summary out/iket/iket/pid_<pid>/iket.decoded_results.json --kernel tile_sum
```

每个区间名一行：所在 warp、次数、p50 / p99 时长、总时长、占所在 warp 生命周期的比例（share），
只统计最后一次 launch。嵌套区间和父区间重叠，比较时只和父区间或同一 warp 上的兄弟区间比。

打点会改变时序，这些数字用来看**比例和先后**（谁在等谁、哪一段占大头），
kernel 的真实耗时请用不打点的版本测。

## 6. 关掉

不加 `-DIKET_ENABLE` 编译即可：所有 marker 和元数据都不会生成，编出来的代码和删掉 marker
完全一样（`tests/test_iket_abi.py::test_disabled_is_free` 逐行比较 SASS 来保证）。
所以 marker 可以一直留在源码里，需要时加一个编译开关就能抓。

## 下一步

- 宏、payload、事件 id 的完整说明与常见编译错误：[api.md](api.md)
- 多进程 / torchrun、JIT、怎么读 trace、开销：[profiling.md](profiling.md)
- 底层原理与升级 cutlass-dsl 时怎么办：[abi.md](abi.md)
