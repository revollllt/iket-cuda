# C++ API 参考（`include/iket/iket.cuh`）

header-only。编译时加 `-I<仓库>/include`；定义 `IKET_ENABLE`（`-DIKET_ENABLE` 或在 include 前
`#define IKET_ENABLE`）才生效，否则下面所有宏展开为空、不生成任何代码和元数据。

与 CuTe DSL 的对应：

| CuTe DSL（`cute.experimental.iket`） | 本头文件 |
|---|---|
| `iket.range_push(name[, payload])` / `iket.range_pop()` | `IKET_RANGE_PUSH(ident[, payload])` / `IKET_RANGE_POP()` |
| `tok = iket.range_start(name[, payload])` / `iket.range_end(tok[, payload])` | `IKET_RANGE_START(ident[, payload])` / `IKET_RANGE_END(ident[, payload])` |
| `iket.mark(name[, payload])` | `IKET_MARK(ident[, payload])` |
| 事件名自动编号 | `IKET_DECLARE_*` 显式声明（见下） |
| `options="iket"` / `CUTE_DSL_COMPILER_OPT=iket` | `-DIKET_ENABLE` |

## 声明事件

每个事件在**全局命名空间**声明一次（声明会生成 `iket_events::<ident>` 类型和 cubin 元数据）：

```cpp
IKET_DECLARE_RANGE(ident, id, "display name", ::iket::Payload::<type>);      // push/pop 区间
IKET_DECLARE_START_END(ident, id, "display name", ::iket::Payload::<type>);  // start/end 区间
IKET_DECLARE_MARK(ident, id, "display name", ::iket::Payload::<type>);       // 时间点
```

| 参数 | 规则 |
|---|---|
| `ident` | C++ 标识符，打点宏用它引用事件 |
| `id` | 模块（编译单元）内唯一。默认 1..30；定义了 `IKET_EXTENDED_EVENT_IDS` 时 64..4095 |
| `"display name"` | Perfetto 里显示的名字，最多 31 字节，可以含 `.` 等字符（如 `"Pull.ChooseToken"`） |
| payload 类型 | `None`、`I32`、`U32`、`F32`、`I64`、`U64` |

**事件多于 30 个**：在 include 之前 `#define IKET_EXTENDED_EVENT_IDS`，id 改用 64 起。整个模块只能二选一，
这和 CuTe DSL 的行为一致（它在名字超过 30 个时把整个模块切到 extended 记录：事件 id 不再塞进时间戳低 5 位，
而是和时间戳一起写成 64 位）。

`id` 为什么要手写：C++ 没法在编译期给整个编译单元里出现的名字自动编号。
推荐做法是把一个 kernel 的事件表集中写在文件开头，按顺序编号。

## 打点

```cpp
IKET_RANGE_PUSH(ident);             // 或 IKET_RANGE_PUSH(ident, payload)
IKET_RANGE_POP();
IKET_RANGE_START(ident);            // 同一个 ident 开始与结束
IKET_RANGE_END(ident);
IKET_MARK(ident);                   // 或 IKET_MARK(ident, payload)
```

语义（与 CuTe DSL 相同）：

- **warp 级**：一个 warp 执行到打点就记录一条，不是每个线程一条。带 payload 时记录第一个活跃 lane 的值；
  只有部分 lane 活跃时（例如 `elect_one_sync()` 块里）也可以打点。
- **push/pop 按 warp 后进先出**：每条执行路径上 push 和 pop 都必须配平；不要在 push 与 pop 之间
  `return`/`break` 出去。pop 不指定名字，自动关掉最近一次 push。
- **start/end** 用同一个 `ident` 配对，不嵌套在 push/pop 栈里，在 Perfetto 里单独一条轨；
  起止的 payload 形式必须一致（都带或都不带，同一类型）。
- **mark** 是时间点，画在 warp 的 Marker 轨上。
- payload 的 C++ 类型须与声明匹配：32 位类型接收 `int`/`unsigned`/`float` 等，64 位类型接收 `long long` 等，
  不匹配时编译期报错。注意 **payload 会拼进 Perfetto 的显示名**（例如每个 tile 一个名字），
  想在 Perfetto 里按名字聚合的区间不要带 payload。

## 编译要求

- sm_90 及以上（用到 `%cluster_ctarank`、`elect.sync`）；已验证 sm_100a / sm_100f（Blackwell）。
- C++17（`if constexpr`）。不需要 `--expt-relaxed-constexpr`、`-rdc`。
- nvcc 已验证：只编 device（`--cubin`，JIT 场景）和普通的 host+device 程序都可以。NVRTC 生成的是同样的
  内联 PTX 和 device 变量，应当同样适用，但尚未单独测试。
- 一个编译单元就是一个模块，事件表在模块内生效；同一个进程里多个模块各自独立编号。

## 放在哪里打点

- 按 **warp 角色**打：warp 专业化的 kernel 里，每个角色（生产者 / MMA / epilogue ……）的主循环各打一组，
  时间线上一眼能看出谁在等谁。
- **等待单独成区间**：把 barrier / mbarrier 等待包成独立区间（如 `mma_wait_ab`），
  它和紧挨着的"干活"区间的比例最能说明瓶颈。
- **粒度**：每个 warp 每次 launch 的事件数决定缓冲大小和开销；K 循环里每次迭代打点是常见的最细粒度。
- 与已有 kernel 做同一件事的阶段**沿用同一个名字**，两边的汇总可以直接横比。

## 常见编译错误

| 报错 | 原因 |
|---|---|
| `IKET event id out of range` | id 不在 1..30（或 extended 下 64..4095）；事件超过 30 个时加 `IKET_EXTENDED_EVENT_IDS` |
| `IKET event name too long` | 显示名超过 31 字节 |
| `this event is declared with a payload` / `without a payload` | 打点时带不带 payload 与声明不一致 |
| `payload wider than the declared 32-bit type` | 给 32 位 payload 传了 64 位值 |
| 找不到 `iket_events` 或其成员 `<ident>` | 事件没声明，或声明不在全局命名空间 |
| 重复定义 `__iket_evt_decl_<ident>_attrs` | 同一 ident 声明了两次 |
