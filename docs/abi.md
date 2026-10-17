# 原理：IKET ABI 与本仓库如何复现它

## 为什么 C++ kernel 也能用 run-iket

CuTe DSL 带 `options="iket"` 编译时，把 `iket.*` 调用降级成**普通的内联 PTX**和几个**全局字节数组**；
`run-iket` 的注入库在模块加载时从 cubin 读元数据、在 SASS 里找到打点序列并打补丁，把记录写进它自己的缓冲。
这个过程只看 cubin 的内容，不关心 cubin 是谁编的。所以只要 C++ 编出同样的内容，
注入、预扫、解码、出图就全部原样可用。`include/iket/iket.cuh` 做的就是这件事。

## ABI 细节（cutlass-dsl 4.6.2 / IKET 0.5 实测）

测法：用 CuTe DSL 编若干探针 kernel（开/关 iket、各种 payload、多于 30 个名字），
`CUTE_DSL_KEEP=ptx,cubin,sass` 导出后比对，再用 `iket.Context(cubin)`（`run-iket` 用的解析器）读出元数据和插桩点。

### 打点序列

native 模式（模块内 ≤ 30 个事件名）：

```
mov.b32   %r, %cluster_ctarank;
mov.u32   %t, %globaltimer_lo;
or.b32    %t, %t, <id>;                       // 事件 id 塞进时间戳低 5 位（globaltimer 粒度 32 ns）
mad.lo.u32 %r, %r, 0x1000000, 0x20;           // 标记用的 shared 地址：cluster 内 CTA 号 << 24 | 0x20
st.weak.shared.u32 [%r], %t;
pmevent.mask <id>;                            // SASS 里是 PMTRIG <id>：注入库的锚点
```

带 payload 时先 `activemask` + `elect.sync` 选一个 lane，由它写时间戳和 payload
（32 位写 `[%r+4]`，64 位写 `[%r+8]`）。

extended 模式（> 30 个名字，整个模块切换）：id 从 64 起，不再 `or` 进时间戳，而是
`mov.b64 %ts_evtid, {%t, %evtid}; st.weak.shared.u64 [%r], %ts_evtid;`，payload 写 `[%r+8]`。

`range_pop` 在两种模式下都是事件 id **31**；id 0 保留。

### 元数据

| 符号 | 大小 | 字段（均为 u32，名字为定长字符数组） |
|---|---|---|
| `__iket_meta_info` | 48 B | size=48, 0, version_minor=5, max_event_id（31 或 4095）, max_name=32, evt_decl_size=60, magic=0xBABEF19D, 0, supported_features=3, 0, 0, 0 |
| `__iket_evt_decl_<名>_attrs` | 60 B | size=60, event_id, method（3 NATIVE_DUMP / 5 EXTENDED_NATIVE_DUMP）, payload_type, event_pos, range_id, name_len, name[32] |
| `__iket_range_decl_<名>_attrs` | 72 B | size=72, scope=0（warp 内）, range_id, color=0xFFFFFFFF（自动）, type（2 push/pop / 1 start/end）, pair_mode（0 / start/end 为 1）, 0, 0, name_len, name[32], 0 |

- `range_id` = 名字的 **FNV-1a 32 位**哈希；
- `event_pos`：push 为 1（区间开始）、start/end 为 4（开始与结束同一事件）、mark 为 0；
- payload_type：0 无、5 I32、6 U32、7 I64、13 F32、16 U64。

解析器**只认符号前缀和内容**：符号名后缀（DSL 里带 id 和哈希）不参与；符号绑定是 GLOBAL 还是 LOCAL 也不影响，
所以 nvcc 默认整程序模式（会把这些变量变成 LOCAL）照样能识别。

### 本仓库的实现要点

- 打点序列用内联 asm 逐条照抄，事件 id 用 `"n"` 约束作为立即数传入；
- 元数据用 `static __device__` 结构体、放在全局命名空间：全局命名空间的变量名不会被 mangle，
  符号名保持 `__iket_*`；host+device 程序里 nvcc 会给 device 变量生成运行时注册代码，
  static 变量在本编译单元有定义，注册和链接都能通过（`extern "C"` 的版本没有 host 端定义，普通程序会链接失败）；
- 没有 `IKET_ENABLE` 时声明宏展开为 `static_assert(true, "")`、打点宏展开为空语句，不生成任何代码和符号。

nvcc 写内联 asm 的一个坑：**没有操作数**的 `asm volatile("...")` 不转义 `%%`，要写 `%`；有操作数时要写 `%%`。
本头文件的打点序列都有 `"n"` 操作数，统一写 `%%`。

## 如何证明一致

`tests/test_iket_abi.py` 用同一份事件清单分别生成 CuTe DSL 和 C++ kernel，编译后比较 `iket.Context` 读到的
meta、事件表、range 表和每个 kernel 的插桩点序列，native 与 extended 两种模式各一组，要求完全相等；
再比较关闭时的 SASS 与不写打点的 kernel 逐行相同。

CPU 测试证明的是"`run-iket` 读 cubin 时看到的和 CuTe DSL 一样"。注入库在 GPU 上的补丁与落盘，
要在 GPU 上实际跑一次才算验证。

## 升级 cutlass-dsl / IKET 时

1. 在新镜像里跑 `python3 -m pytest -q`。
2. `test_cpp_matches_cute_dsl` 失败时，断言差异会指出哪一项变了（meta 版本、事件字段、插桩点）。
   用 `CUTE_DSL_COMPILER_OPT=iket CUTE_DSL_KEEP=ptx,cubin` 编一个小探针，对照新的 PTX 和元数据字节改 `iket.cuh`，
   直到测试重新通过。
3. 更新 README 的兼容性表和本文的版本号。
