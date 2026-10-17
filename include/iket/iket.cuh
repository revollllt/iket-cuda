#pragma once

// IKET (In-Kernel Event Tracing) for CUDA C++ kernels.
//
// CuTe DSL lowers `iket.range_push/pop`, `iket.mark`, `iket.range_start/end` to
// inline PTX plus `__iket_*` metadata globals; NVIDIA's `run-iket` (CUPTI
// injection, SASS patching, tracker pre-pass, decode, Perfetto) works on any
// cubin carrying that ABI. This header emits the same ABI from CUDA C++, so
// nvcc / NVRTC kernels (plain CUDA, CuTe/CUTLASS C++, DeepGEMM, ...) are
// profiled by the unmodified `run-iket`, exactly like CuTe DSL kernels.
//
// Zero cost when off: without IKET_ENABLE every macro below expands to nothing
// (no metadata, no instructions), the counterpart of CuTe DSL's
// strip-iket-ops when the kernel is compiled without options="iket".
//
// Usage (one translation unit = one module):
//
//   #define IKET_ENABLE                  // or -DIKET_ENABLE; omit to strip everything
//   #include <iket/iket.cuh>
//
//   // Declare every event once, in the global namespace, with a module-unique id:
//   // 1..30 by default, or 64..4095 with IKET_EXTENDED_EVENT_IDS defined
//   // (needed for more than 30 names; every event of the module then uses
//   // the extended record, as CuTe DSL does).
//   IKET_DECLARE_RANGE(mma_wait_ab, 1, "mma_wait_ab", ::iket::Payload::None);    // push/pop
//   IKET_DECLARE_RANGE(pull_token, 2, "Pull.Token", ::iket::Payload::U32);       // push/pop + payload
//   IKET_DECLARE_START_END(load, 3, "load", ::iket::Payload::None);              // start/end
//   IKET_DECLARE_MARK(tile_done, 4, "tile_done", ::iket::Payload::U32);          // point event
//
//   __global__ void kernel(...) {
//       IKET_RANGE_PUSH(mma_wait_ab);  ...  IKET_RANGE_POP();
//       IKET_RANGE_PUSH(pull_token, expert_idx);  ...  IKET_RANGE_POP();
//       IKET_RANGE_START(load);  ...  IKET_RANGE_END(load);
//       IKET_MARK(tile_done, tile_idx);
//   }
//
// Semantics are CuTe DSL's: events are warp-level (with a payload, the first
// active lane's value is recorded), push/pop nest per warp in LIFO order, a
// payload's type must match its declaration (checked at compile time).
//
// The byte layouts below mirror what CuTe DSL 4.6.2 emits for IKET 0.5
// (`iket.Context(cubin).get_meta_info()`: version 0.5, magic 0xBABEF19D); a
// newer IKET may change them, which tests/test_iket_abi.py detects by
// comparing against a CuTe DSL build of the same events.

#include <cstdint>

#if defined(IKET_ENABLE)

namespace iket {

enum class Payload : uint32_t { None = 0, I32 = 5, U32 = 6, I64 = 7, F32 = 13, U64 = 16 };

struct EventDecl {
    uint32_t size;
    uint32_t event_id;
    uint32_t instrument_method;
    uint32_t payload_type;
    uint32_t event_pos;
    uint32_t range_id;
    uint32_t name_length;
    char name[32];
};

struct RangeDecl {
    uint32_t size;
    uint32_t range_scope;
    uint32_t range_id;
    uint32_t color;
    uint32_t range_type;
    uint32_t pair_mode;
    uint32_t reserved0;
    uint32_t reserved1;
    uint32_t name_length;
    char name[32];
    uint32_t reserved2;
};

struct MetaInfo {
    uint32_t words[12];
};

constexpr uint32_t kMaxNameLength = 32;
constexpr uint32_t kPopEventId = 31;
constexpr uint32_t kEventPosNotInRange = 0, kEventPosRangeStart = 1, kEventPosRangeStartEnd = 4;
constexpr uint32_t kRangeTypeStartEnd = 1, kRangeTypePushPop = 2;
constexpr uint32_t kPairModeInvalid = 0, kPairModeNameOnly = 1;
constexpr uint32_t kColorAuto = 0xFFFFFFFFu;

#if defined(IKET_EXTENDED_EVENT_IDS)
constexpr bool kExtended = true;
constexpr uint32_t kInstrumentMethod = 5;  // EXTENDED_NATIVE_DUMP
constexpr uint32_t kMinEventId = 64, kMaxEventId = 4095;
#else
constexpr bool kExtended = false;
constexpr uint32_t kInstrumentMethod = 3;  // NATIVE_DUMP
constexpr uint32_t kMinEventId = 1, kMaxEventId = 30;
#endif

// range_id is the FNV-1a hash of the range name, as CuTe DSL computes it.
__host__ __device__ constexpr uint32_t fnv1a(const char* name) {
    uint32_t hash = 0x811C9DC5u;
    for (; *name != '\0'; ++ name)
        hash = (hash ^ static_cast<uint8_t>(*name)) * 0x01000193u;
    return hash;
}

__host__ __device__ constexpr uint32_t name_length(const char* name) {
    uint32_t length = 0;
    while (name[length] != '\0')
        ++ length;
    return length;
}

__host__ __device__ constexpr bool is_32bit(const Payload& payload) {
    return payload == Payload::I32 or payload == Payload::U32 or payload == Payload::F32;
}

__host__ __device__ constexpr bool is_64bit(const Payload& payload) {
    return payload == Payload::I64 or payload == Payload::U64;
}

// The recorded sequence, verbatim from CuTe DSL: timestamp (low 32 bits of
// %globaltimer, with the event id in the low bits or beside it) stored to the
// cluster-rank-tagged shared address 0x20, optional payload after it, then the
// `pmevent` the injection library anchors its SASS patch on.
template <uint32_t kEventId>
__device__ __forceinline__ void record() {
    if constexpr (kExtended) {
        asm volatile(
            "{\n\t"
            ".reg .b32 %%r, %%t, %%evtid;\n\t"
            ".reg .b64 %%ts_evtid;\n\t"
            "mov.b32 %%r, %%cluster_ctarank;\n\t"
            "mov.u32 %%t, %%globaltimer_lo;\n\t"
            "mad.lo.u32 %%r, %%r, 0x1000000, 0x20;\n\t"
            "mov.b32 %%evtid, %0;\n\t"
            "mov.b64 %%ts_evtid, {%%t, %%evtid};\n\t"
            "st.weak.shared.u64 [%%r], %%ts_evtid;\n\t"
            "pmevent.mask %0;\n\t"
            "}\n" :: "n"(kEventId));
    } else {
        asm volatile(
            "{\n\t"
            ".reg .b32 %%r, %%t;\n\t"
            "mov.b32 %%r, %%cluster_ctarank;\n\t"
            "mov.u32 %%t, %%globaltimer_lo;\n\t"
            "or.b32 %%t, %%t, %0;\n\t"
            "mad.lo.u32 %%r, %%r, 0x1000000, 0x20;\n\t"
            "st.weak.shared.u32 [%%r], %%t;\n\t"
            "pmevent.mask %0;\n\t"
            "}\n" :: "n"(kEventId));
    }
}

template <uint32_t kEventId>
__device__ __forceinline__ void record_payload32(const uint32_t& payload) {
    if constexpr (kExtended) {
        asm volatile(
            "{\n\t"
            ".reg .pred %%p;\n\t"
            ".reg .b32 %%r, %%t, %%mask, %%evtid, %%payload32;\n\t"
            ".reg .b64 %%ts_evtid;\n\t"
            "activemask.b32 %%mask;\n\t"
            "elect.sync _|%%p, %%mask;\n\t"
            "mov.b32 %%r, %%cluster_ctarank;\n\t"
            "mov.u32 %%t, %%globaltimer_lo;\n\t"
            "mad.lo.u32 %%r, %%r, 0x1000000, 0x20;\n\t"
            "mov.b32 %%evtid, %0;\n\t"
            "mov.b32 %%payload32, %1;\n\t"
            "mov.b64 %%ts_evtid, {%%t, %%evtid};\n\t"
            "@%%p st.weak.shared.u64 [%%r], %%ts_evtid;\n\t"
            "@%%p st.weak.shared.b32 [%%r+8], %%payload32;\n\t"
            "pmevent.mask %0;\n\t"
            "}\n" :: "n"(kEventId), "r"(payload));
    } else {
        asm volatile(
            "{\n\t"
            ".reg .pred %%p;\n\t"
            ".reg .b32 %%r, %%t, %%mask, %%payload32;\n\t"
            "activemask.b32 %%mask;\n\t"
            "elect.sync _|%%p, %%mask;\n\t"
            "mov.b32 %%r, %%cluster_ctarank;\n\t"
            "mov.u32 %%t, %%globaltimer_lo;\n\t"
            "or.b32 %%t, %%t, %0;\n\t"
            "mad.lo.u32 %%r, %%r, 0x1000000, 0x20;\n\t"
            "mov.b32 %%payload32, %1;\n\t"
            "@%%p st.weak.shared.u32 [%%r], %%t;\n\t"
            "@%%p st.weak.shared.b32 [%%r+4], %%payload32;\n\t"
            "pmevent.mask %0;\n\t"
            "}\n" :: "n"(kEventId), "r"(payload));
    }
}

template <uint32_t kEventId>
__device__ __forceinline__ void record_payload64(const uint64_t& payload) {
    if constexpr (kExtended) {
        asm volatile(
            "{\n\t"
            ".reg .pred %%p;\n\t"
            ".reg .b32 %%r, %%t, %%mask, %%evtid;\n\t"
            ".reg .b64 %%ts_evtid, %%payload64;\n\t"
            "activemask.b32 %%mask;\n\t"
            "elect.sync _|%%p, %%mask;\n\t"
            "mov.b32 %%r, %%cluster_ctarank;\n\t"
            "mov.u32 %%t, %%globaltimer_lo;\n\t"
            "mad.lo.u32 %%r, %%r, 0x1000000, 0x20;\n\t"
            "mov.b32 %%evtid, %0;\n\t"
            "mov.b64 %%payload64, %1;\n\t"
            "mov.b64 %%ts_evtid, {%%t, %%evtid};\n\t"
            "@%%p st.weak.shared.u64 [%%r], %%ts_evtid;\n\t"
            "@%%p st.weak.shared.b64 [%%r+8], %%payload64;\n\t"
            "pmevent.mask %0;\n\t"
            "}\n" :: "n"(kEventId), "l"(payload));
    } else {
        asm volatile(
            "{\n\t"
            ".reg .pred %%p;\n\t"
            ".reg .b32 %%r, %%t, %%mask;\n\t"
            ".reg .b64 %%payload64;\n\t"
            "activemask.b32 %%mask;\n\t"
            "elect.sync _|%%p, %%mask;\n\t"
            "mov.b32 %%r, %%cluster_ctarank;\n\t"
            "mov.u32 %%t, %%globaltimer_lo;\n\t"
            "or.b32 %%t, %%t, %0;\n\t"
            "mad.lo.u32 %%r, %%r, 0x1000000, 0x20;\n\t"
            "mov.b64 %%payload64, %1;\n\t"
            "@%%p st.weak.shared.u32 [%%r], %%t;\n\t"
            "@%%p st.weak.shared.b64 [%%r+8], %%payload64;\n\t"
            "pmevent.mask %0;\n\t"
            "}\n" :: "n"(kEventId), "l"(payload));
    }
}

// Event types come from the IKET_DECLARE_* macros: `kId`, `kPayload`.
template <typename event_t>
__device__ __forceinline__ void emit() {
    static_assert(event_t::kPayload == Payload::None, "this event is declared with a payload");
    record<event_t::kId>();
}

template <typename event_t, typename value_t>
__device__ __forceinline__ void emit(const value_t& value) {
    constexpr Payload kPayload = event_t::kPayload;
    if constexpr (is_32bit(kPayload)) {
        static_assert(sizeof(value_t) <= 4, "payload wider than the declared 32-bit type");
        if constexpr (kPayload == Payload::F32)
            record_payload32<event_t::kId>(__float_as_uint(static_cast<float>(value)));
        else
            record_payload32<event_t::kId>(static_cast<uint32_t>(value));
    } else {
        static_assert(is_64bit(kPayload), "this event is declared without a payload");
        record_payload64<event_t::kId>(static_cast<uint64_t>(value));
    }
}

__device__ __forceinline__ void emit_pop() {
    record<kPopEventId>();
}

}  // namespace iket

// Metadata records are `static __device__` in the global namespace: the
// symbol keeps its plain name (global-namespace variables are not mangled),
// IKET reads it from the cubin whatever its binding, and in a host + device
// program nvcc's runtime registration finds a definition in this translation
// unit (an extern "C" __device__ record gets no host-side definition and fails
// to link there).

// The module-wide header record, once per module.
static __device__ __attribute__((used)) ::iket::MetaInfo __iket_meta_info = {
    {48, 0, 5, ::iket::kExtended ? 4095u : 31u, ::iket::kMaxNameLength, 60, 0xBABEF19Du, 0, 3, 0, 0, 0}};

#define IKET_DETAIL_EVENT(ident, id, name, payload, event_pos, range_id)                                   \
    namespace iket_events {                                                                               \
    struct ident {                                                                                        \
        static constexpr uint32_t kId = id;                                                               \
        static constexpr ::iket::Payload kPayload = payload;                                              \
        static_assert(::iket::kMinEventId <= kId and kId <= ::iket::kMaxEventId,                          \
                      "IKET event id out of range (1..30, or 64..4095 with IKET_EXTENDED_EVENT_IDS)");    \
        static_assert(::iket::name_length(name) < ::iket::kMaxNameLength, "IKET event name too long");    \
    };                                                                                                    \
    }                                                                                                     \
    static __device__ __attribute__((used)) ::iket::EventDecl __iket_evt_decl_##ident##_attrs = {        \
        60, id, ::iket::kInstrumentMethod, static_cast<uint32_t>(payload), event_pos, range_id,           \
        ::iket::name_length(name), name}

#define IKET_DETAIL_RANGE(ident, name, range_type, pair_mode)                                             \
    static __device__ __attribute__((used)) ::iket::RangeDecl __iket_range_decl_##ident##_attrs = {      \
        72, 0, ::iket::fnv1a(name), ::iket::kColorAuto, range_type, pair_mode, 0, 0,                      \
        ::iket::name_length(name), name, 0}

#define IKET_DECLARE_RANGE(ident, id, name, payload)                                                      \
    IKET_DETAIL_EVENT(ident, id, name, payload, ::iket::kEventPosRangeStart, ::iket::fnv1a(name));        \
    IKET_DETAIL_RANGE(ident, name, ::iket::kRangeTypePushPop, ::iket::kPairModeInvalid)

#define IKET_DECLARE_START_END(ident, id, name, payload)                                                  \
    IKET_DETAIL_EVENT(ident, id, name, payload, ::iket::kEventPosRangeStartEnd, ::iket::fnv1a(name));     \
    IKET_DETAIL_RANGE(ident, name, ::iket::kRangeTypeStartEnd, ::iket::kPairModeNameOnly)

#define IKET_DECLARE_MARK(ident, id, name, payload)                                                       \
    IKET_DETAIL_EVENT(ident, id, name, payload, ::iket::kEventPosNotInRange, 0)

#define IKET_RANGE_PUSH(ident, ...) ::iket::emit<::iket_events::ident>(__VA_ARGS__)
#define IKET_RANGE_POP() ::iket::emit_pop()
#define IKET_RANGE_START(ident, ...) ::iket::emit<::iket_events::ident>(__VA_ARGS__)
#define IKET_RANGE_END(ident, ...) ::iket::emit<::iket_events::ident>(__VA_ARGS__)
#define IKET_MARK(ident, ...) ::iket::emit<::iket_events::ident>(__VA_ARGS__)

#else  // IKET_ENABLE

#define IKET_DECLARE_RANGE(ident, id, name, payload) static_assert(true, "")
#define IKET_DECLARE_START_END(ident, id, name, payload) static_assert(true, "")
#define IKET_DECLARE_MARK(ident, id, name, payload) static_assert(true, "")
#define IKET_RANGE_PUSH(ident, ...) do {} while (false)
#define IKET_RANGE_POP() do {} while (false)
#define IKET_RANGE_START(ident, ...) do {} while (false)
#define IKET_RANGE_END(ident, ...) do {} while (false)
#define IKET_MARK(ident, ...) do {} while (false)

#endif  // IKET_ENABLE
