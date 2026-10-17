"""iket.cuh emits the IKET ABI CuTe DSL emits: for one list of events, a CuTe
DSL kernel and a CUDA C++ kernel are compiled (CPU only, sm_100a) and IKET's own
cubin reader must see the same module metadata, events, ranges and per-kernel
instrument points in both. With IKET_ENABLE off the C++ kernel must compile to
exactly the SASS of the same kernel with the markers deleted.

Runs inside the runtime image (CuTe DSL, IKET, nvcc):

    apptainer exec $SIF python3 -m pytest -q tests
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import pytest
from iket import Context

INCLUDE_DIR = Path(__file__).resolve().parents[1] / "include"
NVCC = "/usr/local/cuda/bin/nvcc"
CUOBJDUMP = "/usr/local/cuda/bin/cuobjdump"
ARCH = "sm_100a"


@dataclass(frozen=True)
class Event:
    """One IKET event as both front ends spell it. kind: push (push/pop range),
    start_end (token range), mark; payload: none, i32, i64, f32."""

    name: str
    kind: str
    payload: str


DSL_PAYLOAD_ARG = {"none": None, "i32": "value32", "i64": "value64", "f32": "real"}
CPP_PAYLOAD_TYPE = {"none": "None", "i32": "I32", "i64": "I64", "f32": "F32"}
CPP_PAYLOAD_ARG = {"none": "", "i32": ", value32", "i64": ", value64", "f32": ", real"}
CPP_DECLARE = {"push": "IKET_DECLARE_RANGE", "start_end": "IKET_DECLARE_START_END", "mark": "IKET_DECLARE_MARK"}

BASE_EVENTS = [
    Event("plain_range", "push", "none"),
    Event("payload_range", "push", "i32"),
    Event("token_range", "start_end", "i32"),
    Event("wide_mark", "mark", "i64"),
    Event("real_mark", "mark", "f32"),
]
# More names than the 30 native ids: CuTe DSL switches the module to extended records.
MANY_EVENTS = BASE_EVENTS + [Event(f"many_{index:02d}", "push", "none") for index in range(30)]


def dsl_source(events: list[Event]) -> str:
    body = []
    for event in events:
        arg = DSL_PAYLOAD_ARG[event.payload]
        payload = f", {arg}" if arg else ""
        match event.kind:
            case "push":
                body += [f'iket.range_push("{event.name}"{payload})', "iket.range_pop()"]
            case "start_end":
                body += [
                    f'token = iket.range_start("{event.name}"{payload})',
                    f"iket.range_end(token{payload})",
                ]
            case "mark":
                body += [f'iket.mark("{event.name}"{payload})']
            case _:
                raise ValueError(f"unknown event kind {event.kind}")
    lines = "\n".join(f"    {line}" for line in body)
    return f"""
import cutlass
import cutlass.cute as cute
from cutlass.cute.experimental import iket

@cute.kernel
def probe_kernel(value32: cutlass.Int32, value64: cutlass.Int64, real: cutlass.Float32):
{lines}

@cute.jit
def launch(value32: cutlass.Int32, value64: cutlass.Int64, real: cutlass.Float32):
    probe_kernel(value32, value64, real).launch(grid=(1, 1, 1), block=(64, 1, 1))

cute.compile(launch, cutlass.Int32(7), cutlass.Int64(9), cutlass.Float32(1.5))
"""


def cpp_source(events: list[Event], with_markers: bool) -> str:
    extended = len(events) > 30
    first_id = 64 if extended else 1
    declarations, body = [], []
    for index, event in enumerate(events):
        declarations.append(
            f'{CPP_DECLARE[event.kind]}({event.name}, {first_id + index}, "{event.name}", '
            f"::iket::Payload::{CPP_PAYLOAD_TYPE[event.payload]});"
        )
        arg = CPP_PAYLOAD_ARG[event.payload]
        match event.kind:
            case "push":
                body += [f"IKET_RANGE_PUSH({event.name}{arg});", "IKET_RANGE_POP();"]
            case "start_end":
                body += [f"IKET_RANGE_START({event.name}{arg});", f"IKET_RANGE_END({event.name}{arg});"]
            case "mark":
                body += [f"IKET_MARK({event.name}{arg});"]
            case _:
                raise ValueError(f"unknown event kind {event.kind}")
    marker_lines = "\n".join(f"    {line}" for line in body) if with_markers else ""
    return f"""
{"#define IKET_EXTENDED_EVENT_IDS" if extended else ""}
#include <cstdint>
#include <iket/iket.cuh>

{chr(10).join(declarations) if with_markers else ""}

extern "C" __global__ void probe_kernel(int value32, long long value64, float real, float* out) {{
{marker_lines}
    out[threadIdx.x] = real * value32 + value64;
}}
"""


def compile_dsl(events: list[Event], work_dir: Path) -> bytes:
    script = work_dir / "probe.py"
    script.write_text(dsl_source(events))
    dump_dir = work_dir / "dsl"
    dump_dir.mkdir()
    env = {**os.environ, "CUTE_DSL_ARCH": ARCH, "CUTE_DSL_COMPILER_OPT": "iket",
           "CUTE_DSL_KEEP": "cubin", "CUTE_DSL_DUMP_DIR": str(dump_dir)}
    subprocess.run([sys.executable, str(script)], env=env, check=True, capture_output=True)
    (cubin,) = dump_dir.glob("*.cubin")
    return cubin.read_bytes()


def compile_cpp(source: str, work_dir: Path, name: str, defines: list[str]) -> bytes:
    source_path = work_dir / f"{name}.cu"
    source_path.write_text(source)
    cubin_path = work_dir / f"{name}.cubin"
    subprocess.run(
        [NVCC, "-std=c++17", "-O3", f"-arch={ARCH}", "--cubin", f"-I{INCLUDE_DIR}", *defines,
         str(source_path), "-o", str(cubin_path)],
        check=True, capture_output=True,
    )
    return cubin_path.read_bytes()


def iket_view(cubin: bytes) -> tuple[object, list[tuple], list[tuple], list[str]]:
    """What run-iket reads from a cubin: (meta info, events, ranges, the event
    names of the probe kernel's instrument points in code order)."""
    context = Context(cubin)
    events = sorted(
        (e.event_name, e.event_id, e.instrument_method, e.payload_type, e.event_pos, e.range_id)
        for e in context.get_all_events()
    )
    ranges = sorted(
        (r.range_name, r.range_id, r.range_scope, r.range_type, r.evt_pair_mode, r.color)
        for r in context.get_all_ranges()
    )
    (kernel,) = [k for k in context.get_kernels_instrument_info() if "probe_kernel" in k.kernel_name]
    points = [point.event_attr.event_name for point in sorted(kernel.instrument_points, key=lambda p: p.offset_start)]
    return context.get_meta_info(), events, ranges, points


@pytest.mark.parametrize("events", [BASE_EVENTS, MANY_EVENTS], ids=["native", "extended"])
def test_cpp_matches_cute_dsl(events: list[Event], tmp_path: Path) -> None:
    reference = iket_view(compile_dsl(events, tmp_path))
    ours = iket_view(compile_cpp(cpp_source(events, with_markers=True), tmp_path, "ours", ["-DIKET_ENABLE"]))
    assert ours == reference


def test_disabled_is_free(tmp_path: Path) -> None:
    def sass(cubin: bytes, name: str) -> list[str]:
        path = tmp_path / f"{name}.cubin"
        path.write_bytes(cubin)
        text = subprocess.run([CUOBJDUMP, "-sass", str(path)], check=True, capture_output=True, text=True).stdout
        return [line for line in text.splitlines() if re.match(r"\s+/\*[0-9a-f]{4}\*/", line)]

    disabled = compile_cpp(cpp_source(MANY_EVENTS, with_markers=True), tmp_path, "disabled", [])
    stripped = compile_cpp(cpp_source(MANY_EVENTS, with_markers=False), tmp_path, "stripped", [])
    assert sass(disabled, "disabled") == sass(stripped, "stripped")
    assert b"__iket" not in disabled
