"""The examples build exactly as the docs say, and IKET sees their markers.

    apptainer exec $SIF python3 -m pytest -q tests
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

from iket import Context

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"
CUOBJDUMP = "/usr/local/cuda/bin/cuobjdump"


def extract_cubin(executable: Path, work_dir: Path, kernel_name: bytes) -> bytes:
    """The embedded cubin holding kernel_name (the fatbin also carries a tiny
    runtime module without kernels)."""
    subprocess.run([CUOBJDUMP, "-xelf", "all", str(executable)], cwd=work_dir, check=True, capture_output=True)
    (cubin,) = [path.read_bytes() for path in work_dir.glob("*.cubin") if kernel_name in path.read_bytes()]
    return cubin


def test_quickstart_builds_and_carries_its_markers(tmp_path: Path) -> None:
    out = tmp_path / "out"
    env = {**os.environ, "PATH": f"/usr/local/cuda/bin:{os.environ['PATH']}"}
    subprocess.run([str(EXAMPLES / "01_quickstart" / "run.sh"), "--build-only", "--out", str(out)],
                   env=env, check=True, capture_output=True)

    (tmp_path / "on").mkdir()
    context = Context(extract_cubin(out / "quickstart_iket", tmp_path / "on", b"tile_sum"))
    (kernel,) = [k for k in context.get_kernels_instrument_info() if "tile_sum" in k.kernel_name]
    recorded = [point.event_attr.event_name for point in sorted(kernel.instrument_points, key=lambda p: p.offset_start)]
    assert set(recorded) == {"block_life", "load_tile", "reduce_tile", "tile_done", "pop_range"}
    assert recorded.count("block_life") == 2  # start and end

    (tmp_path / "off").mkdir()
    assert b"__iket" not in extract_cubin(out / "quickstart", tmp_path / "off", b"tile_sum")
