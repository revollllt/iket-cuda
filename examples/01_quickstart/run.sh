#!/usr/bin/env bash
# Build the quickstart with and without IKET, trace it with run-iket, summarize.
#
#   examples/01_quickstart/run.sh [--build-only] [--out DIR]
#
# Needs nvcc (CUDA 13), a Blackwell GPU for the run, and the iket package
# (nvidia-cutlass-dsl) in python3; the SGLang images have all three:
#   apptainer exec --nv $SIF examples/01_quickstart/run.sh     # SIF: lmsysorg/sglang:v0.5.19-cu130-runtime
# ARCH picks the target (default 100f: sm_100 / sm_103).
set -Eeuo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
REPO=$(cd "$HERE/../.." && pwd)
ARCH=${ARCH:-100f}

build_only=0 out=$HERE/out
while [[ $# -gt 0 ]]; do
  case $1 in
    --build-only) build_only=1; shift ;;
    --out) out=$2; shift 2 ;;
    *) echo "unknown flag: $1" >&2; exit 2 ;;
  esac
done
mkdir -p "$out"

nvcc_flags=(-std=c++17 -O3 "-gencode=arch=compute_$ARCH,code=sm_$ARCH" "-I$REPO/include")
nvcc "${nvcc_flags[@]}" -DIKET_ENABLE "$HERE/quickstart.cu" -o "$out/quickstart_iket"
nvcc "${nvcc_flags[@]}" "$HERE/quickstart.cu" -o "$out/quickstart"
echo "built $out/quickstart_iket (IKET on) and $out/quickstart (IKET off)"
if [[ $build_only -eq 1 ]]; then
  exit 0
fi

"$out/quickstart"
# --postprocess and --keep default to none / false in the external release.
"$REPO/bin/run-iket" -o "$out/iket" --clobber profile --postprocess perfetto --keep -- "$out/quickstart_iket"
ls "$out"/iket/*.pftrace
for decoded in "$out"/iket/iket/pid_*/iket.decoded_results.json; do
  PYTHONPATH=$REPO python3 -m iket_cuda.summary "$decoded" --kernel tile_sum
done
