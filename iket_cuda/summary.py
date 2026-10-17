"""Per-range summary of one kernel launch from run-iket's decoded results, the
numeric companion of the Perfetto timeline, identical for CuTe DSL and CUDA C++
kernels (both go through run-iket).

    python -m iket_cuda.summary <out>/iket/pid_<pid>/iket.decoded_results.json [--kernel NAME]

One row per range name: the warps it runs on, count, p50 / p99 duration, total,
and share = total / summed lifetime of the warps it runs on. Nested ranges
overlap their parent (mma_wait_ab sits inside mma_fc1), so compare a range with
its parent or its siblings on the same warps. Rows are ordered by the lowest
warp index they run on, then first appearance.

Instrumentation stretches a kernel (compare with an uninstrumented run of the
same workload), so use the ratios within one trace, not absolute times.
"""

from __future__ import annotations

import argparse
import statistics
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import TypedDict

import orjson

INTRA_WARP_SCOPE = 0


class IketLocation(TypedDict):
    warpId: int


class IketRange(TypedDict):
    startTs: int
    endTs: int
    rangeNameIdx: int
    rangeScope: int
    warpLocIdxs: list[int]


class IketLifetime(TypedDict):
    locIdx: int
    startTs: int
    endTs: int


class IketLaunch(TypedDict):
    kernelName: str
    ranges: list[IketRange]
    warpLifetimes: list[IketLifetime]


class IketResults(TypedDict):
    """The fields of iket.decoded_results.json this summary reads."""

    launches: list[IketLaunch]
    locationTable: list[IketLocation]
    stringTable: list[str]


@dataclass(frozen=True)
class RangeStats:
    name: str
    warp_ids: tuple[int, ...]
    count: int
    p50_us: float
    p99_us: float
    total_ms: float
    share: float
    first_start: int


def select_launch(results: IketResults, kernel_substring: str) -> IketLaunch:
    """The last launch (by first warp entry) of a kernel whose name contains
    kernel_substring: run-iket records warmup and timed calls alike."""
    matching = [
        launch for launch in results["launches"]
        if kernel_substring in launch["kernelName"] and launch["warpLifetimes"]
    ]
    if not matching:
        names = sorted({launch["kernelName"][:80] for launch in results["launches"]})
        raise ValueError(f"no traced launch of a kernel matching {kernel_substring!r}; kernels: {names}")
    return max(matching, key=lambda launch: min(life["startTs"] for life in launch["warpLifetimes"]))


def summarize(results: IketResults, launch: IketLaunch) -> tuple[list[RangeStats], float]:
    """(one RangeStats per range name, kernel span in µs)."""
    locations = results["locationTable"]
    lifetime_ns = {life["locIdx"]: life["endTs"] - life["startTs"] for life in launch["warpLifetimes"]}
    durations_ns: dict[str, list[int]] = defaultdict(list)
    locations_by_name: dict[str, set[int]] = defaultdict(set)
    first_start: dict[str, int] = {}
    for warp_range in launch["ranges"]:
        if warp_range["rangeScope"] != INTRA_WARP_SCOPE:
            continue
        name = results["stringTable"][warp_range["rangeNameIdx"]]
        durations_ns[name].append(warp_range["endTs"] - warp_range["startTs"])
        locations_by_name[name].add(warp_range["warpLocIdxs"][0])
        first_start[name] = min(first_start.get(name, warp_range["startTs"]), warp_range["startTs"])

    stats = []
    for name, durations in durations_ns.items():
        ordered = sorted(durations)
        warp_lifetime_ns = sum(lifetime_ns[location_idx] for location_idx in locations_by_name[name])
        stats.append(
            RangeStats(
                name=name,
                warp_ids=tuple(sorted({locations[location_idx]["warpId"] for location_idx in locations_by_name[name]})),
                count=len(ordered),
                p50_us=statistics.median(ordered) / 1e3,
                p99_us=ordered[min(len(ordered) - 1, int(0.99 * len(ordered)))] / 1e3,
                total_ms=sum(ordered) / 1e6,
                share=sum(ordered) / warp_lifetime_ns,
                first_start=first_start[name],
            )
        )
    stats.sort(key=lambda range_stats: (range_stats.warp_ids[0], range_stats.first_start))
    starts = [life["startTs"] for life in launch["warpLifetimes"]]
    ends = [life["endTs"] for life in launch["warpLifetimes"]]
    return stats, (max(ends) - min(starts)) / 1e3


def format_table(stats: list[RangeStats], span_us: float, title: str) -> str:
    def warp_label(warp_ids: tuple[int, ...]) -> str:
        if len(warp_ids) == 1:
            return f"{warp_ids[0]}"
        contiguous = warp_ids == tuple(range(warp_ids[0], warp_ids[-1] + 1))
        return f"{warp_ids[0]}-{warp_ids[-1]}" if contiguous else ",".join(map(str, warp_ids))

    lines = [
        f"{title}: kernel span {span_us:.1f} us",
        f"{'range':<26}{'warps':>8}{'count':>9}{'p50 us':>10}{'p99 us':>10}{'total ms':>10}{'share':>8}",
    ]
    for range_stats in stats:
        lines.append(
            f"{range_stats.name:<26}{warp_label(range_stats.warp_ids):>8}{range_stats.count:>9}"
            f"{range_stats.p50_us:>10.2f}{range_stats.p99_us:>10.2f}{range_stats.total_ms:>10.2f}{range_stats.share:>8.1%}"
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("decoded_results", type=Path)
    parser.add_argument("--kernel", default="", help="substring of the kernel name (default: any traced kernel)")
    options = parser.parse_args()
    results: IketResults = orjson.loads(options.decoded_results.read_bytes())
    launch = select_launch(results, options.kernel)
    stats, span_us = summarize(results, launch)
    print(format_table(stats, span_us, f"{options.decoded_results} [{launch['kernelName'][:60]}]"))


if __name__ == "__main__":
    main()
