"""Read-only A/B Nsight SQLite analysis; run after nsys export -t sqlite."""

import collections
import json
from pathlib import Path
import sqlite3


def union_ns(intervals):
    end = 0
    total = 0
    for start, stop in sorted(intervals):
        total += max(0, stop - max(start, end))
        end = max(end, stop)
    return total


def summarize(path):
    db = sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)
    db.row_factory = sqlite3.Row
    names = dict(db.execute("SELECT id, value FROM StringIds"))
    tables = {row[0] for row in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}

    def rows(table):
        return [dict(row) for row in db.execute(f'SELECT * FROM "{table}" ORDER BY start')] if table in tables else []

    kernels = rows("CUPTI_ACTIVITY_KIND_KERNEL")
    runtime = rows("CUPTI_ACTIVITY_KIND_RUNTIME")
    copies = rows("CUPTI_ACTIVITY_KIND_MEMCPY")
    memsets = rows("CUPTI_ACTIVITY_KIND_MEMSET")
    nvtx = rows("NVTX_EVENTS")
    for event in nvtx:
        event["label"] = event.get("text") or names.get(event.get("textId"), "")
    ranges = [event for event in nvtx if event["label"].startswith("infer_") and event["end"] is not None]

    def group(events, key):
        result = collections.defaultdict(lambda: {"count": 0, "total_ms": 0})
        for event in events:
            entry = result[key(event)]
            entry["count"] += 1
            entry["total_ms"] += (event["end"] - event["start"]) / 1e6
        return dict(sorted(result.items(), key=lambda item: -item[1]["total_ms"]))

    result = {"file": str(path), "inferences": []}
    measured_kernels = []
    measured_runtime = []
    for region in ranges:
        start, end = region["start"], region["end"]

        def inside(events, start=start, end=end):
            return [
                event
                for event in events
                if event["start"] >= start and event["end"] is not None and event["end"] <= end
            ]

        ks, rs, cs, ms = inside(kernels), inside(runtime), inside(copies), inside(memsets)
        measured_kernels.extend(ks)
        measured_runtime.extend(rs)
        kernel_intervals = [(k["start"], k["end"]) for k in ks]
        gpu_intervals = kernel_intervals + [(c["start"], c["end"]) for c in cs + ms]
        gaps = []
        cursor = start
        for lo, hi in sorted(gpu_intervals):
            if lo > cursor:
                gaps.append({"offset_ms": (cursor - start) / 1e6, "duration_ms": (lo - cursor) / 1e6})
            cursor = max(cursor, hi)
        if cursor < end:
            gaps.append({"offset_ms": (cursor - start) / 1e6, "duration_ms": (end - cursor) / 1e6})
        stages = []
        for stage in inside(nvtx):
            if stage["label"] not in {"embed_prefix", "vlm_prefix", "denoise_step", "sync_wait", "model_preprocess"}:
                continue
            # Project CPU launch ranges to kernels using CUPTI correlation IDs.
            # Used for eager stages only; graph replay nodes need other attribution.
            ids = {
                r["correlationId"]
                for r in rs
                if stage["start"] <= r["start"] and r["end"] <= stage["end"] and r["globalTid"] == stage["globalTid"]
            }
            stage_ks = [k for k in ks if k["correlationId"] in ids]
            stages.append(
                {
                    "name": stage["label"],
                    "cpu_ms": (stage["end"] - stage["start"]) / 1e6,
                    "kernel_count": len(stage_ks),
                    "kernel_ms": union_ns((k["start"], k["end"]) for k in stage_ks) / 1e6,
                }
            )
        graphs = []
        for launch in rs:
            if "GraphLaunch" not in names[launch["nameId"]]:
                continue
            graph_ks = [k for k in ks if k["correlationId"] == launch["correlationId"]]
            graphs.append(
                {
                    "kernel_count": len(graph_ks),
                    "kernel_ms": union_ns((k["start"], k["end"]) for k in graph_ks) / 1e6,
                    "kernel_span_ms": (max(k["end"] for k in graph_ks) - min(k["start"] for k in graph_ks)) / 1e6,
                }
            )
        result["inferences"].append(
            {
                "name": region["label"],
                "start_ns": start,
                "end_ns": end,
                "wall_ms": (end - start) / 1e6,
                "kernel_count": len(ks),
                "kernel_union_ms": union_ns(kernel_intervals) / 1e6,
                "gpu_activity_union_ms": union_ns(gpu_intervals) / 1e6,
                "kernel_span_ms": (max(k["end"] for k in ks) - min(k["start"] for k in ks)) / 1e6,
                "streams": dict(collections.Counter(str(k["streamId"]) for k in ks)),
                "runtime": group(rs, lambda r: names[r["nameId"]]),
                "copies": group(cs, lambda c: f'{c["copyKind"]}:{c["bytes"]}B'),
                "largest_gaps": sorted(gaps, key=lambda g: -g["duration_ms"])[:12],
                "stages": stages,
                "graphs": graphs,
            }
        )
    result["kernels"] = group(measured_kernels, lambda k: names[k["shortName"]])
    result["runtime"] = group(measured_runtime, lambda r: names[r["nameId"]])
    result["kernel_name_count"] = len(result["kernels"])
    result["diagnostics"] = (
        [dict(row) for row in db.execute("SELECT * FROM DIAGNOSTIC_EVENT")] if "DIAGNOSTIC_EVENT" in tables else []
    )
    db.close()
    return result


if __name__ == "__main__":
    root = Path(__file__).resolve().parent
    results = {label: summarize(root / f"{label}_nodes.sqlite") for label in "AB"}
    output = root / "AB_nsys_analysis.json"
    output.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    for label, result in results.items():
        print(label)
        for row in result["inferences"]:
            print({key: row[key] for key in ("name", "wall_ms", "kernel_count", "kernel_union_ms", "streams")})
        print("Top kernels:", list(result["kernels"].items())[:12])
        print("Top runtime:", list(result["runtime"].items())[:12])
    print(output)
