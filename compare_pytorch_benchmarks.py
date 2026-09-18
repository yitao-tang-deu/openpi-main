"""Compare two formal benchmark artifacts and their saved action tensors."""

import argparse
import json
from pathlib import Path
import re


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path, help="Baseline artifact prefix, without .json")
    parser.add_argument("candidate", type=Path, help="Candidate artifact prefix, without .json")
    parser.add_argument("--atol", type=float, default=1e-3)
    parser.add_argument("--rtol", type=float, default=1e-2)
    parser.add_argument("--allow-workload-change", action="store_true", help="For explicit algorithm/step ablations")
    args = parser.parse_args()
    import numpy as np

    def read(prefix):
        return json.loads(Path(str(prefix) + ".json").read_text(encoding="utf-8"))

    baseline, candidate = read(args.baseline), read(args.candidate)
    for result in (baseline, candidate):
        if not result["valid_for_formal_timing"] or not result["output_finite"]:
            parser.error("Both artifacts must be unprofiled measurements with finite outputs")
    workload_keys = ("workload", "weights", "checkpoint_sha256", "batch_size", "num_steps", "seed", "input_sha256")
    differences = [key for key in workload_keys if baseline[key] != candidate[key]]

    # Execution options do not change the workload. Also accept older artifacts
    # created before the camera-batching field was introduced.
    def normalized_config(result):
        config = result["model_config"].replace("pytorch_compile_mode='max-autotune'", "pytorch_compile_mode=None")
        return re.sub(r", pytorch_camera_batching=(?:True|False)", "", config)

    if normalized_config(baseline) != normalized_config(candidate):
        differences.append("model_config")
    if differences and not args.allow_workload_change:
        parser.error(f"Workload mismatch: {differences}; use --allow-workload-change only for deliberate ablations")
    environment_differences = [
        key
        for key in ("torch", "cuda", "transformers", "gpu", "gpu_memory_bytes", "float32_matmul_precision", "env")
        if baseline["environment"][key] != candidate["environment"][key]
    ]
    with np.load(str(args.baseline) + ".output.npz") as archive:
        original = archive["actions"].copy()
    with np.load(str(args.candidate) + ".output.npz") as archive:
        changed = archive["actions"].copy()
    if original.shape != changed.shape:
        parser.error(f"Output shape mismatch: {original.shape} vs {changed.shape}")
    error = np.abs(original - changed)
    print(
        json.dumps(
            {
                "baseline": str(args.baseline),
                "candidate": str(args.candidate),
                "image_batching": {
                    "baseline": baseline.get("image_batching", "serial"),
                    "candidate": candidate.get("image_batching", "serial"),
                },
                "p50_speedup": baseline["latency_ms"]["p50"] / candidate["latency_ms"]["p50"],
                "mean_speedup": baseline["latency_ms"]["mean"] / candidate["latency_ms"]["mean"],
                "baseline_latency_ms": baseline["latency_ms"],
                "candidate_latency_ms": candidate["latency_ms"],
                "max_abs_error": float(error.max()),
                "mean_abs_error": float(error.mean()),
                "rmse": float(np.sqrt(np.mean((original - changed) ** 2))),
                "allclose": bool(np.allclose(original, changed, atol=args.atol, rtol=args.rtol)),
                "atol": args.atol,
                "rtol": args.rtol,
                "workload_differences": differences,
                "environment_differences": environment_differences,
                "note": "Tolerance is diagnostic, not a task-quality acceptance criterion. Check environment differences.",
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
