"""CUDA-only PyTorch baseline. See docs/pytorch_benchmark.md for commands."""

import argparse
import contextlib
import functools
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import statistics
import subprocess
import time


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compile", choices=("off", "max-autotune"), required=True)
    parser.add_argument("--model", choices=("pi0", "pi05"), default="pi05")
    parser.add_argument("--checkpoint", type=Path, help="Local model.safetensors; omitted = random weights")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-steps", type=int, default=10)
    parser.add_argument("--token-length", type=int, help="Defaults: pi0=48, pi05=200; all tokens valid")
    parser.add_argument("--action-horizon", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=5, help="Additional calls AFTER one first call")
    parser.add_argument("--runs", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--profile", choices=("none", "torch", "nsys"), default="none")
    parser.add_argument("--stage-labels", action="store_true", help="Eager profiling only; never formal timing")
    parser.add_argument("--output", type=Path, required=True, help="Artifact prefix, e.g. results/A_r1")
    args = parser.parse_args(argv)
    if min(args.batch_size, args.num_steps, args.action_horizon, args.runs) < 1 or args.warmup < 0:
        parser.error("batch-size, num-steps, action-horizon, runs must be positive; warmup >= 0")
    if args.token_length is not None and args.token_length < 1:
        parser.error("token-length must be positive")
    if args.stage_labels and (args.compile != "off" or args.profile == "none"):
        parser.error("stage-labels requires --compile off and --profile torch/nsys")
    if args.checkpoint and not args.checkpoint.is_file():
        parser.error("checkpoint must be a local model.safetensors file")
    return args


def percentile(values, q):
    ordered = sorted(values)
    index = (len(ordered) - 1) * q
    lower = int(index)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (index - lower)


def file_hash(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_info(root):
    def run(*args):
        result = subprocess.run(["git", "-C", str(root), *args], capture_output=True, text=True, check=False)
        return result.stdout.strip() if result.returncode == 0 else None

    return {"commit": run("rev-parse", "HEAD"), "status": run("status", "--short")}


def main():
    args = parse_args()
    # These modules import JAX types; keep JAX from reserving GPU memory in this Torch benchmark.
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    import numpy as np
    import torch

    from openpi.models.model import Observation
    from openpi.models.pi0_config import Pi0Config
    from openpi.models_pytorch.pi0_pytorch import PI0Pytorch

    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("This benchmark requires a CUDA PyTorch environment")
    torch.cuda.set_device(device)
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("This baseline requires BF16 support")
    if args.compile != "off" and os.environ.get("TORCH_COMPILE_DISABLE", "0") not in ("", "0"):
        raise RuntimeError("Remove TORCH_COMPILE_DISABLE for the compiled baseline")
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def artifact(suffix):
        return Path(str(args.output) + suffix)

    for suffix in (".json", ".output.npz", ".trace.json", ".operators.txt"):
        if artifact(suffix).exists():
            raise FileExistsError(f"Use a fresh --output prefix: {artifact(suffix)}")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    config = Pi0Config(
        pi05=args.model == "pi05",
        action_horizon=args.action_horizon,
        max_token_len=args.token_length,
        pytorch_compile_mode=None if args.compile == "off" else args.compile,
    )
    model = PI0Pytorch(config)
    checkpoint_sha256 = None
    if args.checkpoint:
        from safetensors.torch import load_model

        load_model(model, str(args.checkpoint), strict=True)
        checkpoint_sha256 = file_hash(args.checkpoint)
    # Match policy_config's selected BF16 conversion; projections/norms retain their intended dtype.
    model.paligemma_with_expert.to_bfloat16_for_selected_params("bfloat16")
    model.to(device).eval()

    # Independent CPU generator: model initialization must not change the synthetic workload.
    generator = torch.Generator(device="cpu").manual_seed(args.seed + 1)
    batch = args.batch_size
    camera_keys = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")
    cpu_images = {key: torch.rand(batch, 224, 224, 3, generator=generator) * 2 - 1 for key in camera_keys}
    cpu_state = torch.rand(batch, config.action_dim, generator=generator) * 2 - 1
    cpu_tokens = torch.randint(0, 1000, (batch, config.max_token_len), generator=generator)
    cpu_noise = torch.randn(batch, config.action_horizon, config.action_dim, generator=generator)
    input_digest = hashlib.sha256()
    for tensor in (*cpu_images.values(), cpu_state, cpu_tokens, cpu_noise):
        input_digest.update(tensor.numpy().tobytes())
    observation = Observation(
        # SigLIP's Conv2d expects NCHW. Keep the seeded NHWC source above for
        # reproducibility, and convert outside the timed inference region.
        images={key: value.permute(0, 3, 1, 2).contiguous().to(device) for key, value in cpu_images.items()},
        image_masks={key: torch.ones(batch, dtype=torch.bool, device=device) for key in camera_keys},
        state=cpu_state.to(device),
        tokenized_prompt=cpu_tokens.to(device),
        tokenized_prompt_mask=torch.ones(batch, config.max_token_len, dtype=torch.bool, device=device),
    )
    noise = cpu_noise.to(device)

    @contextlib.contextmanager
    def mark(name):
        if args.profile == "torch":
            with torch.profiler.record_function(name):
                yield
        elif args.profile == "nsys":
            with torch.cuda.nvtx.range(name):
                yield
        else:
            yield

    if args.stage_labels:

        def wrap(owner, attribute, label):
            original = getattr(owner, attribute)

            @functools.wraps(original)
            def traced(*positional, **keywords):
                with mark(label):
                    return original(*positional, **keywords)

            setattr(owner, attribute, traced)

        wrap(model, "_preprocess_observation", "model_preprocess")
        wrap(model, "embed_prefix", "embed_prefix")
        wrap(model, "denoise_step", "denoise_step")
        wrap(model.paligemma_with_expert.paligemma.language_model, "forward", "vlm_prefix")

    def invoke():
        # End the previous CUDA graph iteration explicitly; keep no old GPU output alive.
        if args.compile != "off":
            torch.compiler.cudagraph_mark_step_begin()
        return model.sample_actions(device, observation, noise=noise, num_steps=args.num_steps)

    torch.cuda.synchronize(device)
    with torch.no_grad():
        start = time.perf_counter()
        output = invoke()
        torch.cuda.synchronize(device)
        first_call_ms = (time.perf_counter() - start) * 1000
        del output
        for _ in range(args.warmup):
            output = invoke()
            torch.cuda.synchronize(device)
            del output

        torch.cuda.reset_peak_memory_stats(device)
        memory_at_start = torch.cuda.memory_allocated(device)
        profiler = (
            torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
            )
            if args.profile == "torch"
            else contextlib.nullcontext()
        )
        latencies = []
        if args.profile == "nsys":
            torch.cuda.cudart().cudaProfilerStart()
        try:
            with profiler as prof, mark("measurement"):
                for index in range(args.runs):
                    with mark(f"infer_{index:04d}"):
                        start = time.perf_counter()
                        with mark("sample_actions"):
                            output = invoke()
                        with mark("sync_wait"):
                            torch.cuda.synchronize(device)
                        latencies.append((time.perf_counter() - start) * 1000)
                    if index != args.runs - 1:
                        del output
        finally:
            if args.profile == "nsys":
                torch.cuda.cudart().cudaProfilerStop()

        peak_allocated = torch.cuda.max_memory_allocated(device)
        peak_reserved = torch.cuda.max_memory_reserved(device)
        output_cpu = output.detach().float().cpu().numpy()
        del output

    if args.profile == "torch":
        prof.export_chrome_trace(str(artifact(".trace.json")))
        artifact(".operators.txt").write_text(
            prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=100), encoding="utf-8"
        )
    np.savez(artifact(".output.npz"), actions=output_cpu)
    root = Path(__file__).resolve().parent
    source_paths = sorted((root / "src/openpi/models_pytorch").rglob("*.py"))
    source_paths += [root / "src/openpi/models/pi0_config.py", Path(__file__).resolve()]
    properties = torch.cuda.get_device_properties(device)
    result = {
        "label": str(args.output),
        "workload": "synthetic_device_resident_sample_actions",
        "weights": "checkpoint" if args.checkpoint else "random",
        "checkpoint": str(args.checkpoint.resolve()) if args.checkpoint else None,
        "checkpoint_sha256": checkpoint_sha256,
        "compile_requested": args.compile,
        "compile_verified_by_trace": False,
        "profile": args.profile,
        "valid_for_formal_timing": args.profile == "none",
        "stage_labels": args.stage_labels,
        "model_config": repr(config),
        "batch_size": batch,
        "num_steps": args.num_steps,
        "seed": args.seed,
        "input_sha256": input_digest.hexdigest(),
        "warmup_runs_after_first_call": args.warmup,
        "measured_runs": args.runs,
        "first_call_ms_including_compile_and_execution": first_call_ms,
        "latency_ms": {
            "mean": statistics.mean(latencies),
            "p50": percentile(latencies, 0.5),
            "p90": percentile(latencies, 0.9),
            "p95": percentile(latencies, 0.95),
            "min": min(latencies),
            "max": max(latencies),
            "stddev": statistics.pstdev(latencies),
        },
        "latencies_ms": latencies,
        "serial_samples_per_second": batch * len(latencies) * 1000 / sum(latencies),
        "memory_bytes": {
            "allocated_after_warmup": memory_at_start,
            "peak_allocated_measurement": peak_allocated,
            "peak_reserved_measurement": peak_reserved,
        },
        "output_finite": bool(np.isfinite(output_cpu).all()),
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "transformers": importlib.metadata.version("transformers"),
            "gpu": properties.name,
            "gpu_memory_bytes": properties.total_memory,
            "device": str(device),
            "float32_matmul_precision": torch.get_float32_matmul_precision(),
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
            "env": {key: value for key, value in os.environ.items() if key.startswith(("TORCH", "CUDA", "JAX", "XLA"))},
        },
        "git": git_info(root),
        "source_sha256": {str(path.relative_to(root)): file_hash(path) for path in source_paths},
    }
    artifact(".json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    if not result["output_finite"]:
        raise RuntimeError("Nonfinite output: artifacts saved, but this is not a valid numerical baseline")


if __name__ == "__main__":
    main()
