import argparse
import contextlib
import importlib.metadata
import json
import os
from pathlib import Path
import statistics
import time

import jax
import jax.numpy as jnp

from openpi.models import pi0_config
import openpi.shared.nnx_utils as nnx_utils


def _random_observation(config: pi0_config.Pi0Config, batch_size: int, rng):
    observation_spec, _ = config.inputs_spec(batch_size=batch_size)
    keys = iter(jax.random.split(rng, len(jax.tree.leaves(observation_spec))))

    def make_value(spec):
        key = next(keys)
        if jnp.issubdtype(spec.dtype, jnp.floating):
            return jax.random.uniform(key, spec.shape, dtype=spec.dtype, minval=-1.0, maxval=1.0)
        if jnp.issubdtype(spec.dtype, jnp.bool_):
            return jnp.ones(spec.shape, dtype=spec.dtype)
        return jax.random.randint(key, spec.shape, 0, 1000, dtype=spec.dtype)

    return jax.tree.map(make_value, observation_spec)


def benchmark_inference(
    batch_size: int,
    num_steps: int,
    warmup_runs: int,
    test_runs: int,
    *,
    mode: str = "latency",
    queue_depth: int = 4,
    seed: int = 0,
    nvtx_enabled: bool = False,
    json_path: str | None = None,
) -> dict:
    """Measure device-resident synthetic inference, excluding policy transforms and I/O.

    Throughput mode submits bounded groups and waits for ALL outputs. Asynchronous
    submission does not guarantee concurrent GPU execution or multiple streams.
    """
    if min(batch_size, num_steps, test_runs, queue_depth) < 1 or warmup_runs < 0:
        raise ValueError("batch_size, num_steps, test_runs and queue_depth must be positive; warmup >= 0")
    if mode not in ("latency", "throughput"):
        raise ValueError(f"Unknown mode: {mode}")
    nvtx_module = None
    if nvtx_enabled:
        import nvtx  # Optional: install in the remote profiling environment.

        nvtx_module = nvtx

    def annotate(name):
        return nvtx_module.annotate(name, domain="pi0_benchmark") if nvtx_module else contextlib.nullcontext()

    config = pi0_config.Pi0Config()
    model_rng, obs_rng, sample_rng = jax.random.split(jax.random.key(seed), 3)
    with annotate("setup"):
        model = config.create(model_rng)
        model.eval()
        observation = _random_observation(config, batch_size, obs_rng)
        # Materialize individual keys before timing: split/unstack also dispatch JAX work.
        keys = list(jax.random.split(sample_rng, 1 + warmup_runs + test_runs))
        jax.block_until_ready((observation, keys))
    sample_fn = nnx_utils.module_jit(model.sample_actions)

    with annotate("first_call"):
        start = time.perf_counter()
        jax.block_until_ready(sample_fn(keys[0], observation, num_steps=num_steps))
        first_call_ms = (time.perf_counter() - start) * 1000
    with annotate("warmup"):
        for key in keys[1 : 1 + warmup_runs]:
            jax.block_until_ready(sample_fn(key, observation, num_steps=num_steps))

    latencies_ms = []
    measured_keys = keys[1 + warmup_runs :]
    with annotate("measurement"):
        measurement_start = time.perf_counter()
        if mode == "latency":
            for run_index, key in enumerate(measured_keys):
                with annotate(f"infer_{run_index:04d}"):
                    start = time.perf_counter()
                    output = sample_fn(key, observation, num_steps=num_steps)
                    jax.block_until_ready(output)
                    latencies_ms.append((time.perf_counter() - start) * 1000)
        else:
            for offset in range(0, test_runs, queue_depth):
                with annotate(f"group_{offset:04d}"):
                    outputs = [
                        sample_fn(key, observation, num_steps=num_steps)
                        for key in measured_keys[offset : offset + queue_depth]
                    ]
                    jax.block_until_ready(outputs)
                    del outputs
        measurement_seconds = time.perf_counter() - measurement_start

    result = {
        "workload": "random_weights_synthetic_device_resident_pi0",
        "mode": mode,
        "batch_size": batch_size,
        "num_steps": num_steps,
        "warmup_runs": warmup_runs,
        "test_runs": test_runs,
        "seed": seed,
        "queue_depth": queue_depth if mode == "throughput" else 1,
        "first_call_ms": first_call_ms,  # Includes compilation/autotuning and execution.
        "measurement_seconds": measurement_seconds,
        "samples_per_second": batch_size * test_runs / measurement_seconds,
        "batches_per_second": test_runs / measurement_seconds,
        "jax_version": jax.__version__,
        "jaxlib_version": importlib.metadata.version("jaxlib"),
        "backend": jax.default_backend(),
        "devices": [str(d) for d in jax.devices()],
        "xla_flags": os.environ.get("XLA_FLAGS", ""),
        "nvtx_enabled": nvtx_enabled,
        "model_config": repr(config),
    }
    if latencies_ms:
        ordered = sorted(latencies_ms)

        # Linear interpolation, also defined when only one sample is requested.
        def percentile(q):
            index = (len(ordered) - 1) * q
            lower = int(index)
            upper = min(lower + 1, len(ordered) - 1)
            return ordered[lower] + (ordered[upper] - ordered[lower]) * (index - lower)

        result["latencies_ms"] = latencies_ms
        result["latency_ms"] = {
            "mean": statistics.mean(latencies_ms),
            "min": min(latencies_ms),
            "max": max(latencies_ms),
            "p50": percentile(0.5),
            "p95": percentile(0.95),
            "stddev": statistics.pstdev(latencies_ms),
        }
    print(json.dumps(result, indent=2))
    if json_path:
        Path(json_path).write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark Pi0 inference latency.")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-steps", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--runs", type=int, default=50)
    parser.add_argument("--mode", choices=("latency", "throughput"), default="latency")
    parser.add_argument("--queue-depth", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--nvtx", action="store_true", help="Requires the optional nvtx package")
    parser.add_argument("--json", dest="json_path", help="Write results to a JSON file")
    args = parser.parse_args()
    if min(args.batch_size, args.num_steps, args.runs, args.queue_depth) < 1 or args.warmup < 0:
        parser.error("batch-size, num-steps, runs and queue-depth must be positive; warmup >= 0")
    benchmark_inference(
        args.batch_size,
        args.num_steps,
        args.warmup,
        args.runs,
        mode=args.mode,
        queue_depth=args.queue_depth,
        seed=args.seed,
        nvtx_enabled=args.nvtx,
        json_path=args.json_path,
    )


if __name__ == "__main__":
    main()
