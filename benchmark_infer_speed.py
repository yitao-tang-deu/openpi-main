import argparse
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
            return jax.random.uniform(key, spec.shape, dtype=spec.dtype)
        if jnp.issubdtype(spec.dtype, jnp.bool_):
            return jax.random.bernoulli(key, shape=spec.shape)
        return jax.random.randint(key, spec.shape, 0, 1000, dtype=spec.dtype)

    return jax.tree.map(make_value, observation_spec)


def benchmark_inference(batch_size: int, num_steps: int, warmup_runs: int, test_runs: int) -> None:
    config = pi0_config.Pi0Config()
    rng = jax.random.key(0)
    model = config.create(rng)
    observation = _random_observation(config, batch_size, rng)
    sample_fn = nnx_utils.module_jit(model.sample_actions)

    for _ in range(warmup_runs):
        rng, sample_rng = jax.random.split(rng)
        output = sample_fn(sample_rng, observation, num_steps=num_steps)
        jax.block_until_ready(output)

    latencies_ms = []
    for run_index in range(test_runs):
        rng, sample_rng = jax.random.split(rng)
        start = time.perf_counter()
        output = sample_fn(sample_rng, observation, num_steps=num_steps)
        jax.block_until_ready(output)
        elapsed_ms = (time.perf_counter() - start) * 1000
        latencies_ms.append(elapsed_ms)
        print(f"run {run_index + 1}/{test_runs}: {elapsed_ms:.2f} ms")

    print(f"avg: {statistics.mean(latencies_ms):.2f} ms")
    print(f"min: {min(latencies_ms):.2f} ms")
    print(f"max: {max(latencies_ms):.2f} ms")
    print(f"p50: {statistics.median(latencies_ms):.2f} ms")


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark Pi0 inference latency.")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-steps", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--runs", type=int, default=5)
    args = parser.parse_args()
    benchmark_inference(args.batch_size, args.num_steps, args.warmup, args.runs)


if __name__ == "__main__":
    main()