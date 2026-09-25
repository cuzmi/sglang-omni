"""Compare a MiniCPM-o decode Gate/Up GEMV with SGLang's projection."""

from __future__ import annotations

import argparse
import json
from functools import partial
from importlib.metadata import version
from pathlib import Path
from statistics import mean

import torch
import triton
import triton.testing
from sglang.srt.layers.linear import MergedColumnParallelLinear

from sglang_omni.models.minicpm_o.components.gate_up_gemv import (
    INPUT_FEATURES,
    OUTPUT_FEATURES,
    gate_up_gemv,
)

BLOCK_N = 16
TRIAL_BLOCK_N = 32
BLOCK_K = 256
NUM_WARPS = 4
PROFILE_WARMUP_ITERATIONS = 20
BASELINE_CONFIG = (BLOCK_N, BLOCK_K, NUM_WARPS)
WINNER_CONFIG = (BLOCK_N, 512, 8)
PROFILE_CONFIGS = {"original": BASELINE_CONFIG, "winner": WINNER_CONFIG}
BLOCK_N_CONFIGS = (BASELINE_CONFIG, (TRIAL_BLOCK_N, BLOCK_K, NUM_WARPS))
TILE_WARP_CONFIGS = (
    BASELINE_CONFIG,
    (BLOCK_N, 128, NUM_WARPS),
    (BLOCK_N, 512, NUM_WARPS),
    (BLOCK_N, BLOCK_K, 2),
    (BLOCK_N, BLOCK_K, 8),
)
COMBINED_CONFIGS = (
    BASELINE_CONFIG,
    (BLOCK_N, 512, NUM_WARPS),
    (BLOCK_N, BLOCK_K, 8),
    WINNER_CONFIG,
)
LOCAL_GRID_CONFIGS = (
    WINNER_CONFIG,
    (BLOCK_N, 1024, 8),
    (BLOCK_N, 512, 16),
    (BLOCK_N, 1024, 16),
)
PREFETCH_GRID_CONFIGS = (
    WINNER_CONFIG,
    (BLOCK_N, BLOCK_K, 8),
    (BLOCK_N, 1024, 8),
    (BLOCK_N, 512, NUM_WARPS),
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument(
        "--experiment",
        choices=(
            "block-n",
            "tile-warp",
            "combined",
            "local-grid",
            "prefetch",
            "prefetch-grid",
        ),
        default="block-n",
    )
    parser.add_argument(
        "--profile-kernel",
        choices=("reference", "candidate"),
        help="Launch exactly one selected kernel between CUDA profiler markers",
    )
    parser.add_argument("--profile-config", choices=PROFILE_CONFIGS, default="original")
    args = parser.parse_args()
    profile_config = PROFILE_CONFIGS[args.profile_config]
    if args.profile_kernel is not None:
        configs = ((*profile_config, False),)
    elif args.experiment == "block-n":
        configs = tuple((*config, False) for config in BLOCK_N_CONFIGS)
    elif args.experiment == "tile-warp":
        configs = tuple((*config, False) for config in TILE_WARP_CONFIGS)
    elif args.experiment == "combined":
        configs = tuple((*config, False) for config in COMBINED_CONFIGS)
    elif args.experiment == "local-grid":
        configs = tuple((*config, False) for config in LOCAL_GRID_CONFIGS)
    elif args.experiment == "prefetch-grid":
        configs = tuple((*config, True) for config in PREFETCH_GRID_CONFIGS)
    else:
        configs = ((*WINNER_CONFIG, False), (*WINNER_CONFIG, True))

    if not torch.cuda.is_available():
        raise RuntimeError("This benchmark requires a CUDA GPU")

    torch.manual_seed(0)
    with torch.device("cuda"):
        gate_up_projection = MergedColumnParallelLinear(
            INPUT_FEATURES,
            [OUTPUT_FEATURES // 2, OUTPUT_FEATURES // 2],
            bias=False,
            params_dtype=torch.bfloat16,
            quant_config=None,
            tp_rank=0,
            tp_size=1,
        )
    with torch.no_grad():
        gate_up_projection.weight.copy_(
            torch.randn_like(gate_up_projection.weight)
        )
    input_tensor = torch.randn((1, INPUT_FEATURES), device="cuda", dtype=torch.bfloat16)
    with torch.inference_mode():
        reference, bias = gate_up_projection(input_tensor)
        assert bias is None
        candidate_outputs = {
            config: gate_up_gemv(
                input_tensor,
                gate_up_projection.weight,
                block_n=config[0],
                block_k=config[1],
                num_warps=config[2],
                prefetch=config[3],
            )
            for config in configs
        }
        max_abs_errors = {}
        for config, candidate_output in candidate_outputs.items():
            torch.testing.assert_close(candidate_output, reference, rtol=0.02, atol=0.5)
            max_abs_errors[config] = (
                (candidate_output.float() - reference.float()).abs().max().item()
            )

        if args.profile_kernel is not None:
            for _ in range(PROFILE_WARMUP_ITERATIONS):
                if args.profile_kernel == "reference":
                    gate_up_projection(input_tensor)
                else:
                    gate_up_gemv(
                        input_tensor,
                        gate_up_projection.weight,
                        output=candidate_outputs[configs[0]],
                        block_n=profile_config[0],
                        block_k=profile_config[1],
                        num_warps=profile_config[2],
                        prefetch=False,
                    )
            torch.cuda.synchronize()
            torch.cuda.cudart().cudaProfilerStart()
            if args.profile_kernel == "reference":
                gate_up_projection(input_tensor)
            else:
                gate_up_gemv(
                    input_tensor,
                    gate_up_projection.weight,
                    output=candidate_outputs[configs[0]],
                    block_n=profile_config[0],
                    block_k=profile_config[1],
                    num_warps=profile_config[2],
                    prefetch=False,
                )
            torch.cuda.synchronize()
            torch.cuda.cudart().cudaProfilerStop()
            print(f"Profiled one {args.profile_kernel} kernel launch")
            return

        reference_timings_us = [
            1000
            * triton.testing.do_bench_cudagraph(
                lambda: gate_up_projection(input_tensor)[0], rep=200
            )
        ]
        round_order = (*configs, *reversed(configs))
        if args.experiment == "prefetch":
            rounds = 3
        elif args.experiment in ("combined", "local-grid", "prefetch-grid"):
            rounds = 2
        else:
            rounds = 1
        benchmark_order = round_order * rounds
        candidate_timings_us: dict[tuple[int, int, int, bool], list[float]] = {
            config: [] for config in configs
        }
        for _ in range(rounds):
            for block_n, block_k, num_warps, prefetch in round_order:
                config = (block_n, block_k, num_warps, prefetch)
                candidate_timings_us[config].append(
                    1000
                    * triton.testing.do_bench_cudagraph(
                        partial(
                            gate_up_gemv,
                            input_tensor,
                            gate_up_projection.weight,
                            block_n=block_n,
                            block_k=block_k,
                            num_warps=num_warps,
                            prefetch=prefetch,
                        ),
                        rep=200,
                    )
                )
            reference_timings_us.append(
                1000
                * triton.testing.do_bench_cudagraph(
                    lambda: gate_up_projection(input_tensor)[0],
                    rep=200,
                )
            )
    reference_us = mean(reference_timings_us)
    report = {
        "gpu": torch.cuda.get_device_name(),
        "torch_version": str(torch.__version__),
        "triton_version": triton.__version__,
        "sglang_version": version("sglang"),
        "input_shape": list(input_tensor.shape),
        "weight_shape": list(gate_up_projection.weight.shape),
        "dtype": str(input_tensor.dtype),
        "tp_size": gate_up_projection.tp_size,
        "reference_operator": "MergedColumnParallelLinear.forward",
        "quant_method": type(gate_up_projection.quant_method).__name__,
        "experiment": args.experiment,
        "reference_us": reference_us,
        "baseline_config": list(BASELINE_CONFIG),
        "control_config": list(configs[0]),
        "benchmark_order": [list(config) for config in benchmark_order],
        "reference_timings_us": reference_timings_us,
        "candidates": [
            {
                "block_n": config[0],
                "block_k": config[1],
                "num_warps": config[2],
                "prefetch": config[3],
                "max_abs_error": max_abs_errors[config],
                "timings_us": candidate_timings_us[config],
                "mean_us": mean(candidate_timings_us[config]),
            }
            for config in configs
        ],
        "benchmark": "triton.testing.do_bench_cudagraph",
    }
    result = json.dumps(report, indent=2)
    print(result)
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(f"{result}\n")


if __name__ == "__main__":
    main()
