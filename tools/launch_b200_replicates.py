"""Paired seed replications: one process/GPU, separate configs/results/logs.

Run with nohup using the B200 environment's Python. The default plan adds
seeds 1..4 to the existing seed-0 pilot, retaining its learning controls.
"""

from __future__ import annotations

import argparse
import fcntl
import itertools
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

import yaml


def csv_ints(value: str) -> list[int]:
    values = [int(item) for item in value.split(",")]
    if not values or min(values) < 0 or len(set(values)) != len(values):
        raise argparse.ArgumentTypeError("use distinct nonnegative comma-separated integers")
    return values


def available_memory() -> int | None:
    path = Path("/proc/meminfo")
    if path.is_file():
        for line in path.read_text().splitlines():
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) * 1024
    return None


def csv_floats(value: str) -> list[float]:
    values = [float(item) for item in value.split(",")]
    if not values or any(not math.isfinite(item) or item < 0 for item in values):
        raise argparse.ArgumentTypeError("use finite nonnegative comma-separated values")
    if len(set(values)) != len(values):
        raise argparse.ArgumentTypeError("grid values must be distinct")
    return values


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpus", type=csv_ints, default=csv_ints("0,1,2,3"))
    parser.add_argument("--seeds", type=csv_ints, default=csv_ints("1,2,3,4"))
    parser.add_argument("--useful-n-list", type=csv_floats, default=csv_floats("0"))
    parser.add_argument("--utility-weights", type=csv_floats, default=csv_floats("1"))
    parser.add_argument("--output-root", type=Path, default=Path("outputs/mass_b200_replicates"))
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument(
        "--job-memory-gib",
        type=int,
        default=48,
        help="Scheduling estimate; pilot measured about 39 GiB peak RSS.",
    )
    parser.add_argument("--reserve-memory-gib", type=int, default=16)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.threads < 1 or args.job_memory_gib < 1 or args.reserve_memory_gib < 0:
        parser.error("thread/memory controls must be positive (reserve may be zero)")
    repo = Path(__file__).resolve().parents[1]
    os.chdir(repo)
    output = args.output_root.resolve()
    output.mkdir(parents=True, exist_ok=True)
    lock = (output / "launcher.lock").open("a")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        parser.error("another launcher already owns this output directory")
    capacity = min(len(args.gpus), max(1, (os.cpu_count() or 1) // args.threads))
    memory = available_memory()
    if memory is not None:
        memory_capacity = (memory - args.reserve_memory_gib * 2**30) // (
            args.job_memory_gib * 2**30
        )
        if memory_capacity < 1 and not args.dry_run:
            parser.error("available host RAM is below the configured one-job memory estimate")
        capacity = min(capacity, max(1, memory_capacity))
    jobs = []
    for seed, loss_n, weight in itertools.product(
        args.seeds, args.useful_n_list, args.utility_weights
    ):
        destination = output / f"seed-{seed}-n{loss_n:g}-uw{weight:g}"
        destination.mkdir(exist_ok=True)
        config = {
            "base_config": str(repo / "configs/criteo_prior_art_gpu_pilot.yaml"),
            "seeds": [seed],
            "comparison_pilot_seeds": [seed],
            "mass_seed_list": [seed],
            "mass_pilot_seed_list": [seed],
            "mass_n_list": [loss_n],
            "mass_pilot_n_list": [loss_n],
            "mass_utility_weight_list": [weight],
            "mass_pilot_utility_weight_list": [weight],
            "comparison_output_dir": str(destination),
            "mass_device": "cuda:0",
            "attribute_attack_prior": 0.5,
            "ctr_ranking_mc_replicates": 32,
        }
        config_path = destination / "replication.yaml"
        config_path.write_text(yaml.safe_dump(config, sort_keys=True))
        jobs.append(
            {
                "seed": seed,
                "loss_n": loss_n,
                "utility_weight": weight,
                "config": str(config_path),
                "output": str(destination),
                "log": str(destination / "experiment.log"),
                "status": "queued",
                "command": [
                    sys.executable,
                    "-u",
                    "-m",
                    "capt12.cli",
                    "prior-art-comparison",
                    "--config",
                    str(config_path),
                    "--phase",
                    "pilot",
                ],
            }
        )
    plan = {
        "jobs": jobs,
        "maximum_parallel_jobs": capacity,
        "host_available_bytes": memory,
        "job_memory_gib": args.job_memory_gib,
        "gpus": args.gpus,
    }
    plan_path = output / "launch.json"

    def save():
        plan_path.write_text(json.dumps(plan, indent=2) + "\n")

    save()
    print(json.dumps(plan, indent=2), flush=True)
    if args.dry_run:
        return
    from capt12.utils.artifacts import require_clean_worktree

    plan["source_git_sha"] = require_clean_worktree()
    # Check only the explicitly allocated GPUs; no all-device CUDA initialization.
    for gpu in args.gpus:
        used = subprocess.check_output(
            [
                "nvidia-smi",
                "-i",
                str(gpu),
                "--query-gpu=memory.used",
                "--format=csv,noheader,nounits",
            ],
            text=True,
        ).strip()
        if int(used) > 1024:
            parser.error(
                f"GPU {gpu} already uses {used} MiB; choose a different allocated GPU list"
            )
    running, free = [], list(args.gpus)
    pending = list(jobs)
    try:
        while pending or running:
            while pending and free and len(running) < capacity:
                job, gpu = pending.pop(0), free.pop(0)
                environment = dict(os.environ)
                environment.update(
                    CUDA_VISIBLE_DEVICES=str(gpu),
                    PYTHONUNBUFFERED="1",
                    MPLBACKEND="Agg",
                    OMP_NUM_THREADS=str(args.threads),
                    MKL_NUM_THREADS=str(args.threads),
                    OPENBLAS_NUM_THREADS=str(args.threads),
                )
                log = Path(job["log"]).open("a")
                process = subprocess.Popen(
                    job["command"],
                    cwd=repo,
                    env=environment,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    stdin=subprocess.DEVNULL,
                )
                job.update(status="running", gpu=gpu, pid=process.pid)
                running.append((job, gpu, process, log))
                print(
                    f"Started seed={job['seed']} n={job['loss_n']} useful_weight={job['utility_weight']} on physical GPU={gpu}, PID={process.pid}",
                    flush=True,
                )
                save()
            for entry in running[:]:
                job, gpu, process, log = entry
                code = process.poll()
                if code is None:
                    continue
                log.close()
                job.update(status="complete" if code == 0 else "failed", returncode=code)
                running.remove(entry)
                free.append(gpu)
                print(f"Finished seed={job['seed']} returncode={code}", flush=True)
                save()
            if running:
                time.sleep(5)
    except BaseException:
        for job, _, process, log in running:
            process.terminate()
            process.wait()
            log.close()
            job.update(status="interrupted", returncode=process.returncode)
        save()
        raise
    if any(job["status"] != "complete" for job in jobs):
        raise SystemExit("Some seed jobs failed; inspect launch.json and per-seed experiment.log")
    print("ALL_REPLICATES_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
