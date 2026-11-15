#!/usr/bin/env python3


from __future__ import annotations

import argparse
import atexit
import csv
import json
import math
import os
import random
import statistics
import subprocess
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SYSTEM_PROMPT = (
    "You are SmartTalk, a smart-home command parser. Return only valid JSON "
    "mapping rooms to device actions. Do not include explanations."
)


def make_prompt(command: str) -> str:
    return (
        f"### System:\n{SYSTEM_PROMPT}\n\n"
        f"### Input:\n{command.strip()}\n\n"
        "### Output:\n"
    )


def strict_json_loads(text: str) -> Any:
    def reject_constant(value: str) -> None:
        raise ValueError(f"invalid JSON constant {value!r}")

    def reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key {key!r}")
            result[key] = value
        return result

    return json.loads(
        text,
        object_pairs_hook=reject_duplicate_keys,
        parse_constant=reject_constant,
    )


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=True)


def schema_is_valid(value: Any) -> bool:
    if not isinstance(value, dict) or not value:
        return False
    scalar_types = (str, int, float, bool, type(None))
    for room, devices in value.items():
        if not isinstance(room, str) or not room.strip():
            return False
        if not isinstance(devices, dict) or not devices:
            return False
        for device, state in devices.items():
            if not isinstance(device, str) or not device.strip():
                return False
            if not isinstance(state, scalar_types):
                return False
    return True


def reservoir_sample_csv(
    dataset_path: Path,
    sample_count: int,
    seed: int,
) -> tuple[list[dict[str, Any]], int, int]:
    rng = random.Random(seed)
    sample: list[dict[str, Any]] = []
    valid_rows = 0
    invalid_rows = 0

    with dataset_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError("dataset has no header")
        columns = {name.strip().lower(): name for name in reader.fieldnames}
        if "input" not in columns or "output" not in columns:
            raise ValueError(
                f"dataset must contain input and output columns; found {reader.fieldnames}"
            )
        input_column = columns["input"]
        output_column = columns["output"]

        for csv_row_number, row in enumerate(reader, start=2):
            command = (row.get(input_column) or "").strip()
            expected_text = (row.get(output_column) or "").strip()
            try:
                expected_obj = strict_json_loads(expected_text)
                if not command or not schema_is_valid(expected_obj):
                    raise ValueError("empty command or invalid SmartTALK JSON schema")
            except Exception:
                invalid_rows += 1
                continue

            valid_rows += 1
            record = {
                "csv_row_number": csv_row_number,
                "input": command,
                "expected_text": expected_text,
                "expected_obj": expected_obj,
                "expected_canonical": canonical_json(expected_obj),
            }
            if len(sample) < sample_count:
                sample.append(record)
            else:
                replacement_index = rng.randrange(valid_rows)
                if replacement_index < sample_count:
                    sample[replacement_index] = record

    if valid_rows < sample_count:
        raise ValueError(
            f"requested {sample_count} samples, but only {valid_rows} valid rows exist"
        )
    rng.shuffle(sample)
    return sample, valid_rows, invalid_rows


def percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, math.ceil(fraction * len(ordered)) - 1)
    return ordered[index]


def sample_std(values: list[float]) -> float:
    return statistics.stdev(values) if len(values) > 1 else 0.0


def round_or_none(value: float | None, digits: int = 4) -> float | None:
    return round(value, digits) if value is not None else None


def generated_token_count(model: Any, response: dict[str, Any], text: str) -> int:
    usage = response.get("usage") or {}
    count = usage.get("completion_tokens")
    if isinstance(count, int):
        return count
    try:
        return len(model.tokenize(text.encode("utf-8"), add_bos=False))
    except Exception:
        return len(text.split())


class PythonBackend:
    name = "llama_cpp_python"

    def __init__(self, model_path: Path, args: argparse.Namespace) -> None:
        try:
            from llama_cpp import Llama
        except ImportError as exc:
            raise RuntimeError(
                "llama-cpp-python is unavailable; use --backend server with "
                "--llama-server ~/llama.cpp/build-cuda/bin/llama-server"
            ) from exc
        self.model = Llama(
            model_path=str(model_path),
            n_ctx=args.n_ctx,
            n_batch=args.n_batch,
            n_threads=args.threads,
            n_gpu_layers=args.n_gpu_layers,
            seed=args.seed,
            verbose=args.verbose_model,
        )

    def generate(self, prompt: str, max_tokens: int) -> tuple[str, int, float]:
        start = time.perf_counter()
        response = self.model(
            prompt, max_tokens=max_tokens, temperature=0.0, top_p=1.0, echo=False
        )
        latency_ms = (time.perf_counter() - start) * 1000.0
        text = str(response["choices"][0]["text"]).strip()
        return text, generated_token_count(self.model, response, text), latency_ms

    def close(self) -> None:
        pass


class ServerBackend:
    name = "llama_server_http"

    def __init__(
        self,
        executable: Path,
        model_path: Path,
        output_path: Path,
        args: argparse.Namespace,
    ) -> None:
        if not executable.is_file():
            raise FileNotFoundError(f"llama-server not found: {executable}")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        self.base_url = f"http://127.0.0.1:{args.server_port}"
        self.log_path = output_path.with_suffix(".llama-server.log")
        self.log_handle = self.log_path.open("w", encoding="utf-8")
        gpu_layers = 99 if args.n_gpu_layers < 0 else args.n_gpu_layers
        command = [
            str(executable),
            "-m",
            str(model_path),
            "-ngl",
            str(gpu_layers),
            "-c",
            str(args.n_ctx),
            "-b",
            str(args.n_batch),
            "--threads",
            str(args.threads),
            "--host",
            "127.0.0.1",
            "--port",
            str(args.server_port),
            "--no-warmup",
        ]
        print("Starting persistent llama-server ...")
        print("Server log:", self.log_path)
        self.process = subprocess.Popen(
            command,
            stdout=self.log_handle,
            stderr=subprocess.STDOUT,
        )
        try:
            self._wait_until_ready(args.server_start_timeout)
        except Exception:
            self.close()
            raise

    def _wait_until_ready(self, timeout_seconds: float) -> None:
        deadline = time.monotonic() + timeout_seconds
        last_error = "server did not answer"
        while time.monotonic() < deadline:
            if self.process.poll() is not None:
                self.log_handle.flush()
                raise RuntimeError(
                    f"llama-server exited with code {self.process.returncode}; "
                    f"inspect {self.log_path}"
                )
            try:
                with urllib.request.urlopen(self.base_url + "/health", timeout=2) as response:
                    if response.status == 200:
                        print("llama-server is ready.")
                        return
            except urllib.error.HTTPError as exc:
                last_error = f"HTTP {exc.code} while model loads"
            except Exception as exc:
                last_error = str(exc)
            time.sleep(0.5)
        raise TimeoutError(
            f"llama-server was not ready after {timeout_seconds:.0f}s ({last_error}); "
            f"inspect {self.log_path}"
        )

    def generate(self, prompt: str, max_tokens: int) -> tuple[str, int, float]:
        payload = json.dumps(
            {
                "prompt": prompt,
                "n_predict": max_tokens,
                "temperature": 0.0,
                "top_p": 1.0,
                "stream": False,
                "cache_prompt": False,
            }
        ).encode("utf-8")
        request = urllib.request.Request(
            self.base_url + "/completion",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        start = time.perf_counter()
        with urllib.request.urlopen(request, timeout=600) as response:
            result = json.loads(response.read().decode("utf-8"))
        latency_ms = (time.perf_counter() - start) * 1000.0
        text = str(result.get("content", "")).strip()
        token_count = result.get("tokens_predicted")
        if not isinstance(token_count, int):
            timings = result.get("timings") or {}
            token_count = timings.get("predicted_n")
        if not isinstance(token_count, int):
            token_count = len(text.split())
        return text, token_count, latency_ms

    def close(self) -> None:
        if getattr(self, "process", None) is not None and self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=5)
        if getattr(self, "log_handle", None) is not None:
            self.log_handle.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate a GGUF SmartTALK model on 100 reproducibly random commands "
            "and report exact-match accuracy plus latency mean +/- standard deviation."
        )
    )
    parser.add_argument("--model", type=Path, help="Path to one GGUF model")
    parser.add_argument(
        "--model-dir",
        type=Path,
        help="Evaluate every non-empty *-Q4_K_M.gguf file in this directory",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("eval_100_results"),
        help="Directory for reports when --model-dir is used",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("smart_home_100k_clean.csv"),
        help="CSV containing input and output columns",
    )
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--n-ctx", type=int, default=2048)
    parser.add_argument("--n-batch", type=int, default=128)
    parser.add_argument("--n-gpu-layers", type=int, default=-1)
    parser.add_argument("--threads", type=int, default=max(1, (os.cpu_count() or 4) - 2))
    parser.add_argument(
        "--backend",
        choices=("auto", "server", "python"),
        default="auto",
        help="Use the existing llama-server or the optional llama_cpp Python package",
    )
    parser.add_argument(
        "--llama-server",
        type=Path,
        default=Path("~/llama.cpp/build-cuda/bin/llama-server"),
    )
    parser.add_argument("--server-port", type=int, default=18080)
    parser.add_argument("--server-start-timeout", type=float, default=300.0)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("eval_100_results/gguf_random100_seed42.json"),
    )
    parser.add_argument("--verbose-model", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.model is None and args.model_dir is None:
        raise ValueError("provide either --model or --model-dir")
    if args.model is not None and args.model_dir is not None:
        raise ValueError("use only one of --model or --model-dir")

    if args.model_dir is not None:
        model_dir = args.model_dir.expanduser().resolve()
        if not model_dir.is_dir():
            raise FileNotFoundError(f"GGUF model directory not found: {model_dir}")
        models = sorted(
            path for path in model_dir.glob("*-Q4_K_M.gguf")
            if path.is_file() and path.stat().st_size > 0
        )
        if not models:
            raise FileNotFoundError(f"no non-empty *-Q4_K_M.gguf files found in {model_dir}")
        output_dir = args.output_dir.expanduser().resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        forwarded = list(sys.argv[1:])
        for option in ("--model-dir", "--output-dir"):
            while option in forwarded:
                index = forwarded.index(option)
                del forwarded[index : index + 2]
        completed = 0
        for model in models:
            report = output_dir / f"{model.stem}.json"
            command = [sys.executable, str(Path(__file__).resolve()), *forwarded, "--model", str(model), "--output", str(report)]
            print(f"\n=== Evaluating {model.name} ===", flush=True)
            result = subprocess.run(command, check=False)
            if result.returncode == 0:
                completed += 1
            else:
                print(f"Evaluation failed for {model.name} with exit code {result.returncode}", file=sys.stderr)
        print(f"\nCompleted {completed}/{len(models)} model evaluations.")
        print(f"Reports: {output_dir}")
        return 0 if completed == len(models) else 1
    model_path = args.model.expanduser().resolve()
    dataset_path = args.dataset.expanduser().resolve()
    output_path = args.output.expanduser().resolve()

    if args.samples <= 0:
        raise ValueError("--samples must be positive")
    if args.warmup < 0:
        raise ValueError("--warmup cannot be negative")
    if not model_path.is_file():
        raise FileNotFoundError(f"GGUF model not found: {model_path}")
    if not dataset_path.is_file():
        raise FileNotFoundError(f"dataset not found: {dataset_path}")

    sampled_rows, valid_dataset_rows, invalid_dataset_rows = reservoir_sample_csv(
        dataset_path, args.samples, args.seed
    )
    print(
        f"Selected {len(sampled_rows)} random rows with seed {args.seed} "
        f"from {valid_dataset_rows} valid dataset rows."
    )

    print(f"Loading model: {model_path}")
    server_path = args.llama_server.expanduser().resolve()
    selected_backend = args.backend
    if selected_backend == "auto":
        selected_backend = "server" if server_path.is_file() else "python"
    if selected_backend == "server":
        backend: Any = ServerBackend(server_path, model_path, output_path, args)
    else:
        backend = PythonBackend(model_path, args)
    atexit.register(backend.close)
    print(f"Inference backend: {backend.name}")

    for warmup_index in range(args.warmup):
        row = sampled_rows[warmup_index % len(sampled_rows)]
        print(f"Warm-up {warmup_index + 1}/{args.warmup} ...")
        backend.generate(make_prompt(row["input"]), args.max_tokens)

    results: list[dict[str, Any]] = []
    canonical_matches = 0
    raw_matches = 0
    valid_json_count = 0
    schema_valid_count = 0
    latencies_ms: list[float] = []
    throughputs: list[float] = []
    total_generated_tokens = 0

    for position, row in enumerate(sampled_rows, start=1):
        generated, generated_tokens, latency_ms = backend.generate(
            make_prompt(row["input"]), args.max_tokens
        )
        latencies_ms.append(latency_ms)
        total_generated_tokens += generated_tokens
        tokens_per_second = generated_tokens / (latency_ms / 1000.0) if latency_ms > 0 else 0.0
        throughputs.append(tokens_per_second)

        parsed: Any = None
        parse_error: str | None = None
        try:
            parsed = strict_json_loads(generated)
        except Exception as exc:
            parse_error = str(exc)

        valid_json = parse_error is None
        schema_valid = valid_json and schema_is_valid(parsed)
        canonical_match = valid_json and parsed == row["expected_obj"]
        raw_match = generated == row["expected_text"]

        valid_json_count += int(valid_json)
        schema_valid_count += int(schema_valid)
        canonical_matches += int(canonical_match)
        raw_matches += int(raw_match)

        results.append(
            {
                "sample_position": position,
                "csv_row_number": row["csv_row_number"],
                "input": row["input"],
                "expected": row["expected_canonical"],
                "generated": generated,
                "canonical_json_exact_match": canonical_match,
                "raw_text_exact_match": raw_match,
                "valid_json": valid_json,
                "schema_valid": schema_valid,
                "parse_error": parse_error,
                "generated_tokens": generated_tokens,
                "latency_ms": round(latency_ms, 4),
                "tokens_per_second": round(tokens_per_second, 4),
            }
        )

        running_accuracy = 100.0 * canonical_matches / position
        print(
            f"[{position:3d}/{args.samples}] row={row['csv_row_number']} "
            f"match={canonical_match} latency={latency_ms:.2f} ms "
            f"tok/s={tokens_per_second:.2f} running_accuracy={running_accuracy:.2f}%",
            flush=True,
        )

    mean_latency_ms = statistics.fmean(latencies_ms)
    std_latency_ms = sample_std(latencies_ms)
    total_generation_seconds = sum(latencies_ms) / 1000.0
    aggregate_throughput = (
        total_generated_tokens / total_generation_seconds if total_generation_seconds > 0 else 0.0
    )

    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model": str(model_path),
        "dataset": str(dataset_path),
        "sampling": {
            "method": "uniform reservoir sample without replacement",
            "seed": args.seed,
            "requested_samples": args.samples,
            "valid_dataset_rows": valid_dataset_rows,
            "invalid_dataset_rows_skipped": invalid_dataset_rows,
        },
        "generation": {
            "backend": backend.name,
            "llama_server": str(server_path) if selected_backend == "server" else None,
            "prompt_format": "SmartTALK ### System / ### Input / ### Output",
            "temperature": 0.0,
            "max_tokens": args.max_tokens,
            "warmup_calls_excluded": args.warmup,
            "n_ctx": args.n_ctx,
            "n_batch": args.n_batch,
            "n_gpu_layers": args.n_gpu_layers,
            "threads": args.threads,
        },
        "accuracy": {
            "headline_metric": "canonical_json_exact_match_accuracy",
            "correct": canonical_matches,
            "total": args.samples,
            "canonical_json_exact_match_accuracy": canonical_matches / args.samples,
            "canonical_json_exact_match_percent": 100.0 * canonical_matches / args.samples,
            "raw_text_exact_match_accuracy": raw_matches / args.samples,
            "valid_json_rate": valid_json_count / args.samples,
            "schema_valid_rate": schema_valid_count / args.samples,
        },
        "latency_ms": {
            "definition": "end-to-end batch-1 generation time; model load and warm-up excluded",
            "mean": round(mean_latency_ms, 4),
            "sample_standard_deviation": round(std_latency_ms, 4),
            "mean_plus_minus_standard_deviation": f"{mean_latency_ms:.2f} +/- {std_latency_ms:.2f} ms",
            "median": round(statistics.median(latencies_ms), 4),
            "p95": round_or_none(percentile(latencies_ms, 0.95)),
            "minimum": round(min(latencies_ms), 4),
            "maximum": round(max(latencies_ms), 4),
        },
        "throughput_tokens_per_second": {
            "paper_comparable_aggregate": round(aggregate_throughput, 4),
            "per_sample_mean": round(statistics.fmean(throughputs), 4),
            "per_sample_standard_deviation": round(sample_std(throughputs), 4),
            "generated_tokens_total": total_generated_tokens,
        },
        "samples": results,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)
        handle.write("\n")

    print("\n=== SmartTALK random-sample evaluation ===")
    print(
        f"Exact-match accuracy: {canonical_matches}/{args.samples} "
        f"({100.0 * canonical_matches / args.samples:.2f}%)"
    )
    print(f"Valid JSON: {valid_json_count}/{args.samples} ({100.0 * valid_json_count / args.samples:.2f}%)")
    print(f"Latency: {mean_latency_ms:.2f} +/- {std_latency_ms:.2f} ms (mean +/- sample SD)")
    print(f"Aggregate throughput: {aggregate_throughput:.2f} generated tokens/s")
    print(f"Detailed report: {output_path}")
    backend.close()
    atexit.unregister(backend.close)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        raise SystemExit(130)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
