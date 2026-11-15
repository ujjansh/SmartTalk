#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
import torch
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence


PAPER_MODEL_IDS: Dict[str, str] = {
    "yi_1_5_6b": "01-ai/Yi-1.5-6B-Chat",
    "codellama_7b": "codellama/CodeLlama-7b-Instruct-hf",
    "internlm2_5_7b": "internlm/internlm2_5-7b-chat",
    "mistral_7b_v0_3": "mistralai/Mistral-7B-Instruct-v0.3",
    "qwen2_5_7b": "Qwen/Qwen2.5-7B-Instruct",
    "llama3_8b": "meta-llama/Meta-Llama-3-8B-Instruct",
    "llama3_1_8b": "meta-llama/Llama-3.1-8B-Instruct",
    "yi_1_5_9b": "01-ai/Yi-1.5-9B-Chat",
    "gemma2_9b": "google/gemma-2-9b-it",
}

GATED_MODELS = {"llama3_8b", "llama3_1_8b", "gemma2_9b"}

PRESETS: Dict[str, List[str]] = {
    "7b9b": ["qwen2_5_7b", "yi_1_5_9b"],
    "ungated-large": ["yi_1_5_6b", "codellama_7b", "mistral_7b_v0_3", "qwen2_5_7b", "yi_1_5_9b"],
    "paper-large": [
        "yi_1_5_6b",
        "codellama_7b",
        "internlm2_5_7b",
        "mistral_7b_v0_3",
        "qwen2_5_7b",
        "llama3_8b",
        "llama3_1_8b",
        "yi_1_5_9b",
        "gemma2_9b",
    ],
    "qwen": ["qwen2_5_7b"],
    "llama": ["llama3_8b"],
    "gemma": ["gemma2_9b"],
    "yi9b": ["yi_1_5_9b"],
}


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def jsonable_args(args: argparse.Namespace) -> Dict[str, object]:
    out: Dict[str, object] = {}
    for key, value in vars(args).items():
        if isinstance(value, Path):
            out[key] = str(value)
        else:
            out[key] = value
    return out


def memory_report() -> Dict[str, Optional[float]]:
    report: Dict[str, Optional[float]] = {
        "system_ram_total_gb": None,
        "system_ram_available_gb": None,
        "disk_free_gb": None,
    }
    try:
        import psutil  # type: ignore

        vm = psutil.virtual_memory()
        report["system_ram_total_gb"] = round(vm.total / (1024**3), 2)
        report["system_ram_available_gb"] = round(vm.available / (1024**3), 2)
    except Exception:
        pass

    try:
        usage = shutil.disk_usage(Path.cwd())
        report["disk_free_gb"] = round(usage.free / (1024**3), 2)
    except Exception:
        pass
    return report


def install_deps() -> None:
    packages = [
        "transformers>=4.45.0",
        "accelerate>=0.33.0",
        "datasets>=2.20.0",
        "huggingface_hub>=0.24.0",
        "safetensors>=0.4.3",
        "sentencepiece>=0.2.0",
        "protobuf>=4.25.0",
        "tqdm>=4.66.0",
        "psutil>=5.9.0",
    ]
    print("Installing Python dependencies. This does not install or replace Jetson PyTorch.")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-U", "--upgrade-strategy", "only-if-needed", *packages])


def model_keys_from_args(args: argparse.Namespace) -> List[str]:
    if args.models:
        keys = PRESETS["paper-large"] if args.models == ["all"] else args.models
    else:
        keys = PRESETS[args.preset]

    unknown = [key for key in keys if key not in PAPER_MODEL_IDS]
    if unknown:
        raise SystemExit(f"Unknown model key(s): {unknown}. Valid keys: {sorted(PAPER_MODEL_IDS)}")

    if args.skip_gated:
        keys = [key for key in keys if key not in GATED_MODELS]
    return keys


def download_model(model_key: str, cache_dir: Optional[str]) -> None:
    from huggingface_hub import snapshot_download

    model_id = PAPER_MODEL_IDS[model_key]
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    print(f"Downloading {model_key}: {model_id}")
    snapshot_download(
        repo_id=model_id,
        token=token,
        cache_dir=cache_dir,
        resume_download=True,
    )


def run_model(args: argparse.Namespace, model_key: str, output_dir: Path) -> Dict[str, object]:
    model_output = output_dir / model_key
    command = [
        args.python,
        str(args.prune_script),
        "--dataset-path",
        str(args.dataset_path),
        "--output-dir",
        str(output_dir),
        "--paper-model",
        model_key,
        "--variants",
        args.variant,
        "--prune-ratio",
        str(args.prune_ratio),
        "--calibration-size",
        str(args.calibration_size),
        "--max-length",
        str(args.max_length),
        "--max-new-tokens",
        str(args.max_new_tokens),
        "--device-map",
        args.device_map,
        "--warmup-generations",
        str(args.warmup_generations),
    ]
    if not args.full_eval:
        command += ["--eval-limit", str(args.eval_limit)]
    if args.skip_baseline:
        command.append("--skip-baseline")
    if args.skip_reload_check:
        command.append("--skip-reload-check")
    if args.allow_cpu:
        command.append("--allow-cpu")
    if args.model_revision != "main":
        command += ["--model-revision", args.model_revision]

    started = time.perf_counter()
    print("\n" + "=" * 80)
    print(f"Running {model_key}: {PAPER_MODEL_IDS[model_key]}")
    print("Command:")
    print(" ".join(command))
    print("=" * 80, flush=True)
    completed = subprocess.run(command, check=False)
    elapsed = round(time.perf_counter() - started, 3)

    return {
        "model_key": model_key,
        "model_id": PAPER_MODEL_IDS[model_key],
        "return_code": completed.returncode,
        "elapsed_seconds": elapsed,
        "model_output_dir": str(model_output),
        "comparison_report": str(model_output / args.variant / "paper_comparison_report.json"),
    }


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Run SmartTalk 7B/9B pruning + accuracy evaluation on Jetson.")
    parser.add_argument("--dataset-path", type=Path, required=True, help="Path to smart_home_100k_clean CSV on Jetson.")
    parser.add_argument("--output-dir", type=Path, default=Path("smarttalk_jetson_prune_output"))
    parser.add_argument("--prune-script", type=Path, default=script_dir / "smarttalk_structured_prune.py")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--preset", choices=sorted(PRESETS), default="7b9b")
    parser.add_argument("--models", nargs="+", default=None, help="Explicit model keys. Overrides --preset.")
    parser.add_argument("--skip-gated", action="store_true", help="Skip LLaMA/Gemma gated models if selected.")
    parser.add_argument("--model-revision", default="main")
    parser.add_argument("--variant", choices=["structural50", "paper-budget"], default="structural50")
    parser.add_argument("--prune-ratio", type=float, default=0.50)
    parser.add_argument("--calibration-size", type=int, default=128)
    parser.add_argument("--eval-limit", type=int, default=1000, help="Rows to evaluate unless --full-eval is set.")
    parser.add_argument("--full-eval", action="store_true", help="Evaluate full 10k test split.")
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--device-map", default="auto", help="Usually 'auto' on Jetson. Use 'cuda' only if model fits.")
    parser.add_argument("--warmup-generations", type=int, default=1)
    parser.add_argument("--skip-baseline", action="store_true", help="Only evaluate pruned model; faster but less comparable.")
    parser.add_argument("--skip-reload-check", action="store_true", default=True)
    parser.add_argument("--reload-check", dest="skip_reload_check", action="store_false")
    parser.add_argument("--allow-cpu", action="store_true", help="Debug only. Large models on CPU will be extremely slow.")
    parser.add_argument("--download-only", action="store_true", help="Only pre-download selected HF models.")
    parser.add_argument("--cache-dir", default=None, help="Optional Hugging Face cache directory.")
    parser.add_argument("--install-deps", action="store_true", help="Install non-PyTorch Python dependencies.")
    parser.add_argument("--continue-on-error", action="store_true", default=True)
    parser.add_argument("--stop-on-error", dest="continue_on_error", action="store_false")
    args = parser.parse_args(argv)

    args.dataset_path = args.dataset_path.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    args.prune_script = args.prune_script.expanduser().resolve()

    if not args.dataset_path.exists():
        raise SystemExit(f"Dataset path does not exist: {args.dataset_path}")
    if not args.prune_script.exists():
        raise SystemExit(f"Pruning script does not exist: {args.prune_script}")
    if args.calibration_size <= 0:
        raise SystemExit("--calibration-size must be positive")
    if args.eval_limit <= 0 and not args.full_eval:
        raise SystemExit("--eval-limit must be positive unless --full-eval is set")
    return args


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    if args.install_deps:
        install_deps()

    if args.cache_dir:
        cache_dir = str(Path(args.cache_dir).expanduser().resolve())
        os.environ.setdefault("HF_HOME", cache_dir)
        os.environ.setdefault("TRANSFORMERS_CACHE", cache_dir)
    else:
        cache_dir = None

    args.output_dir.mkdir(parents=True, exist_ok=True)
    selected = model_keys_from_args(args)
    token_present = bool(os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN"))
    gated_selected = [key for key in selected if key in GATED_MODELS]

    manifest = {
        "created_at_utc": now_utc(),
        "dataset_path": str(args.dataset_path),
        "output_dir": str(args.output_dir),
        "prune_script": str(args.prune_script),
        "selected_models": [{"key": key, "model_id": PAPER_MODEL_IDS[key], "gated": key in GATED_MODELS} for key in selected],
        "token_present": token_present,
        "memory_report": memory_report(),
        "args": jsonable_args(args),
        "runs": [],
    }
    write_json(args.output_dir / "jetson_runner_manifest.json", manifest)

    print("Selected models:")
    for key in selected:
        gate_note = " gated" if key in GATED_MODELS else ""
        print(f"  - {key}: {PAPER_MODEL_IDS[key]}{gate_note}")
    if gated_selected and not token_present:
        print("WARNING: selected gated models require HF_TOKEN or huggingface-cli login.")
    print(f"Memory/disk: {manifest['memory_report']}")

    if args.download_only:
        for key in selected:
            download_model(key, cache_dir)
        print("Download-only complete.")
        return 0

    failures = 0
    for key in selected:
        result = run_model(args, key, args.output_dir)
        manifest["runs"].append(result)
        write_json(args.output_dir / "jetson_runner_manifest.json", manifest)
        if result["return_code"] != 0:
            failures += 1
            print(f"FAILED: {key} returned {result['return_code']}", file=sys.stderr)
            if not args.continue_on_error:
                break

    print(f"\nDone. Manifest: {args.output_dir / 'jetson_runner_manifest.json'}")
    if failures:
        print(f"Failures: {failures}. Check each model folder and console logs.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
