#!/usr/bin/env python3


from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import os
import platform
import random
import re
import shutil
import statistics
import subprocess
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


DEFAULT_MODEL_NAME = "HuggingFaceTB/SmolLM2-360M"
DEFAULT_MODEL_REVISION = "main"
SEED = 42
MAX_LENGTH = 512
MAX_NEW_TOKENS = 256
EFFECTIVE_BATCH_SIZE = 32
PAPER_BASELINE_ACCURACY = 0.5771
PAPER_FINETUNED_ACCURACY = 0.9420
PAPER_T4_LATENCY_MS = 149.8
SMOKE_TRAIN_ROWS = 32
SMOKE_VALIDATION_ROWS = 8
SMOKE_TEST_ROWS = 8

LORA_TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]

PAPER_QLORA_MODELS = [
    "SmolLM2-135M",
    "SmolLM2-360M",
    "Qwen2.5-0.5B-Instruct",
    "OLMo-1B",
    "TinyLlama-1.1B-Chat",
    "LLaMA-3.2-1B-Instruct",
    "Phi-1.5",
    "Qwen2.5-1.5B-Instruct",
    "StableLM2-1.6B-Zephyr",
    "SmolLM2-1.7B-Instruct",
    "Gemma-2B-IT",
    "Phi-2",
    "Qwen2.5-3B-Instruct",
    "LLaMA-3.2-3B-Instruct",
]

LORA_TARGET_PROFILES = {
    "llama_qwen_gemma_olmo_stablelm": LORA_TARGET_MODULES,
    "phi": ["q_proj", "k_proj", "v_proj", "dense", "fc1", "fc2"],
    "gpt_neox_stablelm": ["query_key_value", "dense", "dense_h_to_4h", "dense_4h_to_h"],
    "internlm2": ["wqkv", "wo", "w1", "w2", "w3"],
}

GENERIC_LORA_TARGET_NAMES = sorted({
    name
    for profile_targets in LORA_TARGET_PROFILES.values()
    for name in profile_targets
})

SYSTEM_PROMPT = (
    "You are SmartTalk, a smart-home command parser. Return only valid JSON "
    "mapping rooms to device actions. Do not include explanations."
)

JSON_SCALAR_TYPES = (str, int, float, bool, type(None))


@dataclass(frozen=True)
class SmartTalkRow:
    row_id: int
    input_text: str
    output_text: str
    target_obj: Dict[str, Dict[str, Any]]
    canonical_output: str
    action_count: int


@dataclass(frozen=True)
class SplitIndices:
    train: List[int]
    validation: List[int]
    test: List[int]


class SmartTalkError(RuntimeError):
    """User-facing script failure."""


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
        f.write("\n")


def file_sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, ensure_ascii=True)


def strict_json_loads(text: str) -> Any:
    def reject_constant(value: str) -> None:
        raise ValueError(f"invalid JSON constant {value!r}")

    def no_duplicate_keys(pairs: List[Tuple[str, Any]]) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        for key, value in pairs:
            if key in out:
                raise ValueError(f"duplicate JSON key {key!r}")
            out[key] = value
        return out

    return json.loads(
        text,
        object_pairs_hook=no_duplicate_keys,
        parse_constant=reject_constant,
    )


def schema_status(obj: Any) -> Tuple[bool, Optional[str], int]:
    if not isinstance(obj, dict) or not obj:
        return False, "target must be a non-empty JSON object", 0

    action_count = 0
    for room, devices in obj.items():
        if not isinstance(room, str) or not room.strip():
            return False, "room keys must be non-empty strings", action_count
        if not isinstance(devices, dict) or not devices:
            return False, f"room {room!r} must map to a non-empty object", action_count
        for device, state in devices.items():
            if not isinstance(device, str) or not device.strip():
                return False, f"device keys in room {room!r} must be non-empty strings", action_count
            if not isinstance(state, JSON_SCALAR_TYPES):
                return False, (
                    f"action value for {room!r}/{device!r} must be a JSON scalar, "
                    f"got {type(state).__name__}"
                ), action_count
            action_count += 1

    return True, None, action_count


def make_prompt(input_text: str) -> str:
    return f"### System:\n{SYSTEM_PROMPT}\n\n### Input:\n{input_text.strip()}\n\n### Output:\n"


def default_output_dir() -> Path:
    if Path("/kaggle/working").exists():
        return Path("/kaggle/working/smarttalk_qlora_output")
    if Path("/content").exists():
        return Path("/content/smarttalk_qlora_output")
    return Path(__file__).resolve().parent / "smarttalk_qlora_output"


def discover_roots() -> List[Tuple[Path, bool, str]]:
    roots: List[Tuple[Path, bool, str]] = []
    if Path("/kaggle/input").exists():
        roots.append((Path("/kaggle/input"), True, "kaggle_input"))
    if Path("/content").exists():
        roots.append((Path("/content"), True, "colab_content"))

    script_dir = Path(__file__).resolve().parent
    roots.append((script_dir, False, "script_dir"))

    downloads = Path.home() / "Downloads"
    if downloads.exists() and downloads.resolve() != script_dir.resolve():
        roots.append((downloads, False, "windows_downloads"))

    return roots


def csv_header(path: Path) -> List[str]:
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as f:
            reader = csv.reader(f)
            return [cell.strip().lower() for cell in next(reader, [])]
    except UnicodeDecodeError:
        return []
    except OSError:
        return []


def score_candidate(path: Path) -> int:
    name = path.name.lower()
    header = csv_header(path)
    if "input" not in header or "output" not in header:
        return 0

    score = 20
    if "smart_home_100k_clean" in name:
        score += 100
    if "smarttalk" in name or "smart_talk" in name:
        score += 90
    if "smart" in name and "home" in name:
        score += 70
    if "command" in name:
        score += 35
    if "100k" in name:
        score += 20
    if path.stat().st_size >= 10_000_000:
        score += 10
    return score


def collect_csv_candidates() -> List[Dict[str, Any]]:
    seen: set[Path] = set()
    candidates: List[Dict[str, Any]] = []
    for root, recursive, source in discover_roots():
        if not root.exists():
            continue
        iterator = root.rglob("*.csv") if recursive else root.glob("*.csv")
        for path in iterator:
            try:
                resolved = path.resolve()
            except OSError:
                continue
            if resolved in seen or not resolved.is_file():
                continue
            seen.add(resolved)
            try:
                score = score_candidate(resolved)
                stat = resolved.stat()
            except OSError:
                continue
            if score <= 0:
                continue
            candidates.append(
                {
                    "path": str(resolved),
                    "score": score,
                    "source": source,
                    "size_bytes": stat.st_size,
                    "mtime": stat.st_mtime,
                }
            )
    candidates.sort(key=lambda item: (-item["score"], item["path"]))
    return candidates


def discover_dataset_path(explicit_path: Optional[str]) -> Tuple[Path, Dict[str, Any]]:
    if explicit_path:
        path = Path(explicit_path).expanduser().resolve()
        if not path.exists():
            raise SmartTalkError(f"Dataset path does not exist: {path}")
        if not path.is_file():
            raise SmartTalkError(f"Dataset path is not a file: {path}")
        return path, {"mode": "explicit", "path": str(path)}

    candidates = collect_csv_candidates()
    if not candidates:
        roots = [str(root) for root, _, _ in discover_roots()]
        raise SmartTalkError(
            "Could not auto-discover a SmartTalk CSV with input/output columns. "
            "Pass --dataset-path. Searched: " + ", ".join(roots)
        )

    top_score = candidates[0]["score"]
    top = [item for item in candidates if item["score"] == top_score]
    if len(top) == 1:
        return Path(top[0]["path"]), {"mode": "auto", "selected": top[0], "candidates": candidates[:25]}

    hashes: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for item in top:
        path = Path(item["path"])
        item_with_hash = dict(item)
        item_with_hash["sha256"] = file_sha256(path)
        hashes[item_with_hash["sha256"]].append(item_with_hash)

    if len(hashes) == 1:
        preferred_names = ["smart_home_100k_clean (2) (2).csv", "smart_home_100k_clean.csv"]
        preferred = [item for item in top if Path(item["path"]).name in preferred_names]
        selected = sorted(preferred or top, key=lambda item: item["path"])[0]
        selected_path = Path(selected["path"])
        return selected_path, {
            "mode": "auto_same_content",
            "selected": selected,
            "same_content_candidates": top,
            "candidates": candidates[:25],
        }

    rendered = "\n".join(f"  - {item['path']} (score={item['score']})" for item in top[:50])
    raise SmartTalkError(
        "Dataset auto-discovery is ambiguous. Pass --dataset-path with one of:\n" + rendered
    )


def audit_csv(path: Path) -> Tuple[List[SmartTalkRow], Dict[str, Any]]:
    dataset_hash = file_sha256(path)
    rows: List[SmartTalkRow] = []
    malformed_examples: List[Dict[str, Any]] = []
    schema_examples: List[Dict[str, Any]] = []
    unsorted_examples: List[Dict[str, Any]] = []
    empty_examples: List[Dict[str, Any]] = []
    conflict_examples: List[Dict[str, Any]] = []

    row_pair_counts: Counter[Tuple[str, str]] = Counter()
    input_to_targets: Dict[str, set[str]] = defaultdict(set)
    input_word_counts: List[int] = []
    action_counts: List[int] = []
    room_counts: Counter[str] = Counter()
    device_counts: Counter[str] = Counter()
    output_text_counts: Counter[str] = Counter()

    total_rows = 0
    non_empty_pairs = 0
    valid_json_outputs = 0
    schema_valid_outputs = 0
    sorted_json_outputs = 0

    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        columns = list(reader.fieldnames or [])
        lower_columns = [col.strip().lower() for col in columns]
        if "input" not in lower_columns or "output" not in lower_columns:
            raise SmartTalkError(
                f"CSV must contain input and output columns. Found columns: {columns}"
            )
        input_column = columns[lower_columns.index("input")]
        output_column = columns[lower_columns.index("output")]

        for row_id, record in enumerate(reader):
            total_rows += 1
            input_text = (record.get(input_column) or "").strip()
            output_text = (record.get(output_column) or "").strip()
            if not input_text or not output_text:
                if len(empty_examples) < 10:
                    empty_examples.append({"row_id": row_id, "input": input_text, "output": output_text})
                continue
            non_empty_pairs += 1

            try:
                target_obj = strict_json_loads(output_text)
                valid_json_outputs += 1
            except Exception as exc:
                if len(malformed_examples) < 25:
                    malformed_examples.append(
                        {"row_id": row_id, "output": output_text[:500], "error": str(exc)}
                    )
                continue

            schema_ok, schema_error, action_count = schema_status(target_obj)
            if schema_ok:
                schema_valid_outputs += 1
            else:
                if len(schema_examples) < 25:
                    schema_examples.append(
                        {
                            "row_id": row_id,
                            "output": output_text[:500],
                            "error": schema_error,
                        }
                    )
                continue

            canon = canonical_json(target_obj)
            if output_text == canon:
                sorted_json_outputs += 1
            elif len(unsorted_examples) < 25:
                unsorted_examples.append(
                    {
                        "row_id": row_id,
                        "output": output_text[:500],
                        "canonical_output": canon[:500],
                    }
                )

            typed_obj = target_obj
            assert isinstance(typed_obj, dict)
            rows.append(
                SmartTalkRow(
                    row_id=row_id,
                    input_text=input_text,
                    output_text=output_text,
                    target_obj=typed_obj,
                    canonical_output=canon,
                    action_count=action_count,
                )
            )

            row_pair_counts[(input_text, canon)] += 1
            input_to_targets[input_text].add(canon)
            input_word_counts.append(len(input_text.split()))
            action_counts.append(action_count)
            output_text_counts[canon] += 1
            for room, devices in target_obj.items():
                room_counts[str(room)] += 1
                for device in devices.keys():
                    device_counts[str(device)] += 1

    conflict_inputs = {inp: targets for inp, targets in input_to_targets.items() if len(targets) > 1}
    for inp, targets in list(conflict_inputs.items())[:10]:
        conflict_examples.append({"input": inp, "target_count": len(targets), "targets": sorted(targets)[:5]})

    max_actions = max(action_counts) if action_counts else 0
    action_histogram = Counter(action_counts)
    audit = {
        "created_at": now_utc(),
        "dataset_path": str(path),
        "dataset_sha256": dataset_hash,
        "columns": columns if "columns" in locals() else [],
        "required_columns_present": True,
        "total_rows": total_rows,
        "non_empty_pairs": non_empty_pairs,
        "empty_pair_count": total_rows - non_empty_pairs,
        "valid_json_outputs": valid_json_outputs,
        "malformed_json_outputs": non_empty_pairs - valid_json_outputs,
        "schema_valid_outputs": schema_valid_outputs,
        "schema_invalid_outputs": valid_json_outputs - schema_valid_outputs,
        "sorted_json_outputs": sorted_json_outputs,
        "unsorted_json_outputs": schema_valid_outputs - sorted_json_outputs,
        "unique_row_pairs": len(row_pair_counts),
        "duplicate_rows": total_rows - len(row_pair_counts),
        "unique_inputs": len(input_to_targets),
        "inputs_with_conflicting_targets": len(conflict_inputs),
        "unique_outputs": len(output_text_counts),
        "avg_input_words": round(statistics.fmean(input_word_counts), 3) if input_word_counts else 0.0,
        "avg_device_actions": round(statistics.fmean(action_counts), 3) if action_counts else 0.0,
        "min_device_actions": min(action_counts) if action_counts else 0,
        "max_device_actions": max_actions,
        "commands_with_max_device_actions": action_histogram.get(max_actions, 0),
        "action_count_histogram": {str(k): v for k, v in sorted(action_histogram.items())},
        "rooms": sorted(room_counts.keys()),
        "room_count": len(room_counts),
        "room_frequencies": dict(sorted(room_counts.items())),
        "device_types": sorted(device_counts.keys()),
        "device_type_count": len(device_counts),
        "device_type_frequencies": dict(sorted(device_counts.items())),
        "examples": {
            "empty_pairs": empty_examples,
            "malformed_json": malformed_examples,
            "schema_invalid": schema_examples,
            "unsorted_json": unsorted_examples,
            "input_conflicts": conflict_examples,
        },
    }
    return rows, audit


def validate_audit_or_fail(audit: Dict[str, Any]) -> None:
    failures = []
    if audit["total_rows"] <= 0:
        failures.append("CSV contains no data rows")
    if audit["empty_pair_count"] != 0:
        failures.append(f"{audit['empty_pair_count']} rows have empty input/output values")
    if audit["malformed_json_outputs"] != 0:
        failures.append(f"{audit['malformed_json_outputs']} outputs are not strict JSON")
    if audit["schema_invalid_outputs"] != 0:
        failures.append(f"{audit['schema_invalid_outputs']} outputs fail the smart-home JSON schema")
    if audit["inputs_with_conflicting_targets"] != 0:
        failures.append(f"{audit['inputs_with_conflicting_targets']} inputs have conflicting targets")
    if failures:
        raise SmartTalkError("Dataset audit failed before model allocation:\n" + "\n".join(f"  - {x}" for x in failures))


def create_split_indices(n_rows: int, seed: int = SEED) -> SplitIndices:
    if n_rows < 10:
        raise SmartTalkError(f"Need at least 10 rows for an 80/10/10 split, got {n_rows}")
    indices = list(range(n_rows))
    random.Random(seed).shuffle(indices)
    train_end = int(n_rows * 0.8)
    validation_end = int(n_rows * 0.9)
    return SplitIndices(
        train=indices[:train_end],
        validation=indices[train_end:validation_end],
        test=indices[validation_end:],
    )


def split_overlap_audit(rows: Sequence[SmartTalkRow], splits: SplitIndices) -> Dict[str, Any]:
    train_inputs = {rows[idx].input_text for idx in splits.train}
    validation_inputs = {rows[idx].input_text for idx in splits.validation}
    test_seen_flags = [rows[idx].input_text in train_inputs for idx in splits.test]
    test_seen = sum(test_seen_flags)
    test_unseen = len(test_seen_flags) - test_seen
    duplicate_test_row_pairs_in_train = 0
    train_pairs = {(rows[idx].input_text, rows[idx].canonical_output) for idx in splits.train}
    for idx in splits.test:
        if (rows[idx].input_text, rows[idx].canonical_output) in train_pairs:
            duplicate_test_row_pairs_in_train += 1

    return {
        "seed": SEED,
        "split_strategy": "shuffle all row indices with Python random.Random(seed), then slice 80/10/10",
        "train_rows": len(splits.train),
        "validation_rows": len(splits.validation),
        "test_rows": len(splits.test),
        "train_unique_inputs": len(train_inputs),
        "validation_unique_inputs": len(validation_inputs),
        "test_unique_inputs": len({rows[idx].input_text for idx in splits.test}),
        "test_inputs_seen_in_train": test_seen,
        "test_inputs_unseen_in_train": test_unseen,
        "test_row_pairs_present_in_train": duplicate_test_row_pairs_in_train,
        "validation_inputs_seen_in_train": sum(1 for idx in splits.validation if rows[idx].input_text in train_inputs),
    }


def save_split_artifacts(output_dir: Path, splits: SplitIndices, overlap: Dict[str, Any]) -> None:
    split_dir = output_dir / "splits"
    write_json(split_dir / "train_indices.json", splits.train)
    write_json(split_dir / "validation_indices.json", splits.validation)
    write_json(split_dir / "test_indices.json", splits.test)
    write_json(output_dir / "split_overlap_audit.json", overlap)


def select_indices(indices: Sequence[int], limit: Optional[int]) -> List[int]:
    if limit is None:
        return list(indices)
    return list(indices[: max(0, limit)])


def console_progress_iter(iterable: Iterable[Any], total: Optional[int], desc: str) -> Iterable[Any]:
    step = max(1, (total or 100) // 100)
    for count, item in enumerate(iterable, start=1):
        if total and (count == 1 or count == total or count % step == 0):
            print(f"[{desc}] {count}/{total}", flush=True)
        yield item


def progress_iter(iterable: Iterable[Any], total: Optional[int], desc: str) -> Iterable[Any]:
    try:
        from tqdm.auto import tqdm

        return tqdm(iterable, total=total, desc=desc, dynamic_ncols=True, leave=True)
    except Exception:
        return console_progress_iter(iterable, total, desc)


def install_deps_and_restart(argv: Sequence[str]) -> None:
    packages = [
        "transformers>=4.44.0",
        "peft>=0.12.0",
        "datasets>=2.20.0",
        "accelerate>=0.33.0",
        "bitsandbytes>=0.43.3",
        "huggingface_hub>=0.24.0",
        "safetensors>=0.4.3",
        "sentencepiece>=0.2.0",
        "protobuf>=4.25.0",
        "unsloth",
        "tqdm>=4.66.0",
    ]
    print("Installing ML dependencies while leaving PyTorch untouched:")
    for package in packages:
        print(f"  - {package}")
    if package_version("torch") is None:
        raise SmartTalkError("--install-deps will not install PyTorch. Select a CUDA runtime with PyTorch first, then rerun this script.")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-U", "--upgrade-strategy", "only-if-needed", *packages])
    next_argv = [arg for arg in argv if arg != "--install-deps"]
    print("Restarting script after dependency installation.")
    os.execv(sys.executable, [sys.executable, str(Path(__file__).resolve()), *next_argv])


def module_available(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def resolve_backend(requested_backend: str) -> str:
    if requested_backend == "auto":
        return "unsloth" if module_available("unsloth") else "hf"
    return requested_backend


def import_required_ml(backend: str) -> Dict[str, Any]:
    missing = []
    if backend == "unsloth":
        modules = {
            "torch": "torch",
            "unsloth": "unsloth",
            "transformers": "transformers",
            "peft": "peft",
            "datasets": "datasets",
            "accelerate": "accelerate",
            "bitsandbytes": "bitsandbytes",
            "huggingface_hub": "huggingface_hub",
        }
    else:
        modules = {
            "torch": "torch",
            "transformers": "transformers",
            "peft": "peft",
            "datasets": "datasets",
            "accelerate": "accelerate",
            "bitsandbytes": "bitsandbytes",
            "huggingface_hub": "huggingface_hub",
        }
    imported: Dict[str, Any] = {}
    for import_name, package_name in modules.items():
        try:
            imported[import_name] = __import__(import_name)
        except Exception as exc:
            missing.append(f"{package_name} ({exc})")
    if missing:
        raise SmartTalkError(
            "Missing or unusable ML dependencies. Re-run with --install-deps. "
            "PyTorch is intentionally not installed by this script. Missing: " + "; ".join(missing)
        )
    return imported

def package_version(package: str) -> Optional[str]:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return None


def hf_token_from_env() -> Optional[str]:
    return os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")


def platform_report(torch_module: Optional[Any] = None, precision: Optional[str] = None) -> Dict[str, Any]:
    report: Dict[str, Any] = {
        "created_at": now_utc(),
        "platform": platform.platform(),
        "system": platform.system(),
        "release": platform.release(),
        "machine": platform.machine(),
        "python_version": platform.python_version(),
        "python_executable": sys.executable,
        "packages": {
            "torch": package_version("torch"),
            "transformers": package_version("transformers"),
            "peft": package_version("peft"),
            "datasets": package_version("datasets"),
            "accelerate": package_version("accelerate"),
            "bitsandbytes": package_version("bitsandbytes"),
            "huggingface_hub": package_version("huggingface_hub"),
            "safetensors": package_version("safetensors"),
            "unsloth": package_version("unsloth"),
            "unsloth_zoo": package_version("unsloth_zoo"),        },
        "precision": precision,
    }
    if torch_module is None:
        return report

    cuda_available = bool(torch_module.cuda.is_available())
    cuda_report: Dict[str, Any] = {
        "available": cuda_available,
        "torch_cuda_version": getattr(torch_module.version, "cuda", None),
        "device_count": torch_module.cuda.device_count() if cuda_available else 0,
    }
    if cuda_available:
        devices = []
        for idx in range(torch_module.cuda.device_count()):
            props = torch_module.cuda.get_device_properties(idx)
            devices.append(
                {
                    "index": idx,
                    "name": torch_module.cuda.get_device_name(idx),
                    "capability": list(torch_module.cuda.get_device_capability(idx)),
                    "total_memory_gb": round(props.total_memory / (1024**3), 3),
                }
            )
        cuda_report["devices"] = devices
        cuda_report["bf16_supported"] = bool(torch_module.cuda.is_bf16_supported())
    report["cuda"] = cuda_report
    return report


def require_cuda(torch_module: Any) -> None:
    if not torch_module.cuda.is_available():
        raise SmartTalkError(
            "CUDA is required for bitsandbytes 4-bit QLoRA training/evaluation. "
            "Run --analyze-only on CPU, or use a CUDA runtime such as Kaggle/Colab GPU."
        )


def choose_compute_dtype(torch_module: Any) -> Tuple[Any, str]:
    if torch_module.cuda.is_bf16_supported():
        return torch_module.bfloat16, "bf16"
    return torch_module.float16, "fp16"


def resolve_model_revision(model_name: str, revision: str, token: Optional[str]) -> Dict[str, Any]:
    try:
        from huggingface_hub import HfApi

        info = HfApi().model_info(model_name, revision=revision, token=token)
        return {
            "model_name": model_name,
            "requested_revision": revision,
            "resolved_revision": info.sha,
            "resolution_error": None,
        }
    except Exception as exc:
        return {
            "model_name": model_name,
            "requested_revision": revision,
            "resolved_revision": None,
            "resolution_error": str(exc),
        }


def load_tokenizer(model_name: str, revision: str, token: Optional[str]) -> Any:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        revision=revision,
        token=token,
        trust_remote_code=False,
    )
    if tokenizer.eos_token is None:
        raise SmartTalkError("Tokenizer has no EOS token; cannot append EOS to targets.")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return tokenizer


def encode_prompt_and_target(tokenizer: Any, row: SmartTalkRow, max_length: int) -> Dict[str, List[int]]:
    prompt = make_prompt(row.input_text)
    target = row.canonical_output + tokenizer.eos_token
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    target_ids = tokenizer(target, add_special_tokens=False)["input_ids"]
    input_ids = prompt_ids + target_ids
    if len(input_ids) > max_length:
        raise SmartTalkError(
            "Example exceeds max length without target truncation: "
            f"row_id={row.row_id}, prompt_tokens={len(prompt_ids)}, "
            f"target_tokens={len(target_ids)}, combined={len(input_ids)}, max_length={max_length}. "
            "Increase --max-length or inspect the row."
        )
    return {
        "input_ids": input_ids,
        "attention_mask": [1] * len(input_ids),
        "labels": [-100] * len(prompt_ids) + target_ids,
    }


def token_length_audit(
    rows: Sequence[SmartTalkRow],
    tokenizer: Any,
    max_length: int,
) -> Dict[str, Any]:
    prompt_lengths: List[int] = []
    target_lengths: List[int] = []
    combined_lengths: List[int] = []
    too_long: List[Dict[str, Any]] = []

    for row in rows:
        prompt_len = len(tokenizer(make_prompt(row.input_text), add_special_tokens=False)["input_ids"])
        target_len = len(tokenizer(row.canonical_output + tokenizer.eos_token, add_special_tokens=False)["input_ids"])
        combined = prompt_len + target_len
        prompt_lengths.append(prompt_len)
        target_lengths.append(target_len)
        combined_lengths.append(combined)
        if combined > max_length and len(too_long) < 25:
            too_long.append(
                {
                    "row_id": row.row_id,
                    "prompt_tokens": prompt_len,
                    "target_tokens": target_len,
                    "combined_tokens": combined,
                    "input": row.input_text[:500],
                    "target": row.canonical_output[:500],
                }
            )

    def summary(values: Sequence[int]) -> Dict[str, Any]:
        if not values:
            return {"min": 0, "max": 0, "mean": 0.0, "p50": 0, "p95": 0}
        sorted_values = sorted(values)
        p95_idx = min(len(sorted_values) - 1, math.ceil(0.95 * len(sorted_values)) - 1)
        return {
            "min": min(values),
            "max": max(values),
            "mean": round(statistics.fmean(values), 3),
            "p50": int(statistics.median(values)),
            "p95": sorted_values[p95_idx],
        }

    return {
        "max_length": max_length,
        "prompt_tokens": summary(prompt_lengths),
        "target_tokens": summary(target_lengths),
        "combined_tokens": summary(combined_lengths),
        "examples_exceeding_max_length_count": sum(1 for value in combined_lengths if value > max_length),
        "examples_exceeding_max_length": too_long,
    }


class CausalLMDataCollator:
    def __init__(self, tokenizer: Any, pad_to_multiple_of: int = 8) -> None:
        self.tokenizer = tokenizer
        self.pad_to_multiple_of = pad_to_multiple_of

    def __call__(self, features: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
        import torch

        max_len = max(len(feature["input_ids"]) for feature in features)
        if self.pad_to_multiple_of:
            max_len = int(math.ceil(max_len / self.pad_to_multiple_of) * self.pad_to_multiple_of)

        batch_input_ids: List[List[int]] = []
        batch_attention: List[List[int]] = []
        batch_labels: List[List[int]] = []
        for feature in features:
            pad_len = max_len - len(feature["input_ids"])
            batch_input_ids.append(feature["input_ids"] + [self.tokenizer.pad_token_id] * pad_len)
            batch_attention.append(feature["attention_mask"] + [0] * pad_len)
            batch_labels.append(feature["labels"] + [-100] * pad_len)
        return {
            "input_ids": torch.tensor(batch_input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(batch_attention, dtype=torch.long),
            "labels": torch.tensor(batch_labels, dtype=torch.long),
        }


def build_hf_dataset(
    rows: Sequence[SmartTalkRow],
    indices: Sequence[int],
    tokenizer: Any,
    max_length: int,
    desc: str = "Encoding dataset",
) -> Any:
    from datasets import Dataset

    iterator = progress_iter(indices, total=len(indices), desc=desc)
    records = [encode_prompt_and_target(tokenizer, rows[idx], max_length) for idx in iterator]
    return Dataset.from_list(records)


def ensure_masked_prompt_labels(rows: Sequence[SmartTalkRow], indices: Sequence[int], tokenizer: Any, max_length: int) -> Dict[str, Any]:
    if not indices:
        raise SmartTalkError("No training rows selected; cannot validate masked labels.")
    encoded = encode_prompt_and_target(tokenizer, rows[indices[0]], max_length)
    prompt_len = len(tokenizer(make_prompt(rows[indices[0]].input_text), add_special_tokens=False)["input_ids"])
    prompt_masked = all(value == -100 for value in encoded["labels"][:prompt_len])
    response_supervised = all(value != -100 for value in encoded["labels"][prompt_len:])
    return {
        "row_id": rows[indices[0]].row_id,
        "prompt_tokens": prompt_len,
        "response_tokens": len(encoded["labels"]) - prompt_len,
        "prompt_labels_all_masked": prompt_masked,
        "response_labels_all_supervised": response_supervised,
    }


def load_quantized_model(
    model_name: str,
    revision: str,
    token: Optional[str],
    compute_dtype: Any,
) -> Any:
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig

    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=compute_dtype,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        revision=revision,
        token=token,
        trust_remote_code=False,
        quantization_config=quant_config,
        device_map="auto",
    )
    model.config.use_cache = False
    return model


def load_unsloth_quantized_model(
    model_name: str,
    revision: str,
    token: Optional[str],
    max_length: int,
    compute_dtype: Any,
) -> Tuple[Any, Any]:
    from unsloth import FastLanguageModel

    kwargs: Dict[str, Any] = {
        "model_name": model_name,
        "max_seq_length": max_length,
        "dtype": compute_dtype,
        "load_in_4bit": True,
        "token": token,
    }
    if revision != DEFAULT_MODEL_REVISION:
        kwargs["revision"] = revision
    try:
        model, tokenizer = FastLanguageModel.from_pretrained(**kwargs)
    except TypeError:
        kwargs.pop("revision", None)
        model, tokenizer = FastLanguageModel.from_pretrained(**kwargs)

    if tokenizer.eos_token is None:
        raise SmartTalkError("Tokenizer has no EOS token; cannot append EOS to targets.")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    model.config.use_cache = False
    return model, tokenizer

def module_leaf_counts(model: Any) -> Counter[str]:
    counts: Counter[str] = Counter()
    for name, _module in model.named_modules():
        if not name:
            continue
        counts[name.rsplit(".", 1)[-1]] += 1
    return counts


def available_lora_targets(model: Any, targets: Sequence[str]) -> Dict[str, Any]:
    leaf_counts = module_leaf_counts(model)
    found = {target: int(leaf_counts.get(target, 0)) for target in targets}
    missing = [target for target, count in found.items() if count == 0]
    return {"target_modules": list(targets), "module_counts": found, "missing_targets": missing}


def resolve_lora_targets(model: Any, requested: str) -> Dict[str, Any]:
    requested = (requested or "auto").strip()
    if requested.lower() in {"paper", "default"}:
        report = available_lora_targets(model, LORA_TARGET_MODULES)
        if report["missing_targets"]:
            raise SmartTalkError(
                "The base model is missing the paper LoRA target modules: "
                + ", ".join(report["missing_targets"])
                + ". Use --lora-target-modules auto for other paper model families."
            )
        return {**report, "selection_mode": requested, "selected_profile": "paper_qkvo_gate_up_down"}

    if requested.lower() not in {"auto", "paper-auto"}:
        targets = [part.strip() for part in requested.split(",") if part.strip()]
        if not targets:
            raise SmartTalkError("--lora-target-modules must be 'auto', 'paper', or a comma-separated module list")
        report = available_lora_targets(model, targets)
        if report["missing_targets"]:
            raise SmartTalkError(
                "The base model is missing requested LoRA target modules: "
                + ", ".join(report["missing_targets"])
            )
        return {**report, "selection_mode": "manual", "selected_profile": "manual"}

    profile_reports: List[Dict[str, Any]] = []
    for profile_name, targets in LORA_TARGET_PROFILES.items():
        report = available_lora_targets(model, targets)
        present_count = len(targets) - len(report["missing_targets"])
        profile_reports.append(
            {
                **report,
                "selection_mode": "auto",
                "selected_profile": profile_name,
                "present_target_count": present_count,
            }
        )

    complete_profiles = [report for report in profile_reports if not report["missing_targets"]]
    if complete_profiles:
        complete_profiles.sort(key=lambda report: (-len(report["target_modules"]), report["selected_profile"]))
        return complete_profiles[0]

    generic_report = available_lora_targets(model, GENERIC_LORA_TARGET_NAMES)
    selected = [target for target, count in generic_report["module_counts"].items() if count > 0]
    if selected:
        report = available_lora_targets(model, selected)
        return {
            **report,
            "selection_mode": "auto_generic",
            "selected_profile": "available_projection_names",
            "profile_attempts": profile_reports,
        }

    raise SmartTalkError(
        "Could not find known attention/MLP projection module names for this model. "
        "Pass --lora-target-modules with a comma-separated list after inspecting model.named_modules()."
    )


def apply_lora(model: Any, lora_target_modules: str, gradient_checkpointing: bool = True) -> Tuple[Any, Dict[str, Any]]:
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    target_report = resolve_lora_targets(model, lora_target_modules)
    selected_targets = list(target_report["target_modules"])

    model = prepare_model_for_kbit_training(
        model,
        use_gradient_checkpointing=gradient_checkpointing,
    )
    if gradient_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    config = LoraConfig(
        r=32,
        lora_alpha=64,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=selected_targets,
    )
    model = get_peft_model(model, config)
    trainable, total = count_trainable_parameters(model)
    report = {
        **target_report,
        "paper_qlora_models": PAPER_QLORA_MODELS,
        "lora_rank": 32,
        "lora_alpha": 64,
        "lora_dropout": 0.05,
        "lora_bias": "none",
        "gradient_checkpointing": gradient_checkpointing,
        "trainable_parameters": trainable,
        "total_parameters": total,
        "trainable_percent": round(100 * trainable / total, 6) if total else 0.0,
    }
    return model, report


def apply_lora_unsloth(model: Any, lora_target_modules: str, max_length: int) -> Tuple[Any, Dict[str, Any]]:
    from unsloth import FastLanguageModel

    target_report = resolve_lora_targets(model, lora_target_modules)
    selected_targets = list(target_report["target_modules"])
    model = FastLanguageModel.get_peft_model(
        model,
        r=32,
        target_modules=selected_targets,
        lora_alpha=64,
        lora_dropout=0.05,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=SEED,
        max_seq_length=max_length,
        use_rslora=False,
        loftq_config=None,
    )
    trainable, total = count_trainable_parameters(model)
    report = {
        "backend": "unsloth",
        **target_report,
        "paper_qlora_models": PAPER_QLORA_MODELS,
        "lora_rank": 32,
        "lora_alpha": 64,
        "lora_dropout": 0.05,
        "lora_bias": "none",
        "gradient_checkpointing": "unsloth",
        "trainable_parameters": trainable,
        "total_parameters": total,
        "trainable_percent": round(100 * trainable / total, 6) if total else 0.0,
    }
    return model, report


def count_trainable_parameters(model: Any) -> Tuple[int, int]:
    trainable = 0
    total = 0
    for param in model.parameters():
        count = param.numel()
        total += count
        if param.requires_grad:
            trainable += count
    return trainable, total


def select_optimizer(torch_module: Any) -> Tuple[str, Dict[str, Any]]:
    report: Dict[str, Any] = {"requested_optimizer": "adamw_torch_fused"}
    try:
        dummy = torch_module.zeros(1, device="cuda", requires_grad=True)
        optimizer = torch_module.optim.AdamW([dummy], lr=1e-3, fused=True)
        optimizer.zero_grad(set_to_none=True)
        del optimizer
        del dummy
        report["selected_optimizer"] = "adamw_torch_fused"
        report["fallback_reason"] = None
        return "adamw_torch_fused", report
    except Exception as exc:
        report["selected_optimizer"] = "adamw_torch"
        report["fallback_reason"] = str(exc)
        return "adamw_torch", report


def training_args_kwargs(args: argparse.Namespace, output_dir: Path, optim_name: str) -> Dict[str, Any]:
    import inspect
    from transformers import TrainingArguments

    eval_key = "eval_strategy" if "eval_strategy" in inspect.signature(TrainingArguments.__init__).parameters else "evaluation_strategy"

    common: Dict[str, Any] = {
        "output_dir": str(output_dir),
        "overwrite_output_dir": False,
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "per_device_eval_batch_size": 1,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "learning_rate": 2e-4,
        "weight_decay": 0.01,
        "lr_scheduler_type": "cosine",
        "warmup_ratio": 0.05,
        "logging_steps": 10 if not args.smoke_test else 1,
        "save_total_limit": 3,
        "optim": optim_name,
        "fp16": args.precision == "fp16",
        "bf16": args.precision == "bf16",
        "gradient_checkpointing": True,
        "remove_unused_columns": False,
        "report_to": [],
        "seed": SEED,
        "data_seed": SEED,
        "disable_tqdm": False,
        "logging_first_step": True,
    }

    if args.smoke_test:
        common.update(
            {
                "max_steps": 1,
                "num_train_epochs": 1,
                "save_strategy": "steps",
                "save_steps": 1,
            }
        )
        if args.finetune_only:
            common[eval_key] = "no"
        else:
            common[eval_key] = "steps"
            common["eval_steps"] = 1
    else:
        common.update(
            {
                "num_train_epochs": 3,
                "save_strategy": "epoch",
            }
        )
        common[eval_key] = "no" if args.finetune_only else "epoch"

    return common


def resolve_resume_checkpoint(value: Optional[str], trainer_output_dir: Path) -> Optional[str]:
    if value is None:
        return None
    if value != "auto":
        path = Path(value).expanduser().resolve()
        if not path.exists():
            raise SmartTalkError(f"--resume-from-checkpoint path does not exist: {path}")
        return str(path)

    checkpoints = sorted(
        [path for path in trainer_output_dir.glob("checkpoint-*") if path.is_dir()],
        key=lambda path: int(path.name.split("-")[-1]) if path.name.split("-")[-1].isdigit() else -1,
    )
    if not checkpoints:
        raise SmartTalkError(
            f"--resume-from-checkpoint auto requested, but no checkpoint-* directories exist in {trainer_output_dir}"
        )
    return str(checkpoints[-1])


def first_model_device(model: Any) -> Any:
    import torch

    for param in model.parameters():
        if param.device.type != "meta":
            return param.device
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def parse_generated_json(text: str) -> Tuple[Optional[Any], Optional[str]]:
    stripped = text.strip()
    if not stripped:
        return None, "empty generation"
    try:
        return strict_json_loads(stripped), None
    except Exception as exc:
        return None, str(exc)


def percentile(values: Sequence[float], pct: float) -> Optional[float]:
    if not values:
        return None
    sorted_values = sorted(values)
    idx = min(len(sorted_values) - 1, math.ceil(pct * len(sorted_values)) - 1)
    return sorted_values[idx]


def evaluate_generation(
    model: Any,
    tokenizer: Any,
    rows: Sequence[SmartTalkRow],
    test_indices: Sequence[int],
    train_inputs: set[str],
    output_dir: Path,
    run_name: str,
    eval_limit: Optional[int],
    max_new_tokens: int,
    torch_module: Any,
) -> Dict[str, Any]:
    selected_indices = select_indices(test_indices, eval_limit)
    if not selected_indices:
        raise SmartTalkError("Evaluation selected zero rows.")

    model.eval()
    device = first_model_device(model)

    canonical_matches = 0
    raw_matches = 0
    valid_json = 0
    schema_valid = 0
    seen_total = 0
    seen_matches = 0
    unseen_total = 0
    unseen_matches = 0
    generated_tokens_total = 0
    latencies_ms: List[float] = []
    failures: List[Dict[str, Any]] = []

    generation_params = {
        "max_new_tokens": max_new_tokens,
        "do_sample": False,
        "num_beams": 1,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "batch_size": 1,
    }

    progress = progress_iter(selected_indices, total=len(selected_indices), desc=f"{run_name} generation")
    for position, row_idx in enumerate(progress):
        row = rows[row_idx]
        prompt = make_prompt(row.input_text)
        encoded = tokenizer(prompt, add_special_tokens=False, return_tensors="pt")
        encoded = {key: value.to(device) for key, value in encoded.items()}
        prompt_tokens = int(encoded["input_ids"].shape[-1])

        if torch_module.cuda.is_available():
            torch_module.cuda.synchronize()
        start = time.perf_counter()
        with torch_module.inference_mode():
            generated = model.generate(
                **encoded,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                num_beams=1,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        if torch_module.cuda.is_available():
            torch_module.cuda.synchronize()
        latency_ms = (time.perf_counter() - start) * 1000
        latencies_ms.append(latency_ms)

        new_token_ids = generated[0, prompt_tokens:]
        generated_tokens = int(new_token_ids.shape[-1])
        generated_tokens_total += generated_tokens
        generated_text = tokenizer.decode(new_token_ids, skip_special_tokens=True).strip()

        parsed, parse_error = parse_generated_json(generated_text)
        is_valid_json = parsed is not None and parse_error is None
        if is_valid_json:
            valid_json += 1
        is_schema_valid = False
        schema_error = None
        if is_valid_json:
            is_schema_valid, schema_error, _ = schema_status(parsed)
            if is_schema_valid:
                schema_valid += 1

        canonical_match = bool(is_valid_json and parsed == row.target_obj)
        raw_match = generated_text == row.output_text
        canonical_matches += int(canonical_match)
        raw_matches += int(raw_match)

        seen = row.input_text in train_inputs
        if seen:
            seen_total += 1
            seen_matches += int(canonical_match)
        else:
            unseen_total += 1
            unseen_matches += int(canonical_match)

        if not canonical_match and len(failures) < 100:
            failures.append(
                {
                    "eval_position": position,
                    "row_id": row.row_id,
                    "seen_in_train": seen,
                    "input": row.input_text,
                    "expected": row.canonical_output,
                    "generated": generated_text,
                    "valid_json": is_valid_json,
                    "schema_valid": is_schema_valid,
                    "parse_error": parse_error,
                    "schema_error": schema_error,
                    "latency_ms": round(latency_ms, 3),
                }
            )

        if hasattr(progress, "set_postfix") and (position + 1) % 10 == 0:
            progress.set_postfix(
                accuracy=f"{canonical_matches / (position + 1):.4f}",
                valid_json=f"{valid_json / (position + 1):.4f}",
            )
        elif (position + 1) % 100 == 0:
            print(
                f"[{run_name}] evaluated {position + 1}/{len(selected_indices)} rows; "
                f"canonical_accuracy={canonical_matches / (position + 1):.4f}",
                flush=True,
            )
    total = len(selected_indices)
    total_latency_s = sum(latencies_ms) / 1000
    report = {
        "run_name": run_name,
        "created_at": now_utc(),
        "evaluated_rows": total,
        "full_test_rows": len(test_indices),
        "eval_limit": eval_limit,
        "generation_params": generation_params,
        "headline_metric": "canonical_json_exact_match_accuracy",
        "canonical_json_exact_match_accuracy": canonical_matches / total,
        "raw_text_exact_match_accuracy": raw_matches / total,
        "valid_json_rate": valid_json / total,
        "schema_valid_rate": schema_valid / total,
        "seen_inputs": {
            "rows": seen_total,
            "canonical_json_exact_match_accuracy": seen_matches / seen_total if seen_total else None,
        },
        "unseen_inputs": {
            "rows": unseen_total,
            "canonical_json_exact_match_accuracy": unseen_matches / unseen_total if unseen_total else None,
        },
        "latency_ms": {
            "average": statistics.fmean(latencies_ms) if latencies_ms else None,
            "p50": statistics.median(latencies_ms) if latencies_ms else None,
            "p95": percentile(latencies_ms, 0.95),
            "min": min(latencies_ms) if latencies_ms else None,
            "max": max(latencies_ms) if latencies_ms else None,
        },
        "generated_tokens": {
            "total": generated_tokens_total,
            "tokens_per_second": generated_tokens_total / total_latency_s if total_latency_s > 0 else None,
        },
        "failure_examples_path": str(output_dir / f"{run_name}_failure_examples.json"),
    }
    write_json(output_dir / f"{run_name}_report.json", report)
    write_json(output_dir / f"{run_name}_failure_examples.json", failures)
    return report


def paper_comparison_report(
    baseline_report: Optional[Dict[str, Any]],
    finetuned_report: Dict[str, Any],
    platform_info: Dict[str, Any],
    output_dir: Path,
) -> Dict[str, Any]:
    def latency_average(report: Optional[Dict[str, Any]]) -> Optional[float]:
        if not report:
            return None
        return report.get("latency_ms", {}).get("average")

    def accuracy(report: Optional[Dict[str, Any]]) -> Optional[float]:
        if not report:
            return None
        return report.get("canonical_json_exact_match_accuracy")

    gpu_names = [
        device.get("name", "")
        for device in platform_info.get("cuda", {}).get("devices", [])
        if isinstance(device, dict)
    ]
    has_t4 = any("T4" in name for name in gpu_names)
    baseline_accuracy = accuracy(baseline_report)
    finetuned_accuracy = accuracy(finetuned_report)
    finetuned_latency = latency_average(finetuned_report)

    comparison = {
        "created_at": now_utc(),
        "paper": {
            "baseline_accuracy": PAPER_BASELINE_ACCURACY,
            "finetuned_accuracy": PAPER_FINETUNED_ACCURACY,
            "t4_latency_ms": PAPER_T4_LATENCY_MS,
        },
        "this_run": {
            "baseline_accuracy": baseline_accuracy,
            "finetuned_accuracy": finetuned_accuracy,
            "finetuned_average_latency_ms": finetuned_latency,
            "gpu_names": gpu_names,
        },
        "deltas": {
            "baseline_accuracy_delta_percentage_points": (
                (baseline_accuracy - PAPER_BASELINE_ACCURACY) * 100 if baseline_accuracy is not None else None
            ),
            "finetuned_accuracy_delta_percentage_points": (
                (finetuned_accuracy - PAPER_FINETUNED_ACCURACY) * 100 if finetuned_accuracy is not None else None
            ),
            "latency_ms_delta_vs_t4": (
                finetuned_latency - PAPER_T4_LATENCY_MS if finetuned_latency is not None else None
            ),
        },
        "latency_comparability_note": (
            "GPU includes T4; latency is still software/runtime dependent."
            if has_t4
            else "GPU is not reported as T4, so latency should not be claimed as directly comparable to the paper."
        ),
    }
    write_json(output_dir / "paper_comparison_report.json", comparison)
    return comparison


def print_analysis_summary(audit: Dict[str, Any], overlap: Dict[str, Any], output_dir: Path) -> None:
    print("Dataset audit complete.")
    print(f"  rows: {audit['total_rows']}")
    print(f"  non-empty pairs: {audit['non_empty_pairs']}")
    print(f"  valid JSON outputs: {audit['valid_json_outputs']}")
    print(f"  sorted JSON outputs: {audit['sorted_json_outputs']}")
    print(f"  duplicate rows: {audit['duplicate_rows']}")
    print(f"  unique inputs: {audit['unique_inputs']}")
    print(f"  conflicting inputs: {audit['inputs_with_conflicting_targets']}")
    print(f"  avg input words: {audit['avg_input_words']}")
    print(f"  avg device actions: {audit['avg_device_actions']}")
    print(f"  rooms/device types: {audit['room_count']}/{audit['device_type_count']}")
    print(f"  max device actions: {audit['max_device_actions']}")
    print(
        "  split rows: "
        f"train={overlap['train_rows']}, validation={overlap['validation_rows']}, test={overlap['test_rows']}"
    )
    print(
        "  test overlap: "
        f"seen={overlap['test_inputs_seen_in_train']}, unseen={overlap['test_inputs_unseen_in_train']}"
    )
    print(f"Reports written to: {output_dir}")


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit and reproduce SmartTalk QLoRA fine-tuning for SmolLM2-360M."
    )
    parser.add_argument("--dataset-path", default=None, help="Path to the SmartTalk CSV.")
    parser.add_argument("--output-dir", default=None, help="Directory for reports, checkpoints, and adapter.")
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME, help="Hugging Face model id.")
    parser.add_argument("--model-revision", default=DEFAULT_MODEL_REVISION, help="Model branch, tag, or commit.")
    parser.add_argument(
        "--backend",
        choices=["auto", "hf", "unsloth"],
        default="auto",
        help="Training backend. auto uses Unsloth when installed, otherwise Hugging Face/PEFT.",
    )
    parser.add_argument(
        "--lora-target-modules",
        default="auto",
        help="LoRA targets: auto, paper, or a comma-separated module list. Auto covers the model families in the paper.",
    )
    parser.add_argument(
        "--per-device-train-batch-size",
        type=int,
        default=4,
        help="Micro-batch size. Gradient accumulation is recomputed to preserve effective batch 32.",
    )
    parser.add_argument("--eval-limit", type=int, default=None, help="Limit generation evaluation rows.")
    parser.add_argument("--skip-baseline", action="store_true", help="Skip untouched baseline generation evaluation.")
    parser.add_argument(
        "--finetune-only",
        action="store_true",
        default=True,
        help="Train and save the LoRA adapter only; skip baseline, final generation evaluation, and paper comparison. This is the default.",
    )
    parser.add_argument(
        "--evaluate-after-training",
        dest="finetune_only",
        action="store_false",
        help="Run final generation evaluation and paper comparison after training. Also runs baseline unless --skip-baseline is set.",
    )
    parser.add_argument(
        "--resume-from-checkpoint",
        nargs="?",
        const="auto",
        default=None,
        help="Resume Trainer from a checkpoint path, or auto-detect the latest checkpoint.",
    )
    parser.add_argument("--analyze-only", action="store_true", help="Only audit CSV and write split artifacts.")
    parser.add_argument("--smoke-test", action="store_true", help="Train a deterministic tiny subset for one optimizer step.")
    parser.add_argument("--install-deps", action="store_true", help="Install ML dependencies except PyTorch, then restart.")
    parser.add_argument("--max-length", type=int, default=MAX_LENGTH, help=argparse.SUPPRESS)
    parser.add_argument("--max-new-tokens", type=int, default=MAX_NEW_TOKENS, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)

    if args.per_device_train_batch_size <= 0:
        raise SmartTalkError("--per-device-train-batch-size must be positive")
    if EFFECTIVE_BATCH_SIZE % args.per_device_train_batch_size != 0:
        raise SmartTalkError(
            "--per-device-train-batch-size must divide effective batch size 32. "
            f"Got {args.per_device_train_batch_size}."
        )
    args.gradient_accumulation_steps = EFFECTIVE_BATCH_SIZE // args.per_device_train_batch_size
    if args.eval_limit is not None and args.eval_limit <= 0:
        raise SmartTalkError("--eval-limit must be positive when provided")
    if args.max_length <= 0:
        raise SmartTalkError("--max-length must be positive")
    if args.max_new_tokens <= 0:
        raise SmartTalkError("--max-new-tokens must be positive")
    args.backend = resolve_backend(args.backend)
    if args.finetune_only:
        args.skip_baseline = True
    return args


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    if args.install_deps:
        install_deps_and_restart(sys.argv[1:] if argv is None else argv)

    dataset_path, discovery_report = discover_dataset_path(args.dataset_path)
    output_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else default_output_dir().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    rows, audit = audit_csv(dataset_path)
    audit["discovery"] = discovery_report
    write_json(output_dir / "dataset_audit.json", audit)
    validate_audit_or_fail(audit)

    splits = create_split_indices(len(rows), seed=SEED)
    overlap = split_overlap_audit(rows, splits)
    write_json(output_dir / "dataset_audit.json", audit)
    save_split_artifacts(output_dir, splits, overlap)

    if args.analyze_only:
        print_analysis_summary(audit, overlap, output_dir)
        return 0

    ml_modules = import_required_ml(args.backend)
    torch_module = ml_modules["torch"]
    require_cuda(torch_module)
    torch_module.manual_seed(SEED)
    if torch_module.cuda.is_available():
        torch_module.cuda.manual_seed_all(SEED)

    compute_dtype, precision = choose_compute_dtype(torch_module)
    args.precision = precision
    token = hf_token_from_env()
    model_revision_report = resolve_model_revision(args.model_name, args.model_revision, token)
    model = None
    if args.backend == "unsloth":
        print("Loading 4-bit base model with Unsloth.", flush=True)
        model, tokenizer = load_unsloth_quantized_model(
            args.model_name,
            args.model_revision,
            token,
            args.max_length,
            compute_dtype,
        )
    else:
        tokenizer = load_tokenizer(args.model_name, args.model_revision, token)
    token_audit = token_length_audit(rows, tokenizer, args.max_length)
    audit["token_lengths"] = token_audit
    write_json(output_dir / "dataset_audit.json", audit)
    if token_audit["examples_exceeding_max_length_count"] != 0:
        raise SmartTalkError(
            f"{token_audit['examples_exceeding_max_length_count']} examples exceed max length {args.max_length}; "
            "targets are not truncated by design. See dataset_audit.json."
        )

    train_indices = list(splits.train)
    validation_indices = list(splits.validation)
    test_indices = list(splits.test)
    effective_eval_limit = args.eval_limit
    if args.smoke_test:
        train_indices = train_indices[:SMOKE_TRAIN_ROWS]
        validation_indices = validation_indices[:SMOKE_VALIDATION_ROWS]
        test_indices = test_indices[:SMOKE_TEST_ROWS]
        effective_eval_limit = args.eval_limit or SMOKE_TEST_ROWS

    mask_report = ensure_masked_prompt_labels(rows, train_indices, tokenizer, args.max_length)
    platform_info = platform_report(torch_module, precision)
    optimizer_name, optimizer_report = select_optimizer(torch_module)

    run_config = {
        "created_at": now_utc(),
        "dataset_path": str(dataset_path),
        "output_dir": str(output_dir),
        "model": model_revision_report,
        "seed": SEED,
        "smoke_test": args.smoke_test,
        "analyze_only": args.analyze_only,
        "finetune_only": args.finetune_only,
        "skip_baseline": args.skip_baseline,
        "backend": args.backend,
        "precision": precision,
        "max_length": args.max_length,
        "generation": {
            "max_new_tokens": args.max_new_tokens,
            "do_sample": False,
            "num_beams": 1,
            "batch_size": 1,
        },
        "training": {
            "epochs": 1 if args.smoke_test else 3,
            "effective_batch_size": EFFECTIVE_BATCH_SIZE,
            "per_device_train_batch_size": args.per_device_train_batch_size,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "learning_rate": 2e-4,
            "lr_scheduler_type": "cosine",
            "warmup_ratio": 0.05,
            "weight_decay": 0.01,
            "gradient_checkpointing": True,
            "optimizer": optimizer_report,
        },
        "qlora": {
            "lora_target_modules_requested": args.lora_target_modules,
            "load_in_4bit": True,
            "bnb_4bit_quant_type": "nf4",
            "bnb_4bit_use_double_quant": True,
            "bnb_4bit_compute_dtype": precision,
        },
        "prompt_format": {
            "system_prompt": SYSTEM_PROMPT,
            "template": "### System:\\n{system_prompt}\\n\\n### Input:\\n{input}\\n\\n### Output:\\n",
            "completion": "{canonical_json}{eos_token}",
        },
        "masked_labels": mask_report,
        "platform": platform_info,
    }
    write_json(output_dir / "run_config.json", run_config)

    if model is None:
        print("Loading 4-bit base model with Hugging Face Transformers/bitsandbytes.", flush=True)
        model = load_quantized_model(args.model_name, args.model_revision, token, compute_dtype)

    train_inputs = {rows[idx].input_text for idx in splits.train}
    baseline_report = None
    if not args.skip_baseline:
        print("Evaluating untouched baseline.", flush=True)
        baseline_report = evaluate_generation(
            model=model,
            tokenizer=tokenizer,
            rows=rows,
            test_indices=test_indices,
            train_inputs=train_inputs,
            output_dir=output_dir,
            run_name="baseline",
            eval_limit=effective_eval_limit,
            max_new_tokens=args.max_new_tokens,
            torch_module=torch_module,
        )
    else:
        write_json(
            output_dir / "baseline_report.json",
            {"run_name": "baseline", "skipped": True, "created_at": now_utc()},
        )

    print("Applying LoRA adapter.", flush=True)
    if args.backend == "unsloth":
        model, lora_report = apply_lora_unsloth(model, args.lora_target_modules, args.max_length)
    else:
        model, lora_report = apply_lora(model, args.lora_target_modules, gradient_checkpointing=True)
    run_config["lora"] = lora_report
    write_json(output_dir / "run_config.json", run_config)

    trainer_output_dir = output_dir / "trainer"
    resume_checkpoint = resolve_resume_checkpoint(args.resume_from_checkpoint, trainer_output_dir)
    try:
        train_dataset = build_hf_dataset(rows, train_indices, tokenizer, args.max_length, desc="Encoding train dataset")
        validation_dataset = None if args.finetune_only else build_hf_dataset(rows, validation_indices, tokenizer, args.max_length, desc="Encoding validation dataset")
        from transformers import Trainer, TrainingArguments

        training_kwargs = training_args_kwargs(args, trainer_output_dir, optimizer_name)
        run_config["training_arguments"] = training_kwargs
        run_config["resume_from_checkpoint"] = resume_checkpoint
        write_json(output_dir / "run_config.json", run_config)

        trainer = Trainer(
            model=model,
            args=TrainingArguments(**training_kwargs),
            train_dataset=train_dataset,
            eval_dataset=validation_dataset,
            data_collator=CausalLMDataCollator(tokenizer),
        )

        print("Starting adapter training.", flush=True)
        train_result = trainer.train(resume_from_checkpoint=resume_checkpoint)
        train_metrics = dict(train_result.metrics)
        trainer.log_metrics("train", train_metrics)
        trainer.save_metrics("train", train_metrics)
        trainer.save_state()
    except Exception as exc:
        checkpoint_candidates = sorted(str(path) for path in trainer_output_dir.glob("checkpoint-*"))
        raise SmartTalkError(
            "Training failed. If this was an interrupted Colab/Kaggle run, retry with "
            f"--resume-from-checkpoint auto. Available checkpoints: {checkpoint_candidates}. "
            f"Original error: {exc}"
        ) from exc

    trainer_state = {}
    state_path = trainer_output_dir / "trainer_state.json"
    if state_path.exists():
        with state_path.open("r", encoding="utf-8") as f:
            trainer_state = json.load(f)
    write_json(output_dir / "training_history.json", trainer_state.get("log_history", []))

    adapter_dir = output_dir / "lora_adapter"
    adapter_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)
    print(f"Saved LoRA adapter to {adapter_dir}", flush=True)

    if args.finetune_only:
        summary = {
            "created_at": now_utc(),
            "run_name": "finetune_only",
            "adapter_dir": str(adapter_dir),
            "trainer_dir": str(trainer_output_dir),
            "backend": args.backend,
            "train_metrics": train_metrics,
            "generation_evaluation_skipped": True,
            "baseline_evaluation_skipped": True,
        }
        write_json(output_dir / "finetune_only_summary.json", summary)
        print("Fine-tuning complete; generation evaluation was skipped because --finetune-only is set.")
        print(f"  adapter: {adapter_dir}")
        print(f"  reports: {output_dir}")
        return 0

    print("Evaluating fine-tuned adapter.", flush=True)
    finetuned_report = evaluate_generation(
        model=model,
        tokenizer=tokenizer,
        rows=rows,
        test_indices=test_indices,
        train_inputs=train_inputs,
        output_dir=output_dir,
        run_name="finetuned",
        eval_limit=effective_eval_limit,
        max_new_tokens=args.max_new_tokens,
        torch_module=torch_module,
    )

    comparison = paper_comparison_report(
        baseline_report=baseline_report,
        finetuned_report=finetuned_report,
        platform_info=platform_info,
        output_dir=output_dir,
    )

    print("Run complete.")
    if baseline_report:
        print(f"  baseline canonical accuracy: {baseline_report['canonical_json_exact_match_accuracy']:.4f}")
    print(f"  fine-tuned canonical accuracy: {finetuned_report['canonical_json_exact_match_accuracy']:.4f}")
    print(f"  fine-tuned average latency ms: {finetuned_report['latency_ms']['average']:.3f}")
    print(
        "  paper deltas: "
        f"accuracy_pp={comparison['deltas']['finetuned_accuracy_delta_percentage_points']}, "
        f"latency_ms={comparison['deltas']['latency_ms_delta_vs_t4']}"
    )
    print(f"Reports written to: {output_dir}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SmartTalkError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2)
