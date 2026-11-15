#!/usr/bin/env python
"""Prune transformer attention heads and MLP channels.

Importance combines weight norms with mean calibration activation norms.
Pruning is uniform across layers; reloading requires head_dim support.
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import importlib
import json
import math
import os
import platform
import random
import statistics
import subprocess
import sys
import time
import torch 
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


SEED = 42
DEFAULT_MODEL_REVISION = "main"
PAPER_PRUNE_RATIO = 0.50
PAPER_T4_HARDWARE = "Google Colab NVIDIA T4 16GB"
PAPER_CALIBRATION_NOTE = (
    "The paper states structured pruning uses calibration activations, but it "
    "does not disclose calibration size, split seed, prompt, decoding config, "
    "model revisions, or quantization implementation."
)


@dataclass(frozen=True)
class PaperModel:
    key: str
    model_id: str
    label: str
    paper_original_b: float
    paper_pruned_b: float
    paper_original_accuracy: float
    paper_pruned_accuracy: float
    paper_original_latency_ms: float
    paper_pruned_latency_ms: float
    trust_remote_code: bool = False
    family_hint: str = "auto"
    gated: bool = False


PAPER_MODELS: Dict[str, PaperModel] = {
    "yi_1_5_6b": PaperModel(
        "yi_1_5_6b",
        "01-ai/Yi-1.5-6B-Chat",
        "Yi-1.5-6B-Chat",
        6.0,
        3.0,
        0.8176,
        0.8176,
        992.9,
        546.0,
        family_hint="llama",
    ),
    "codellama_7b": PaperModel(
        "codellama_7b",
        "codellama/CodeLlama-7b-Instruct-hf",
        "CodeLlama-7B-Instruct",
        7.0,
        3.5,
        0.8324,
        0.8321,
        1143.1,
        629.0,
        family_hint="llama",
    ),
    "internlm2_5_7b": PaperModel(
        "internlm2_5_7b",
        "internlm/internlm2_5-7b-chat",
        "InternLM2.5-7B-Chat",
        7.0,
        3.5,
        0.8416,
        0.8016,
        1125.5,
        619.0,
        trust_remote_code=True,
        family_hint="internlm",
    ),
    "mistral_7b_v0_3": PaperModel(
        "mistral_7b_v0_3",
        "mistralai/Mistral-7B-Instruct-v0.3",
        "Mistral-7B-v0.3-Instruct",
        7.24,
        3.62,
        0.9295,
        0.9095,
        1158.1,
        637.0,
        family_hint="mistral",
    ),
    "qwen2_5_7b": PaperModel(
        "qwen2_5_7b",
        "Qwen/Qwen2.5-7B-Instruct",
        "Qwen2.5-7B-Instruct",
        7.62,
        3.81,
        0.9195,
        0.9190,
        1229.3,
        676.0,
        family_hint="qwen2",
    ),
    "llama3_8b": PaperModel(
        "llama3_8b",
        "meta-llama/Meta-Llama-3-8B-Instruct",
        "LLaMA-3-8B-Instruct",
        8.03,
        4.015,
        0.9623,
        0.9620,
        1276.4,
        702.0,
        family_hint="llama",
        gated=True,
    ),
    "llama3_1_8b": PaperModel(
        "llama3_1_8b",
        "meta-llama/Llama-3.1-8B-Instruct",
        "LLaMA-3.1-8B-Instruct",
        8.03,
        4.015,
        0.8577,
        0.8574,
        1269.7,
        698.0,
        family_hint="llama",
        gated=True,
    ),
    "yi_1_5_9b": PaperModel(
        "yi_1_5_9b",
        "01-ai/Yi-1.5-9B-Chat",
        "Yi-1.5-9B-Chat",
        9.0,
        4.5,
        0.8670,
        0.8670,
        1420.0,
        781.0,
        family_hint="llama",
    ),
    "gemma2_9b": PaperModel(
        "gemma2_9b",
        "google/gemma-2-9b-it",
        "Gemma-2-9B-IT",
        9.24,
        4.62,
        0.8731,
        0.8731,
        1461.9,
        804.0,
        family_hint="gemma2",
        gated=True,
    ),
}

MODEL_ALIASES = {
    "yi6b": "yi_1_5_6b",
    "yi-6b": "yi_1_5_6b",
    "codellama": "codellama_7b",
    "internlm": "internlm2_5_7b",
    "mistral": "mistral_7b_v0_3",
    "qwen": "qwen2_5_7b",
    "qwen7b": "qwen2_5_7b",
    "llama3": "llama3_8b",
    "llama3.1": "llama3_1_8b",
    "yi9b": "yi_1_5_9b",
    "gemma": "gemma2_9b",
}


@dataclass
class SmartTalkRow:
    row_id: int
    input_text: str
    output_text: str
    target_obj: Any
    canonical_output: str
    input_words: int
    action_count: int
    rooms: Tuple[str, ...]
    devices: Tuple[str, ...]


@dataclass
class SplitIndices:
    train: List[int]
    validation: List[int]
    test: List[int]


@dataclass
class ArchAdapter:
    family: str
    attention_attr: str
    mlp_attr: str
    fused_qkv: bool
    q_proj_name: Optional[str]
    k_proj_name: Optional[str]
    v_proj_name: Optional[str]
    o_proj_name: str
    qkv_proj_name: Optional[str]
    gate_proj_name: str
    up_proj_name: str
    down_proj_name: str


class SmartTalkError(RuntimeError):
    pass


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")


def file_sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
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

    return json.loads(text, object_pairs_hook=no_duplicate_keys, parse_constant=reject_constant)


def schema_status(obj: Any) -> Tuple[bool, Optional[str], int, Tuple[str, ...], Tuple[str, ...]]:
    if not isinstance(obj, dict):
        return False, "top-level JSON value must be an object mapping rooms to devices", 0, tuple(), tuple()
    rooms = set()
    devices = set()
    action_count = 0
    for room, device_map in obj.items():
        if not isinstance(room, str) or not room.strip():
            return False, "room key must be a non-empty string", action_count, tuple(), tuple()
        if not isinstance(device_map, dict) or not device_map:
            return False, f"room {room!r} must map to a non-empty device object", action_count, tuple(), tuple()
        rooms.add(room)
        for device, state in device_map.items():
            if not isinstance(device, str) or not device.strip():
                return False, f"device key under room {room!r} must be a non-empty string", action_count, tuple(), tuple()
            if isinstance(state, (dict, list)) or state is None:
                return False, f"state for {room!r}/{device!r} must be a scalar value", action_count, tuple(), tuple()
            devices.add(device)
            action_count += 1
    return True, None, action_count, tuple(sorted(rooms)), tuple(sorted(devices))

def make_plain_prompt(input_text: str) -> str:
    return (
        "You are SmartTalk, a smart-home command parser. Return only valid JSON "
        "mapping rooms to device actions. Do not include explanation.\n\n"
        f"### Input:\n{input_text}\n\n### Output:\n"
    )


def make_messages(input_text: str) -> List[Dict[str, str]]:
    return [
        {
            "role": "system",
            "content": (
                "You are SmartTalk, a smart-home command parser. Return only "
                "valid JSON as an object mapping rooms to device state objects. Do not include "
                "explanation."
            ),
        },
        {"role": "user", "content": f"### Input:\n{input_text}\n\n### Output:"},
    ]


def format_prompt(tokenizer: Any, input_text: str, add_generation_prompt: bool = True) -> str:
    if getattr(tokenizer, "chat_template", None):
        try:
            return tokenizer.apply_chat_template(
                make_messages(input_text),
                tokenize=False,
                add_generation_prompt=add_generation_prompt,
            )
        except Exception:
            pass
    return make_plain_prompt(input_text)


def default_output_dir() -> Path:
    if Path("/kaggle/working").exists():
        return Path("/kaggle/working/smarttalk_structured_prune_output")
    if Path("/content").exists():
        return Path("/content/smarttalk_structured_prune_output")
    return Path(__file__).resolve().parent / "smarttalk_structured_prune_output"


def discover_roots() -> List[Tuple[Path, bool, str]]:
    roots: List[Tuple[Path, bool, str]] = []
    if Path("/kaggle/input").exists():
        roots.append((Path("/kaggle/input"), True, "kaggle_input"))
    if Path("/content").exists():
        roots.append((Path("/content"), True, "colab_content"))
    roots.append((Path(__file__).resolve().parent, False, "script_directory"))
    roots.append((Path.cwd(), False, "current_directory"))
    downloads = Path.home() / "Downloads"
    if downloads.exists():
        roots.append((downloads, False, "home_downloads"))
    deduped: List[Tuple[Path, bool, str]] = []
    seen = set()
    for root, recursive, source in roots:
        try:
            key = str(root.resolve())
        except OSError:
            key = str(root)
        if key not in seen and root.exists():
            deduped.append((root, recursive, source))
            seen.add(key)
    return deduped


def csv_header(path: Path) -> List[str]:
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.reader(handle)
            return [str(x).strip() for x in next(reader)]
    except Exception:
        return []


def score_candidate(path: Path) -> int:
    name = path.name.lower()
    header = {h.lower() for h in csv_header(path)}
    score = 0
    if "smart" in name:
        score += 4
    if "home" in name:
        score += 2
    if "100k" in name:
        score += 3
    if {"input", "output"}.issubset(header):
        score += 10
    return score


def collect_csv_candidates() -> List[Dict[str, Any]]:
    candidates: List[Dict[str, Any]] = []
    for root, recursive, source in discover_roots():
        paths = root.rglob("*.csv") if recursive else root.glob("*.csv")
        for path in paths:
            try:
                header = csv_header(path)
                score = score_candidate(path)
                if score <= 0:
                    continue
                candidates.append(
                    {
                        "path": str(path),
                        "source": source,
                        "score": score,
                        "header": header,
                        "size_bytes": path.stat().st_size,
                    }
                )
            except OSError:
                continue
    return sorted(candidates, key=lambda x: (-x["score"], x["path"]))


def discover_dataset_path(explicit_path: Optional[str]) -> Tuple[Path, Dict[str, Any]]:
    if explicit_path:
        path = Path(explicit_path).expanduser()
        if not path.exists():
            raise SmartTalkError(f"Dataset path does not exist: {path}")
        return path, {"mode": "explicit", "selected": str(path), "candidates": []}

    candidates = collect_csv_candidates()
    valid = [c for c in candidates if {"input", "output"}.issubset({h.lower() for h in c["header"]})]
    if not valid:
        roots = [str(root) for root, _, _ in discover_roots()]
        raise SmartTalkError(
            "Could not auto-discover a SmartTalk CSV with input/output columns. "
            f"Searched: {roots}"
        )
    top_score = valid[0]["score"]
    top = [c for c in valid if c["score"] == top_score]
    if len(top) > 1:
        rendered = "\n".join(f"  - {c['path']} score={c['score']}" for c in top[:20])
        raise SmartTalkError(
            "Dataset auto-discovery is ambiguous. Pass --dataset-path with one of:\n" + rendered
        )
    return Path(valid[0]["path"]), {"mode": "auto", "selected": valid[0]["path"], "candidates": valid[:20]}


def audit_csv(path: Path) -> Tuple[List[SmartTalkRow], Dict[str, Any]]:
    rows: List[SmartTalkRow] = []
    malformed = 0
    schema_invalid = 0
    empty_pairs = 0
    sorted_json_outputs = 0
    target_by_input: Dict[str, set] = defaultdict(set)
    pair_counts: Counter[Tuple[str, str]] = Counter()
    action_counts = Counter()
    room_counter = Counter()
    device_counter = Counter()
    malformed_examples: List[Dict[str, Any]] = []
    schema_examples: List[Dict[str, Any]] = []

    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        lower_to_real = {str(h).lower(): h for h in (reader.fieldnames or [])}
        if "input" not in lower_to_real or "output" not in lower_to_real:
            raise SmartTalkError(f"CSV must contain input and output columns. Header: {reader.fieldnames}")
        input_col = lower_to_real["input"]
        output_col = lower_to_real["output"]

        for row_id, record in enumerate(reader):
            input_text = str(record.get(input_col, "")).strip()
            output_text = str(record.get(output_col, "")).strip()
            if not input_text or not output_text:
                empty_pairs += 1
                continue
            try:
                target_obj = strict_json_loads(output_text)
            except Exception as exc:
                malformed += 1
                if len(malformed_examples) < 20:
                    malformed_examples.append({"row_id": row_id, "error": str(exc), "output": output_text[:500]})
                continue

            is_schema_valid, schema_error, action_count, rooms, devices = schema_status(target_obj)
            if not is_schema_valid:
                schema_invalid += 1
                if len(schema_examples) < 20:
                    schema_examples.append({"row_id": row_id, "error": schema_error, "output": output_text[:500]})
                continue

            canon = canonical_json(target_obj)
            if canon == output_text:
                sorted_json_outputs += 1
            target_by_input[input_text].add(canon)
            pair_counts[(input_text, canon)] += 1
            action_counts[action_count] += 1
            room_counter.update(rooms)
            device_counter.update(devices)
            rows.append(
                SmartTalkRow(
                    row_id=row_id,
                    input_text=input_text,
                    output_text=output_text,
                    target_obj=target_obj,
                    canonical_output=canon,
                    input_words=len(input_text.split()),
                    action_count=action_count,
                    rooms=rooms,
                    devices=devices,
                )
            )

    duplicate_rows = sum(count - 1 for count in pair_counts.values() if count > 1)
    conflicts = {text: sorted(targets) for text, targets in target_by_input.items() if len(targets) > 1}
    total_actions = sum(row.action_count for row in rows)
    audit = {
        "dataset_path": str(path),
        "dataset_sha256": file_sha256(path),
        "created_at_utc": now_utc(),
        "total_rows": len(rows) + malformed + schema_invalid + empty_pairs,
        "non_empty_pairs": len(rows) + malformed + schema_invalid,
        "usable_rows": len(rows),
        "valid_json_outputs": len(rows) + schema_invalid,
        "schema_valid_outputs": len(rows),
        "malformed_json_outputs": malformed,
        "schema_invalid_outputs": schema_invalid,
        "empty_pair_count": empty_pairs,
        "sorted_json_outputs": sorted_json_outputs,
        "duplicate_rows": duplicate_rows,
        "unique_inputs": len(target_by_input),
        "inputs_with_conflicting_targets": len(conflicts),
        "conflicting_input_examples": dict(list(conflicts.items())[:20]),
        "avg_input_words": round(statistics.mean([r.input_words for r in rows]), 3) if rows else None,
        "avg_device_actions": round(total_actions / len(rows), 3) if rows else None,
        "max_device_actions": max(action_counts) if action_counts else 0,
        "action_count_distribution": {str(k): v for k, v in sorted(action_counts.items())},
        "room_count": len(room_counter),
        "device_type_count": len(device_counter),
        "rooms": sorted(room_counter),
        "devices": sorted(device_counter),
        "malformed_examples": malformed_examples,
        "schema_invalid_examples": schema_examples,
    }
    return rows, audit


def validate_audit_or_fail(audit: Dict[str, Any]) -> None:
    failures = []
    if audit["usable_rows"] <= 0:
        failures.append("no usable rows found")
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
    indices = list(range(n_rows))
    rng = random.Random(seed)
    rng.shuffle(indices)
    train_end = int(n_rows * 0.80)
    validation_end = train_end + int(n_rows * 0.10)
    return SplitIndices(
        train=indices[:train_end],
        validation=indices[train_end:validation_end],
        test=indices[validation_end:],
    )


def split_overlap_audit(rows: Sequence[SmartTalkRow], splits: SplitIndices) -> Dict[str, Any]:
    train_inputs = {rows[idx].input_text for idx in splits.train}
    train_pairs = {(rows[idx].input_text, rows[idx].canonical_output) for idx in splits.train}
    seen = 0
    pair_seen = 0
    for idx in splits.test:
        if rows[idx].input_text in train_inputs:
            seen += 1
        if (rows[idx].input_text, rows[idx].canonical_output) in train_pairs:
            pair_seen += 1
    return {
        "split_seed": SEED,
        "train_rows": len(splits.train),
        "validation_rows": len(splits.validation),
        "test_rows": len(splits.test),
        "test_inputs_seen_in_train": seen,
        "test_inputs_unseen_in_train": len(splits.test) - seen,
        "test_exact_pairs_seen_in_train": pair_seen,
        "note": (
            "The paper states category-balanced splits with personalized commands only in test, "
            "but this CSV has no category column. This script uses deterministic row-level 80/10/10."
        ),
    }


def save_split_artifacts(output_dir: Path, splits: SplitIndices, overlap: Dict[str, Any]) -> None:
    split_dir = output_dir / "splits"
    split_dir.mkdir(parents=True, exist_ok=True)
    write_json(split_dir / "train_indices.json", splits.train)
    write_json(split_dir / "validation_indices.json", splits.validation)
    write_json(split_dir / "test_indices.json", splits.test)
    write_json(output_dir / "split_overlap_audit.json", overlap)


def select_indices(indices: Sequence[int], limit: Optional[int]) -> List[int]:
    if limit is None or limit < 0:
        return list(indices)
    return list(indices[:limit])


def package_version(package: str) -> Optional[str]:
    try:
        module = importlib.import_module(package)
        return getattr(module, "__version__", "unknown")
    except Exception:
        return None


def install_deps_and_restart(argv: Sequence[str]) -> None:
    packages = [
        "transformers>=4.45.0",
        "accelerate>=0.33.0",
        "datasets>=2.20.0",
        "safetensors>=0.4.3",
        "sentencepiece>=0.2.0",
        "protobuf>=4.25.0",
        "peft>=0.12.0",
        "bitsandbytes>=0.43.3",
    ]
    print("Installing dependencies without replacing the runtime PyTorch build...")
    cmd = [sys.executable, "-m", "pip", "install", "-U", *packages]
    subprocess.check_call(cmd)
    remaining = [arg for arg in argv if arg != "--install-deps"]
    os.execv(sys.executable, [sys.executable, __file__, *remaining])


def import_required_ml() -> Dict[str, Any]:
    missing = []
    modules = {}
    for name in ("torch", "transformers", "datasets"):
        try:
            modules[name] = importlib.import_module(name)
        except Exception as exc:
            missing.append(f"{name}: {exc}")
    if missing:
        raise SmartTalkError(
            "Missing ML dependencies. Run with --install-deps first.\n" + "\n".join(f"  - {x}" for x in missing)
        )
    return modules


def hf_token_from_env() -> Optional[str]:
    return os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")


def platform_report(torch_module: Optional[Any] = None, precision: Optional[str] = None) -> Dict[str, Any]:
    report = {
        "created_at_utc": now_utc(),
        "platform": platform.platform(),
        "python": sys.version.replace("\n", " "),
        "executable": sys.executable,
        "cwd": str(Path.cwd()),
        "packages": {
            "torch": package_version("torch"),
            "transformers": package_version("transformers"),
            "accelerate": package_version("accelerate"),
            "datasets": package_version("datasets"),
            "peft": package_version("peft"),
            "bitsandbytes": package_version("bitsandbytes"),
            "safetensors": package_version("safetensors"),
        },
        "precision": precision,
        "cuda": None,
    }
    if torch_module is not None:
        cuda = {
            "available": bool(torch_module.cuda.is_available()),
            "torch_cuda": getattr(torch_module.version, "cuda", None),
            "device_count": torch_module.cuda.device_count() if torch_module.cuda.is_available() else 0,
            "devices": [],
        }
        if torch_module.cuda.is_available():
            for i in range(torch_module.cuda.device_count()):
                props = torch_module.cuda.get_device_properties(i)
                cuda["devices"].append(
                    {
                        "index": i,
                        "name": props.name,
                        "total_memory_gb": round(props.total_memory / (1024**3), 3),
                        "major": props.major,
                        "minor": props.minor,
                    }
                )
        report["cuda"] = cuda
    return report


def require_cuda(torch_module: Any, allow_cpu: bool = False) -> None:
    if allow_cpu:
        return
    if not torch_module.cuda.is_available():
        raise SmartTalkError(
            "CUDA is required for practical pruning/evaluation of the paper models. "
            "Use Kaggle/Colab GPU or pass --allow-cpu only for tiny smoke tests."
        )


def choose_compute_dtype(torch_module: Any) -> Tuple[Any, str]:
    if torch_module.cuda.is_available() and torch_module.cuda.is_bf16_supported():
        return torch_module.bfloat16, "bf16"
    if torch_module.cuda.is_available():
        return torch_module.float16, "fp16"
    return torch_module.float32, "fp32"


def resolve_model_selection(args: argparse.Namespace) -> List[PaperModel]:
    if args.model_id:
        key = "custom"
        return [
            PaperModel(
                key=key,
                model_id=args.model_id,
                label=args.model_id,
                paper_original_b=0.0,
                paper_pruned_b=0.0,
                paper_original_accuracy=0.0,
                paper_pruned_accuracy=0.0,
                paper_original_latency_ms=0.0,
                paper_pruned_latency_ms=0.0,
                trust_remote_code=args.trust_remote_code,
                family_hint=args.family_hint,
            )
        ]
    requested = args.paper_model.lower()
    if requested == "all":
        return list(PAPER_MODELS.values())
    requested = MODEL_ALIASES.get(requested, requested)
    if requested not in PAPER_MODELS:
        valid = ", ".join(["all", *sorted(PAPER_MODELS)])
        raise SmartTalkError(f"Unknown --paper-model {args.paper_model!r}. Valid values: {valid}")
    return [PAPER_MODELS[requested]]


def resolve_model_revision(model_id: str, revision: str, token: Optional[str]) -> Dict[str, Any]:
    try:
        from huggingface_hub import HfApi

        info = HfApi(token=token).model_info(model_id, revision=revision)
        return {
            "model_id": model_id,
            "requested_revision": revision,
            "resolved_sha": getattr(info, "sha", None),
        }
    except Exception as exc:
        return {
            "model_id": model_id,
            "requested_revision": revision,
            "resolved_sha": None,
            "resolution_error": str(exc),
        }


def first_model_device(model: Any) -> Any:
    for parameter in model.parameters():
        return parameter.device
    raise SmartTalkError("Model has no parameters")


def tokenizer_device_inputs(tokenizer: Any, text: str, max_length: int, device: Any) -> Dict[str, Any]:
    encoded = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
        padding=False,
        add_special_tokens=True,
    )
    return {key: value.to(device) for key, value in encoded.items()}


def load_tokenizer(model: PaperModel, revision: str, token: Optional[str]) -> Any:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        model.model_id,
        revision=revision,
        token=token,
        trust_remote_code=model.trust_remote_code,
        use_fast=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    return tokenizer


def load_model(
    model: PaperModel,
    revision: str,
    dtype: Any,
    token: Optional[str],
    device_map: str,
    load_in_4bit: bool = False,
) -> Any:
    from transformers import AutoModelForCausalLM

    kwargs: Dict[str, Any] = {
        "revision": revision,
        "token": token,
        "trust_remote_code": model.trust_remote_code,
        "torch_dtype": dtype,
        "low_cpu_mem_usage": True,
    }
    if load_in_4bit:
        try:
            from transformers import BitsAndBytesConfig

            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=dtype,
            )
            kwargs["device_map"] = "auto"
        except Exception as exc:
            raise SmartTalkError(f"bitsandbytes 4-bit loading is unavailable: {exc}") from exc
    elif device_map == "auto":
        kwargs["device_map"] = "auto"
    model_obj = AutoModelForCausalLM.from_pretrained(model.model_id, **kwargs)
    if not load_in_4bit and device_map != "auto":
        model_obj.to(device_map)
    model_obj.eval()
    if hasattr(model_obj.config, "use_cache"):
        model_obj.config.use_cache = True
    return model_obj


def get_decoder_layers(model: Any) -> List[Any]:
    candidates = [
        ("model", "layers"),
        ("transformer", "h"),
        ("gpt_neox", "layers"),
        ("model", "decoder", "layers"),
    ]
    for path in candidates:
        obj = model
        ok = True
        for attr in path:
            if not hasattr(obj, attr):
                ok = False
                break
            obj = getattr(obj, attr)
        if ok:
            return list(obj)
    raise SmartTalkError("Could not locate decoder layers on model")


def get_nested_attr(obj: Any, names: Sequence[str]) -> Any:
    current = obj
    for name in names:
        current = getattr(current, name)
    return current


def first_existing_attr(obj: Any, names: Sequence[str]) -> Optional[str]:
    for name in names:
        if hasattr(obj, name):
            return name
    return None


def detect_arch_adapter(model: Any, family_hint: str = "auto") -> ArchAdapter:
    layers = get_decoder_layers(model)
    if not layers:
        raise SmartTalkError("Model has no decoder layers")
    layer = layers[0]
    attention_attr = first_existing_attr(layer, ("self_attn", "attention", "attn"))
    mlp_attr = first_existing_attr(layer, ("mlp", "feed_forward", "ffn"))
    if attention_attr is None or mlp_attr is None:
        raise SmartTalkError("Could not detect attention/MLP modules on first layer")
    attn = getattr(layer, attention_attr)
    mlp = getattr(layer, mlp_attr)

    q_name = first_existing_attr(attn, ("q_proj", "wq"))
    k_name = first_existing_attr(attn, ("k_proj", "wk"))
    v_name = first_existing_attr(attn, ("v_proj", "wv"))
    o_name = first_existing_attr(attn, ("o_proj", "wo"))
    qkv_name = first_existing_attr(attn, ("wqkv", "query_key_value", "qkv_proj"))
    fused = qkv_name is not None and (q_name is None or k_name is None or v_name is None)

    gate_name = first_existing_attr(mlp, ("gate_proj", "w1"))
    up_name = first_existing_attr(mlp, ("up_proj", "w3"))
    down_name = first_existing_attr(mlp, ("down_proj", "w2"))

    if o_name is None or gate_name is None or up_name is None or down_name is None:
        raise SmartTalkError(
            "Could not detect required projection modules. "
            f"attention={type(attn).__name__}, mlp={type(mlp).__name__}"
        )
    if not fused and (q_name is None or k_name is None or v_name is None):
        raise SmartTalkError(f"Could not detect q/k/v projections on {type(attn).__name__}")

    return ArchAdapter(
        family=family_hint,
        attention_attr=attention_attr,
        mlp_attr=mlp_attr,
        fused_qkv=fused,
        q_proj_name=q_name,
        k_proj_name=k_name,
        v_proj_name=v_name,
        o_proj_name=o_name,
        qkv_proj_name=qkv_name,
        gate_proj_name=gate_name,
        up_proj_name=up_name,
        down_proj_name=down_name,
    )


def get_head_config(config: Any) -> Tuple[int, int, int, int, int]:
    hidden_size = int(config.hidden_size)
    num_heads = int(config.num_attention_heads)
    num_kv_heads = int(getattr(config, "num_key_value_heads", num_heads))
    head_dim = int(getattr(config, "head_dim", hidden_size // num_heads))
    intermediate_size = int(config.intermediate_size)
    if num_heads % num_kv_heads != 0:
        raise SmartTalkError(f"num_attention_heads={num_heads} is not divisible by num_key_value_heads={num_kv_heads}")
    if num_heads * head_dim <= 0:
        raise SmartTalkError(
            f"Invalid head config: num_heads={num_heads}, head_dim={head_dim}"
        )
    return hidden_size, num_heads, num_kv_heads, head_dim, intermediate_size


class HeadActivationCollector:
    def __init__(self, num_heads: int, head_dim: int):
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.total = None
        self.count = 0

    def hook_fn(self, module: Any, inputs: Tuple[Any, ...]) -> None:
        hidden = inputs[0].detach().float()
        if hidden.numel() == 0:
            return
        per_head = hidden.reshape(-1, self.num_heads, self.head_dim)
        norms = per_head.norm(dim=-1).mean(dim=0).cpu()
        if self.total is None:
            self.total = norms
        else:
            self.total += norms
        self.count += 1

    def mean(self) -> Any:
        if self.total is None or self.count == 0:
            raise SmartTalkError("No attention activations were collected")
        return self.total / self.count


class NeuronActivationCollector:
    def __init__(self):
        self.total = None
        self.count = 0

    def hook_fn(self, module: Any, inputs: Tuple[Any, ...]) -> None:
        hidden = inputs[0].detach().float()
        if hidden.numel() == 0:
            return
        values = hidden.reshape(-1, hidden.shape[-1]).abs().mean(dim=0).cpu()
        if self.total is None:
            self.total = values
        else:
            self.total += values
        self.count += 1

    def mean(self) -> Any:
        if self.total is None or self.count == 0:
            raise SmartTalkError("No MLP activations were collected")
        return self.total / self.count


def get_layer_modules(layer: Any, adapter: ArchAdapter) -> Tuple[Any, Any, Any, Any]:
    attn = getattr(layer, adapter.attention_attr)
    mlp = getattr(layer, adapter.mlp_attr)
    o_proj = getattr(attn, adapter.o_proj_name)
    down_proj = getattr(mlp, adapter.down_proj_name)
    return attn, mlp, o_proj, down_proj


def collect_importance_scores(
    model: Any,
    tokenizer: Any,
    rows: Sequence[SmartTalkRow],
    indices: Sequence[int],
    adapter: ArchAdapter,
    calibration_size: int,
    max_length: int,
    seed: int,
) -> Dict[str, Any]:
    hidden_size, num_heads, num_kv_heads, head_dim, intermediate_size = get_head_config(model.config)
    layers = get_decoder_layers(model)
    rng = random.Random(seed)
    selected = list(indices)
    rng.shuffle(selected)
    selected = selected[: min(calibration_size, len(selected))]

    head_collectors = [HeadActivationCollector(num_heads, head_dim) for _ in layers]
    neuron_collectors = [NeuronActivationCollector() for _ in layers]
    handles = []
    for i, layer in enumerate(layers):
        _, _, o_proj, down_proj = get_layer_modules(layer, adapter)
        handles.append(o_proj.register_forward_pre_hook(head_collectors[i].hook_fn))
        handles.append(down_proj.register_forward_pre_hook(neuron_collectors[i].hook_fn))

    device = first_model_device(model)
    model.eval()
    start = time.perf_counter()
    try:
        for pos, row_idx in enumerate(selected):
            prompt = format_prompt(tokenizer, rows[row_idx].input_text, add_generation_prompt=True)
            inputs = tokenizer_device_inputs(tokenizer, prompt, max_length=max_length, device=device)
            with torch.no_grad():
                _ = model(**inputs, use_cache=False)
            if torch.cuda.is_available() and (pos + 1) % 32 == 0:
                torch.cuda.empty_cache()
            if (pos + 1) % 100 == 0:
                print(f"  calibration forward {pos + 1}/{len(selected)}")
    finally:
        for handle in handles:
            handle.remove()

    head_scores = []
    neuron_scores = []
    for layer_idx, layer in enumerate(layers):
        attn, mlp, o_proj, down_proj = get_layer_modules(layer, adapter)
        o_weight = o_proj.weight.detach().float().cpu()
        expected_cols = num_heads * head_dim
        if o_weight.shape[1] != expected_cols:
            raise SmartTalkError(
                f"Layer {layer_idx} o_proj input width {o_weight.shape[1]} does not match heads*head_dim {expected_cols}"
            )
        o_by_head = o_weight.reshape(o_weight.shape[0], num_heads, head_dim)
        head_weight_norm = o_by_head.norm(dim=(0, 2))
        head_scores.append(head_weight_norm * head_collectors[layer_idx].mean())

        down_weight = down_proj.weight.detach().float().cpu()
        if down_weight.shape[1] != intermediate_size:
            raise SmartTalkError(
                f"Layer {layer_idx} down projection input width {down_weight.shape[1]} "
                f"does not match intermediate_size {intermediate_size}"
            )
        neuron_weight_norm = down_weight.norm(dim=0)
        neuron_scores.append(neuron_weight_norm * neuron_collectors[layer_idx].mean())

    elapsed = time.perf_counter() - start
    all_heads = torch.cat(head_scores)
    all_neurons = torch.cat(neuron_scores)
    return {
        "head_scores": head_scores,
        "neuron_scores": neuron_scores,
        "metadata": {
            "calibration_rows": len(selected),
            "calibration_indices": selected,
            "calibration_seed": seed,
            "calibration_seconds": elapsed,
            "activation_hook": "attention output-projection pre-hook and MLP down-projection pre-hook",
            "attention_score_min": float(all_heads.min().item()),
            "attention_score_max": float(all_heads.max().item()),
            "attention_score_mean": float(all_heads.mean().item()),
            "mlp_score_min": float(all_neurons.min().item()),
            "mlp_score_max": float(all_neurons.max().item()),
            "mlp_score_mean": float(all_neurons.mean().item()),
            "hidden_size": hidden_size,
            "num_attention_heads": num_heads,
            "num_key_value_heads": num_kv_heads,
            "head_dim": head_dim,
            "intermediate_size": intermediate_size,
            "layers": len(layers),
        },
    }


def replace_linear_rows(linear: Any, keep_idx: Any) -> Any:
    import torch
    from torch import nn

    keep_idx = keep_idx.to(device=linear.weight.device)
    bias = linear.bias is not None
    new_linear = nn.Linear(
        linear.in_features,
        int(keep_idx.numel()),
        bias=bias,
        device=linear.weight.device,
        dtype=linear.weight.dtype,
    )
    with torch.no_grad():
        new_linear.weight.copy_(linear.weight.index_select(0, keep_idx))
        if bias:
            new_linear.bias.copy_(linear.bias.index_select(0, keep_idx))
    return new_linear


def replace_linear_cols(linear: Any, keep_idx: Any) -> Any:
    import torch
    from torch import nn

    keep_idx = keep_idx.to(device=linear.weight.device)
    bias = linear.bias is not None
    new_linear = nn.Linear(
        int(keep_idx.numel()),
        linear.out_features,
        bias=bias,
        device=linear.weight.device,
        dtype=linear.weight.dtype,
    )
    with torch.no_grad():
        new_linear.weight.copy_(linear.weight.index_select(1, keep_idx))
        if bias:
            new_linear.bias.copy_(linear.bias)
    return new_linear


def head_indices_to_rows(head_indices: Sequence[int], head_dim: int, torch_module: Any) -> Any:
    rows = []
    for head in head_indices:
        rows.extend(range(int(head) * head_dim, (int(head) + 1) * head_dim))
    return torch_module.tensor(rows, dtype=torch_module.long)


def set_module_attr(parent: Any, name: str, value: Any) -> None:
    setattr(parent, name, value)


def keep_groups_from_scores(scores: Any, num_heads: int, num_kv_heads: int, keep_kv_heads: int) -> List[int]:
    group_size = num_heads // num_kv_heads
    group_scores = []
    for kv in range(num_kv_heads):
        start = kv * group_size
        end = start + group_size
        group_scores.append((float(scores[start:end].mean().item()), kv))
    group_scores.sort(key=lambda x: x[0], reverse=True)
    return sorted(kv for _, kv in group_scores[:keep_kv_heads])


def prune_attention_layer(
    layer: Any,
    adapter: ArchAdapter,
    keep_q_heads: List[int],
    keep_kv_heads: List[int],
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
) -> Dict[str, Any]:
    import torch

    attn = getattr(layer, adapter.attention_attr)
    q_rows = head_indices_to_rows(keep_q_heads, head_dim, torch)
    kv_rows = head_indices_to_rows(keep_kv_heads, head_dim, torch)

    if adapter.fused_qkv:
        assert adapter.qkv_proj_name is not None
        qkv = getattr(attn, adapter.qkv_proj_name)
        if adapter.qkv_proj_name == "wqkv" or adapter.family.lower().startswith("internlm"):
            group_size = num_heads // num_kv_heads
            per_kv_width = (group_size + 2) * head_dim
            fused_chunks = []
            for kv in keep_kv_heads:
                base = int(kv) * per_kv_width
                fused_chunks.append(torch.arange(base, base + per_kv_width, dtype=torch.long))
            fused_rows = torch.cat(fused_chunks)
        else:
            q_width = num_heads * head_dim
            kv_width = num_kv_heads * head_dim
            fused_rows = torch.cat([q_rows, q_width + kv_rows, q_width + kv_width + kv_rows])
        set_module_attr(attn, adapter.qkv_proj_name, replace_linear_rows(qkv, fused_rows))
    else:
        assert adapter.q_proj_name and adapter.k_proj_name and adapter.v_proj_name
        set_module_attr(attn, adapter.q_proj_name, replace_linear_rows(getattr(attn, adapter.q_proj_name), q_rows))
        set_module_attr(attn, adapter.k_proj_name, replace_linear_rows(getattr(attn, adapter.k_proj_name), kv_rows))
        set_module_attr(attn, adapter.v_proj_name, replace_linear_rows(getattr(attn, adapter.v_proj_name), kv_rows))

    o_proj = getattr(attn, adapter.o_proj_name)
    set_module_attr(attn, adapter.o_proj_name, replace_linear_cols(o_proj, q_rows))

    kept_q = len(keep_q_heads)
    kept_kv = len(keep_kv_heads)
    for attr, value in (
        ("num_heads", kept_q),
        ("num_key_value_heads", kept_kv),
        ("num_key_value_groups", kept_q // kept_kv),
        ("head_dim", head_dim),
        ("hidden_size", kept_q * head_dim),
    ):
        if hasattr(attn, attr):
            setattr(attn, attr, value)
    return {
        "q_heads_kept": kept_q,
        "kv_heads_kept": kept_kv,
        "q_heads": keep_q_heads,
        "kv_heads": keep_kv_heads,
    }


def prune_mlp_layer(layer: Any, adapter: ArchAdapter, keep_neurons: Any) -> Dict[str, Any]:
    mlp = getattr(layer, adapter.mlp_attr)
    set_module_attr(mlp, adapter.gate_proj_name, replace_linear_rows(getattr(mlp, adapter.gate_proj_name), keep_neurons))
    set_module_attr(mlp, adapter.up_proj_name, replace_linear_rows(getattr(mlp, adapter.up_proj_name), keep_neurons))
    set_module_attr(mlp, adapter.down_proj_name, replace_linear_cols(getattr(mlp, adapter.down_proj_name), keep_neurons))
    if hasattr(mlp, "intermediate_size"):
        setattr(mlp, "intermediate_size", int(keep_neurons.numel()))
    return {"mlp_neurons_kept": int(keep_neurons.numel())}


def prune_model_structured(
    model: Any,
    adapter: ArchAdapter,
    head_scores: Sequence[Any],
    neuron_scores: Sequence[Any],
    prune_ratio: float,
) -> Dict[str, Any]:
    import torch

    hidden_size, num_heads, num_kv_heads, head_dim, intermediate_size = get_head_config(model.config)
    layers = get_decoder_layers(model)
    keep_kv_heads = max(1, round(num_kv_heads * (1.0 - prune_ratio)))
    keep_neurons = max(1, round(intermediate_size * (1.0 - prune_ratio)))
    keep_neurons = min(intermediate_size, keep_neurons)
    keep_kv_heads = min(num_kv_heads, keep_kv_heads)
    group_size = num_heads // num_kv_heads
    keep_q_heads_count = keep_kv_heads * group_size

    layer_stats = []
    for layer_idx, layer in enumerate(layers):
        kv_heads = keep_groups_from_scores(head_scores[layer_idx], num_heads, num_kv_heads, keep_kv_heads)
        q_heads: List[int] = []
        for kv in kv_heads:
            q_heads.extend(range(kv * group_size, (kv + 1) * group_size))
        q_heads = sorted(q_heads)
        neuron_keep = torch.argsort(neuron_scores[layer_idx], descending=True)[:keep_neurons].sort().values
        attn_stats = prune_attention_layer(layer, adapter, q_heads, kv_heads, num_heads, num_kv_heads, head_dim)
        mlp_stats = prune_mlp_layer(layer, adapter, neuron_keep)
        layer_stats.append(
            {
                "layer": layer_idx,
                **attn_stats,
                **mlp_stats,
                "removed_q_heads": sorted(set(range(num_heads)) - set(q_heads)),
                "removed_kv_heads": sorted(set(range(num_kv_heads)) - set(kv_heads)),
            }
        )

    model.config.num_attention_heads = keep_q_heads_count
    model.config.num_key_value_heads = keep_kv_heads
    model.config.intermediate_size = keep_neurons
    setattr(model.config, "head_dim", head_dim)
    if hasattr(model.config, "num_key_value_groups"):
        model.config.num_key_value_groups = keep_q_heads_count // keep_kv_heads
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = True

    return {
        "prune_ratio_requested": prune_ratio,
        "original_hidden_size": hidden_size,
        "original_num_attention_heads": num_heads,
        "original_num_key_value_heads": num_kv_heads,
        "original_head_dim": head_dim,
        "original_intermediate_size": intermediate_size,
        "kept_num_attention_heads": keep_q_heads_count,
        "kept_num_key_value_heads": keep_kv_heads,
        "kept_intermediate_size": keep_neurons,
        "kept_q_head_fraction": keep_q_heads_count / num_heads,
        "kept_kv_head_fraction": keep_kv_heads / num_kv_heads,
        "kept_mlp_fraction": keep_neurons / intermediate_size,
        "layer_stats": layer_stats,
    }


def count_parameters(model: Any) -> Tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def estimate_prune_ratio_for_paper_budget(model: Any, target_fraction: float = 0.50) -> Dict[str, Any]:
    hidden_size, num_heads, num_kv_heads, head_dim, intermediate_size = get_head_config(model.config)
    layers = get_decoder_layers(model)
    total_params, _ = count_parameters(model)
    transformer_structural = 0
    for layer in layers:
        transformer_structural += (
            hidden_size * (num_heads * head_dim)
            + hidden_size * (num_kv_heads * head_dim)
            + hidden_size * (num_kv_heads * head_dim)
            + hidden_size * (num_heads * head_dim)
            + hidden_size * intermediate_size * 3
        )
    non_structural = max(0, total_params - transformer_structural)
    target_params = total_params * target_fraction
    if transformer_structural <= 0 or target_params <= non_structural:
        prune_ratio = PAPER_PRUNE_RATIO
    else:
        keep_fraction = (target_params - non_structural) / transformer_structural
        prune_ratio = max(0.0, min(0.95, 1.0 - keep_fraction))
    return {
        "target_total_fraction": target_fraction,
        "estimated_total_params": total_params,
        "estimated_transformer_structural_params": transformer_structural,
        "estimated_non_structural_params": non_structural,
        "estimated_prune_ratio": prune_ratio,
        "note": (
            "This is an ablation because the paper says 50% heads/MLP structures, "
            "but Table V reports exactly half total parameters."
        ),
    }


def parse_generated_json(text: str) -> Tuple[Optional[Any], Optional[str]]:
    stripped = text.strip()
    if not stripped:
        return None, "empty generation"
    decoder = json.JSONDecoder()
    try:
        obj, _ = decoder.raw_decode(stripped)
        return obj, None
    except Exception:
        pass
    first = min([idx for idx in [stripped.find("["), stripped.find("{")] if idx >= 0], default=-1)
    if first >= 0:
        try:
            obj, _ = decoder.raw_decode(stripped[first:])
            return obj, None
        except Exception as exc:
            return None, str(exc)
    return None, "no JSON object or list found"


def percentile(values: Sequence[float], pct: float) -> Optional[float]:
    if not values:
        return None
    ordered = sorted(values)
    pos = (len(ordered) - 1) * pct / 100.0
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return ordered[lo]
    return ordered[lo] * (hi - pos) + ordered[hi] * (pos - lo)


def evaluate_generation(
    model: Any,
    tokenizer: Any,
    rows: Sequence[SmartTalkRow],
    indices: Sequence[int],
    train_inputs: set,
    max_length: int,
    max_new_tokens: int,
    limit: Optional[int],
    run_name: str,
    output_dir: Path,
    warmup: int = 5,
) -> Dict[str, Any]:
    import torch

    selected_indices = select_indices(indices, limit)
    if not selected_indices:
        raise SmartTalkError(f"No rows selected for {run_name} evaluation")
    model.eval()
    device = first_model_device(model)
    eos_token_id = tokenizer.eos_token_id
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else eos_token_id

    warmup_indices = selected_indices[: min(warmup, len(selected_indices))]
    for idx in warmup_indices:
        prompt = format_prompt(tokenizer, rows[idx].input_text, add_generation_prompt=True)
        inputs = tokenizer_device_inputs(tokenizer, prompt, max_length=max_length, device=device)
        with torch.no_grad():
            _ = model.generate(
                **inputs,
                max_new_tokens=min(16, max_new_tokens),
                do_sample=False,
                pad_token_id=pad_token_id,
                eos_token_id=eos_token_id,
                use_cache=True,
            )
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    canonical_matches = 0
    raw_matches = 0
    valid_json = 0
    schema_valid = 0
    seen_total = 0
    seen_matches = 0
    unseen_total = 0
    unseen_matches = 0
    latencies_ms = []
    generated_tokens = []
    failures = []
    started = time.perf_counter()

    for position, idx in enumerate(selected_indices):
        row = rows[idx]
        prompt = format_prompt(tokenizer, row.input_text, add_generation_prompt=True)
        inputs = tokenizer_device_inputs(tokenizer, prompt, max_length=max_length, device=device)
        input_len = int(inputs["input_ids"].shape[-1])
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                num_beams=1,
                pad_token_id=pad_token_id,
                eos_token_id=eos_token_id,
                use_cache=True,
            )
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        gen_ids = output_ids[0, input_len:]
        generated_tokens.append(int(gen_ids.numel()))
        latencies_ms.append(elapsed_ms)
        decoded = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
        parsed, parse_error = parse_generated_json(decoded)
        is_valid_json = parsed is not None
        if is_valid_json:
            valid_json += 1
        schema_ok = False
        if is_valid_json:
            schema_ok, _, _, _, _ = schema_status(parsed)
            schema_valid += int(schema_ok)
        canonical_match = bool(is_valid_json and parsed == row.target_obj)
        raw_match = decoded == row.canonical_output or decoded == row.output_text
        canonical_matches += int(canonical_match)
        raw_matches += int(raw_match)

        is_seen = row.input_text in train_inputs
        if is_seen:
            seen_total += 1
            seen_matches += int(canonical_match)
        else:
            unseen_total += 1
            unseen_matches += int(canonical_match)

        if not canonical_match and len(failures) < 100:
            failures.append(
                {
                    "row_id": row.row_id,
                    "input": row.input_text,
                    "expected": row.canonical_output,
                    "generated": decoded[:1000],
                    "parse_error": parse_error,
                    "valid_json": is_valid_json,
                    "schema_valid": schema_ok,
                    "seen_in_train": is_seen,
                }
            )
        if (position + 1) % 250 == 0:
            print(
                f"  [{run_name}] evaluated {position + 1}/{len(selected_indices)} rows; "
                f"canonical_accuracy={canonical_matches / (position + 1):.4f}"
            )

    total = len(selected_indices)
    elapsed = time.perf_counter() - started
    latency_avg = statistics.mean(latencies_ms) if latencies_ms else None
    tokens_total = sum(generated_tokens)
    report = {
        "run_name": run_name,
        "evaluated_rows": total,
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
            "average": latency_avg,
            "p50": percentile(latencies_ms, 50),
            "p95": percentile(latencies_ms, 95),
            "min": min(latencies_ms) if latencies_ms else None,
            "max": max(latencies_ms) if latencies_ms else None,
            "warmup_generations": len(warmup_indices),
        },
        "generated_tokens": {
            "total": tokens_total,
            "average": statistics.mean(generated_tokens) if generated_tokens else None,
            "tokens_per_second": tokens_total / elapsed if elapsed > 0 else None,
        },
        "wall_time_seconds": elapsed,
        "max_length": max_length,
        "max_new_tokens": max_new_tokens,
        "decoding": {"do_sample": False, "num_beams": 1},
    }
    write_json(output_dir / f"{run_name}_report.json", report)
    write_json(output_dir / f"{run_name}_failures.json", failures)
    return report


def build_paper_comparison(
    paper_model: PaperModel,
    baseline_report: Optional[Dict[str, Any]],
    pruned_report: Dict[str, Any],
    prune_stats: Dict[str, Any],
    original_params: int,
    pruned_params: int,
    variant: str,
) -> Dict[str, Any]:
    baseline_accuracy = (
        baseline_report.get("canonical_json_exact_match_accuracy") if baseline_report else None
    )
    baseline_latency = (
        baseline_report.get("latency_ms", {}).get("average") if baseline_report else None
    )
    pruned_accuracy = pruned_report.get("canonical_json_exact_match_accuracy")
    pruned_latency = pruned_report.get("latency_ms", {}).get("average")
    return {
        "paper_model": asdict(paper_model),
        "variant": variant,
        "paper_method": {
            "importance_score": "||W_j||_2 * E_x[||h_j(x)||_2]",
            "reported_prune_setting": "50% of selected attention heads and MLP structures",
            "hardware": PAPER_T4_HARDWARE,
            "limitations": PAPER_CALIBRATION_NOTE,
        },
        "script_method": {
            "attention_unit": "GQA-safe KV group plus associated Q heads",
            "mlp_unit": "down-projection input neuron",
            "strict_variant": variant == "structural50",
            "prune_stats": prune_stats,
        },
        "parameters": {
            "script_original": original_params,
            "script_pruned": pruned_params,
            "script_pruned_fraction": pruned_params / original_params if original_params else None,
            "paper_original_b": paper_model.paper_original_b,
            "paper_pruned_b": paper_model.paper_pruned_b,
            "paper_pruned_fraction": (
                paper_model.paper_pruned_b / paper_model.paper_original_b
                if paper_model.paper_original_b
                else None
            ),
        },
        "accuracy": {
            "paper_original": paper_model.paper_original_accuracy or None,
            "paper_pruned": paper_model.paper_pruned_accuracy or None,
            "script_original": baseline_accuracy,
            "script_pruned": pruned_accuracy,
            "script_original_delta_percentage_points": (
                (baseline_accuracy - paper_model.paper_original_accuracy) * 100
                if baseline_accuracy is not None and paper_model.paper_original_accuracy
                else None
            ),
            "script_pruned_delta_percentage_points": (
                (pruned_accuracy - paper_model.paper_pruned_accuracy) * 100
                if pruned_accuracy is not None and paper_model.paper_pruned_accuracy
                else None
            ),
        },
        "latency_ms": {
            "paper_original": paper_model.paper_original_latency_ms or None,
            "paper_pruned": paper_model.paper_pruned_latency_ms or None,
            "script_original_average": baseline_latency,
            "script_pruned_average": pruned_latency,
            "script_original_delta_ms": (
                baseline_latency - paper_model.paper_original_latency_ms
                if baseline_latency is not None and paper_model.paper_original_latency_ms
                else None
            ),
            "script_pruned_delta_ms": (
                pruned_latency - paper_model.paper_pruned_latency_ms
                if pruned_latency is not None and paper_model.paper_pruned_latency_ms
                else None
            ),
        },
    }


def save_model_and_tokenizer(model: Any, tokenizer: Any, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir, safe_serialization=True)
    tokenizer.save_pretrained(output_dir)


def reload_check(saved_dir: Path, dtype: Any, trust_remote_code: bool, max_length: int, prompt: str) -> Dict[str, Any]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    result = {"attempted": True, "ok": False, "error": None}
    try:
        tokenizer = AutoTokenizer.from_pretrained(saved_dir, trust_remote_code=trust_remote_code)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            saved_dir,
            trust_remote_code=trust_remote_code,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
        )
        if torch.cuda.is_available():
            model.to("cuda")
        model.eval()
        device = first_model_device(model)
        inputs = tokenizer_device_inputs(tokenizer, prompt, max_length=max_length, device=device)
        with torch.no_grad():
            _ = model.generate(
                **inputs,
                max_new_tokens=8,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                use_cache=True,
            )
        result["ok"] = True
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception as exc:
        result["error"] = str(exc)
    return result


def maybe_run_recovery_lora(
    args: argparse.Namespace,
    model: Any,
    tokenizer: Any,
    rows: Sequence[SmartTalkRow],
    splits: SplitIndices,
    output_dir: Path,
) -> Optional[Dict[str, Any]]:
    if "recovery" not in args.variants:
        return None
    try:
        import torch
        from datasets import Dataset
        from peft import LoraConfig, get_peft_model
        from transformers import Trainer, TrainingArguments
    except Exception as exc:
        raise SmartTalkError(f"Recovery LoRA requires datasets, peft, and transformers Trainer: {exc}") from exc

    def encode(row: SmartTalkRow) -> Dict[str, List[int]]:
        prompt = format_prompt(tokenizer, row.input_text, add_generation_prompt=True)
        target = row.canonical_output + (tokenizer.eos_token or "")
        prompt_ids = tokenizer(prompt, add_special_tokens=True, truncation=True, max_length=args.max_length)["input_ids"]
        target_ids = tokenizer(target, add_special_tokens=False)["input_ids"]
        input_ids = prompt_ids + target_ids
        if len(input_ids) > args.max_length:
            raise SmartTalkError(
                f"Recovery example row {row.row_id} exceeds --max-length {args.max_length}; "
                "increase max length because targets are not truncated."
            )
        labels = [-100] * len(prompt_ids) + target_ids
        return {"input_ids": input_ids, "attention_mask": [1] * len(input_ids), "labels": labels}

    class Collator:
        def __call__(self, features: List[Dict[str, List[int]]]) -> Dict[str, Any]:
            max_len = max(len(f["input_ids"]) for f in features)
            pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
            batch = {"input_ids": [], "attention_mask": [], "labels": []}
            for feature in features:
                pad = max_len - len(feature["input_ids"])
                batch["input_ids"].append([pad_id] * pad + feature["input_ids"])
                batch["attention_mask"].append([0] * pad + feature["attention_mask"])
                batch["labels"].append([-100] * pad + feature["labels"])
            return {k: torch.tensor(v, dtype=torch.long) for k, v in batch.items()}

    train_limit = args.recovery_train_limit or len(splits.train)
    train_rows = [encode(rows[idx]) for idx in splits.train[:train_limit]]
    dataset = Dataset.from_list(train_rows)
    target_modules = [name for name in ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj", "wqkv", "wo", "w1", "w2", "w3")]
    lora_config = LoraConfig(
        r=args.recovery_lora_rank,
        lora_alpha=args.recovery_lora_alpha,
        lora_dropout=args.recovery_lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=target_modules,
    )
    model = get_peft_model(model, lora_config)
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    training_args = TrainingArguments(
        output_dir=str(output_dir / "recovery_trainer"),
        num_train_epochs=args.recovery_epochs,
        per_device_train_batch_size=args.recovery_batch_size,
        gradient_accumulation_steps=max(1, math.ceil(32 / args.recovery_batch_size)),
        learning_rate=args.recovery_lr,
        warmup_ratio=0.05,
        lr_scheduler_type="cosine",
        logging_steps=25,
        save_strategy="epoch",
        report_to=[],
        fp16=bool(torch.cuda.is_available() and not torch.cuda.is_bf16_supported()),
        bf16=bool(torch.cuda.is_available() and torch.cuda.is_bf16_supported()),
        remove_unused_columns=False,
    )
    trainer = Trainer(model=model, args=training_args, train_dataset=dataset, data_collator=Collator())
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    adapter_dir = output_dir / "recovery_lora_adapter"
    model.save_pretrained(adapter_dir)
    return {
        "adapter_dir": str(adapter_dir),
        "train_rows": len(train_rows),
        "epochs": args.recovery_epochs,
        "lora_rank": args.recovery_lora_rank,
        "note": "Recovery LoRA is an ablation; it is not part of the strict Table V pruning result.",
    }


def run_bnb4_eval_if_requested(
    args: argparse.Namespace,
    paper_model: PaperModel,
    saved_dir: Path,
    tokenizer: Any,
    rows: Sequence[SmartTalkRow],
    splits: SplitIndices,
    train_inputs: set,
    dtype: Any,
    output_dir: Path,
) -> Optional[Dict[str, Any]]:
    if "bnb4" not in args.variants:
        return None
    import torch
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig

    try:
        qconfig = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=dtype,
        )
        model = AutoModelForCausalLM.from_pretrained(
            saved_dir,
            quantization_config=qconfig,
            device_map="auto",
            trust_remote_code=paper_model.trust_remote_code,
        )
        report = evaluate_generation(
            model,
            tokenizer,
            rows,
            splits.test,
            train_inputs,
            args.max_length,
            args.max_new_tokens,
            args.eval_limit,
            "pruned_bnb4",
            output_dir,
            warmup=args.warmup_generations,
        )
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return report
    except Exception as exc:
        return {"error": str(exc), "note": "bnb4 is optional and was not used for strict Table V comparison."}


def run_gptq4_if_requested(args: argparse.Namespace, saved_dir: Path, tokenizer: Any, rows: Sequence[SmartTalkRow], splits: SplitIndices, output_dir: Path) -> Optional[Dict[str, Any]]:
    if "gptq4" not in args.variants:
        return None
    try:
        from gptqmodel import GPTQModel, QuantizeConfig
    except Exception as exc:
        return {"error": f"GPTQModel is not installed: {exc}", "note": "Install gptqmodel manually for this optional ablation."}

    try:
        prompts = [
            format_prompt(tokenizer, rows[idx].input_text, add_generation_prompt=True)
            for idx in splits.train[: min(args.calibration_size, len(splits.train))]
        ]
        quant_dir = output_dir / "pruned_gptq4"
        qconfig = QuantizeConfig(bits=4, group_size=128)
        model = GPTQModel.load(str(saved_dir), quantize_config=qconfig)
        model.quantize(prompts, tokenizer=tokenizer)
        model.save(str(quant_dir))
        return {
            "quantized_dir": str(quant_dir),
            "bits": 4,
            "group_size": 128,
            "calibration_rows": len(prompts),
            "note": "GPTQ4 is an optional ablation; it is not the strict Table V score unless explicitly chosen.",
        }
    except Exception as exc:
        return {"error": str(exc), "note": "GPTQ4 optional ablation failed."}


def print_analysis_summary(audit: Dict[str, Any], overlap: Dict[str, Any], output_dir: Path) -> None:
    print("Dataset audit complete.")
    print(f"  rows: {audit['total_rows']}")
    print(f"  usable rows: {audit['usable_rows']}")
    print(f"  valid JSON outputs: {audit['valid_json_outputs']}")
    print(f"  duplicate rows: {audit['duplicate_rows']}")
    print(f"  unique inputs: {audit['unique_inputs']}")
    print(f"  conflicting inputs: {audit['inputs_with_conflicting_targets']}")
    print(f"  avg input words: {audit['avg_input_words']}")
    print(f"  avg device actions: {audit['avg_device_actions']}")
    print(f"  rooms/device types: {audit['room_count']}/{audit['device_type_count']}")
    print(f"  max device actions: {audit['max_device_actions']}")
    print(f"  split train/val/test: {overlap['train_rows']}/{overlap['validation_rows']}/{overlap['test_rows']}")
    print(f"  test seen/unseen: {overlap['test_inputs_seen_in_train']}/{overlap['test_inputs_unseen_in_train']}")
    print(f"  reports: {output_dir}")


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SmartTalk paper structured pruning reproduction.")
    parser.add_argument("--dataset-path", default=None, help="Path to SmartTalk CSV. Auto-discovered when omitted.")
    parser.add_argument("--output-dir", default=None, help="Directory for reports and pruned model outputs.")
    parser.add_argument("--paper-model", default="qwen2_5_7b", help="Paper model key or 'all'.")
    parser.add_argument("--model-id", default=None, help="Override with an arbitrary Hugging Face model id.")
    parser.add_argument("--model-revision", default=DEFAULT_MODEL_REVISION, help="Model branch, tag, or commit.")
    parser.add_argument("--trust-remote-code", action="store_true", help="Trust remote code for --model-id overrides.")
    parser.add_argument("--family-hint", default="auto", help="Architecture hint for --model-id overrides.")
    parser.add_argument(
        "--variants",
        nargs="+",
        default=["structural50"],
        choices=["structural50", "paper-budget", "bnb4", "gptq4", "recovery"],
        help=(
            "Experiment variants. structural50 is the paper-faithful headline. "
            "paper-budget adjusts ratio to approach half total parameters. "
            "bnb4/gptq4/recovery are optional ablations."
        ),
    )
    parser.add_argument("--prune-ratio", type=float, default=PAPER_PRUNE_RATIO, help="Fraction of structures to remove.")
    parser.add_argument("--calibration-size", type=int, default=512, help="Training rows used for activation calibration.")
    parser.add_argument("--max-length", type=int, default=512, help="Maximum prompt length for calibration/evaluation.")
    parser.add_argument("--max-new-tokens", type=int, default=256, help="Maximum generated tokens per command.")
    parser.add_argument("--eval-limit", type=int, default=None, help="Limit test rows for faster debugging. Omit for paper comparison.")
    parser.add_argument("--skip-baseline", action="store_true", help="Skip untouched baseline evaluation.")
    parser.add_argument("--skip-reload-check", action="store_true", help="Skip reload/generation check after saving pruned model.")
    parser.add_argument("--device-map", default="auto", help="Use 'auto', 'cuda', or 'cpu'.")
    parser.add_argument("--allow-cpu", action="store_true", help="Allow CPU execution for tiny smoke tests.")
    parser.add_argument("--analyze-only", action="store_true", help="Only audit CSV and write split artifacts.")
    parser.add_argument("--smoke-test", action="store_true", help="Use tiny calibration/eval settings.")
    parser.add_argument("--install-deps", action="store_true", help="Install compatible ML packages, preserving runtime PyTorch.")
    parser.add_argument("--continue-on-error", action="store_true", help="Continue to next model if one model fails.")
    parser.add_argument("--resume-from-checkpoint", default=None, help="Trainer checkpoint for optional recovery LoRA.")
    parser.add_argument("--warmup-generations", type=int, default=5, help="Warmup generations before measuring latency.")
    parser.add_argument("--recovery-epochs", type=float, default=2.0, help="Optional LoRA recovery epochs.")
    parser.add_argument("--recovery-train-limit", type=int, default=None, help="Limit rows for optional recovery LoRA.")
    parser.add_argument("--recovery-batch-size", type=int, default=4, help="Micro-batch for optional recovery LoRA.")
    parser.add_argument("--recovery-lr", type=float, default=1e-4, help="Learning rate for optional recovery LoRA.")
    parser.add_argument("--recovery-lora-rank", type=int, default=8, help="LoRA rank for optional recovery.")
    parser.add_argument("--recovery-lora-alpha", type=int, default=16, help="LoRA alpha for optional recovery.")
    parser.add_argument("--recovery-lora-dropout", type=float, default=0.05, help="LoRA dropout for optional recovery.")
    args = parser.parse_args(argv)
    if args.smoke_test:
        args.calibration_size = min(args.calibration_size, 4)
        args.eval_limit = min(args.eval_limit or 4, 4)
        args.max_new_tokens = min(args.max_new_tokens, 64)
        args.warmup_generations = 1
        args.allow_cpu = True
        if args.model_id is None:
            args.model_id = "hf-internal-testing/tiny-random-LlamaForCausalLM"
            args.family_hint = "llama"
            args.skip_baseline = True
            args.warmup_generations = 0
    if not 0.0 <= args.prune_ratio < 1.0:
        parser.error("--prune-ratio must be in [0, 1)")
    return args


def run_one_model(
    args: argparse.Namespace,
    paper_model: PaperModel,
    rows: Sequence[SmartTalkRow],
    splits: SplitIndices,
    output_root: Path,
    dtype: Any,
    precision: str,
    torch_module: Any,
    token: Optional[str],
    audit: Dict[str, Any],
    overlap: Dict[str, Any],
) -> Dict[str, Any]:
    model_dir_name = paper_model.key if paper_model.key != "custom" else paper_model.model_id.replace("/", "__")
    model_output = output_root / model_dir_name
    model_output.mkdir(parents=True, exist_ok=True)
    write_json(model_output / "paper_model.json", asdict(paper_model))
    revision_report = resolve_model_revision(paper_model.model_id, args.model_revision, token)
    write_json(model_output / "model_revision.json", revision_report)

    print(f"\n=== {paper_model.label} ({paper_model.model_id}) ===")
    if paper_model.gated and not token:
        print("  note: this model may require HF_TOKEN/HUGGINGFACE_HUB_TOKEN or prior Hugging Face login.")

    tokenizer = load_tokenizer(paper_model, args.model_revision, token)
    model = load_model(
        paper_model,
        args.model_revision,
        dtype,
        token,
        device_map=args.device_map,
        load_in_4bit=False,
    )
    adapter = detect_arch_adapter(model, paper_model.family_hint)
    original_params, _ = count_parameters(model)
    train_inputs = {rows[idx].input_text for idx in splits.train}
    run_config = {
        "args": vars(args),
        "paper_model": asdict(paper_model),
        "adapter": asdict(adapter),
        "platform": platform_report(torch_module, precision),
        "model_revision": revision_report,
        "dataset_sha256": audit["dataset_sha256"],
        "split_overlap": overlap,
        "original_parameter_count": original_params,
    }
    write_json(model_output / "run_config.json", run_config)

    baseline_report = None
    if not args.skip_baseline:
        print("  evaluating untouched baseline...")
        baseline_report = evaluate_generation(
            model,
            tokenizer,
            rows,
            splits.test,
            train_inputs,
            args.max_length,
            args.max_new_tokens,
            args.eval_limit,
            "baseline",
            model_output,
            warmup=args.warmup_generations,
        )

    print(f"  collecting importance scores on {args.calibration_size} calibration rows...")
    importance = collect_importance_scores(
        model,
        tokenizer,
        rows,
        splits.train,
        adapter,
        args.calibration_size,
        args.max_length,
        seed=SEED,
    )
    write_json(model_output / "importance_metadata.json", importance["metadata"])

    strict_reports: Dict[str, Any] = {}
    variants_to_prune = [v for v in args.variants if v in ("structural50", "paper-budget")]
    if not variants_to_prune:
        variants_to_prune = ["structural50"]

    for variant_index, variant in enumerate(variants_to_prune):
        if variant_index > 0:
            del model
            gc.collect()
            if torch_module.cuda.is_available():
                torch_module.cuda.empty_cache()
            model = load_model(
                paper_model,
                args.model_revision,
                dtype,
                token,
                device_map=args.device_map,
                load_in_4bit=False,
            )
            adapter = detect_arch_adapter(model, paper_model.family_hint)

        variant_output = model_output / variant
        variant_output.mkdir(parents=True, exist_ok=True)
        ratio = args.prune_ratio
        budget_report = None
        if variant == "paper-budget":
            budget_report = estimate_prune_ratio_for_paper_budget(model, target_fraction=0.50)
            ratio = float(budget_report["estimated_prune_ratio"])
            write_json(variant_output / "paper_budget_ratio.json", budget_report)

        print(f"  pruning variant={variant} ratio={ratio:.4f}...")
        prune_stats = prune_model_structured(
            model,
            adapter,
            importance["head_scores"],
            importance["neuron_scores"],
            prune_ratio=ratio,
        )
        pruned_params, _ = count_parameters(model)
        prune_stats["original_parameter_count"] = original_params
        prune_stats["pruned_parameter_count"] = pruned_params
        prune_stats["pruned_parameter_fraction"] = pruned_params / original_params if original_params else None
        if budget_report:
            prune_stats["paper_budget"] = budget_report
        write_json(variant_output / "prune_stats.json", prune_stats)

        pruned_model_dir = variant_output / "pruned_model"
        print(f"  saving pruned model to {pruned_model_dir}...")
        save_model_and_tokenizer(model, tokenizer, pruned_model_dir)

        reload_result = {"attempted": False, "ok": None}
        if not args.skip_reload_check:
            sample_prompt = format_prompt(tokenizer, rows[splits.test[0]].input_text, add_generation_prompt=True)
            print("  checking saved model reload...")
            reload_result = reload_check(pruned_model_dir, dtype, paper_model.trust_remote_code, args.max_length, sample_prompt)
            write_json(variant_output / "reload_check.json", reload_result)
            if not reload_result.get("ok"):
                print(f"  reload check failed: {reload_result.get('error')}")

        print("  evaluating pruned model...")
        pruned_report = evaluate_generation(
            model,
            tokenizer,
            rows,
            splits.test,
            train_inputs,
            args.max_length,
            args.max_new_tokens,
            args.eval_limit,
            f"pruned_{variant}",
            variant_output,
            warmup=args.warmup_generations,
        )
        comparison = build_paper_comparison(
            paper_model,
            baseline_report,
            pruned_report,
            prune_stats,
            original_params,
            pruned_params,
            variant,
        )
        comparison["reload_check"] = reload_result
        write_json(variant_output / "paper_comparison_report.json", comparison)

        recovery_report = maybe_run_recovery_lora(args, model, tokenizer, rows, splits, variant_output)
        if recovery_report is not None:
            write_json(variant_output / "recovery_lora_report.json", recovery_report)

        bnb4_report = run_bnb4_eval_if_requested(
            args,
            paper_model,
            pruned_model_dir,
            tokenizer,
            rows,
            splits,
            train_inputs,
            dtype,
            variant_output,
        )
        if bnb4_report is not None:
            write_json(variant_output / "bnb4_report.json", bnb4_report)

        gptq4_report = run_gptq4_if_requested(args, pruned_model_dir, tokenizer, rows, splits, variant_output)
        if gptq4_report is not None:
            write_json(variant_output / "gptq4_report.json", gptq4_report)

        strict_reports[variant] = comparison

    del model
    gc.collect()
    if torch_module.cuda.is_available():
        torch_module.cuda.empty_cache()
    summary = {
        "model_key": paper_model.key,
        "model_id": paper_model.model_id,
        "baseline_report_path": str(model_output / "baseline_report.json") if baseline_report else None,
        "variant_reports": {
            variant: str(model_output / variant / "paper_comparison_report.json") for variant in strict_reports
        },
    }
    write_json(model_output / "model_summary.json", summary)
    return summary


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    if args.install_deps:
        install_deps_and_restart(sys.argv[1:])

    output_dir = Path(args.output_dir).expanduser() if args.output_dir else default_output_dir()
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_path, discovery_report = discover_dataset_path(args.dataset_path)
    rows, audit = audit_csv(dataset_path)
    audit["discovery"] = discovery_report
    write_json(output_dir / "dataset_audit.json", audit)
    validate_audit_or_fail(audit)
    splits = create_split_indices(len(rows), seed=SEED)
    overlap = split_overlap_audit(rows, splits)
    save_split_artifacts(output_dir, splits, overlap)
    print_analysis_summary(audit, overlap, output_dir)
    if args.analyze_only:
        return 0

    ml = import_required_ml()
    torch_module = ml["torch"]
    require_cuda(torch_module, allow_cpu=args.allow_cpu)
    dtype, precision = choose_compute_dtype(torch_module)
    write_json(output_dir / "platform_report.json", platform_report(torch_module, precision))
    token = hf_token_from_env()
    selected_models = resolve_model_selection(args)
    summaries = []
    failures = []
    for paper_model in selected_models:
        try:
            summaries.append(
                run_one_model(
                    args,
                    paper_model,
                    rows,
                    splits,
                    output_dir,
                    dtype,
                    precision,
                    torch_module,
                    token,
                    audit,
                    overlap,
                )
            )
        except Exception as exc:
            failure = {"model_key": paper_model.key, "model_id": paper_model.model_id, "error": str(exc)}
            failures.append(failure)
            write_json(output_dir / "failures.json", failures)
            print(f"ERROR for {paper_model.label}: {exc}", file=sys.stderr)
            if not args.continue_on_error:
                raise

    final_report = {
        "created_at_utc": now_utc(),
        "dataset_path": str(dataset_path),
        "dataset_sha256": audit["dataset_sha256"],
        "selected_models": [asdict(m) for m in selected_models],
        "summaries": summaries,
        "failures": failures,
        "strict_match_guidance": {
            "headline_variant": "structural50",
            "headline_metric": "canonical_json_exact_match_accuracy",
            "full_test_rows_required": len(splits.test),
            "latency_comparable_only_on": PAPER_T4_HARDWARE,
            "paper_omissions": PAPER_CALIBRATION_NOTE,
        },
    }
    write_json(output_dir / "final_pruning_report.json", final_report)
    print(f"\nPruning run complete. Final report: {output_dir / 'final_pruning_report.json'}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SmartTalkError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2)
