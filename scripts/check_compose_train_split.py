from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import yaml


TRAIN_NAME_RE = re.compile(r"^(p[1-6]_.*|p8_.*|p5_.*train.*|.*calibrat.*|.*export.*onnx.*)$", re.IGNORECASE)
EDGE_NAME_RE = re.compile(r"^(p9_.*|p10_.*|p11_.*|.*api.*|.*service.*)$", re.IGNORECASE)
P5_TRAIN_RE = re.compile(r"^p5_.*train.*$", re.IGNORECASE)
P4_NAME_RE = re.compile(r"^p4_.*$", re.IGNORECASE)

CMD_TOKENS_TRAIN = [
    "ingest",
    "features",
    "label",
    "iforest",
    "meta train",
    "calibrate",
    "export onnx",
]


def _collect_compose_files(cwd: Path) -> List[Path]:
    patterns = [
        "docker-compose*.yaml",
        "docker-compose*.yml",
        "compose/*.yaml",
        "compose/*.yml",
    ]
    files: List[Path] = []
    for pat in patterns:
        files.extend(sorted(cwd.glob(pat)))
    return files


def _load_yaml(path: Path) -> Dict[str, Any]:
    try:
        with path.open("r") as f:
            data = yaml.safe_load(f) or {}
        if not isinstance(data, dict):
            return {}
        return data
    except Exception:
        return {}


def _merge_services(files: List[Path]) -> Dict[str, Dict[str, Any]]:
    services: Dict[str, Dict[str, Any]] = {}
    for fp in files:
        data = _load_yaml(fp)
        svcs = data.get("services", {}) or {}
        if not isinstance(svcs, dict):
            continue
        for name, svc in svcs.items():
            # last writer wins replacement
            if isinstance(svc, dict):
                services[name] = svc
    return services


def _command_string(svc: Dict[str, Any]) -> str:
    cmd = svc.get("command")
    if cmd is None:
        return ""
    if isinstance(cmd, list):
        return " ".join(str(x) for x in cmd).lower()
    return str(cmd).lower()


def _env_dict(svc: Dict[str, Any]) -> Dict[str, str]:
    env = svc.get("environment")
    out: Dict[str, str] = {}
    if env is None:
        return out
    if isinstance(env, dict):
        return {str(k): str(v) for k, v in env.items()}
    if isinstance(env, list):
        for item in env:
            s = str(item)
            if "=" in s:
                k, v = s.split("=", 1)
                out[k] = v
            else:
                out[s] = ""
    return out


def _has_gpu_config(svc: Dict[str, Any]) -> bool:
    if svc.get("gpus"):
        return True
    if str(svc.get("runtime", "")).lower() == "nvidia":
        return True
    # deploy.resources.reservations.devices.*.driver == nvidia
    try:
        devices = (
            svc.get("deploy", {})
            .get("resources", {})
            .get("reservations", {})
            .get("devices", [])
        )
        for d in devices:
            if isinstance(d, dict) and str(d.get("driver", "")).lower() == "nvidia":
                return True
            # also allow capabilities include 'gpu'
            caps = d.get("capabilities", []) if isinstance(d, dict) else []
            if any(str(c).lower() == "gpu" for c in caps):
                return True
    except Exception:
        pass
    return False


def _ports_contains_8080(svc: Dict[str, Any], expect_port: int) -> bool:
    ports = svc.get("ports")
    if not ports:
        return False
    def has_8080(p: Any) -> bool:
        if isinstance(p, str):
            s = p.split("/")[0]
            parts = s.split(":")
            try:
                if len(parts) == 1:
                    return int(parts[0]) == expect_port
                if len(parts) >= 2:
                    return int(parts[0]) == expect_port or int(parts[1]) == expect_port
            except Exception:
                return False
        if isinstance(p, dict):
            # compose format: target/published
            tgt = p.get("target")
            pub = p.get("published")
            return tgt == expect_port or pub == expect_port
        return False
    return any(has_8080(p) for p in ports)


def _profiles(svc: Dict[str, Any]) -> List[str]:
    prof = svc.get("profiles") or []
    if isinstance(prof, str):
        return [prof]
    if isinstance(prof, list):
        return [str(x) for x in prof]
    return []


def _classify(name: str, svc: Dict[str, Any]) -> Tuple[bool, bool]:
    profs = set(_profiles(svc))
    cmd = _command_string(svc)
    is_train = ("train" in profs) or bool(TRAIN_NAME_RE.match(name)) or any(tok in cmd for tok in CMD_TOKENS_TRAIN)
    is_edge = ("edge" in profs) or bool(EDGE_NAME_RE.match(name)) or ("service api" in cmd or "api" in cmd)
    return is_train, is_edge


def check_repo(assert_edge_port: int = 8080, pretty: bool = False, cwd: str | Path = ".") -> Dict[str, Any]:
    cwd = Path(cwd)
    files = _collect_compose_files(cwd)
    services = _merge_services(files)

    train_names: List[str] = []
    edge_names: List[str] = []
    for name, svc in services.items():
        is_train, is_edge = _classify(name, svc)
        if is_train:
            train_names.append(name)
        if is_edge:
            edge_names.append(name)

    errors: List[str] = []
    # R1
    if not train_names or not edge_names:
        errors.append("R1:both TRAIN and EDGE groups must be non-empty")

    # R2
    p5_train_like = [n for n in train_names if P5_TRAIN_RE.match(n)]
    if not p5_train_like:
        errors.append("R2:no TRAIN service named like p5_*train* found")

    # R3
    gpu_ok = False
    env_ok = False
    if p5_train_like:
        # Prefer exact 'p5_train'
        p5_name = "p5_train" if "p5_train" in services else p5_train_like[0]
        svc = services.get(p5_name, {})
        gpu_ok = _has_gpu_config(svc)
        env = _env_dict(svc)
        env_ok = any(k in env for k in ("CUDA_VISIBLE_DEVICES", "NVIDIA_VISIBLE_DEVICES"))
        if not gpu_ok:
            errors.append("R3:p5_train missing GPU config (gpus/runtime/deploy.devices)")
        if not env_ok:
            errors.append("R3:p5_train missing CUDA/NVIDIA_VISIBLE_DEVICES in environment")

    # R4: p4_* no GPU
    for n in services:
        if P4_NAME_RE.match(n):
            if _has_gpu_config(services[n]):
                errors.append(f"R4:{n} must not require GPU")

    # R5: profiles presence
    profiles_ok = True
    for n in train_names:
        if "train" not in _profiles(services[n]):
            profiles_ok = False
            errors.append(f"R5:service {n} missing profile 'train'")
    for n in edge_names:
        if "edge" not in _profiles(services[n]):
            profiles_ok = False
            errors.append(f"R5:service {n} missing profile 'edge'")

    # R6: edge exposes port
    ports_8080_edge = [n for n in edge_names if _ports_contains_8080(services[n], assert_edge_port)]
    if not ports_8080_edge:
        errors.append(f"R6:no EDGE service exposes port {assert_edge_port}")

    ok = len(errors) == 0

    payload = {
        "pass": ok,
        "errors": errors,
        "summary": {
            "train_services": sorted(train_names),
            "edge_services": sorted(edge_names),
            "gpu_in_p5_train": bool(gpu_ok and env_ok),
            "profiles_ok": profiles_ok,
            "ports_8080_edge": sorted(ports_8080_edge),
        },
    }

    # Print JSON to stdout
    if pretty:
        print(json.dumps(payload, indent=2))
    else:
        print(json.dumps(payload))

    # Human summary to stderr
    human = (
        f"PASS={ok} | TRAIN={len(train_names)} EDGE={len(edge_names)} "
        f"GPU_P5={gpu_ok and env_ok} PROFILES_OK={profiles_ok} EDGE_8080={ports_8080_edge}"
    )
    print(human, file=sys.stderr)

    return payload


def main() -> None:
    ap = argparse.ArgumentParser(description="Check Docker Compose train vs edge split")
    ap.add_argument("--assert-edge-port", type=int, default=8080)
    ap.add_argument("--pretty", action="store_true")
    args = ap.parse_args()
    payload = check_repo(assert_edge_port=args.assert_edge_port, pretty=args.pretty)
    sys.exit(0 if payload["pass"] else 1)


if __name__ == "__main__":
    main()

