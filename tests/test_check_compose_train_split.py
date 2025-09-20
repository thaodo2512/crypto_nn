from __future__ import annotations

import json
import os
from pathlib import Path

import yaml

from scripts.check_compose_train_split import check_repo


def write_compose(tmp: Path, name: str, services: dict) -> Path:
    data = {"version": "3.9", "services": services}
    fp = tmp / name
    fp.write_text(yaml.safe_dump(data))
    return fp


def test_fail_when_no_profiles(tmp_path, monkeypatch):
    # Missing profiles for both train and edge
    services = {
        "p5_train": {
            "command": ["python", "-m", "meta", "train"],
            "environment": ["NVIDIA_VISIBLE_DEVICES=all"],
            "runtime": "nvidia",
        },
        "edge_api": {
            "command": ["python", "service", "api"],
            "ports": ["8080:8080"],
        },
    }
    write_compose(tmp_path, "docker-compose.yml", services)
    monkeypatch.chdir(tmp_path)
    out = check_repo()
    assert not out["pass"]
    errs = "\n".join(out["errors"]) 
    assert "R5:service p5_train missing profile 'train'" in errs
    assert "R5:service edge_api missing profile 'edge'" in errs


def test_gpu_required_for_p5(tmp_path, monkeypatch):
    services = {
        "p5_train": {
            "profiles": ["train"],
            "command": ["python", "-m", "meta", "train"],
            # Missing GPU config and env
        },
        "edge_api": {
            "profiles": ["edge"],
            "command": ["python", "service", "api"],
            "ports": ["8080:8080"],
        },
    }
    write_compose(tmp_path, "docker-compose.yml", services)
    monkeypatch.chdir(tmp_path)
    out = check_repo()
    assert not out["pass"]
    errs = "\n".join(out["errors"]) 
    assert "R3:p5_train missing GPU config" in errs
    assert "R3:p5_train missing CUDA/NVIDIA_VISIBLE_DEVICES in environment" in errs


def test_p4_cpu_only(tmp_path, monkeypatch):
    services = {
        "p4_smote": {
            "profiles": ["train"],
            "gpus": 1,
            "command": ["python", "cli_p4.py", "smote-windows"],
        },
        "edge_api": {
            "profiles": ["edge"],
            "command": ["python", "service", "api"],
            "ports": ["8080:8080"],
        },
    }
    write_compose(tmp_path, "docker-compose.yml", services)
    monkeypatch.chdir(tmp_path)
    out = check_repo()
    assert not out["pass"]
    errs = "\n".join(out["errors"]) 
    assert "R4:p4_smote must not require GPU" in errs


def test_detect_services_by_command(tmp_path, monkeypatch):
    services = {
        "custom_trainer": {
            "profiles": ["train"],
            "command": "python -m meta train --arg x",
        },
        "custom_edge": {
            "profiles": ["edge"],
            "command": "python service api --port 8080",
            "ports": ["8080:8080"],
        },
        "p5_train": {
            "profiles": ["train"],
            "runtime": "nvidia",
            "environment": ["NVIDIA_VISIBLE_DEVICES=all"],
            "command": ["python", "-m", "meta", "train"],
        },
    }
    write_compose(tmp_path, "docker-compose.yml", services)
    monkeypatch.chdir(tmp_path)
    out = check_repo()
    assert out["summary"]["train_services"]
    assert out["summary"]["edge_services"]


def test_pass_minimal_example(tmp_path, monkeypatch):
    services = {
        "p5_train": {
            "profiles": ["train"],
            "runtime": "nvidia",
            "environment": ["NVIDIA_VISIBLE_DEVICES=all"],
            "command": ["python", "-m", "meta", "train"],
            # typical training mounts
            "volumes": ["./data:/app/data", "./artifacts:/app/artifacts"],
        },
        "p4_iforest": {
            "profiles": ["train"],
            "command": ["python", "cli_p4.py", "iforest-train"],
        },
        "edge_api": {
            "profiles": ["edge"],
            "command": ["python", "service", "api"],
            "ports": ["8080:8080"],
        },
    }
    write_compose(tmp_path, "docker-compose.yml", services)
    monkeypatch.chdir(tmp_path)
    out = check_repo()
    assert out["pass"], out["errors"]
    assert out["summary"]["gpu_in_p5_train"]
    assert out["summary"]["profiles_ok"]
    assert out["summary"]["ports_8080_edge"] == ["edge_api"]

