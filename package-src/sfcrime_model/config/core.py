from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import yaml

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = PACKAGE_ROOT.parents[1]
DEFAULT_CONFIG_PATH = PACKAGE_ROOT / "config.yml"


@dataclass(frozen=True)
class AppConfig:
    config: Dict[str, Any]

    @property
    def project(self) -> Dict[str, Any]:
        return self.config.get("project", {})

    @property
    def features(self) -> Dict[str, Any]:
        return self.config.get("features", {})

    @property
    def models(self) -> Dict[str, Any]:
        return self.config.get("models", {})

    @property
    def mlflow(self) -> Dict[str, Any]:
        return self.config.get("mlflow", {})


def read_yaml_config(path: Path = DEFAULT_CONFIG_PATH) -> AppConfig:
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    return AppConfig(config=cfg)
