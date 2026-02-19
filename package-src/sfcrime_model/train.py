from __future__ import annotations

from sfcrime_model.config.core import read_yaml_config


def main() -> None:
    cfg = read_yaml_config()
    exp = cfg.mlflow.get("experiment_name", "sf-crime-classification")
    print(f"Config loaded OK. MLflow experiment: {exp}")


if __name__ == "__main__":
    main()
