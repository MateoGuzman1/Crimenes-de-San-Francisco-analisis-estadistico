from sfcrime_model.config.core import read_yaml_config


def test_config_loads():
    cfg = read_yaml_config()
    assert isinstance(cfg.config, dict)
    assert "project" in cfg.config
    assert "mlflow" in cfg.config


def test_package_import():
    import sfcrime_model  # noqa: F401
