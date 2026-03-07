import pytest
from sfcrime_model.predict import make_prediction
from sfcrime_model.config.core import config


def test_make_prediction(sample_input_data):
    result = make_prediction(input_data=sample_input_data)

    predictions = result.get("predictions")
    assert result.get("errors") is None
    assert isinstance(predictions, list)
    assert len(predictions) == len(sample_input_data)

    first = predictions[0]
    assert "category" in first
    assert "probability" in first
    assert isinstance(first["category"], str)
    assert 0.0 <= first["probability"] <= 1.0


def test_prediction_category_is_valid(sample_input_data):
    from sfcrime_model.processing.data_manager import load_label_classes
    valid_categories = set(load_label_classes())

    result = make_prediction(input_data=sample_input_data)
    for pred in result["predictions"]:
        assert pred["category"] in valid_categories


def test_config_loads():
    assert config.app_config.package_name == "sfcrime-model"
    assert config.model_config.num_classes == 39


def test_package_import():
    import sfcrime_model  # noqa: F401
