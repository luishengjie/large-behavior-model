import pytest

from behavior_modeling.utils import load_config


def test_load_config(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text("seed: 42\ndata:\n  dataset_id: example/data\n")

    config = load_config(config_path)

    assert config["seed"] == 42
    assert config["data"]["dataset_id"] == "example/data"


def test_load_config_does_not_require_data_mapping(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text("seed: 42\n")

    assert load_config(config_path) == {"seed": 42}


def test_load_config_requires_yaml_mapping(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text("- one\n- two\n")

    with pytest.raises(TypeError, match="mapping"):
        load_config(config_path)
