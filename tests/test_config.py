"""
Tests for NeuroMap Configuration and Environment Handling
"""
import os
import importlib
import pytest
import config as config_module
from config import Config, DevelopmentConfig, TestingConfig, ProductionConfig, get_config
from app import create_app


def test_default_config():
    """Verify default base configuration attributes."""
    config = Config()
    assert config.ALLOWED_EXTENSIONS == {"png", "jpg", "jpeg", "webp"}
    assert config.ALLOWED_PIL_FORMATS == {"PNG", "JPEG", "WEBP"}
    assert config.CLASS_NAMES == ["glioma", "meningioma", "notumor", "pituitary"]
    assert config.IMG_SIZE == (224, 224)
    assert config.INPUT_CHANNELS == 3
    assert config.MAX_CONTENT_LENGTH == 16 * 1024 * 1024
    assert config.TEMP_FILE_TTL_SECONDS == 3600
    assert config.MAX_TEMP_FILES == 500


def test_testing_config():
    """Verify testing configuration defaults."""
    assert TestingConfig.TESTING is True
    assert TestingConfig.DEBUG is False
    assert TestingConfig.MODEL_PATH == ""
    assert TestingConfig.TEMP_FILE_TTL_SECONDS == 0


def test_development_config():
    """Verify development configuration defaults."""
    assert DevelopmentConfig.DEBUG is True
    assert DevelopmentConfig.TESTING is False


def test_production_config():
    """Verify production configuration defaults."""
    assert ProductionConfig.TESTING is False
    assert ProductionConfig.DEBUG is False


def test_get_config_by_name():
    """Verify get_config resolves environment configurations correctly."""
    assert get_config("testing") == TestingConfig
    assert get_config("development") == DevelopmentConfig
    assert get_config("production") == ProductionConfig
    # Fallback is ProductionConfig for secure production-by-default behavior
    assert get_config("unknown_fallback") == ProductionConfig


def test_env_var_overrides(monkeypatch):
    """Verify environment variables are read upon config initialization or reload."""
    monkeypatch.setenv("NEUROMAP_LOG_LEVEL", "DEBUG")
    monkeypatch.setenv("NEUROMAP_GRADCAM_LAYER", "custom_conv_layer")
    monkeypatch.setenv("NEUROMAP_TEMP_TTL_SECONDS", "7200")
    
    # Reload config module so class attributes re-evaluate with monkeypatched env
    reloaded_config = importlib.reload(config_module)
    cfg = reloaded_config.Config()
    
    assert cfg.LOG_LEVEL == "DEBUG"
    assert cfg.GRADCAM_LAYER == "custom_conv_layer"
    assert cfg.TEMP_FILE_TTL_SECONDS == 7200


def test_create_app_with_custom_config():
    """Verify app factory respects custom configuration object."""
    test_app = create_app(TestingConfig)
    assert test_app.config["TESTING"] is True
    assert test_app.config["IMG_SIZE"] == (224, 224)
