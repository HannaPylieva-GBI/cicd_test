"""Configuration of shopper persuasion service."""

from pathlib import Path, PosixPath

from pydantic import BaseSettings


class Config(BaseSettings):
    """Configuration of shopper persuasion service."""
    MODEL_FILE_NAME: str = 'model.joblib'
    MODEL_DIR: str = 'gs://groupby-developmentaip-20220726183540'


def get_config() -> Config:
    """Return persuasion service config.

    :return: config
    """
    return Config()
