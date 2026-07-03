import json
from importlib.resources import files

from pydantic_settings import BaseSettings
from pydantic import BaseModel


class ModelConfig(BaseModel):
    model: str
    region: str
    batch_file: str


class Config(BaseSettings):
    models: dict[str, ModelConfig]
    job_id_file: str = ".job_id.json"

    def __init__(self) -> None:
        super().__init__(
            models=json.loads(
                files("docsynth").joinpath("config/config.json").read_text()
            )
        )
