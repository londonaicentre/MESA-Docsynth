from yaml import safe_load
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import BaseModel


class LLMProvider(BaseModel):
    base_url: str | None = None
    model: str
    temperature: float
    max_tokens: int
    api_key: str | None = None


class LLM(BaseModel):
    enabled: bool
    provider: str
    anthropic: LLMProvider
    gemini: LLMProvider
    local: LLMProvider


class ProfileSelection(BaseModel):
    mode: Literal["random", "sequential"]
    count: int
    file: list[str] | None


class StructureSelection(BaseModel):
    enabled_structures: list[str] | None = None


class PromptConfig(BaseModel):
    include_style: bool
    include_content: bool


class Output(BaseModel):
    subdirectory: str
    skip_existing: bool = False
    domain: str | None = None
    description: str | None = None
    upload_enabled: bool = False


class PipelineConfig(BaseSettings):
    llm: LLM
    profile_selection: ProfileSelection
    structure_selection: StructureSelection
    prompt_config: PromptConfig
    output: Output

    model_config = SettingsConfigDict(
        env_file=".env",
        env_nested_delimiter="__",
        extra="allow",
    )

    def __init__(self) -> None:
        with open("pipeline.yml") as pipeline:
            super().__init__(**safe_load(pipeline))
