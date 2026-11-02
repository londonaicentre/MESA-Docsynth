from yaml import safe_load
from typing import Any, Literal

from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import BaseModel, model_validator


class LLMProvider(BaseModel):
    base_url: str | None = None
    model: str
    temperature: float
    max_tokens: int
    api_key: str | None = None
    username: str | None = None
    password: str | None = None
    split_across_models: bool = False


class LLM(BaseModel):
    enabled: bool
    provider: str
    anthropic: LLMProvider | None = None
    gemini: LLMProvider | None = None
    local: LLMProvider | None = None
    mesa_runtime: LLMProvider | None = None

    @model_validator(mode="before")
    @classmethod
    def drop_unselected_providers(cls, data: Any) -> Any:
        """Drop the blocks of providers other than the selected one.

        Args:
            data (Any): Raw LLM settings, merged from pipeline.yml and .env.

        Returns:
            Any: The settings without blocks for unselected providers.

        """
        return {
            key: value
            for key, value in data.items()
            if key in ("enabled", "provider", data.get("provider"))
        }


class ProfileSelection(BaseModel):
    domain: str | None = None
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
