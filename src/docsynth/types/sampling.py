from importlib.resources.abc import Traversable

from pydantic import BaseModel, ConfigDict, RootModel
import yaml


class SamplingAttribute(BaseModel):
    probability: float
    description: str


class SamplingElement(BaseModel):
    mutually_exclusive: bool = False
    __pydantic_extra__: dict[str, SamplingAttribute]

    model_config = ConfigDict(
        extra="allow",
    )

    @property
    def options(self) -> dict[str, SamplingAttribute]:
        """The section's options, by key.

        Returns:
            dict[str, SamplingAttribute]: The options

        """
        return self.__pydantic_extra__


class Style(RootModel[dict[str, SamplingElement]]):
    def __init__(self, file_path: Traversable) -> None:
        super().__init__(yaml.safe_load(file_path.read_text()))


class Content(RootModel[dict[str, SamplingElement]]):
    def __init__(self, file_path: Traversable) -> None:
        super().__init__(yaml.safe_load(file_path.read_text()))
