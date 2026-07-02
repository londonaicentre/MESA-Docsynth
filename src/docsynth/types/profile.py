from importlib.resources.abc import Traversable
from typing import Generic, TypeVar

from pydantic import BaseModel
import yaml


class Profile(BaseModel):
    profile_id: str | None = None


ProfileT = TypeVar("ProfileT", bound=Profile)


class Profiles(BaseModel, Generic[ProfileT]):
    items: dict[str, ProfileT] = {}

    def __init__(self, file_path: Traversable) -> None:
        super().__init__(items=yaml.safe_load(file_path.read_text()))
