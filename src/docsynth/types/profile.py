from importlib.resources.abc import Traversable
from litellm import BaseModel

from pydantic import ConfigDict
import yaml


class Profile(BaseModel):
    profile_id: str | None = None
    morphology: str = ""
    descriptive_name: str = ""
    biomarker_profile: str = ""

    model_config = ConfigDict(
        extra="allow",
    )


class Profiles(BaseModel):
    items: dict[str, Profile] = {}

    def __init__(self, file_path: Traversable) -> None:
        super().__init__(items=yaml.safe_load(file_path.read_text()))
