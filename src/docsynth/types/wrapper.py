from abc import ABC, abstractmethod
from importlib.resources import files
from importlib.resources.abc import Traversable
from pathlib import Path
from typing import cast

from docsynth.types.profile import Profile
from docsynth.types.sampling import Content, Style


class DocsynthAssets(ABC):
    def __init__(self, base_dir: str) -> None:
        self._base_dir: Traversable = files(base_dir)

    def _load(self, folder: str, file: str) -> str:
        return self._base_dir.joinpath(f"{folder}/{file}").read_text()

    def load_user_prompt_template(self, template_name: str) -> str:
        """Load a user prompt template from wrapped assets.
                Template is assumed to be stored in the form
                `userprompt_{template}.md`

        Args:
            template_name (str): The name of the user prompt template
                to load

        Returns:
            str: The text of the template as a string

        """
        return self._load("prompts", f"userprompt_{template_name}.md")

    def load_all_profiles(self) -> list[Profile]:
        """Load all profiles

        Returns:
            list: Loaded profiles

        """
        all_profiles: list[Profile] = []
        items: list[Traversable] = cast(
            list[Traversable],
            sorted(self._base_dir.joinpath("profiles").iterdir(), key=lambda x: x.name),
        )
        item: Traversable
        for item in items:
            if item.is_file() and item.name.endswith(".yml"):
                profiles: list[Profile] = self._load_profiles_from_file(item)
                all_profiles.extend(profiles)
        return all_profiles

    def load_profiles_from_files(self, filenames: list[str]) -> list[Profile]:
        """Load profiles by file name

        Returns:
            list: Loaded profiles

        """
        all_profiles: list[Profile] = []
        for filename in filenames:
            file_path: str = "profiles/" + filename
            try:
                profiles: list[Profile] = self._load_profiles_from_file(
                    self._base_dir.joinpath(file_path)
                )
                all_profiles.extend(profiles)
            except FileNotFoundError:
                raise FileNotFoundError(f"Profile file not found: {file_path}")
        return all_profiles

    @abstractmethod
    def _load_profiles_from_file(self, file_path: Traversable) -> list[Profile]:
        pass

    @abstractmethod
    def format_profile_prompt(self, profile: Profile) -> str:
        pass

    def load_style_data(self) -> Style:
        """Load data from style file into Style class

        Returns:
            Style: Loaded style data

        """
        return Style(self._base_dir.joinpath("style.yml"))

    def load_content_data(self) -> Content:
        """Load data from content file into Content class

        Returns:
            Content: Loaded content data

        """
        return Content(self._base_dir.joinpath("content.yml"))

    def load_structures(self, enabled_structures: list[str]) -> dict[str, str]:
        """Load structures

        Args:
            enabled_structures (list): Specified files containing example
                structures to include when building a prompt

        Returns:
            dict: A mapping between structure file names and content

        """
        self.structures: dict[str, str] = {}
        for filename in enabled_structures:
            file_path: str = "structure/" + filename
            self.structures[filename] = self._base_dir.joinpath(file_path).read_text()
        return self.structures

    def get_structure_name_without_extension(self, filename: str) -> str:
        """Return a copy of a structure filename without its extension

        Args:
            filename (str): Filename with extension

        Returns:
            str: Structure filename

        """
        return Path(self._base_dir.joinpath("structure/" + filename).name).stem
