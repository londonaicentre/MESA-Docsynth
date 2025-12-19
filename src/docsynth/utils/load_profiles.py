import random
from typing import Generator

from docsynth.types.wrapper import DocsynthAssets
from docsynth.types.profile import Profile


class ProfileLoader:
    """Loads in condition & molecular profiles

    Args:
        assets (SchemaLlamaAssets): An assets wrapper object extending
            the SchemaLlamaAssets type

    """

    def __init__(self, assets: DocsynthAssets):
        self.__assets: DocsynthAssets = assets
        self.__all_profiles: list[Profile] = []

    def load_all_profiles(self) -> list[Profile]:
        """Load all profiles present in the asset wrapper

        Returns:
            list: Loaded profiles

        """
        self.__all_profiles.extend(self.__assets.load_all_profiles())
        return self.__all_profiles

    def load_profiles_from_files(self, filenames: list[str]) -> list[Profile]:
        """Load profiles present in the asset wrapper by file name

        Returns:
            list: Loaded profiles

        """
        self.__all_profiles.extend(self.__assets.load_profiles_from_files(filenames))
        return self.__all_profiles

    def get_random_profile(self) -> Profile:
        """Return a random loaded profile

        Returns:
            Profile: Random profile

        """
        if not self.__all_profiles:
            raise ValueError("No profiles loaded.")
        return random.choice(self.__all_profiles)

    def get_sequential_profiles(self) -> Generator[Profile, None, None]:
        """Return a generator containing all loaded profiles

        Returns:
            Generator: Loaded profiles

        """
        if not self.__all_profiles:
            raise ValueError("No profiles loaded.")
        for profile in self.__all_profiles:
            yield profile

    def format_profile_prompt(self, profile: Profile) -> str:
        """Use the asset wrapper formatter to format a profile for use in a prompt

        Args:
            profile (Profile): The profile to format

        Returns:
            str: The formatted profile

        """
        return self.__assets.format_profile_prompt(profile)

    def get_profile_count(self) -> int:
        """Return the number of loaded profiles

        Returns:
            int: Loaded profile number

        """
        return len(self.__all_profiles)
