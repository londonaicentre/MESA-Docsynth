from typing import Generator

from schemallama_types.assets import Profile, SchemaLlamaAssets
from docsynth.utils.load_sampling import ConfigSampler
from docsynth.utils.load_profiles import ProfileLoader
from docsynth.utils.load_structure import StructureLoader


class PromptBuilder:
    """Assembles complete prompts from all components

    Args:
        assets (SchemaLlamaAssets): An assets wrapper object extending
            the SchemaLlamaAssets type
        enabled_structures (list): Specified files containing example
            structures to include when building a prompt

    """

    def __init__(self, assets: SchemaLlamaAssets, enabled_structures: list[str]):
        self.config_sampler: ConfigSampler = ConfigSampler(assets)
        self.profile_loader: ProfileLoader = ProfileLoader(assets)
        self.structure_loader: StructureLoader = StructureLoader(
            enabled_structures, assets
        )
        self.structure_loader.load_structures()
        self.template: str = assets.load_prompt_template()

    def load_profiles(self, profile_files: list[str] = []) -> None:
        """Load profiles from specified file(s) or all profiles

        Args:
            profile_files (list, optional): Specified files

        """
        if profile_files:
            self.profile_loader.load_profiles_from_files(profile_files)
        else:
            self.profile_loader.load_all_profiles()

    def get_profile_count(self) -> int:
        """Get total number of loaded profiles

        Returns:
            int: total number of loaded profiles

        """
        return self.profile_loader.get_profile_count()

    def get_random_profile(self) -> Profile:
        """Get random profile when using random mode

        Returns:
            Profile: Profile object

        """
        return self.profile_loader.get_random_profile()

    def get_sequential_profiles(self) -> Generator[Profile, None, None]:
        """Get generator for sequential mode

        Returns:
            Profile: Profile object generator

        """
        return self.profile_loader.get_sequential_profiles()

    def build_prompt(
        self, profile: Profile, include_style: bool = True, include_content: bool = True
    ) -> tuple[str, str, str]:
        """Assemble complete prompt for a given profile

        Args:
            profile (Profile): Given profile
            include_style (bool): Whether to include style instructions in
                the prompt
            include_style (bool): Whether to include content instructions in
                the prompt

        Returns:
            tuple: The prompt, chosen structure file name, and the chosen profile id

        """
        # style / content
        style_prompt: str
        content_prompt: str
        style_prompt, content_prompt = self.config_sampler.generate_prompts()

        # profile
        profile_prompt: str = self.profile_loader.format_profile_prompt(profile)

        # get structure
        structure_filename: str
        structure_content: str
        structure_filename, structure_content = (
            self.structure_loader.get_random_structure()
        )
        structure_name: str = (
            self.structure_loader.get_structure_name_without_extension(
                structure_filename
            )
        )
        structure_prompt: str = self.structure_loader.format_structure_prompt(
            structure_content
        )

        # assemble!
        components: list[str] = []

        if include_style:
            components.append(style_prompt)

        if include_content:
            components.append(content_prompt)

        components.extend([profile_prompt, structure_prompt])
        specific_instructions: str = "\n\n".join(components)
        complete_prompt: str = self.template.format(
            specific_instructions=specific_instructions
        )

        return complete_prompt, structure_name, str(profile.profile_id)
