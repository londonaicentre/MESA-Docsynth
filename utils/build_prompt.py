from pathlib import Path
from utils.load_sampling import ConfigSampler
from utils.load_profiles import ProfileLoader
from utils.load_structure import StructureLoader

"""
build_prompt.py - assembles complete prompts from all components
"""


class PromptBuilder:
    def __init__(
        self,
        template_name="default",
        enabled_structures=None,
        style_file=None,
        content_file=None,
        domain=None,
    ):
        """
        Initialise with config parameters
        """
        self.config_sampler = ConfigSampler(style_file, content_file)
        self.profile_loader = ProfileLoader(domain)
        self.structure_loader = StructureLoader(enabled_structures)
        self.structure_loader.load_structures()

        template_path = Path(__file__).parent.parent / "prompts" / f"{template_name}.md"
        with open(template_path, "r") as f:
            self.template = f.read()

    def load_profiles(self, profile_files=None):
        """
        Load profiles from specified file(s) or all profiles
        """
        if profile_files:
            self.profile_loader.load_profiles_from_files(profile_files)
        else:
            self.profile_loader.load_all_profiles()

    def get_profile_count(self):
        """
        Get total number of loaded profiles
        """
        return self.profile_loader.get_profile_count()

    def get_random_profile(self):
        """
        Get random profile when using random mode
        """
        return self.profile_loader.get_random_profile()

    def get_sequential_profiles(self):
        """
        Get generator for sequential mode
        """
        return self.profile_loader.get_sequential_profiles()

    def build_prompt(self, profile, include_style=True, include_content=True):
        """
        Assemble complete prompt for a given profile
        """
        # style / content
        style_prompt, content_prompt = self.config_sampler.generate_prompts()

        # profile
        profile_prompt = self.profile_loader.format_profile_prompt(profile)

        # get structure
        structure_filename, structure_content = (
            self.structure_loader.get_random_structure()
        )
        structure_name = self.structure_loader.get_structure_name_without_extension(
            structure_filename
        )
        structure_prompt = self.structure_loader.format_structure_prompt(
            structure_content
        )

        # assemble!
        components = []

        if include_style:
            components.append(style_prompt)

        if include_content:
            components.append(content_prompt)

        components.extend([profile_prompt, structure_prompt])

        specific_instructions = "\n\n".join(components)
        complete_prompt = self.template.format(
            specific_instructions=specific_instructions
        )

        return complete_prompt, structure_name, profile["profile_id"]
