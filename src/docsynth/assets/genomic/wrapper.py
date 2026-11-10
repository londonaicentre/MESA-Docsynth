import random
from importlib.resources.abc import Traversable

from docsynth.types.profile import Profiles, Profile
from docsynth.types.wrapper import DocsynthAssets


class GenomicProfile(Profile):
    patient_description: str
    test_methodology: str
    genetic_finding: str
    condition: str = ""
    source_file: str = ""


class GenomicAssets(DocsynthAssets):
    def __init__(self) -> None:
        super().__init__("docsynth.assets.genomic")

    def _load_profiles_from_file(self, file_path: Traversable) -> list[Profile]:
        """Load genomic profiles from a given file path"""
        genomic_profiles: list[Profile] = []
        profiles: Profiles[GenomicProfile] = Profiles[GenomicProfile](file_path)
        for profile_id, profile_data in profiles.items.items():
            genomic_profiles.append(
                profile_data.model_copy(
                    update={
                        "profile_id": profile_id,
                        "condition": file_path.name.removesuffix(".yml"),
                        "source_file": file_path.name,
                    }
                )
            )
        return genomic_profiles

    def format_profile_prompt(self, profile: Profile) -> str:
        """Create the profile portion of a docsynth user prompt

        Args:
            profile (Profile): The profile object containing the information
                to include in the prompt

        Returns:
            str: The profile prompt

        """
        assert isinstance(profile, GenomicProfile)
        lines: list[str] = ["## USE THIS GENETIC TEST PROFILE"]
        lines.append("")
        lines.append(f"**Patient History & Context:** {profile.patient_description}")
        lines.append("")
        lines.append(f"**Genetic test methodology:** {profile.test_methodology}")
        lines.append("")
        lines.append(f"**Test results and clinical interpretation:** {profile.genetic_finding}")
        return "\n".join(lines).strip()
