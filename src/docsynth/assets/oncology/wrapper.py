from importlib.resources.abc import Traversable

from docsynth.types.profile import Profiles, Profile
from docsynth.types.wrapper import DocsynthAssets


class OncologyAssets(DocsynthAssets):
    def __init__(self) -> None:
        super().__init__("docsynth.assets.oncology")

    def _load_profiles_from_file(self, file_path: Traversable) -> list[Profile]:
        """Load cancer profiles from a given file path"""
        cancer_profiles: list[Profile] = []
        cancer_type: str = file_path.name
        profiles: Profiles = Profiles(file_path)
        for profile_id, profile_data in profiles.items.items():
            cancer_profiles.append(
                Profile(
                    **{
                        "profile_id": profile_id,
                        "morphology": profile_data.morphology or "UNKNOWN",
                        "descriptive_name": profile_data.descriptive_name or "",
                        "biomarker_profile": profile_data.biomarker_profile or "",
                        "cancer_type": cancer_type,
                        "source_file": file_path.name,
                    }
                )
            )
        return cancer_profiles

    def format_profile_prompt(self, profile: Profile) -> str:
        """Create the profile portion of a docsynth user prompt

        Args:
            profile (Profile): The profile object containing the information
                to include in the prompt

        Returns:
            str: The profile prompt

        """
        lines: list[str] = ["## USE THIS PRIMARY CANCER PROFILE"]
        lines.append("")
        lines.append(
            f"**Primary Diagnosis that should appear verbatim in document:** {profile.descriptive_name}"
        )
        lines.append("")
        lines.append(
            f"**Biomarker Profile - Note that these are the ONLY molecular biomarker results that should be given for this patient:** {profile.biomarker_profile}"
        )
        lines.append("")
        return "\n".join(lines)
