from importlib.resources.abc import Traversable

from docsynth.types.profile import Profiles, Profile
from docsynth.types.wrapper import DocsynthAssets


class GeneralProfile(Profile):
    primary_diagnosis: str
    presenting_complaint: str
    summary_of_course: str
    topic: str = ""
    source_file: str = ""


class GeneralAssets(DocsynthAssets):
    def __init__(self) -> None:
        super().__init__("docsynth.assets.general")

    def _load_profiles_from_file(self, file_path: Traversable) -> list[Profile]:
        """Load general clinical profiles from a given file path"""
        general_profiles: list[Profile] = []
        topic: str = file_path.name
        profiles: Profiles[GeneralProfile] = Profiles[GeneralProfile](file_path)
        for profile_id, profile_data in profiles.items.items():
            general_profiles.append(
                profile_data.model_copy(
                    update={
                        "profile_id": profile_id,
                        "topic": topic,
                        "source_file": file_path.name,
                    }
                )
            )
        return general_profiles

    def format_profile_prompt(self, profile: Profile) -> str:
        """Create the profile portion of a docsynth user prompt

        Args:
            profile (Profile): The profile object containing the information
                to include in the prompt

        Returns:
            str: The profile prompt

        """
        assert isinstance(profile, GeneralProfile)
        lines: list[str] = ["## USE THIS CLINICAL PROFILE"]
        lines.append("")
        lines.append(
            f"**Primary Diagnosis that should appear verbatim in document:** {profile.primary_diagnosis}"
        )
        lines.append("")
        lines.append(f"**Presenting Complaint:** {profile.presenting_complaint}")
        lines.append("")
        lines.append(f"**Summary of Course:** {profile.summary_of_course}")
        lines.append("")
        return "\n".join(lines)
