from importlib.resources.abc import Traversable

from docsynth.types.profile import Profiles, Profile
from docsynth.types.wrapper import DocsynthAssets


class OncoRadProfile(Profile):
    modality: str = ""
    diagnosis: str = ""
    description: str = ""
    cancer_type: str = ""
    source_file: str = ""


class OncoRadAssets(DocsynthAssets):
    def __init__(self) -> None:
        super().__init__("docsynth.assets.oncorad")

    def _load_profiles_from_file(self, file_path: Traversable) -> list[Profile]:
        """Load oncology radiology profiles from a given file path"""
        oncorad_profiles: list[Profile] = []
        cancer_type: str = file_path.name
        profiles: Profiles[OncoRadProfile] = Profiles[OncoRadProfile](file_path)
        for profile_id, profile_data in profiles.items.items():
            oncorad_profiles.append(
                profile_data.model_copy(
                    update={
                        "profile_id": profile_id,
                        "cancer_type": cancer_type,
                        "source_file": file_path.name,
                    }
                )
            )
        return oncorad_profiles

    def format_profile_prompt(self, profile: Profile) -> str:
        """Create the profile portion of a docsynth user prompt

        Args:
            profile (Profile): The profile object containing the information
                to include in the prompt

        Returns:
            str: The profile prompt

        """
        assert isinstance(profile, OncoRadProfile)
        lines: list[str] = ["## USE THIS ONCOLOGY RADIOLOGY PROFILE"]
        lines.append("")
        if profile.modality:
            lines.append(f"**Imaging Examination / Modality:** {profile.modality}")
            lines.append("")
        if profile.diagnosis:
            lines.append(f"**Diagnosis / Cancer Type:** {profile.diagnosis}")
            lines.append("")
        if profile.description:
            lines.append(f"**Patient History & Context:** {profile.description}")
            lines.append("")
        return "\n".join(lines).strip()
