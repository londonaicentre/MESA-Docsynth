from importlib.resources.abc import Traversable

from docsynth.types.profile import Profiles, Profile
from docsynth.types.wrapper import DocsynthAssets


class PaedAcuteProfile(Profile):
    label: bool
    primary_diagnosis: str
    presenting_complaint: str
    summary_of_note: str
    symptom: str = ""
    source_file: str = ""


class PaedAcuteAssets(DocsynthAssets):
    def __init__(self) -> None:
        super().__init__("docsynth.assets.paedacute")

    def _load_profiles_from_file(self, file_path: Traversable) -> list[Profile]:
        """Load paediatric acute triage profiles from a given file path"""
        paedacute_profiles: list[Profile] = []
        symptom: str = file_path.name
        profiles: Profiles[PaedAcuteProfile] = Profiles[PaedAcuteProfile](file_path)
        for profile_id, profile_data in profiles.items.items():
            paedacute_profiles.append(
                profile_data.model_copy(
                    update={
                        "profile_id": profile_id,
                        "symptom": symptom,
                        "source_file": file_path.name,
                    }
                )
            )
        return paedacute_profiles

    def format_profile_prompt(self, profile: Profile) -> str:
        """Create the profile portion of a docsynth user prompt

        Args:
            profile (Profile): The profile object containing the information
                to include in the prompt

        Returns:
            str: The profile prompt

        """
        assert isinstance(profile, PaedAcuteProfile)
        lines: list[str] = ["## USE THIS PAEDIATRIC PRESENTATION PROFILE"]
        lines.append("")
        lines.append(
            f"**Primary Diagnosis that should appear verbatim in document:** {profile.primary_diagnosis}"
        )
        lines.append("")
        lines.append(f"**Presenting Complaint:** {profile.presenting_complaint}")
        lines.append("")
        lines.append(f"**Summary of Note:** {profile.summary_of_note}")
        lines.append("")
        return "\n".join(lines)
