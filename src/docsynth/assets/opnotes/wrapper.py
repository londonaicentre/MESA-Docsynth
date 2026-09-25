from importlib.resources.abc import Traversable

from docsynth.types.profile import Profiles, Profile
from docsynth.types.wrapper import DocsynthAssets


class OpNoteProfile(Profile):
    operation: str
    description: str
    complications: str
    implants: str = ""
    specialty: str = ""
    source_file: str = ""


class OpNoteAssets(DocsynthAssets):
    def __init__(self) -> None:
        super().__init__("docsynth.assets.opnotes")

    def _load_profiles_from_file(self, file_path: Traversable) -> list[Profile]:
        """Load operation note profiles from a given file path"""
        opnote_profiles: list[Profile] = []
        specialty: str = file_path.name
        profiles: Profiles[OpNoteProfile] = Profiles[OpNoteProfile](file_path)
        for profile_id, profile_data in profiles.items.items():
            opnote_profiles.append(
                profile_data.model_copy(
                    update={
                        "profile_id": profile_id,
                        "specialty": specialty,
                        "source_file": file_path.name,
                    }
                )
            )
        return opnote_profiles

    def format_profile_prompt(self, profile: Profile) -> str:
        """Create the profile portion of a docsynth user prompt

        Args:
            profile (Profile): The profile object containing the information
                to include in the prompt

        Returns:
            str: The profile prompt

        """
        assert isinstance(profile, OpNoteProfile)
        lines: list[str] = ["## USE THIS OPERATION PROFILE"]
        lines.append("")
        lines.append(f"**Operation(s) Performed:** {profile.operation}")
        lines.append("")
        lines.append(f"**Patient & Clinical Context:** {profile.description}")
        lines.append("")
        if profile.implants:
            lines.append(f"**Implants / Devices Left In Place:** {profile.implants}")
            lines.append("")
        lines.append(
            f"**Complications (the note must be consistent with this):** {profile.complications}"
        )
        lines.append("")
        return "\n".join(lines).strip()
