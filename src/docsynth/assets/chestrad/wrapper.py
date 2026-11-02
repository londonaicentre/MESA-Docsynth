import random
from importlib.resources.abc import Traversable

from docsynth.types.profile import Profiles, Profile
from docsynth.types.wrapper import DocsynthAssets


class ChestRadProfile(Profile):
    modality_protocol: str
    diagnosis: str
    description: str
    possible_findings: list[str]
    unremarkable: bool = False
    category: str = ""
    source_file: str = ""


class ChestRadAssets(DocsynthAssets):
    def __init__(self) -> None:
        super().__init__("docsynth.assets.chestrad")

    def _load_profiles_from_file(self, file_path: Traversable) -> list[Profile]:
        """Load chest radiology profiles from a given file path"""
        chestrad_profiles: list[Profile] = []
        profiles: Profiles[ChestRadProfile] = Profiles[ChestRadProfile](file_path)
        for profile_id, profile_data in profiles.items.items():
            chestrad_profiles.append(
                profile_data.model_copy(
                    update={
                        "profile_id": profile_id,
                        "category": file_path.name.removesuffix(".yml"),
                        "source_file": file_path.name,
                    }
                )
            )
        return chestrad_profiles

    def format_profile_prompt(self, profile: Profile) -> str:
        """Create the profile portion of a docsynth user prompt

        Args:
            profile (Profile): The profile object containing the information
                to include in the prompt

        Returns:
            str: The profile prompt

        """
        assert isinstance(profile, ChestRadProfile)
        lines: list[str] = ["## USE THIS CHEST RADIOLOGY PROFILE"]
        lines.append("")
        lines.append(f"**Imaging Examination / Protocol:** {profile.modality_protocol}")
        lines.append("")
        lines.append(f"**Known or Suspected Diagnosis:** {profile.diagnosis}")
        lines.append("")
        lines.append(f"**Patient History & Context:** {profile.description}")
        lines.append("")
        lines.append(
            "**Possible Findings (use only as the content instructions direct):**"
        )
        # shuffled so the generator does not always pick the first-listed findings
        findings: list[str] = random.sample(
            profile.possible_findings, k=len(profile.possible_findings)
        )
        lines.extend(f"- {finding}" for finding in findings)
        if profile.unremarkable:
            lines.append("")
            lines.append(
                "**Expected Outcome:** This study is normal / unremarkable. Report no "
                "abnormal findings: none of the possible findings is present, and they may "
                "be mentioned only as explicitly absent. This overrides the content "
                "requirements above wherever they ask for findings, diagnoses, differentials, extent, "
                "anatomical detail, scores or additional findings; follow the rest (history, "
                "negatives, technical quality) as written. Any comparison with a prior study "
                "may only describe appearances as unchanged or a previous abnormality as "
                "resolved."
            )
        return "\n".join(lines).strip()
