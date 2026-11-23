import random
from importlib.resources.abc import Traversable

from docsynth.types.profile import Profiles, Profile
from docsynth.types.wrapper import DocsynthAssets


class OncoPathProfile(Profile):
    working_diagnosis: str
    immutable_features: list[str] = []
    clinical_history: str
    possible_features: list[str] = []
    category: str = ""
    source_file: str = ""


class OncoPathAssets(DocsynthAssets):
    def __init__(self) -> None:
        super().__init__("docsynth.assets.oncopath")

    def _load_profiles_from_file(self, file_path: Traversable) -> list[Profile]:
        """Load oncopath profiles from a given file path"""
        oncopath_profiles: list[Profile] = []
        profiles: Profiles[OncoPathProfile] = Profiles[OncoPathProfile](file_path)
        for profile_id, profile_data in profiles.items.items():
            oncopath_profiles.append(
                profile_data.model_copy(
                    update={
                        "profile_id": profile_id,
                        "category": file_path.name.removesuffix(".yml"),
                        "source_file": file_path.name,
                    }
                )
            )
        return oncopath_profiles

    def format_profile_prompt(self, profile: Profile) -> str:
        """Create the profile portion of a docsynth user prompt

        Args:
            profile (Profile): The profile object containing the information
                to include in the prompt

        Returns:
            str: The profile prompt

        """
        assert isinstance(profile, OncoPathProfile)
        lines: list[str] = ["## USE THIS PATIENT PROFILE"]
        lines.append("")
        lines.append(
            "The profile describes the patient and the type of disease only. It does not "
            "say what the report contains, which specimens were taken, or where the "
            "disease is in its course: the style and content requirements decide those. "
            "The profile fixes the type of disease, its fixed features and the patient's "
            "history. The requirements fix what the report says about them."
        )
        lines.append("")
        lines.append(
            f"**Working Diagnosis (the type of disease; keep the report consistent with it):** {profile.working_diagnosis}"
        )
        lines.append("")
        if profile.immutable_features:
            lines.append(
                "**Fixed Features of This Disease (never contradict these; mention them only as the requirements direct):**"
            )
            lines.extend(f"- {feature}" for feature in profile.immutable_features)
            lines.append("")
        lines.append(f"**Clinical History:** {profile.clinical_history}")
        lines.append("")
        if profile.possible_features:
            lines.append(
                "**Optional Features (draw on these where the requirements call for "
                "an additional named lesion or feature; none is required):**"
            )
            # shuffled so the generator does not always pick the first-listed features
            features: list[str] = random.sample(
                profile.possible_features, k=len(profile.possible_features)
            )
            lines.extend(f"- {feature}" for feature in features)
        return "\n".join(lines).strip()
