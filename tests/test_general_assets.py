import pytest

from docsynth.types.profile import Profile
from docsynth.assets.general.wrapper import GeneralAssets, GeneralProfile


@pytest.fixture(scope="session")
def general_assets() -> GeneralAssets:
    return GeneralAssets()


@pytest.fixture(scope="session")
def general_profiles(general_assets: GeneralAssets) -> list[Profile]:
    return general_assets.load_all_profiles()


def test_load_profiles_from_file(general_profiles: list[Profile]) -> None:
    assert len(general_profiles) == 14699
    first_profile: Profile = general_profiles[0]
    assert isinstance(first_profile, GeneralProfile)
    assert first_profile.primary_diagnosis == "Acute Appendicitis"


def test_format_profile_prompt(
    general_assets: GeneralAssets, general_profiles: list[Profile]
) -> None:
    profile_prompt_portion: str = general_assets.format_profile_prompt(
        general_profiles[0]
    )
    # contains boilerplate text
    assert "## USE THIS CLINICAL PROFILE" in profile_prompt_portion
    # contains an example
    assert "Acute Appendicitis" in profile_prompt_portion
