import pytest

from docsynth.types.profile import Profile
from docsynth.assets.oncology.wrapper import OncologyAssets


@pytest.fixture(scope="session")
def oncology_assets() -> OncologyAssets:
    return OncologyAssets()


def test_load_profiles_from_file(oncology_assets: OncologyAssets) -> None:
    profiles: list[Profile] = oncology_assets.load_all_profiles()
    assert len(profiles) == 2550
    assert profiles[0].descriptive_name == "Cholangiocarcinoma"


def test_format_profile_prompt(oncology_assets: OncologyAssets) -> None:
    profiles: list[Profile] = oncology_assets.load_all_profiles()
    profile_prompt_portion: str = oncology_assets.format_profile_prompt(profiles[0])
    # contains boilerplate text
    assert "## USE THIS PRIMARY CANCER PROFILE" in profile_prompt_portion
    # contains an example
    assert "Cholangiocarcinoma" in profile_prompt_portion
