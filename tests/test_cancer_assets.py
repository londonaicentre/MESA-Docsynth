import pytest

from docsynth.types.profile import Profile
from docsynth.assets.cancer.wrapper import CancerAssets, CancerProfile


@pytest.fixture(scope="session")
def cancer_assets() -> CancerAssets:
    return CancerAssets()


def test_load_profiles_from_file(cancer_assets: CancerAssets) -> None:
    profiles: list[Profile] = cancer_assets.load_all_profiles()
    assert len(profiles) == 2550
    first_profile: Profile = profiles[0]
    assert isinstance(first_profile, CancerProfile)
    assert first_profile.descriptive_name == "Cholangiocarcinoma"


def test_format_profile_prompt(cancer_assets: CancerAssets) -> None:
    profiles: list[Profile] = cancer_assets.load_all_profiles()
    profile_prompt_portion: str = cancer_assets.format_profile_prompt(profiles[0])
    # contains boilerplate text
    assert "## USE THIS PRIMARY CANCER PROFILE" in profile_prompt_portion
    # contains an example
    assert "Cholangiocarcinoma" in profile_prompt_portion
