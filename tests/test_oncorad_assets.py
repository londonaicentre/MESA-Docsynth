import pytest

from docsynth.types.profile import Profile
from docsynth.assets.oncorad.wrapper import OncoRadAssets, OncoRadProfile


@pytest.fixture(scope="session")
def oncorad_assets() -> OncoRadAssets:
    return OncoRadAssets()


@pytest.fixture(scope="session")
def oncorad_profiles(oncorad_assets: OncoRadAssets) -> list[Profile]:
    return oncorad_assets.load_all_profiles()


def test_load_profiles_from_file(oncorad_profiles: list[Profile]) -> None:
    assert len(oncorad_profiles) == 1285
    first_profile: Profile = oncorad_profiles[0]
    assert isinstance(first_profile, OncoRadProfile)
    assert first_profile.diagnosis != ""
    assert first_profile.modality != ""


def test_load_profiles_from_specific_files(oncorad_assets: OncoRadAssets) -> None:
    profiles = oncorad_assets.load_profiles_from_files(["breast_idc.yml"])
    assert len(profiles) == 55
    assert all(isinstance(p, OncoRadProfile) for p in profiles)


def test_format_profile_prompt(
    oncorad_assets: OncoRadAssets, oncorad_profiles: list[Profile]
) -> None:
    first_profile = oncorad_profiles[0]
    assert isinstance(first_profile, OncoRadProfile)
    profile_prompt_portion: str = oncorad_assets.format_profile_prompt(
        first_profile
    )
    # contains boilerplate text
    assert "## USE THIS ONCOLOGY RADIOLOGY PROFILE" in profile_prompt_portion
    assert first_profile.diagnosis in profile_prompt_portion


def test_asset_domain_resolution() -> None:
    from docsynth.types.wrapper import DocsynthAssets

    assets = DocsynthAssets.from_domain("oncorad")
    assert isinstance(assets, OncoRadAssets)
    assert assets.get_domain() == "oncorad"
