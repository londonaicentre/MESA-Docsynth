import pytest

from docsynth.types.profile import Profile
from docsynth.assets.paedacute.wrapper import PaedAcuteAssets, PaedAcuteProfile


@pytest.fixture(scope="session")
def paedacute_assets() -> PaedAcuteAssets:
    return PaedAcuteAssets()


@pytest.fixture(scope="session")
def paedacute_profiles(paedacute_assets: PaedAcuteAssets) -> list[Profile]:
    return paedacute_assets.load_all_profiles()


def test_load_profiles_from_file(paedacute_profiles: list[Profile]) -> None:
    assert len(paedacute_profiles) == 396
    first_profile: Profile = paedacute_profiles[0]
    assert isinstance(first_profile, PaedAcuteProfile)
    assert first_profile.primary_diagnosis == "Neonatal Sepsis (GBS)"
    assert first_profile.label is True


def test_format_profile_prompt(
    paedacute_assets: PaedAcuteAssets, paedacute_profiles: list[Profile]
) -> None:
    profile_prompt_portion: str = paedacute_assets.format_profile_prompt(
        paedacute_profiles[0]
    )
    # contains boilerplate text
    assert "## USE THIS PAEDIATRIC PRESENTATION PROFILE" in profile_prompt_portion
    # contains an example
    assert "Neonatal Sepsis (GBS)" in profile_prompt_portion
