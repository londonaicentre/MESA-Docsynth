import pytest

from docsynth.types.profile import Profile
from docsynth.assets.opnotes.wrapper import OpNoteAssets, OpNoteProfile


@pytest.fixture(scope="session")
def opnote_assets() -> OpNoteAssets:
    return OpNoteAssets()


@pytest.fixture(scope="session")
def opnote_profiles(opnote_assets: OpNoteAssets) -> list[Profile]:
    return opnote_assets.load_all_profiles()


def test_load_profiles_from_file(opnote_profiles: list[Profile]) -> None:
    assert len(opnote_profiles) > 0
    for profile in opnote_profiles:
        assert isinstance(profile, OpNoteProfile)
        assert profile.operation != ""
        assert profile.description != ""
        assert profile.complications != ""


def test_load_profiles_from_specific_files(opnote_assets: OpNoteAssets) -> None:
    profiles = opnote_assets.load_profiles_from_files(["colorectal_uncomplicated.yml"])
    assert len(profiles) > 0
    assert all(isinstance(p, OpNoteProfile) for p in profiles)


def test_format_profile_prompt(
    opnote_assets: OpNoteAssets, opnote_profiles: list[Profile]
) -> None:
    first_profile = opnote_profiles[0]
    assert isinstance(first_profile, OpNoteProfile)
    profile_prompt_portion: str = opnote_assets.format_profile_prompt(first_profile)
    # contains boilerplate text
    assert "## USE THIS OPERATION PROFILE" in profile_prompt_portion
    assert first_profile.operation in profile_prompt_portion
    assert first_profile.complications in profile_prompt_portion


def test_asset_domain_resolution() -> None:
    from docsynth.types.wrapper import DocsynthAssets

    assets = DocsynthAssets.from_domain("opnotes")
    assert isinstance(assets, OpNoteAssets)
    assert assets.get_domain() == "opnotes"
