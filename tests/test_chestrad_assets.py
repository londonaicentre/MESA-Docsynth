import math

import pytest

from docsynth.types.profile import Profile
from docsynth.assets.chestrad.wrapper import ChestRadAssets, ChestRadProfile


@pytest.fixture(scope="session")
def chestrad_assets() -> ChestRadAssets:
    return ChestRadAssets()


@pytest.fixture(scope="session")
def chestrad_profiles(chestrad_assets: ChestRadAssets) -> list[Profile]:
    return chestrad_assets.load_all_profiles()


def _first_with(profiles: list[Profile], unremarkable: bool) -> ChestRadProfile:
    for profile in profiles:
        assert isinstance(profile, ChestRadProfile)
        if profile.unremarkable is unremarkable:
            return profile
    raise AssertionError(f"no profile with unremarkable={unremarkable}")


def test_load_profiles_from_file(chestrad_profiles: list[Profile]) -> None:
    assert len(chestrad_profiles) == 2000
    first_profile: Profile = chestrad_profiles[0]
    assert isinstance(first_profile, ChestRadProfile)
    assert first_profile.modality_protocol != ""
    assert first_profile.diagnosis != ""
    assert first_profile.description != ""
    assert first_profile.possible_findings


def test_profile_ids_unique(chestrad_profiles: list[Profile]) -> None:
    ids = [profile.profile_id for profile in chestrad_profiles]
    assert len(ids) == len(set(ids))


def test_every_file_has_unremarkable_profiles(
    chestrad_profiles: list[Profile],
) -> None:
    files: dict[str, int] = {}
    for profile in chestrad_profiles:
        assert isinstance(profile, ChestRadProfile)
        files.setdefault(profile.source_file, 0)
        files[profile.source_file] += profile.unremarkable
    assert all(count >= 2 for count in files.values())


def test_load_profiles_from_specific_files(chestrad_assets: ChestRadAssets) -> None:
    profiles = chestrad_assets.load_profiles_from_files(["vascular_acute_pe.yml"])
    assert len(profiles) == 70
    assert all(
        isinstance(p, ChestRadProfile) and p.category == "vascular_acute_pe"
        for p in profiles
    )


def test_format_profile_prompt(
    chestrad_assets: ChestRadAssets, chestrad_profiles: list[Profile]
) -> None:
    profile = _first_with(chestrad_profiles, unremarkable=False)
    prompt: str = chestrad_assets.format_profile_prompt(profile)
    assert "## USE THIS CHEST RADIOLOGY PROFILE" in prompt
    assert profile.modality_protocol in prompt
    assert profile.diagnosis in prompt
    assert all(f"- {finding}" in prompt for finding in profile.possible_findings)
    assert "Expected Outcome" not in prompt


def test_format_unremarkable_profile_prompt(
    chestrad_assets: ChestRadAssets, chestrad_profiles: list[Profile]
) -> None:
    profile = _first_with(chestrad_profiles, unremarkable=True)
    prompt: str = chestrad_assets.format_profile_prompt(profile)
    assert "**Expected Outcome:** This study is normal / unremarkable." in prompt


def test_sampling_weights_sum_to_one(chestrad_assets: ChestRadAssets) -> None:
    for config in (
        chestrad_assets.load_style_data(),
        chestrad_assets.load_content_data(),
    ):
        for name, section in config.root.items():
            weights = [
                item.probability for item in (section.__pydantic_extra__ or {}).values()
            ]
            assert weights, name
            if section.mutually_exclusive:
                assert math.isclose(sum(weights), 1.0), name
            else:
                assert all(0 <= w <= 1 for w in weights), name


def test_asset_domain_resolution() -> None:
    from docsynth.types.wrapper import DocsynthAssets

    assets = DocsynthAssets.from_domain("chestrad")
    assert isinstance(assets, ChestRadAssets)
    assert assets.get_domain() == "chestrad"
