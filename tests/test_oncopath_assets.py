import re

import pytest

from docsynth.assets.oncopath.wrapper import OncoPathAssets, OncoPathProfile
from docsynth.types.profile import Profile

# The profile holds disease type, fixed features and history only. These stop
# profiles stating what the content requirements decide (course, treatment, stage,
# specimen, nodes, margins, size, certainty).
COURSE_AND_REPORT_TERMS = re.compile(
    r"recurr|relaps|metasta|neoadjuvant|adjuvant|\bmargins?\b|lymph node|nodal|sentinel|"
    r"\bpT\d|\bpN\d|\bstage\b|biops|resect|excis|\bmm\b|\bcm\b|suspect|\?|"
    r"respon|remission|\bprogress(ed|ion)\b|planned|pending|not performed|multifocal|unifocal",
    re.IGNORECASE,
)
# Features whose status a content section sets, which immutable features must not fix.
CONTENT_OWNED_FEATURES = re.compile(
    r"lymphovascular|perineural|vascular invasion|inflammatory infiltrate|"
    r"treatment effect|extranodal|fibrosis",
    re.IGNORECASE,
)


@pytest.fixture(scope="session")
def oncopath_assets() -> OncoPathAssets:
    return OncoPathAssets()


@pytest.fixture(scope="session")
def oncopath_profiles(oncopath_assets: OncoPathAssets) -> list[Profile]:
    return oncopath_assets.load_all_profiles()


def _profiles(profiles: list[Profile]) -> list[OncoPathProfile]:
    typed: list[OncoPathProfile] = []
    for profile in profiles:
        assert isinstance(profile, OncoPathProfile)
        typed.append(profile)
    return typed


def test_load_profiles(oncopath_profiles: list[Profile]) -> None:
    assert oncopath_profiles
    for profile in _profiles(oncopath_profiles):
        assert profile.working_diagnosis.strip() != ""
        assert profile.clinical_history.strip() != ""
        assert profile.category != ""


def test_profile_ids_unique(oncopath_profiles: list[Profile]) -> None:
    ids = [profile.profile_id for profile in oncopath_profiles]
    assert len(ids) == len(set(ids))


def test_possible_features_present(oncopath_profiles: list[Profile]) -> None:
    for profile in _profiles(oncopath_profiles):
        assert len(profile.possible_features) >= 3, profile.profile_id


def test_clinical_history_holds_background_only(
    oncopath_profiles: list[Profile],
) -> None:
    for profile in _profiles(oncopath_profiles):
        match = COURSE_AND_REPORT_TERMS.search(profile.clinical_history)
        assert match is None, (profile.profile_id, match and match.group())


def test_working_diagnosis_is_unhedged(oncopath_profiles: list[Profile]) -> None:
    for profile in _profiles(oncopath_profiles):
        assert not re.search(
            r"suspect|\?|possible|probable|query|recurr|metasta|respon",
            profile.working_diagnosis,
            re.IGNORECASE,
        ), profile.profile_id


def test_immutable_features_leave_content_to_requirements(
    oncopath_profiles: list[Profile],
) -> None:
    for profile in _profiles(oncopath_profiles):
        for feature in profile.immutable_features:
            assert not COURSE_AND_REPORT_TERMS.search(feature), (
                profile.profile_id,
                feature,
            )
            assert not CONTENT_OWNED_FEATURES.search(feature), (
                profile.profile_id,
                feature,
            )


def test_load_profiles_from_specific_files(oncopath_assets: OncoPathAssets) -> None:
    profiles = oncopath_assets.load_profiles_from_files(["breast_invasive.yml"])
    assert profiles
    assert all(
        isinstance(p, OncoPathProfile) and p.category == "breast_invasive"
        for p in profiles
    )


def test_format_profile_prompt(
    oncopath_assets: OncoPathAssets, oncopath_profiles: list[Profile]
) -> None:
    profile = _profiles(oncopath_profiles)[0]
    prompt: str = oncopath_assets.format_profile_prompt(profile)
    assert "## USE THIS PATIENT PROFILE" in prompt
    assert profile.working_diagnosis in prompt
    assert profile.clinical_history in prompt
    assert all(f"- {feature}" in prompt for feature in profile.immutable_features)
    assert all(f"- {feature}" in prompt for feature in profile.possible_features)


def test_style_and_content_load(oncopath_assets: OncoPathAssets) -> None:
    for data in (
        oncopath_assets.load_style_data(),
        oncopath_assets.load_content_data(),
    ):
        assert data.root
        for section in data.root.values():
            options = section.get_options()
            assert options
            for option in options.values():
                if section.mutually_exclusive:
                    assert option.probability > 0
                else:
                    assert 0 < option.probability <= 1


def test_user_prompt_template(oncopath_assets: OncoPathAssets) -> None:
    template = oncopath_assets.load_user_prompt_template("docsynth")
    assert "{specific_instructions}" in template
    assert "<output>" in template
