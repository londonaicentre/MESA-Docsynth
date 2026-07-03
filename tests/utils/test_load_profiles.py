from dataclasses import dataclass
from unittest.mock import Mock

import pytest
from pytest_mock import MockerFixture

from docsynth.types.wrapper import DocsynthAssets
from docsynth.types.profile import Profile
from docsynth.utils.load_profiles import ProfileLoader


def make_profiles(*profile_ids: str) -> list[Profile]:
    return [Profile(profile_id=profile_id) for profile_id in profile_ids]


@dataclass
class ProfileLoaderMocks:
    assets: Mock


@pytest.fixture
def profile_loader_mocks(mocker: MockerFixture) -> ProfileLoaderMocks:
    return ProfileLoaderMocks(assets=mocker.Mock(spec=DocsynthAssets))


class TestLoadAllProfiles:
    def test_load_all_profiles_assets_return_profiles_extends_and_returns_them(
        self, profile_loader_mocks: ProfileLoaderMocks
    ) -> None:
        profiles: list[Profile] = make_profiles("foo", "bar")
        profile_loader_mocks.assets.load_all_profiles.return_value = profiles
        loader: ProfileLoader = ProfileLoader(profile_loader_mocks.assets)
        assert loader.load_all_profiles() == profiles
        assert loader.get_profile_count() == 2


class TestLoadProfilesFromFiles:
    def test_load_profiles_from_files_filenames_given_extends_and_returns_them(
        self, profile_loader_mocks: ProfileLoaderMocks
    ) -> None:
        profiles: list[Profile] = make_profiles("foo")
        profile_loader_mocks.assets.load_profiles_from_files.return_value = profiles
        assert (
            ProfileLoader(profile_loader_mocks.assets).load_profiles_from_files(
                ["foo.yml"]
            )
            == profiles
        )
        profile_loader_mocks.assets.load_profiles_from_files.assert_called_once_with(
            ["foo.yml"]
        )


class TestGetRandomProfile:
    def test_get_random_profile_profiles_loaded_returns_loaded_profile(
        self, profile_loader_mocks: ProfileLoaderMocks
    ) -> None:
        profiles: list[Profile] = make_profiles("foo", "bar", "baz")
        profile_loader_mocks.assets.load_all_profiles.return_value = profiles
        loader: ProfileLoader = ProfileLoader(profile_loader_mocks.assets)
        loader.load_all_profiles()
        assert loader.get_random_profile() in profiles

    def test_get_random_profile_no_profiles_loaded_raises_value_error(
        self, profile_loader_mocks: ProfileLoaderMocks
    ) -> None:
        with pytest.raises(ValueError, match="No profiles loaded"):
            ProfileLoader(profile_loader_mocks.assets).get_random_profile()


class TestGetSequentialProfiles:
    def test_get_sequential_profiles_profiles_loaded_yields_all_profiles(
        self, profile_loader_mocks: ProfileLoaderMocks
    ) -> None:
        profiles: list[Profile] = make_profiles("foo", "bar")
        profile_loader_mocks.assets.load_all_profiles.return_value = profiles
        loader: ProfileLoader = ProfileLoader(profile_loader_mocks.assets)
        loader.load_all_profiles()
        assert list(loader.get_sequential_profiles()) == profiles

    def test_get_sequential_profiles_no_profiles_loaded_raises_value_error(
        self, profile_loader_mocks: ProfileLoaderMocks
    ) -> None:
        with pytest.raises(ValueError, match="No profiles loaded"):
            list(ProfileLoader(profile_loader_mocks.assets).get_sequential_profiles())


class TestFormatProfilePrompt:
    def test_format_profile_prompt_profile_given_delegates_to_assets(
        self, profile_loader_mocks: ProfileLoaderMocks
    ) -> None:
        profile: Profile = Profile(profile_id="foo")
        profile_loader_mocks.assets.format_profile_prompt.return_value = "foobar"
        assert (
            ProfileLoader(profile_loader_mocks.assets).format_profile_prompt(profile)
            == "foobar"
        )
        profile_loader_mocks.assets.format_profile_prompt.assert_called_once_with(
            profile
        )


class TestFilterExistingProfiles:
    def test_filter_existing_profiles_matching_ids_removes_expected_count(
        self, profile_loader_mocks: ProfileLoaderMocks
    ) -> None:
        profile_loader_mocks.assets.load_all_profiles.return_value = make_profiles(
            "foo", "bar", "baz"
        )
        loader: ProfileLoader = ProfileLoader(profile_loader_mocks.assets)
        loader.load_all_profiles()
        assert loader.filter_existing_profiles({"foo", "baz"}) == 2
        assert loader.get_profile_count() == 1

    def test_filter_existing_profiles_no_matching_ids_removes_none(
        self, profile_loader_mocks: ProfileLoaderMocks
    ) -> None:
        profile_loader_mocks.assets.load_all_profiles.return_value = make_profiles(
            "foo"
        )
        loader: ProfileLoader = ProfileLoader(profile_loader_mocks.assets)
        loader.load_all_profiles()
        assert loader.filter_existing_profiles({"garply"}) == 0
        assert loader.get_profile_count() == 1
