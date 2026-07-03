from dataclasses import dataclass
from typing import Generator
from unittest.mock import MagicMock, Mock

import pytest
from pytest_mock import MockerFixture

from docsynth.types.profile import Profile
from docsynth.types.wrapper import DocsynthAssets
from docsynth.utils.build_prompt import PromptBuilder


@dataclass
class PromptBuilderMocks:
    assets: Mock
    config_sampler: MagicMock
    profile_loader: MagicMock
    names_locations_loader: MagicMock
    structure_loader: MagicMock


@pytest.fixture
def prompt_builder_mocks(mocker: MockerFixture) -> PromptBuilderMocks:
    assets: Mock = mocker.Mock(spec=DocsynthAssets)
    assets.load_user_prompt_template.return_value = "{specific_instructions}"
    config_sampler: MagicMock = mocker.patch(
        "docsynth.utils.build_prompt.ConfigSampler"
    )
    config_sampler.return_value.generate_prompts.return_value = (
        "foo bar",
        "baz qux",
    )
    profile_loader: MagicMock = mocker.patch(
        "docsynth.utils.build_prompt.ProfileLoader"
    )
    profile_loader.return_value.format_profile_prompt.return_value = "waldo grault"
    names_locations_loader: MagicMock = mocker.patch(
        "docsynth.utils.build_prompt.NamesLocationsLoader"
    )
    names_locations_loader.return_value.sample.return_value = {"patient_name": "foo"}
    names_locations_loader.return_value.format_prompt.return_value = "corge grault"
    structure_loader: MagicMock = mocker.patch(
        "docsynth.utils.build_prompt.StructureLoader"
    )
    structure_loader.return_value.get_random_structure.return_value = (None, None)
    return PromptBuilderMocks(
        assets=assets,
        config_sampler=config_sampler,
        profile_loader=profile_loader,
        names_locations_loader=names_locations_loader,
        structure_loader=structure_loader,
    )


class TestInit:
    def test_init_assets_and_structures_given_constructs_collaborators(
        self, prompt_builder_mocks: PromptBuilderMocks
    ) -> None:
        builder: PromptBuilder = PromptBuilder(prompt_builder_mocks.assets, ["foo.txt"])
        prompt_builder_mocks.config_sampler.assert_called_once_with(
            prompt_builder_mocks.assets
        )
        prompt_builder_mocks.profile_loader.assert_called_once_with(
            prompt_builder_mocks.assets
        )
        prompt_builder_mocks.names_locations_loader.assert_called_once_with()
        prompt_builder_mocks.structure_loader.assert_called_once_with(
            ["foo.txt"], prompt_builder_mocks.assets
        )
        prompt_builder_mocks.structure_loader.return_value.load_structures.assert_called_once_with()
        prompt_builder_mocks.assets.load_user_prompt_template.assert_called_once_with(
            "docsynth"
        )
        assert builder.template == "{specific_instructions}"


class TestLoadProfiles:
    def test_load_profiles_files_given_loads_from_files(
        self, prompt_builder_mocks: PromptBuilderMocks
    ) -> None:
        PromptBuilder(prompt_builder_mocks.assets, []).load_profiles(["foo.yml"])
        prompt_builder_mocks.profile_loader.return_value.load_profiles_from_files.assert_called_once_with(
            ["foo.yml"]
        )
        prompt_builder_mocks.profile_loader.return_value.load_all_profiles.assert_not_called()

    def test_load_profiles_no_files_given_loads_all_profiles(
        self, prompt_builder_mocks: PromptBuilderMocks
    ) -> None:
        PromptBuilder(prompt_builder_mocks.assets, []).load_profiles()
        prompt_builder_mocks.profile_loader.return_value.load_all_profiles.assert_called_once_with()
        prompt_builder_mocks.profile_loader.return_value.load_profiles_from_files.assert_not_called()


class TestGetProfileCount:
    def test_get_profile_count_returns_loader_count(
        self, prompt_builder_mocks: PromptBuilderMocks
    ) -> None:
        prompt_builder_mocks.profile_loader.return_value.get_profile_count.return_value = 2
        assert PromptBuilder(prompt_builder_mocks.assets, []).get_profile_count() == 2


class TestFilterExistingProfiles:
    def test_filter_existing_profiles_ids_given_delegates_to_loader(
        self, prompt_builder_mocks: PromptBuilderMocks
    ) -> None:
        prompt_builder_mocks.profile_loader.return_value.filter_existing_profiles.return_value = 1
        assert (
            PromptBuilder(prompt_builder_mocks.assets, []).filter_existing_profiles(
                {"foo"}
            )
            == 1
        )
        prompt_builder_mocks.profile_loader.return_value.filter_existing_profiles.assert_called_once_with(
            {"foo"}
        )


class TestGetRandomProfile:
    def test_get_random_profile_returns_loader_profile(
        self, prompt_builder_mocks: PromptBuilderMocks
    ) -> None:
        profile: Profile = Profile(profile_id="foo")
        prompt_builder_mocks.profile_loader.return_value.get_random_profile.return_value = profile
        assert (
            PromptBuilder(prompt_builder_mocks.assets, []).get_random_profile()
            is profile
        )


class TestGetSequentialProfiles:
    def test_get_sequential_profiles_returns_loader_generator(
        self, prompt_builder_mocks: PromptBuilderMocks
    ) -> None:
        profiles: Generator[Profile, None, None] = (
            profile for profile in [Profile(profile_id="foo")]
        )
        prompt_builder_mocks.profile_loader.return_value.get_sequential_profiles.return_value = profiles
        assert (
            PromptBuilder(prompt_builder_mocks.assets, []).get_sequential_profiles()
            is profiles
        )


class TestBuildPrompt:
    @pytest.mark.parametrize(
        ("include_style", "include_content", "expected_components"),
        [
            (
                True,
                True,
                ["foo bar", "baz qux", "waldo grault", "corge grault"],
            ),
            (False, True, ["baz qux", "waldo grault", "corge grault"]),
            (True, False, ["foo bar", "waldo grault", "corge grault"]),
            (False, False, ["waldo grault", "corge grault"]),
        ],
    )
    def test_build_prompt_no_structure_found_joins_expected_components(
        self,
        prompt_builder_mocks: PromptBuilderMocks,
        include_style: bool,
        include_content: bool,
        expected_components: list[str],
    ) -> None:
        profile: Profile = Profile(profile_id="foo")
        assert PromptBuilder(prompt_builder_mocks.assets, []).build_prompt(
            profile, include_style, include_content
        ) == ("\n\n".join(expected_components), "nostructure", "foo")
        prompt_builder_mocks.profile_loader.return_value.format_profile_prompt.assert_called_once_with(
            profile
        )
        prompt_builder_mocks.names_locations_loader.return_value.format_prompt.assert_called_once_with(
            {"patient_name": "foo"}
        )
        prompt_builder_mocks.structure_loader.return_value.get_structure_name_without_extension.assert_not_called()
        prompt_builder_mocks.structure_loader.return_value.format_structure_prompt.assert_not_called()

    def test_build_prompt_structure_found_appends_structure_prompt(
        self, prompt_builder_mocks: PromptBuilderMocks
    ) -> None:
        prompt_builder_mocks.structure_loader.return_value.get_random_structure.return_value = (
            "foo.txt",
            "structure content",
        )
        prompt_builder_mocks.structure_loader.return_value.get_structure_name_without_extension.return_value = "foo"
        prompt_builder_mocks.structure_loader.return_value.format_structure_prompt.return_value = "plugh xyzzy"
        assert PromptBuilder(prompt_builder_mocks.assets, []).build_prompt(
            Profile(profile_id="foo")
        ) == (
            "\n\n".join(
                [
                    "foo bar",
                    "baz qux",
                    "waldo grault",
                    "corge grault",
                    "plugh xyzzy",
                ]
            ),
            "foo",
            "foo",
        )
        prompt_builder_mocks.structure_loader.return_value.get_structure_name_without_extension.assert_called_once_with(
            "foo.txt"
        )
        prompt_builder_mocks.structure_loader.return_value.format_structure_prompt.assert_called_once_with(
            "structure content"
        )
