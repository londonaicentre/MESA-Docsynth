from dataclasses import dataclass
from importlib.resources.abc import Traversable
import types
from unittest.mock import MagicMock, Mock, call

import pytest
from pytest_mock import MockerFixture

from docsynth.assets.general.wrapper import GeneralAssets
from docsynth.types.profile import Profile
from docsynth.types.wrapper import DocsynthAssets


class DocsynthAssetsFixture(DocsynthAssets):
    def _load_profiles_from_file(self, file_path: Traversable) -> list[Profile]:
        raise NotImplementedError

    def format_profile_prompt(self, profile: Profile) -> str:
        raise NotImplementedError


@dataclass
class DocsynthAssetsMocks:
    base_dir: Mock
    files: MagicMock


@pytest.fixture
def docsynth_assets_mocks(mocker: MockerFixture) -> DocsynthAssetsMocks:
    base_dir: Mock = mocker.Mock(spec=Traversable)
    return DocsynthAssetsMocks(
        base_dir=base_dir,
        files=mocker.patch("docsynth.types.wrapper.files", return_value=base_dir),
    )


class TestInit:
    def test_init_base_dir_given_resolves_traversable_via_importlib_resources(
        self, docsynth_assets_mocks: DocsynthAssetsMocks
    ) -> None:
        DocsynthAssetsFixture("foo.assets")
        docsynth_assets_mocks.files.assert_called_once_with("foo.assets")


class TestFromDomain:
    def test_from_domain_known_domain_returns_matching_assets_instance(self) -> None:
        assert isinstance(DocsynthAssets.from_domain("general"), GeneralAssets)

    def test_from_domain_unknown_domain_raises_module_not_found_error(self) -> None:
        with pytest.raises(ModuleNotFoundError):
            DocsynthAssets.from_domain("foo")

    def test_from_domain_no_matching_class_raises_value_error(
        self, mocker: MockerFixture
    ) -> None:
        mocker.patch(
            "docsynth.types.wrapper.importlib.import_module",
            return_value=types.ModuleType("docsynth.assets.foo.wrapper"),
        )
        with pytest.raises(ValueError, match="found 0"):
            DocsynthAssets.from_domain("foo")

    def test_from_domain_multiple_matching_classes_raises_value_error(
        self, mocker: MockerFixture
    ) -> None:
        module: types.ModuleType = types.ModuleType("docsynth.assets.foo.wrapper")
        setattr(module, "qux", DocsynthAssetsFixture)
        setattr(module, "quux", DocsynthAssetsFixture)
        mocker.patch(
            "docsynth.types.wrapper.importlib.import_module", return_value=module
        )
        with pytest.raises(ValueError, match="found 2"):
            DocsynthAssets.from_domain("foo")


class TestLoad:
    def test_load_folder_and_file_given_reads_joined_path(
        self, docsynth_assets_mocks: DocsynthAssetsMocks
    ) -> None:
        docsynth_assets_mocks.base_dir.joinpath.return_value.read_text.return_value = (
            "foo foobar"
        )
        assert (
            DocsynthAssetsFixture("foo.assets")._load("foo", "bar.txt") == "foo foobar"
        )
        docsynth_assets_mocks.base_dir.joinpath.assert_called_once_with("foo/bar.txt")


class TestLoadUserPromptTemplate:
    def test_load_user_prompt_template_template_name_given_delegates_to_load(
        self, mocker: MockerFixture, docsynth_assets_mocks: DocsynthAssetsMocks
    ) -> None:
        assets: DocsynthAssetsFixture = DocsynthAssetsFixture("foo.assets")
        load: MagicMock = mocker.patch.object(assets, "_load", return_value="foo quux")
        assert assets.load_user_prompt_template("foo") == "foo quux"
        load.assert_called_once_with("prompts", "userprompt_foo.md")


class TestLoadAllProfiles:
    def test_load_all_profiles_mixed_items_loads_only_yml_files_in_sorted_order(
        self, mocker: MockerFixture, docsynth_assets_mocks: DocsynthAssetsMocks
    ) -> None:
        foo_yml: Mock = mocker.Mock(spec=Traversable)
        foo_yml.name = "foo.yml"
        foo_yml.is_file.return_value = True
        bar_yml: Mock = mocker.Mock(spec=Traversable)
        bar_yml.name = "bar.yml"
        bar_yml.is_file.return_value = True
        foo_txt: Mock = mocker.Mock(spec=Traversable)
        foo_txt.name = "foo.txt"
        foo_txt.is_file.return_value = True
        foo_dir: Mock = mocker.Mock(spec=Traversable)
        foo_dir.name = "foo_bar"
        foo_dir.is_file.return_value = False
        docsynth_assets_mocks.base_dir.joinpath.return_value.iterdir.return_value = [
            foo_yml,
            bar_yml,
            foo_txt,
            foo_dir,
        ]
        assets: DocsynthAssetsFixture = DocsynthAssetsFixture("foo.assets")
        load_profiles_from_file: MagicMock = mocker.patch.object(
            assets,
            "_load_profiles_from_file",
            side_effect=[["bar-garply"], ["foo-garply"]],
        )
        assert assets.load_all_profiles() == ["bar-garply", "foo-garply"]
        load_profiles_from_file.assert_has_calls([call(bar_yml), call(foo_yml)])
        docsynth_assets_mocks.base_dir.joinpath.assert_called_once_with("profiles")

    def test_load_all_profiles_no_items_returns_empty_list(
        self, docsynth_assets_mocks: DocsynthAssetsMocks
    ) -> None:
        docsynth_assets_mocks.base_dir.joinpath.return_value.iterdir.return_value = []
        assert DocsynthAssetsFixture("foo.assets").load_all_profiles() == []


class TestLoadProfilesFromFiles:
    def test_load_profiles_from_files_filenames_given_extends_and_returns_them(
        self, mocker: MockerFixture, docsynth_assets_mocks: DocsynthAssetsMocks
    ) -> None:
        assets: DocsynthAssetsFixture = DocsynthAssetsFixture("foo.assets")
        mocker.patch.object(
            assets, "_load_profiles_from_file", return_value=["foo-garply"]
        )
        assert assets.load_profiles_from_files(["foo.yml"]) == ["foo-garply"]
        docsynth_assets_mocks.base_dir.joinpath.assert_called_once_with(
            "profiles/foo.yml"
        )

    def test_load_profiles_from_files_file_not_found_raises_file_not_found_error(
        self, mocker: MockerFixture, docsynth_assets_mocks: DocsynthAssetsMocks
    ) -> None:
        assets: DocsynthAssetsFixture = DocsynthAssetsFixture("foo.assets")
        mocker.patch.object(
            assets, "_load_profiles_from_file", side_effect=FileNotFoundError
        )
        with pytest.raises(
            FileNotFoundError, match="Profile file not found: profiles/foo.yml"
        ):
            assets.load_profiles_from_files(["foo.yml"])


class TestLoadStyleData:
    def test_load_style_data_style_file_given_returns_parsed_style(
        self, mocker: MockerFixture, docsynth_assets_mocks: DocsynthAssetsMocks
    ) -> None:
        style: MagicMock = mocker.patch("docsynth.types.wrapper.Style")
        assert (
            DocsynthAssetsFixture("foo.assets").load_style_data() == style.return_value
        )
        style.assert_called_once_with(
            docsynth_assets_mocks.base_dir.joinpath.return_value
        )
        docsynth_assets_mocks.base_dir.joinpath.assert_called_once_with("style.yml")


class TestLoadContentData:
    def test_load_content_data_content_file_given_returns_parsed_content(
        self, mocker: MockerFixture, docsynth_assets_mocks: DocsynthAssetsMocks
    ) -> None:
        content: MagicMock = mocker.patch("docsynth.types.wrapper.Content")
        assert (
            DocsynthAssetsFixture("foo.assets").load_content_data()
            == content.return_value
        )
        content.assert_called_once_with(
            docsynth_assets_mocks.base_dir.joinpath.return_value
        )
        docsynth_assets_mocks.base_dir.joinpath.assert_called_once_with("content.yml")


class TestLoadStructures:
    def test_load_structures_enabled_structures_given_reads_and_returns_mapping(
        self, docsynth_assets_mocks: DocsynthAssetsMocks
    ) -> None:
        docsynth_assets_mocks.base_dir.joinpath.return_value.read_text.side_effect = [
            "foo foobar",
            "bar quux",
        ]
        assert DocsynthAssetsFixture("foo.assets").load_structures(
            ["foo.txt", "bar.txt"]
        ) == {"foo.txt": "foo foobar", "bar.txt": "bar quux"}
        assert docsynth_assets_mocks.base_dir.joinpath.call_args_list == [
            call("structure/foo.txt"),
            call("structure/bar.txt"),
        ]

    def test_load_structures_no_enabled_structures_returns_empty_mapping(
        self, docsynth_assets_mocks: DocsynthAssetsMocks
    ) -> None:
        assert DocsynthAssetsFixture("foo.assets").load_structures([]) == {}


class TestGetStructureNameWithoutExtension:
    def test_get_structure_name_without_extension_filename_given_returns_stem(
        self, docsynth_assets_mocks: DocsynthAssetsMocks
    ) -> None:
        docsynth_assets_mocks.base_dir.joinpath.return_value.name = "foo.txt"
        assert (
            DocsynthAssetsFixture("foo.assets").get_structure_name_without_extension(
                "foo.txt"
            )
            == "foo"
        )
        docsynth_assets_mocks.base_dir.joinpath.assert_called_once_with(
            "structure/foo.txt"
        )
