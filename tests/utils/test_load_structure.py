from dataclasses import dataclass
from unittest.mock import Mock

import pytest
from pytest_mock import MockerFixture

from docsynth.types.wrapper import DocsynthAssets
from docsynth.utils.load_structure import StructureLoader


@dataclass
class StructureLoaderMocks:
    assets: Mock


@pytest.fixture
def structure_loader_mocks(mocker: MockerFixture) -> StructureLoaderMocks:
    return StructureLoaderMocks(assets=mocker.Mock(spec=DocsynthAssets))


class TestLoadStructures:
    def test_load_structures_assets_return_structures_stores_and_returns_them(
        self, structure_loader_mocks: StructureLoaderMocks
    ) -> None:
        structure_loader_mocks.assets.load_structures.return_value = {
            "foo.txt": "foo quux"
        }
        loader: StructureLoader = StructureLoader(
            ["foo.txt"], structure_loader_mocks.assets
        )
        assert loader.load_structures() == {"foo.txt": "foo quux"}
        assert loader.structures == {"foo.txt": "foo quux"}
        structure_loader_mocks.assets.load_structures.assert_called_once_with(
            ["foo.txt"]
        )

    @pytest.mark.parametrize(
        ("structure_filter", "expected"),
        [("foo.*", ["foo.txt", "foobar.txt"]), ("foo\\.txt", ["foo.txt"])],
    )
    def test_load_structures_filter_given_loads_only_full_matches(
        self,
        structure_loader_mocks: StructureLoaderMocks,
        structure_filter: str,
        expected: list[str],
    ) -> None:
        StructureLoader(
            ["foo.txt", "foobar.txt", "bar.txt"],
            structure_loader_mocks.assets,
            structure_filter,
        ).load_structures()
        structure_loader_mocks.assets.load_structures.assert_called_once_with(expected)

    @pytest.mark.parametrize("enabled_structures", [["foo.txt"], []])
    def test_load_structures_filter_matches_nothing_raises_value_error(
        self,
        structure_loader_mocks: StructureLoaderMocks,
        enabled_structures: list[str],
    ) -> None:
        with pytest.raises(ValueError, match="No structures match filter: bar"):
            StructureLoader(
                enabled_structures, structure_loader_mocks.assets, "bar"
            ).load_structures()
        structure_loader_mocks.assets.load_structures.assert_not_called()


class TestGetRandomStructure:
    def test_get_random_structure_no_structures_loaded_returns_none_tuple(
        self, structure_loader_mocks: StructureLoaderMocks
    ) -> None:
        filename, content = StructureLoader(
            [], structure_loader_mocks.assets
        ).get_random_structure()
        assert filename is None
        assert content is None

    def test_get_random_structure_structures_loaded_returns_valid_pair(
        self, structure_loader_mocks: StructureLoaderMocks
    ) -> None:
        structure_loader_mocks.assets.load_structures.return_value = {
            "foo.txt": "foo quux"
        }
        loader: StructureLoader = StructureLoader(
            ["foo.txt"], structure_loader_mocks.assets
        )
        loader.load_structures()
        filename, content = loader.get_random_structure()
        assert filename == "foo.txt"
        assert content == "foo quux"


class TestFormatStructurePrompt:
    def test_format_structure_prompt_content_given_returns_formatted_prompt(
        self, structure_loader_mocks: StructureLoaderMocks
    ) -> None:
        formatted: str = StructureLoader(
            [], structure_loader_mocks.assets
        ).format_structure_prompt("foo quux")
        assert "## MIMIC THIS DOCUMENT STRUCTURE" in formatted
        assert "foo quux" in formatted


class TestGetStructureCount:
    def test_get_structure_count_no_structures_loaded_returns_zero(
        self, structure_loader_mocks: StructureLoaderMocks
    ) -> None:
        assert (
            StructureLoader([], structure_loader_mocks.assets).get_structure_count()
            == 0
        )

    def test_get_structure_count_structures_loaded_returns_count(
        self, structure_loader_mocks: StructureLoaderMocks
    ) -> None:
        structure_loader_mocks.assets.load_structures.return_value = {
            "foo.txt": "foo quux",
            "bar.txt": "bar content",
        }
        loader: StructureLoader = StructureLoader(
            ["foo.txt", "bar.txt"], structure_loader_mocks.assets
        )
        loader.load_structures()
        assert loader.get_structure_count() == 2


class TestGetStructureNameWithoutExtension:
    def test_get_structure_name_without_extension_filename_given_delegates_to_assets(
        self, structure_loader_mocks: StructureLoaderMocks
    ) -> None:
        structure_loader_mocks.assets.get_structure_name_without_extension.return_value = "foo"
        assert (
            StructureLoader(
                [], structure_loader_mocks.assets
            ).get_structure_name_without_extension("foo.txt")
            == "foo"
        )
        structure_loader_mocks.assets.get_structure_name_without_extension.assert_called_once_with(
            "foo.txt"
        )
