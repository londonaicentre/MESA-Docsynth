from dataclasses import dataclass
from typing import Any
from unittest.mock import MagicMock, Mock, call

import pytest
from pytest_mock import MockerFixture

from docsynth.types.wrapper import DocsynthAssets
from docsynth.types.sampling import Content, SamplingElement, Style
from docsynth.utils.load_sampling import ConfigSampler


def make_style(sections: dict[str, SamplingElement]) -> Style:
    return Style.model_construct(root=sections)


def make_content(sections: dict[str, SamplingElement]) -> Content:
    return Content.model_construct(root=sections)


class ConfigSamplerFixture(ConfigSampler):
    def sample_section(
        self, section_data: dict[str, Any], option: str | None = None
    ) -> list[dict[str, str]]:
        return self._sample_section(section_data, option)

    @staticmethod
    def check_filter(pattern: str | None, data: Style | Content) -> None:
        ConfigSampler._check_filter(pattern, data)

    @staticmethod
    def filter_section(
        section_name: str, section_data: dict[str, Any], pattern: str | None
    ) -> dict[str, Any]:
        return ConfigSampler._filter_section(section_name, section_data, pattern)


@dataclass
class ConfigSamplerMocks:
    assets: Mock
    random: MagicMock


@pytest.fixture
def config_sampler_mocks(mocker: MockerFixture) -> ConfigSamplerMocks:
    assets: Mock = mocker.Mock(spec=DocsynthAssets)
    assets.load_style_data.return_value = make_style({})
    assets.load_content_data.return_value = make_content({})
    return ConfigSamplerMocks(
        assets=assets,
        random=mocker.patch("docsynth.utils.load_sampling.random"),
    )


@pytest.fixture(scope="module")
def sections() -> dict[str, SamplingElement]:
    return {
        "foo": SamplingElement.model_validate(
            {
                "mutually_exclusive": True,
                "bar": {"probability": 0.5, "description": "bar foobar"},
                "baz": {"probability": 0.3, "description": "baz foobar"},
                "none": {"probability": 0.2, "description": ""},
            }
        ),
        "qux": SamplingElement.model_validate(
            {"quux": {"probability": 0.4, "description": "quux foobar"}}
        ),
    }


class TestInit:
    @pytest.mark.parametrize("pattern", [None, "foo\\..*", "qux\\.quux"])
    def test_init_filters_match_options_loads_data(
        self,
        config_sampler_mocks: ConfigSamplerMocks,
        sections: dict[str, SamplingElement],
        pattern: str | None,
    ) -> None:
        config_sampler_mocks.assets.load_style_data.return_value = make_style(sections)
        config_sampler_mocks.assets.load_content_data.return_value = make_content(
            sections
        )
        assert (
            ConfigSampler(config_sampler_mocks.assets, pattern, pattern).style_data.root
            == sections
        )

    @pytest.mark.parametrize(
        ("style_filter", "content_filter"), [("garply", None), (None, "garply")]
    )
    def test_init_filter_matches_nothing_raises_value_error(
        self,
        config_sampler_mocks: ConfigSamplerMocks,
        sections: dict[str, SamplingElement],
        style_filter: str | None,
        content_filter: str | None,
    ) -> None:
        config_sampler_mocks.assets.load_style_data.return_value = make_style(sections)
        config_sampler_mocks.assets.load_content_data.return_value = make_content(
            sections
        )
        with pytest.raises(ValueError, match="No options match filter: garply"):
            ConfigSampler(config_sampler_mocks.assets, style_filter, content_filter)


class TestCheckFilter:
    @pytest.mark.parametrize("pattern", [None, "foo\\.bar", "foo\\..*", ".*"])
    def test_check_filter_pattern_unset_or_matching_does_not_raise(
        self, sections: dict[str, SamplingElement], pattern: str | None
    ) -> None:
        ConfigSamplerFixture.check_filter(pattern, make_style(sections))

    @pytest.mark.parametrize("pattern", ["garply", "foo", "foo\\.ba", "bar"])
    def test_check_filter_pattern_matches_nothing_raises_value_error(
        self, sections: dict[str, SamplingElement], pattern: str
    ) -> None:
        with pytest.raises(ValueError, match="No options match filter"):
            ConfigSamplerFixture.check_filter(pattern, make_style(sections))


class TestFilterSection:
    @pytest.mark.parametrize(
        ("pattern", "expected"),
        [
            (
                "foo\\.(bar|baz)",
                {
                    "mutually_exclusive": True,
                    "bar": {"probability": 0.5, "description": "bar foobar"},
                    "baz": {"probability": 0.3, "description": "baz foobar"},
                },
            ),
            (
                "foo\\.bar",
                {
                    "mutually_exclusive": True,
                    "bar": {"probability": 0.5, "description": "bar foobar"},
                },
            ),
        ],
    )
    def test_filter_section_mutually_exclusive_matched_keeps_only_matches(
        self,
        sections: dict[str, SamplingElement],
        pattern: str,
        expected: dict[str, Any],
    ) -> None:
        assert (
            ConfigSamplerFixture.filter_section(
                "foo", sections["foo"].model_dump(), pattern
            )
            == expected
        )

    def test_filter_section_independent_matched_keeps_probabilities(
        self, sections: dict[str, SamplingElement]
    ) -> None:
        assert ConfigSamplerFixture.filter_section(
            "qux", sections["qux"].model_dump(), "qux\\..*"
        ) == {
            "mutually_exclusive": False,
            "quux": {"probability": 0.4, "description": "quux foobar"},
        }

    @pytest.mark.parametrize("pattern", [None, "qux\\..*"])
    def test_filter_section_pattern_unset_or_unmatched_returns_section_unchanged(
        self, sections: dict[str, SamplingElement], pattern: str | None
    ) -> None:
        assert (
            ConfigSamplerFixture.filter_section(
                "foo", sections["foo"].model_dump(), pattern
            )
            == sections["foo"].model_dump()
        )


class TestSampleSection:
    @pytest.mark.parametrize(
        ("description", "expected"),
        [
            ("foo foobar", [{"key": "foo", "description": "foo foobar"}]),
            ("", []),
        ],
    )
    def test_sample_section_mutually_exclusive_returns_expected_selection(
        self,
        config_sampler_mocks: ConfigSamplerMocks,
        description: str,
        expected: list[dict[str, str]],
    ) -> None:
        config_sampler_mocks.random.choices.return_value = ["foo"]
        assert (
            ConfigSamplerFixture(config_sampler_mocks.assets).sample_section(
                {
                    "mutually_exclusive": True,
                    "foo": {"probability": 0.5, "description": description},
                }
            )
            == expected
        )
        config_sampler_mocks.random.choices.assert_called_once_with(
            ["foo"], weights=[0.5], k=1
        )

    @pytest.mark.parametrize(
        ("random_value", "description", "expected"),
        [
            (
                0.0,
                "foo foobar",
                [{"key": "foo", "description": "foo foobar"}],
            ),
            (0.0, "", []),
            (1.0, "foo foobar", []),
        ],
    )
    def test_sample_section_not_mutually_exclusive_returns_expected_selection(
        self,
        config_sampler_mocks: ConfigSamplerMocks,
        random_value: float,
        description: str,
        expected: list[dict[str, str]],
    ) -> None:
        config_sampler_mocks.random.random.return_value = random_value
        assert (
            ConfigSamplerFixture(config_sampler_mocks.assets).sample_section(
                {"foo": {"probability": 0.5, "description": description}}
            )
            == expected
        )

    @pytest.mark.parametrize(
        ("description", "expected"),
        [
            ("bar foobar", [{"key": "bar", "description": "bar foobar"}]),
            ("", []),
        ],
    )
    def test_sample_section_option_given_selects_option(
        self,
        config_sampler_mocks: ConfigSamplerMocks,
        description: str,
        expected: list[dict[str, str]],
    ) -> None:
        assert (
            ConfigSamplerFixture(config_sampler_mocks.assets).sample_section(
                {
                    "mutually_exclusive": True,
                    "foo": {"probability": 1.0, "description": "foo foobar"},
                    "bar": {"probability": 0.0, "description": description},
                },
                "bar",
            )
            == expected
        )
        config_sampler_mocks.random.choices.assert_not_called()

    def test_sample_section_unknown_option_raises(
        self, config_sampler_mocks: ConfigSamplerMocks
    ) -> None:
        with pytest.raises(ValueError):
            ConfigSamplerFixture(config_sampler_mocks.assets).sample_section(
                {"foo": {"probability": 0.5, "description": "foo foobar"}}, "bar"
            )


class TestSampleStyleConfig:
    def test_sample_style_config_filter_given_samples_filtered_sections(
        self,
        mocker: MockerFixture,
        config_sampler_mocks: ConfigSamplerMocks,
        sections: dict[str, SamplingElement],
    ) -> None:
        config_sampler_mocks.assets.load_style_data.return_value = make_style(sections)
        sample_section: MagicMock = mocker.patch.object(
            ConfigSampler, "_sample_section", return_value=[]
        )
        filter_section: MagicMock = mocker.patch.object(
            ConfigSampler, "_filter_section", side_effect=["corge", "grault"]
        )
        ConfigSampler(
            config_sampler_mocks.assets,
            "foo\\.bar",
            None,
        ).sample_style_config({"qux": "quux"})
        filter_section.assert_has_calls(
            [
                call("foo", sections["foo"].model_dump(), "foo\\.bar"),
                call("qux", sections["qux"].model_dump(), "foo\\.bar"),
            ]
        )
        sample_section.assert_has_calls([call("corge", None), call("grault", "quux")])

    def test_sample_style_config_style_data_loaded_maps_sections_to_selections(
        self, mocker: MockerFixture, config_sampler_mocks: ConfigSamplerMocks
    ) -> None:
        config_sampler_mocks.assets.load_style_data.return_value = make_style(
            {
                "foo": SamplingElement.model_construct(),
                "bar": SamplingElement.model_construct(),
            }
        )
        sample_section: MagicMock = mocker.patch.object(
            ConfigSampler,
            "_sample_section",
            side_effect=[
                [{"key": "foo", "description": "foo foobar"}],
                [{"key": "bar", "description": "bar quux"}],
            ],
        )
        assert ConfigSampler(config_sampler_mocks.assets).sample_style_config(
            {"bar": "baz"}
        ) == {
            "foo": [{"key": "foo", "description": "foo foobar"}],
            "bar": [{"key": "bar", "description": "bar quux"}],
        }
        sample_section.assert_has_calls(
            [
                call({"mutually_exclusive": False}, None),
                call({"mutually_exclusive": False}, "baz"),
            ]
        )


class TestSampleContentConfig:
    def test_sample_content_config_filter_given_samples_filtered_sections(
        self,
        mocker: MockerFixture,
        config_sampler_mocks: ConfigSamplerMocks,
        sections: dict[str, SamplingElement],
    ) -> None:
        config_sampler_mocks.assets.load_content_data.return_value = make_content(
            sections
        )
        sample_section: MagicMock = mocker.patch.object(
            ConfigSampler, "_sample_section", return_value=[]
        )
        filter_section: MagicMock = mocker.patch.object(
            ConfigSampler, "_filter_section", side_effect=["corge", "grault"]
        )
        ConfigSampler(
            config_sampler_mocks.assets,
            None,
            "foo\\.bar",
        ).sample_content_config({"qux": "quux"})
        filter_section.assert_has_calls(
            [
                call("foo", sections["foo"].model_dump(), "foo\\.bar"),
                call("qux", sections["qux"].model_dump(), "foo\\.bar"),
            ]
        )
        sample_section.assert_has_calls([call("corge", None), call("grault", "quux")])

    def test_sample_content_config_content_data_loaded_maps_sections_to_selections(
        self, mocker: MockerFixture, config_sampler_mocks: ConfigSamplerMocks
    ) -> None:
        config_sampler_mocks.assets.load_content_data.return_value = make_content(
            {
                "foo": SamplingElement.model_construct(),
                "bar": SamplingElement.model_construct(),
            }
        )
        sample_section: MagicMock = mocker.patch.object(
            ConfigSampler,
            "_sample_section",
            side_effect=[
                [{"key": "foo", "description": "foo foobar"}],
                [{"key": "bar", "description": "bar quux"}],
            ],
        )
        assert ConfigSampler(config_sampler_mocks.assets).sample_content_config(
            {"bar": "baz"}
        ) == {
            "foo": [{"key": "foo", "description": "foo foobar"}],
            "bar": [{"key": "bar", "description": "bar quux"}],
        }
        sample_section.assert_has_calls(
            [
                call({"mutually_exclusive": False}, None),
                call({"mutually_exclusive": False}, "baz"),
            ]
        )


class TestFormatStylePrompt:
    def test_format_style_prompt_section_with_items_returns_formatted_bullets(
        self, config_sampler_mocks: ConfigSamplerMocks
    ) -> None:
        assert ConfigSampler(config_sampler_mocks.assets).format_style_prompt(
            {
                "foo_bar": [
                    {"key": "foo", "description": "foo foobar"},
                    {"key": "bar", "description": "bar quux"},
                ]
            }
        ) == (
            "## FOLLOW THESE STYLE REQUIREMENTS\n\n"
            "**Foo Bar:**\n"
            "- foo foobar\n"
            "- bar quux"
        )

    def test_format_style_prompt_section_with_no_items_omits_section(
        self, config_sampler_mocks: ConfigSamplerMocks
    ) -> None:
        assert (
            ConfigSampler(config_sampler_mocks.assets).format_style_prompt(
                {"foo_bar": []}
            )
            == "## FOLLOW THESE STYLE REQUIREMENTS"
        )


class TestFormatContentPrompt:
    def test_format_content_prompt_section_with_items_returns_formatted_bullets(
        self, config_sampler_mocks: ConfigSamplerMocks
    ) -> None:
        assert ConfigSampler(config_sampler_mocks.assets).format_content_prompt(
            {
                "foo_bar": [
                    {"key": "foo", "description": "foo foobar"},
                ]
            }
        ) == ("## FOLLOW THESE CONTENT REQUIREMENTS\n\n**Foo Bar:**\n- foo foobar")

    def test_format_content_prompt_section_with_no_items_omits_section(
        self, config_sampler_mocks: ConfigSamplerMocks
    ) -> None:
        assert (
            ConfigSampler(config_sampler_mocks.assets).format_content_prompt(
                {"foo_bar": []}
            )
            == "## FOLLOW THESE CONTENT REQUIREMENTS"
        )


class TestGeneratePrompts:
    @pytest.mark.parametrize(
        ("requirements", "expected_requirements"),
        [(None, {}), ({"foo": "baz", "bar": "qux"}, {"foo": "baz", "bar": "qux"})],
    )
    def test_generate_prompts_delegates_to_sampling_and_formatting_methods(
        self,
        mocker: MockerFixture,
        config_sampler_mocks: ConfigSamplerMocks,
        requirements: dict[str, str] | None,
        expected_requirements: dict[str, str],
    ) -> None:
        config_sampler_mocks.assets.load_style_data.return_value = make_style(
            {"foo": SamplingElement.model_construct()}
        )
        config_sampler_mocks.assets.load_content_data.return_value = make_content(
            {"bar": SamplingElement.model_construct()}
        )
        sampler: ConfigSampler = ConfigSampler(config_sampler_mocks.assets)
        sample_style_config: MagicMock = mocker.patch.object(
            sampler, "sample_style_config", return_value={"foo": []}
        )
        sample_content_config: MagicMock = mocker.patch.object(
            sampler, "sample_content_config", return_value={"bar": []}
        )
        format_style_prompt: MagicMock = mocker.patch.object(
            sampler, "format_style_prompt", return_value="qux quux"
        )
        format_content_prompt: MagicMock = mocker.patch.object(
            sampler, "format_content_prompt", return_value="bar baz"
        )
        assert sampler.generate_prompts(requirements) == ("qux quux", "bar baz")
        sample_style_config.assert_called_once_with(expected_requirements)
        sample_content_config.assert_called_once_with(expected_requirements)
        format_style_prompt.assert_called_once_with({"foo": []})
        format_content_prompt.assert_called_once_with({"bar": []})

    def test_generate_prompts_unknown_section_raises(
        self, config_sampler_mocks: ConfigSamplerMocks
    ) -> None:
        with pytest.raises(ValueError):
            ConfigSampler(config_sampler_mocks.assets).generate_prompts({"foo": "bar"})
