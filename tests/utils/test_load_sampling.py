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
    def sample_section(self, section_data: dict[str, Any]) -> list[dict[str, str]]:
        return self._sample_section(section_data)


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


class TestSampleStyleConfig:
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
        assert ConfigSampler(config_sampler_mocks.assets).sample_style_config() == {
            "foo": [{"key": "foo", "description": "foo foobar"}],
            "bar": [{"key": "bar", "description": "bar quux"}],
        }
        sample_section.assert_has_calls(
            [
                call({"mutually_exclusive": False}),
                call({"mutually_exclusive": False}),
            ]
        )


class TestSampleContentConfig:
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
        assert ConfigSampler(config_sampler_mocks.assets).sample_content_config() == {
            "foo": [{"key": "foo", "description": "foo foobar"}],
            "bar": [{"key": "bar", "description": "bar quux"}],
        }
        sample_section.assert_has_calls(
            [
                call({"mutually_exclusive": False}),
                call({"mutually_exclusive": False}),
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
    def test_generate_prompts_delegates_to_sampling_and_formatting_methods(
        self, mocker: MockerFixture, config_sampler_mocks: ConfigSamplerMocks
    ) -> None:
        sampler: ConfigSampler = ConfigSampler(config_sampler_mocks.assets)
        mocker.patch.object(sampler, "sample_style_config", return_value={"foo": []})
        mocker.patch.object(sampler, "sample_content_config", return_value={"bar": []})
        format_style_prompt: MagicMock = mocker.patch.object(
            sampler, "format_style_prompt", return_value="qux quux"
        )
        format_content_prompt: MagicMock = mocker.patch.object(
            sampler, "format_content_prompt", return_value="bar baz"
        )
        assert sampler.generate_prompts() == ("qux quux", "bar baz")
        format_style_prompt.assert_called_once_with({"foo": []})
        format_content_prompt.assert_called_once_with({"bar": []})
