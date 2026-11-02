from pathlib import Path
from unittest.mock import MagicMock, mock_open

import pytest
from pytest_mock import MockerFixture

from docsynth.pipeline import LLMProvider, PipelineConfig


class TestInit:
    PIPELINE_CONFIG_DATA = {
        "llm": {
            "enabled": True,
            "provider": "mesa_runtime",
            "anthropic": {"model": "foo-model", "temperature": 0.5, "max_tokens": 100},
            "gemini": {"model": "bar-model", "temperature": 0.5, "max_tokens": 100},
            "local": {"model": "baz-model", "temperature": 0.5, "max_tokens": 100},
            "mesa_runtime": {
                "model": "qux-model",
                "temperature": 0.5,
                "max_tokens": 100,
            },
        },
        "profile_selection": {"mode": "random", "count": 1, "file": []},
        "structure_selection": {},
        "prompt_config": {"include_style": True, "include_content": True},
        "output": {"subdirectory": "foo"},
    }

    def test_init_pipeline_yml_present_loads_and_parses_config(
        self, mocker: MockerFixture
    ) -> None:
        open_mock: MagicMock = mocker.patch("builtins.open", mock_open())
        safe_load: MagicMock = mocker.patch(
            "docsynth.pipeline.safe_load", return_value=TestInit.PIPELINE_CONFIG_DATA
        )
        config: PipelineConfig = PipelineConfig()
        assert config.llm.provider == "mesa_runtime"
        assert config.output.subdirectory == "foo"
        open_mock.assert_called_once_with("pipeline.yml")
        safe_load.assert_called_once_with(open_mock.return_value)

    def test_init_mesa_runtime_given_merges_credentials_from_environment(
        self, mocker: MockerFixture, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("LLM__MESA_RUNTIME__BASE_URL", "https://foo.bar")
        monkeypatch.setenv("LLM__MESA_RUNTIME__USERNAME", "baz")
        monkeypatch.setenv("LLM__MESA_RUNTIME__PASSWORD", "qux")
        mocker.patch("builtins.open", mock_open())
        mocker.patch(
            "docsynth.pipeline.safe_load", return_value=TestInit.PIPELINE_CONFIG_DATA
        )
        assert PipelineConfig().llm.mesa_runtime == LLMProvider(
            base_url="https://foo.bar",
            model="qux-model",
            temperature=0.5,
            max_tokens=100,
            username="baz",
            password="qux",
        )

    @pytest.mark.parametrize("provider", ["anthropic", "gemini", "local"])
    def test_init_unselected_provider_blocks_dropped(
        self, mocker: MockerFixture, provider: str
    ) -> None:
        mocker.patch("builtins.open", mock_open())
        mocker.patch(
            "docsynth.pipeline.safe_load", return_value=TestInit.PIPELINE_CONFIG_DATA
        )
        assert getattr(PipelineConfig().llm, provider) is None

    def test_init_unselected_provider_environment_ignored(
        self, mocker: MockerFixture, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("LLM__LOCAL__BASE_URL", "https://foo.bar")
        mocker.patch("builtins.open", mock_open())
        mocker.patch(
            "docsynth.pipeline.safe_load",
            return_value={
                **TestInit.PIPELINE_CONFIG_DATA,
                "llm": {
                    key: value
                    for key, value in TestInit.PIPELINE_CONFIG_DATA["llm"].items()
                    if key != "local"
                },
            },
        )
        assert PipelineConfig().llm.local is None

    def test_init_selected_provider_block_missing_loads_as_none(
        self, mocker: MockerFixture
    ) -> None:
        mocker.patch("builtins.open", mock_open())
        mocker.patch(
            "docsynth.pipeline.safe_load",
            return_value={
                **TestInit.PIPELINE_CONFIG_DATA,
                "llm": {"enabled": True, "provider": "mesa_runtime"},
            },
        )
        assert PipelineConfig().llm.mesa_runtime is None
