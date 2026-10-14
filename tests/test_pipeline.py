from pathlib import Path
from unittest.mock import MagicMock, mock_open

import pytest
from pytest_mock import MockerFixture

from docsynth.pipeline import LLMProvider, PipelineConfig


class TestInit:
    PIPELINE_CONFIG_DATA = {
        "llm": {
            "enabled": True,
            "provider": "foo",
            "anthropic": {"model": "foo-model", "temperature": 0.5, "max_tokens": 100},
            "gemini": {"model": "bar-model", "temperature": 0.5, "max_tokens": 100},
            "local": {"model": "baz-model", "temperature": 0.5, "max_tokens": 100},
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
        assert config.llm.provider == "foo"
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
            "docsynth.pipeline.safe_load",
            return_value={
                **TestInit.PIPELINE_CONFIG_DATA,
                "llm": {
                    **TestInit.PIPELINE_CONFIG_DATA["llm"],
                    "mesa_runtime": {
                        "model": "qux-model",
                        "temperature": 0.5,
                        "max_tokens": 100,
                    },
                },
            },
        )
        assert PipelineConfig().llm.mesa_runtime == LLMProvider(
            base_url="https://foo.bar",
            model="qux-model",
            temperature=0.5,
            max_tokens=100,
            username="baz",
            password="qux",
        )

    def test_init_mesa_runtime_not_given_defaults_to_none(
        self, mocker: MockerFixture, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.chdir(tmp_path)
        mocker.patch("builtins.open", mock_open())
        mocker.patch(
            "docsynth.pipeline.safe_load", return_value=TestInit.PIPELINE_CONFIG_DATA
        )
        assert PipelineConfig().llm.mesa_runtime is None
