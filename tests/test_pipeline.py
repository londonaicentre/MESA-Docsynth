from unittest.mock import MagicMock, mock_open

from pytest_mock import MockerFixture

from docsynth.pipeline import PipelineConfig


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
