from unittest.mock import MagicMock, mock_open

import pytest
from pytest_mock import MockerFixture

from docsynth.pipeline import PipelineConfig, S3Upload


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


class TestS3UploadValidation:
    def test_s3_upload_disabled_missing_bucket_and_region_does_not_raise(self) -> None:
        assert S3Upload().enabled is False

    def test_s3_upload_enabled_bucket_and_region_given_does_not_raise(self) -> None:
        s3: S3Upload = S3Upload(enabled=True, bucket="foo", region="bar")
        assert s3.bucket == "foo"
        assert s3.region == "bar"

    def test_s3_upload_enabled_missing_bucket_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="output.s3.bucket and output.s3.region"):
            S3Upload(enabled=True, region="bar")

    def test_s3_upload_enabled_missing_region_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="output.s3.bucket and output.s3.region"):
            S3Upload(enabled=True, bucket="foo")
