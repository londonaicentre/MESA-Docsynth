import json
from unittest.mock import MagicMock, Mock

from pytest_mock import MockerFixture

from docsynth.config import Config


class TestInit:
    CONFIG_DATA = {
        "foo": {"model": "foo-model", "region": "eu-west-2", "batch_file": "foo.jsonl"}
    }

    def test_init_config_json_present_loads_and_parses_models(
        self, mocker: MockerFixture
    ) -> None:
        base_dir: Mock = mocker.Mock()
        base_dir.joinpath.return_value.read_text.return_value = json.dumps(
            TestInit.CONFIG_DATA
        )
        files: MagicMock = mocker.patch("docsynth.config.files", return_value=base_dir)
        config: Config = Config()
        assert config.models["foo"].model == "foo-model"
        assert config.job_id_file == ".job_id.json"
        files.assert_called_once_with("docsynth")
        base_dir.joinpath.assert_called_once_with("config/config.json")
