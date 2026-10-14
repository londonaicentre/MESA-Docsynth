from collections.abc import Callable
from dataclasses import dataclass
import json
from pathlib import Path
from unittest.mock import MagicMock, mock_open

import pytest
from pytest_mock import MockerFixture

from docsynth.config import Config, ModelConfig
from docsynth.utils.llm_clients import (
    AnthropicClient,
    BedrockBatchClient,
    GeminiClient,
    LocalClient,
    MESARuntimeClient,
)


class BedrockBatchClientFixture(BedrockBatchClient):
    def generate(self, prompt: str, batch_entry_id: str | None = None) -> str | None:
        return None

    def run_batch_inference(
        self, bucket: str, bedrock_execution_role: str | None = None
    ) -> bool:
        return False

    def get_model_config(self) -> ModelConfig:
        return self._get_model_config()

    def store_batch_entry(self, batch_entry_id: str, prompt: str) -> None:
        self._store_batch_entry(batch_entry_id, prompt)

    def submit_batch(self, submit: Callable[[str], object]) -> bool:
        return self._submit_batch(submit)

    def resolve_job_id(self) -> str:
        return self._resolve_job_id()


def model_response(content: str | None, finish_reason: str = "STOP") -> MagicMock:
    return MagicMock(
        choices=[
            MagicMock(message=MagicMock(content=content), finish_reason=finish_reason)
        ]
    )


def submit_immediately(submit: Callable[[str], object]) -> bool:
    submit("docsynth/2026-01-01-0000")
    return True


@dataclass
class LlmCompletionMocks:
    llm: MagicMock


@dataclass
class AnthropicClientMocks:
    config: MagicMock
    aws: MagicMock
    datetime: MagicMock
    open: MagicMock
    path_exists: MagicMock


@dataclass
class MESARuntimeClientMocks:
    config: MagicMock


@pytest.fixture
def llm_completion_mocks(mocker: MockerFixture) -> LlmCompletionMocks:
    return LlmCompletionMocks(llm=mocker.patch("docsynth.utils.llm_clients.LLM"))


@pytest.fixture
def anthropic_client_mocks(mocker: MockerFixture) -> AnthropicClientMocks:
    config: MagicMock = mocker.Mock(spec=Config)
    config.models = {
        "sonnet4": ModelConfig(
            model="foo/bar", region="eu-west-2", batch_file="garply.jsonl"
        )
    }
    config.job_id_file = ".job_id.json"
    mocker.patch("docsynth.utils.llm_clients.Config", return_value=config)
    datetime: MagicMock = mocker.patch("docsynth.utils.llm_clients.datetime")
    datetime.now.return_value.strftime.return_value = "2026-01-01-0000"
    return AnthropicClientMocks(
        config=config,
        aws=mocker.patch("docsynth.utils.llm_clients.AWS"),
        datetime=datetime,
        open=mocker.patch("builtins.open", mock_open()),
        path_exists=mocker.patch.object(Path, "exists", return_value=True),
    )


@pytest.fixture
def mesa_runtime_client_mocks(mocker: MockerFixture) -> MESARuntimeClientMocks:
    config: MagicMock = mocker.Mock(spec=Config)
    config.models = {
        "deepseek4flash": ModelConfig(
            model="Foo-V4-Bar-4bit", region="eu-west-2", batch_file="garply.jsonl"
        )
    }
    mocker.patch("docsynth.utils.llm_clients.Config", return_value=config)
    return MESARuntimeClientMocks(config=config)


def make_mesa_runtime_client() -> MESARuntimeClient:
    return MESARuntimeClient("https://foo.bar", "baz", "qux", "deepseek4flash")


class TestGeminiClientInit:
    def test_init_package_installed_stores_module(self) -> None:
        GeminiClient("gemini-2.5-flash")

    def test_init_package_not_installed_raises_import_error(
        self, mocker: MockerFixture
    ) -> None:
        mocker.patch.dict("sys.modules", {"google.generativeai": None})
        with pytest.raises(ImportError, match="google-generativeai package"):
            GeminiClient("gemini-2.5-flash")


class TestGeminiClientGenerate:
    def test_generate_valid_prompt_returns_content(
        self, llm_completion_mocks: LlmCompletionMocks
    ) -> None:
        llm_completion_mocks.llm.completion.return_value = model_response("foo")
        assert GeminiClient("gemini-2.5-flash").generate("foo") == "foo"

    def test_generate_no_response_returns_none(
        self, llm_completion_mocks: LlmCompletionMocks
    ) -> None:
        llm_completion_mocks.llm.completion.return_value = None
        assert GeminiClient("gemini-2.5-flash").generate("foo") is None

    def test_generate_blocked_by_safety_filter_raises_value_error(
        self, llm_completion_mocks: LlmCompletionMocks
    ) -> None:
        llm_completion_mocks.llm.completion.return_value = model_response(
            None, finish_reason="SAFETY"
        )
        with pytest.raises(ValueError, match="Response blocked by Gemini"):
            GeminiClient("gemini-2.5-flash").generate("foo")

    def test_generate_completion_raises_reraises_exception(
        self, llm_completion_mocks: LlmCompletionMocks
    ) -> None:
        llm_completion_mocks.llm.completion.side_effect = Exception("thud")
        with pytest.raises(Exception, match="thud"):
            GeminiClient("gemini-2.5-flash").generate("foo")


class TestGeminiClientBatchInference:
    def test_run_batch_inference_called_returns_false(self) -> None:
        assert (
            GeminiClient("gemini-2.5-flash").run_batch_inference("foo-bar", "bar-baz")
            is False
        )

    def test_get_batch_inference_outputs_called_returns_none(self) -> None:
        assert (
            GeminiClient("gemini-2.5-flash").get_batch_inference_outputs("foo-bar")
            is None
        )

    def test_check_batch_output_status_called_returns_false(self) -> None:
        assert (
            GeminiClient("gemini-2.5-flash").check_batch_output_status("foo-bar")
            is False
        )


class TestAnthropicClientGenerate:
    def test_generate_batch_entry_id_given_stores_entry_and_returns_none(
        self, anthropic_client_mocks: AnthropicClientMocks
    ) -> None:
        assert AnthropicClient("sonnet4").generate("foo", "foo-1") is None
        anthropic_client_mocks.aws.create_anthropic_bedrock_batch_entry.assert_called_once_with(
            "foo-1", None, "foo"
        )
        anthropic_client_mocks.aws.bedrock_completion.assert_not_called()

    def test_generate_no_batch_entry_id_returns_content(
        self, anthropic_client_mocks: AnthropicClientMocks
    ) -> None:
        anthropic_client_mocks.aws.bedrock_completion.return_value = model_response(
            "qux quux"
        )
        assert (
            AnthropicClient("sonnet4", api_key="foo-key").generate("foo") == "qux quux"
        )

    def test_generate_no_batch_entry_id_missing_api_key_raises_value_error(
        self, anthropic_client_mocks: AnthropicClientMocks
    ) -> None:
        with pytest.raises(
            ValueError,
            match="LLM__ANTHROPIC__API_KEY not found in environment variables",
        ):
            AnthropicClient("sonnet4").generate("foo")
        anthropic_client_mocks.aws.bedrock_completion.assert_not_called()

    def test_generate_no_response_returns_none(
        self, anthropic_client_mocks: AnthropicClientMocks
    ) -> None:
        anthropic_client_mocks.aws.bedrock_completion.return_value = None
        assert AnthropicClient("sonnet4", api_key="foo-key").generate("foo") is None

    def test_generate_empty_response_raises_value_error(
        self, anthropic_client_mocks: AnthropicClientMocks
    ) -> None:
        anthropic_client_mocks.aws.bedrock_completion.return_value = model_response(
            None
        )
        with pytest.raises(ValueError, match="Response not provided by Bedrock"):
            AnthropicClient("sonnet4", api_key="foo-key").generate("foo")

    def test_generate_completion_raises_reraises_exception(
        self, anthropic_client_mocks: AnthropicClientMocks
    ) -> None:
        anthropic_client_mocks.aws.bedrock_completion.side_effect = Exception("thud")
        with pytest.raises(Exception, match="thud"):
            AnthropicClient("sonnet4", api_key="foo-key").generate("foo")


class TestAnthropicClientRunBatchInference:
    def test_run_batch_inference_valid_params_writes_batch_file_and_manifest(
        self, anthropic_client_mocks: AnthropicClientMocks
    ) -> None:
        anthropic_client_mocks.aws.create_anthropic_bedrock_batch_entry.return_value = {
            "waldoFoo": "foo-1"
        }
        client: AnthropicClient = AnthropicClient("sonnet4")
        client.generate("prompt one", "foo-1")
        assert client.run_batch_inference("foo-bar", "bar-baz") is True
        anthropic_client_mocks.open.assert_any_call("garply.jsonl", "w")
        anthropic_client_mocks.open.assert_any_call(".job_id.json", "w")
        written: str = "".join(
            call.args[0]
            for call in anthropic_client_mocks.open.return_value.write.call_args_list
        )
        assert '{"waldoFoo": "foo-1"}' in written
        assert json.dumps({"job_id": "docsynth/2026-01-01-0000"}) in written
        anthropic_client_mocks.aws.run_batch_inference.assert_called_once_with(
            "docsynth/2026-01-01-0000",
            "foo/bar",
            "garply.jsonl",
            "foo-bar",
            "bar-baz",
            "eu-west-2",
        )

    def test_run_batch_inference_no_role_raises_value_error(
        self, anthropic_client_mocks: AnthropicClientMocks
    ) -> None:
        with pytest.raises(ValueError, match="A Bedrock execution role is required"):
            AnthropicClient("sonnet4").run_batch_inference("foo-bar")
        anthropic_client_mocks.aws.run_batch_inference.assert_not_called()


class TestBedrockBatchClient:
    def test_get_model_config_model_configured_returns_model_config(
        self, anthropic_client_mocks: AnthropicClientMocks
    ) -> None:
        assert (
            BedrockBatchClientFixture("sonnet4").get_model_config()
            == anthropic_client_mocks.config.models["sonnet4"]
        )

    def test_store_batch_entry_valid_entry_creates_bedrock_batch_entry(
        self, anthropic_client_mocks: AnthropicClientMocks
    ) -> None:
        BedrockBatchClientFixture("sonnet4").store_batch_entry("foo-1", "foo")
        anthropic_client_mocks.aws.create_anthropic_bedrock_batch_entry.assert_called_once_with(
            "foo-1", None, "foo"
        )

    def test_submit_batch_no_entries_writes_files_and_submits_job_id(
        self, anthropic_client_mocks: AnthropicClientMocks
    ) -> None:
        submit: MagicMock = MagicMock()
        assert BedrockBatchClientFixture("sonnet4").submit_batch(submit) is True
        anthropic_client_mocks.open.assert_any_call("garply.jsonl", "w")
        anthropic_client_mocks.open.assert_any_call(".job_id.json", "w")
        anthropic_client_mocks.open.return_value.write.assert_called_once_with(
            json.dumps({"job_id": "docsynth/2026-01-01-0000"})
        )
        submit.assert_called_once_with("docsynth/2026-01-01-0000")

    def test_resolve_job_id_batch_submitted_returns_in_memory_job_id(
        self, anthropic_client_mocks: AnthropicClientMocks
    ) -> None:
        anthropic_client_mocks.path_exists.return_value = False
        client: BedrockBatchClientFixture = BedrockBatchClientFixture("sonnet4")
        client.submit_batch(MagicMock())
        assert client.resolve_job_id() == "docsynth/2026-01-01-0000"

    def test_resolve_job_id_fresh_process_reads_job_id_from_file(
        self, mocker: MockerFixture, anthropic_client_mocks: AnthropicClientMocks
    ) -> None:
        mocker.patch(
            "builtins.open",
            mock_open(read_data=json.dumps({"job_id": "docsynth/2025-12-25-1200"})),
        )
        assert (
            BedrockBatchClientFixture("sonnet4").resolve_job_id()
            == "docsynth/2025-12-25-1200"
        )

    def test_resolve_job_id_no_job_id_raises_value_error(
        self, anthropic_client_mocks: AnthropicClientMocks
    ) -> None:
        anthropic_client_mocks.path_exists.return_value = False
        with pytest.raises(ValueError, match="No batch job id found"):
            BedrockBatchClientFixture("sonnet4").resolve_job_id()


class TestMESARuntimeClient:
    def test_generate_no_batch_entry_id_raises_value_error(
        self, mesa_runtime_client_mocks: MESARuntimeClientMocks
    ) -> None:
        with pytest.raises(ValueError, match="supports batch inference only"):
            make_mesa_runtime_client().generate("foo")

    def test_generate_batch_entry_id_given_stores_entry_and_returns_none(
        self, mocker: MockerFixture, mesa_runtime_client_mocks: MESARuntimeClientMocks
    ) -> None:
        store_batch_entry: MagicMock = mocker.patch.object(
            MESARuntimeClient, "_store_batch_entry"
        )
        assert make_mesa_runtime_client().generate("foo", "foo-1") is None
        store_batch_entry.assert_called_once_with("foo-1", "foo")

    def test_run_batch_inference_bucket_given_dispatches_job_to_bucket(
        self, mocker: MockerFixture, mesa_runtime_client_mocks: MESARuntimeClientMocks
    ) -> None:
        mocker.patch.object(
            MESARuntimeClient, "_submit_batch", side_effect=submit_immediately
        )
        dispatch: MagicMock = mocker.patch.object(MESARuntimeClient, "_dispatch")
        assert make_mesa_runtime_client().run_batch_inference("foo-bar") is True
        dispatch.assert_called_once_with("docsynth/2026-01-01-0000", "foo-bar")


class TestAnthropicClientGetBatchInferenceOutputs:
    def test_get_batch_inference_outputs_same_session_uses_in_memory_job_id(
        self, anthropic_client_mocks: AnthropicClientMocks
    ) -> None:
        client: AnthropicClient = AnthropicClient("sonnet4")
        client.run_batch_inference("foo-bar", "bar-baz")
        anthropic_client_mocks.aws.get_batch_inference_outputs.return_value = "foobar"
        assert client.get_batch_inference_outputs("foo-bar") == "foobar"
        anthropic_client_mocks.aws.get_batch_inference_outputs.assert_called_once_with(
            "eu-west-2", "foo-bar", "docsynth/2026-01-01-0000", "garply.jsonl.out"
        )

    def test_get_batch_inference_outputs_fresh_process_reads_job_id_from_manifest(
        self, mocker: MockerFixture, anthropic_client_mocks: AnthropicClientMocks
    ) -> None:
        mocker.patch(
            "builtins.open",
            mock_open(read_data=json.dumps({"job_id": "docsynth/2025-12-25-1200"})),
        )
        anthropic_client_mocks.aws.get_batch_inference_outputs.return_value = "foobar"
        assert (
            AnthropicClient("sonnet4").get_batch_inference_outputs("foo-bar")
            == "foobar"
        )
        anthropic_client_mocks.aws.get_batch_inference_outputs.assert_called_once_with(
            "eu-west-2", "foo-bar", "docsynth/2025-12-25-1200", "garply.jsonl.out"
        )

    def test_get_batch_inference_outputs_no_manifest_raises_value_error(
        self, anthropic_client_mocks: AnthropicClientMocks
    ) -> None:
        anthropic_client_mocks.path_exists.return_value = False
        with pytest.raises(ValueError, match="No batch job id found"):
            AnthropicClient("sonnet4").get_batch_inference_outputs("foo-bar")


class TestAnthropicClientCheckBatchOutputStatus:
    def test_check_batch_output_status_same_session_matching_key_returns_true(
        self, anthropic_client_mocks: AnthropicClientMocks
    ) -> None:
        client: AnthropicClient = AnthropicClient("sonnet4")
        client.run_batch_inference("foo-bar", "bar-baz")
        anthropic_client_mocks.aws.list_s3_objects.return_value = [
            {"Key": "docsynth/2026-01-01-0000/output/grault/garply.jsonl.out"}
        ]
        assert client.check_batch_output_status("foo-bar") is True
        anthropic_client_mocks.aws.list_s3_objects.assert_called_once_with(
            "eu-west-2", "foo-bar", "docsynth/2026-01-01-0000/output/"
        )

    def test_check_batch_output_status_no_matching_key_returns_false(
        self, anthropic_client_mocks: AnthropicClientMocks
    ) -> None:
        client: AnthropicClient = AnthropicClient("sonnet4")
        client.run_batch_inference("foo-bar", "bar-baz")
        anthropic_client_mocks.aws.list_s3_objects.return_value = [
            {"Key": "docsynth/2026-01-01-0000/output/grault/manifest.json.out"}
        ]
        assert client.check_batch_output_status("foo-bar") is False

    def test_check_batch_output_status_no_objects_returns_false(
        self, anthropic_client_mocks: AnthropicClientMocks
    ) -> None:
        client: AnthropicClient = AnthropicClient("sonnet4")
        client.run_batch_inference("foo-bar", "bar-baz")
        anthropic_client_mocks.aws.list_s3_objects.return_value = []
        assert client.check_batch_output_status("foo-bar") is False

    def test_check_batch_output_status_fresh_process_reads_job_id_from_manifest(
        self, mocker: MockerFixture, anthropic_client_mocks: AnthropicClientMocks
    ) -> None:
        mocker.patch(
            "builtins.open",
            mock_open(read_data=json.dumps({"job_id": "docsynth/2025-12-25-1200"})),
        )
        anthropic_client_mocks.aws.list_s3_objects.return_value = [
            {"Key": "docsynth/2025-12-25-1200/output/grault/garply.jsonl.out"}
        ]
        assert AnthropicClient("sonnet4").check_batch_output_status("foo-bar") is True
        anthropic_client_mocks.aws.list_s3_objects.assert_called_once_with(
            "eu-west-2", "foo-bar", "docsynth/2025-12-25-1200/output/"
        )

    def test_check_batch_output_status_no_manifest_raises_value_error(
        self, anthropic_client_mocks: AnthropicClientMocks
    ) -> None:
        anthropic_client_mocks.path_exists.return_value = False
        with pytest.raises(ValueError, match="No batch job id found"):
            AnthropicClient("sonnet4").check_batch_output_status("foo-bar")


class TestLocalClientGenerate:
    def test_generate_valid_prompt_returns_content(
        self, llm_completion_mocks: LlmCompletionMocks
    ) -> None:
        llm_completion_mocks.llm.completion.return_value = model_response("foo")
        assert (
            LocalClient("http://localhost:1234/v1", "foo-bar").generate("foo") == "foo"
        )

    def test_generate_no_response_returns_none(
        self, llm_completion_mocks: LlmCompletionMocks
    ) -> None:
        llm_completion_mocks.llm.completion.return_value = None

        assert (
            LocalClient("http://localhost:1234/v1", "foo-bar").generate("foo") is None
        )

    def test_generate_empty_response_raises_value_error(
        self, llm_completion_mocks: LlmCompletionMocks
    ) -> None:
        llm_completion_mocks.llm.completion.return_value = model_response(None)
        with pytest.raises(ValueError, match="Response not provided by local client"):
            LocalClient("http://localhost:1234/v1", "foo-bar").generate("foo")

    def test_generate_completion_raises_reraises_exception(
        self, llm_completion_mocks: LlmCompletionMocks
    ) -> None:
        llm_completion_mocks.llm.completion.side_effect = Exception("thud")
        with pytest.raises(Exception, match="thud"):
            LocalClient("http://localhost:1234/v1", "foo-bar").generate("foo")


class TestLocalClientBatchInference:
    def test_run_batch_inference_called_returns_false(self) -> None:
        assert (
            LocalClient("http://localhost:1234/v1", "foo-bar").run_batch_inference(
                "foo-bar", "bar-baz"
            )
            is False
        )

    def test_get_batch_inference_outputs_called_returns_none(self) -> None:
        assert (
            LocalClient(
                "http://localhost:1234/v1", "foo-bar"
            ).get_batch_inference_outputs("foo-bar")
            is None
        )

    def test_check_batch_output_status_called_returns_false(self) -> None:
        assert (
            LocalClient(
                "http://localhost:1234/v1", "foo-bar"
            ).check_batch_output_status("foo-bar")
            is False
        )
