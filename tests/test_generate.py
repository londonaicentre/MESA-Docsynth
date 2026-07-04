from dataclasses import dataclass
from importlib.resources.abc import Traversable
import io
import json
from pathlib import Path
import tarfile
from unittest.mock import MagicMock, Mock
import zipfile

import pytest
from pytest_mock import MockerFixture

from docsynth.generate import Generator
from docsynth.pipeline import (
    LLM,
    LLMProvider,
    Output,
    PipelineConfig,
    ProfileSelection,
    PromptConfig,
    StructureSelection,
)
from docsynth.types.batch_metadata import BatchMetadata
from docsynth.types.documents import DocsynthDocument
from docsynth.types.profile import Profile
from docsynth.types.sampling import Content, Style
from docsynth.types.wrapper import DocsynthAssets
from docsynth.utils.llm_clients import LLMClient


def _empty_yaml_traversable() -> Traversable:
    buffer: io.BytesIO = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("sampling.yml", "{}")
    return zipfile.Path(zipfile.ZipFile(buffer), "sampling.yml")


class DocsynthAssetsFixture(DocsynthAssets):
    def __init__(self, profile_ids: list[str] | None = None) -> None:
        self.__profiles: list[Profile] = [
            Profile(profile_id=profile_id)
            for profile_id in (profile_ids or ["foo_001"])
        ]

    def load_all_profiles(self) -> list[Profile]:
        return self.__profiles

    def load_profiles_from_files(self, filenames: list[str]) -> list[Profile]:
        return self.__profiles

    def _load_profiles_from_file(self, file_path: object) -> list[Profile]:
        return self.__profiles

    def format_profile_prompt(self, profile: Profile) -> str:
        return "## USE THIS MOCK PROFILE"

    def load_style_data(self) -> Style:
        return Style(_empty_yaml_traversable())

    def load_content_data(self) -> Content:
        return Content(_empty_yaml_traversable())

    def load_structures(self, enabled_structures: list[str]) -> dict[str, str]:
        return {}

    def get_structure_name_without_extension(self, filename: str) -> str:
        return filename

    def load_user_prompt_template(self, template_name: str) -> str:
        return "{specific_instructions}"

    def get_domain(self) -> str:
        return "foo"


class GeneratorFixture(Generator):
    def init_llm_client(self, llm_config: LLM) -> LLMClient | None:
        return self._init_llm_client(llm_config)

    def create_llm_client(self, llm_config: LLM) -> LLMClient | None:
        return self._create_llm_client(llm_config)

    def generate_batch_id(
        self, output_config: Output, bucket: str | None, region: str
    ) -> str:
        return self._generate_zipped_batch_id(output_config, bucket, region)

    def publish_batch(
        self,
        output_dir: str,
        output_config: Output,
        bucket: str | None,
        region: str,
        domain: str,
    ) -> None:
        return self._publish_zipped_batch(
            output_dir, output_config, bucket, region, domain
        )


def make_profile_selection(
    mocker: MockerFixture,
    mode: str = "sequential",
    count: int = -1,
    file: list[str] | None = None,
) -> Mock:
    return mocker.Mock(spec=ProfileSelection, mode=mode, count=count, file=file or [])


def mock_datetime(mocker: MockerFixture, date: str = "2026-01-17") -> MagicMock:
    datetime_mock: MagicMock = mocker.patch("docsynth.generate.datetime")
    datetime_mock.now.return_value.strftime.return_value = date
    datetime_mock.now.return_value.isoformat.return_value = f"{date}T00:00:00"
    return datetime_mock


def document_files(output_dir: Path) -> list[Path]:
    return [
        json_file
        for json_file in output_dir.glob("*.json")
        if json_file.name != "metadata.json"
    ]


def make_output(
    mocker: MockerFixture,
    subdirectory: str = "test_batch",
    skip_existing: bool = False,
    domain: str | None = "foo",
    description: str | None = None,
    upload_enabled: bool = False,
) -> Mock:
    return mocker.Mock(
        spec=Output,
        subdirectory=subdirectory,
        skip_existing=skip_existing,
        domain=domain,
        description=description,
        upload_enabled=upload_enabled,
    )


@dataclass
class GeneratorMocks:
    pipeline_config: PipelineConfig
    from_domain: MagicMock


@pytest.fixture
def generator_mocks(
    mocker: MockerFixture, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> GeneratorMocks:
    monkeypatch.chdir(tmp_path)
    pipeline_config: PipelineConfig = mocker.Mock(spec=PipelineConfig)
    pipeline_config.llm = mocker.Mock(spec=LLM, enabled=False)
    pipeline_config.profile_selection = make_profile_selection(mocker)
    pipeline_config.structure_selection = mocker.Mock(
        spec=StructureSelection, enabled_structures=None
    )
    pipeline_config.prompt_config = mocker.Mock(
        spec=PromptConfig, include_style=False, include_content=False
    )
    pipeline_config.output = make_output(mocker)
    mocker.patch("docsynth.generate.PipelineConfig", return_value=pipeline_config)
    from_domain: MagicMock = mocker.patch(
        "docsynth.generate.DocsynthAssets.from_domain",
        return_value=DocsynthAssetsFixture(),
    )
    return GeneratorMocks(pipeline_config=pipeline_config, from_domain=from_domain)


class TestInit:
    def test_init_assets_given_uses_assets_get_domain(
        self, generator_mocks: GeneratorMocks, tmp_path: Path
    ) -> None:
        Generator(assets=DocsynthAssetsFixture()).generate()
        metadata: BatchMetadata = BatchMetadata.model_validate(
            json.loads(
                (tmp_path / "output" / "test_batch" / "metadata.json").read_text()
            )
        )
        assert metadata.domain == "foo"
        generator_mocks.from_domain.assert_not_called()

    def test_init_no_assets_given_derives_both_from_pipeline_domain(
        self, mocker: MockerFixture, generator_mocks: GeneratorMocks, tmp_path: Path
    ) -> None:
        generator_mocks.pipeline_config.output = make_output(mocker, domain="bar")
        Generator().generate()
        metadata: BatchMetadata = BatchMetadata.model_validate(
            json.loads(
                (tmp_path / "output" / "test_batch" / "metadata.json").read_text()
            )
        )
        assert metadata.domain == "bar"
        generator_mocks.from_domain.assert_called_once_with("bar")

    def test_init_no_assets_given_pipeline_domain_not_set_raises_value_error(
        self, mocker: MockerFixture, generator_mocks: GeneratorMocks
    ) -> None:
        generator_mocks.pipeline_config.output = make_output(mocker, domain=None)
        with pytest.raises(ValueError, match="output.domain must be set"):
            Generator()


class TestCreateLlmClient:
    def test_create_llm_client_disabled_returns_none(
        self, mocker: MockerFixture, generator_mocks: GeneratorMocks
    ) -> None:
        assert (
            GeneratorFixture().create_llm_client(mocker.Mock(spec=LLM, enabled=False))
            is None
        )

    def test_create_llm_client_provider_none_returns_none(
        self, mocker: MockerFixture, generator_mocks: GeneratorMocks
    ) -> None:
        assert (
            GeneratorFixture().create_llm_client(
                mocker.Mock(spec=LLM, enabled=True, provider="none")
            )
            is None
        )

    def test_create_llm_client_provider_unknown_raises_value_error(
        self, mocker: MockerFixture, generator_mocks: GeneratorMocks
    ) -> None:
        with pytest.raises(ValueError, match="Unknown LLM provider: foo"):
            GeneratorFixture().create_llm_client(
                mocker.Mock(spec=LLM, enabled=True, provider="foo")
            )

    def test_create_llm_client_provider_gemini_missing_api_key_raises_value_error(
        self, mocker: MockerFixture, generator_mocks: GeneratorMocks
    ) -> None:
        with pytest.raises(
            ValueError, match="llm__gemini__api_key not found in environment variables"
        ):
            GeneratorFixture().create_llm_client(
                mocker.Mock(
                    spec=LLM,
                    enabled=True,
                    provider="gemini",
                    gemini=LLMProvider(
                        model="foo-model", temperature=0.5, max_tokens=100
                    ),
                )
            )

    def test_create_llm_client_provider_gemini_returns_gemini_client(
        self, mocker: MockerFixture, generator_mocks: GeneratorMocks
    ) -> None:
        gemini_client: MagicMock = mocker.patch("docsynth.generate.GeminiClient")
        assert (
            GeneratorFixture().create_llm_client(
                mocker.Mock(
                    spec=LLM,
                    enabled=True,
                    provider="gemini",
                    gemini=LLMProvider(
                        model="foo-model",
                        temperature=0.5,
                        max_tokens=100,
                        api_key="foo-key",
                    ),
                )
            )
            == gemini_client.return_value
        )
        gemini_client.assert_called_once_with(
            model="foo-model", temperature=0.5, max_tokens=100, api_key="foo-key"
        )

    def test_create_llm_client_provider_anthropic_missing_api_key_raises_value_error(
        self, mocker: MockerFixture, generator_mocks: GeneratorMocks
    ) -> None:
        with pytest.raises(
            ValueError,
            match="llm__anthropic__api_key not found in environment variables",
        ):
            GeneratorFixture().create_llm_client(
                mocker.Mock(
                    spec=LLM,
                    enabled=True,
                    provider="anthropic",
                    anthropic=LLMProvider(
                        model="foo-model", temperature=0.5, max_tokens=100
                    ),
                )
            )

    def test_create_llm_client_provider_anthropic_returns_anthropic_client(
        self, mocker: MockerFixture, generator_mocks: GeneratorMocks
    ) -> None:
        anthropic_client: MagicMock = mocker.patch("docsynth.generate.AnthropicClient")

        assert (
            GeneratorFixture().create_llm_client(
                mocker.Mock(
                    spec=LLM,
                    enabled=True,
                    provider="anthropic",
                    anthropic=LLMProvider(
                        model="foo-model",
                        temperature=0.5,
                        max_tokens=100,
                        api_key="foo-key",
                    ),
                )
            )
            == anthropic_client.return_value
        )
        anthropic_client.assert_called_once_with(
            model="foo-model", temperature=0.5, max_tokens=100, api_key="foo-key"
        )

    def test_create_llm_client_provider_local_missing_base_url_raises_value_error(
        self, mocker: MockerFixture, generator_mocks: GeneratorMocks
    ) -> None:
        with pytest.raises(
            ValueError, match="llm__local__base_url not found in environment variables"
        ):
            GeneratorFixture().create_llm_client(
                mocker.Mock(
                    spec=LLM,
                    enabled=True,
                    provider="local",
                    local=LLMProvider(
                        model="foo-model", temperature=0.5, max_tokens=100
                    ),
                )
            )

    def test_create_llm_client_provider_local_returns_local_client(
        self, mocker: MockerFixture, generator_mocks: GeneratorMocks
    ) -> None:
        local_client: MagicMock = mocker.patch("docsynth.generate.LocalClient")
        assert (
            GeneratorFixture().create_llm_client(
                mocker.Mock(
                    spec=LLM,
                    enabled=True,
                    provider="local",
                    local=LLMProvider(
                        base_url="http://localhost:1234/v1",
                        model="foo-model",
                        temperature=0.5,
                        max_tokens=100,
                    ),
                )
            )
            == local_client.return_value
        )
        local_client.assert_called_once_with(
            base_url="http://localhost:1234/v1",
            model="foo-model",
            temperature=0.5,
            max_tokens=100,
            api_key="not-needed",
        )


class TestInitLlmClient:
    def test_init_llm_client_disabled_returns_none(
        self, mocker: MockerFixture, generator_mocks: GeneratorMocks
    ) -> None:
        assert (
            GeneratorFixture().init_llm_client(mocker.Mock(spec=LLM, enabled=False))
            is None
        )

    def test_init_llm_client_enabled_create_succeeds_returns_client(
        self, mocker: MockerFixture, generator_mocks: GeneratorMocks
    ) -> None:
        llm_client: Mock = mocker.Mock(spec=LLMClient)
        mocker.patch.object(
            GeneratorFixture, "_create_llm_client", return_value=llm_client
        )
        assert (
            GeneratorFixture().init_llm_client(
                mocker.Mock(spec=LLM, enabled=True, provider="foo")
            )
            is llm_client
        )

    def test_init_llm_client_enabled_create_returns_none_returns_none(
        self, mocker: MockerFixture, generator_mocks: GeneratorMocks
    ) -> None:
        mocker.patch.object(GeneratorFixture, "_create_llm_client", return_value=None)

        assert (
            GeneratorFixture().init_llm_client(
                mocker.Mock(spec=LLM, enabled=True, provider="none")
            )
            is None
        )

    def test_init_llm_client_enabled_create_raises_returns_none(
        self, mocker: MockerFixture, generator_mocks: GeneratorMocks
    ) -> None:
        mocker.patch.object(
            GeneratorFixture, "_create_llm_client", side_effect=ValueError("thud")
        )
        assert (
            GeneratorFixture().init_llm_client(
                mocker.Mock(spec=LLM, enabled=True, provider="gemini")
            )
            is None
        )


class TestGenerateBatchId:
    def test_generate_batch_id_upload_disabled_defaults_to_sequence_one(
        self, mocker: MockerFixture, generator_mocks: GeneratorMocks
    ) -> None:
        mock_datetime(mocker)
        assert (
            GeneratorFixture().generate_batch_id(make_output(mocker), None, "eu-west-2")
            == "test_batch-2026-01-17-001"
        )

    def test_generate_batch_id_upload_enabled_no_existing_batches_returns_sequence_one(
        self, mocker: MockerFixture, generator_mocks: GeneratorMocks
    ) -> None:
        mock_datetime(mocker)
        list_s3_objects: MagicMock = mocker.patch(
            "docsynth.generate.AWS.list_s3_objects", return_value=[]
        )
        assert (
            GeneratorFixture().generate_batch_id(
                make_output(mocker, upload_enabled=True), "foo-bar", "eu-west-2"
            )
            == "test_batch-2026-01-17-001"
        )
        list_s3_objects.assert_called_once_with(
            "eu-west-2", "foo-bar", "documents/test_batch-2026-01-17"
        )

    def test_generate_batch_id_upload_enabled_existing_batches_increments_sequence(
        self, mocker: MockerFixture, generator_mocks: GeneratorMocks
    ) -> None:
        mock_datetime(mocker)
        mocker.patch(
            "docsynth.generate.AWS.list_s3_objects",
            return_value=[
                {"Key": "documents/test_batch-2026-01-17-001.tar.gz"},
                {"Key": "documents/test_batch-2026-01-17-002.tar.gz"},
            ],
        )
        assert (
            GeneratorFixture().generate_batch_id(
                make_output(mocker, upload_enabled=True), "foo-bar", "eu-west-2"
            )
            == "test_batch-2026-01-17-003"
        )


class TestPublishBatch:
    def test_publish_batch_no_documents_does_nothing(
        self, generator_mocks: GeneratorMocks, tmp_path: Path
    ) -> None:
        output_dir: str = str(tmp_path / "output" / "test_batch")
        GeneratorFixture().publish_batch(
            output_dir,
            generator_mocks.pipeline_config.output,
            None,
            "eu-west-2",
            "foo",
        )
        assert not Path(output_dir).exists()

    def test_publish_batch_documents_present_writes_metadata_files(
        self, mocker: MockerFixture, generator_mocks: GeneratorMocks, tmp_path: Path
    ) -> None:
        mock_datetime(mocker)
        output_dir: Path = tmp_path / "output" / "test_batch"
        output_dir.mkdir(parents=True)
        (output_dir / "foo.json").write_text("{}")
        GeneratorFixture().publish_batch(
            str(output_dir),
            make_output(mocker, description="bar"),
            None,
            "eu-west-2",
            "foo",
        )
        metadata: BatchMetadata = BatchMetadata.model_validate(
            json.loads((output_dir / "metadata.json").read_text())
        )
        assert metadata.num_documents == 1
        assert metadata.source_type == "DocSynth"
        assert metadata.domain == "foo"
        assert metadata.description == "bar"
        assert metadata.created_at == "2026-01-17T00:00:00"
        assert (output_dir / "documentbatch.txt").read_text() == metadata.to_text()

    def test_publish_batch_excludes_prior_metadata_json_from_document_count(
        self, mocker: MockerFixture, generator_mocks: GeneratorMocks, tmp_path: Path
    ) -> None:
        output_dir: Path = tmp_path / "output" / "test_batch"
        output_dir.mkdir(parents=True)
        (output_dir / "foo.json").write_text("{}")
        (output_dir / "metadata.json").write_text(
            BatchMetadata(
                batch_id="test_batch-2026-01-01-001",
                created_at="2026-01-01T00:00:00",
                num_documents=1,
                source_type="DocSynth",
            ).model_dump_json()
        )
        GeneratorFixture().publish_batch(
            str(output_dir),
            make_output(mocker),
            None,
            "eu-west-2",
            "foo",
        )
        assert (
            BatchMetadata.model_validate(
                json.loads((output_dir / "metadata.json").read_text())
            ).num_documents
            == 1
        )

    def test_publish_batch_upload_disabled_does_not_upload(
        self, mocker: MockerFixture, generator_mocks: GeneratorMocks, tmp_path: Path
    ) -> None:
        upload_file: MagicMock = mocker.patch("docsynth.generate.AWS.upload_file")
        output_dir: Path = tmp_path / "output" / "test_batch"
        output_dir.mkdir(parents=True)
        (output_dir / "foo.json").write_text("{}")
        GeneratorFixture().publish_batch(
            str(output_dir),
            make_output(mocker),
            None,
            "eu-west-2",
            "foo",
        )
        upload_file.assert_not_called()

    def test_publish_batch_upload_enabled_archives_folder_and_uploads_it(
        self, mocker: MockerFixture, generator_mocks: GeneratorMocks, tmp_path: Path
    ) -> None:
        upload_file: MagicMock = mocker.patch("docsynth.generate.AWS.upload_file")
        mocker.patch("docsynth.generate.AWS.list_s3_objects", return_value=[])
        mock_datetime(mocker)
        output_dir: Path = tmp_path / "output" / "test_batch"
        output_dir.mkdir(parents=True)
        (output_dir / "foo.json").write_text("{}")
        GeneratorFixture().publish_batch(
            str(output_dir),
            make_output(mocker, upload_enabled=True),
            "foo-bar",
            "eu-west-2",
            "foo",
        )
        archive_path: Path = tmp_path / "output" / "test_batch-2026-01-17-001.tar.gz"
        with tarfile.open(archive_path) as archive:
            assert sorted(archive.getnames()) == [
                "documentbatch.txt",
                "foo.json",
                "metadata.json",
            ]
        upload_file.assert_called_once_with(
            region_name="eu-west-2",
            file_name=str(archive_path),
            bucket="foo-bar",
            object_name="test_batch-2026-01-17-001.tar.gz",
            path="documents",
        )


class TestGenerate:
    def test_generate_llm_disabled_saves_prompt_only_document_with_expected_schema(
        self, generator_mocks: GeneratorMocks, tmp_path: Path
    ) -> None:
        Generator().generate()
        output_files: list[Path] = document_files(tmp_path / "output" / "test_batch")
        assert len(output_files) == 1
        doc_id: str = output_files[0].stem
        assert len(doc_id) == 32
        int(doc_id, 16)  # raises ValueError if doc_id is not valid hex (MD5)
        document: DocsynthDocument = DocsynthDocument.model_validate(
            json.loads(output_files[0].read_text())
        )
        assert document.doc_id == doc_id
        assert document.document_name == "nostructure"
        assert document.source == "DocSynth"
        assert document.profile == "foo_001"
        assert document.content == ""

    def test_generate_random_mode_llm_disabled_saves_prompt_only_document(
        self, mocker: MockerFixture, generator_mocks: GeneratorMocks, tmp_path: Path
    ) -> None:
        generator_mocks.pipeline_config.profile_selection = make_profile_selection(
            mocker, mode="random"
        )
        Generator().generate()
        output_files: list[Path] = document_files(tmp_path / "output" / "test_batch")
        assert len(output_files) == 1
        assert (
            DocsynthDocument.model_validate(
                json.loads(output_files[0].read_text())
            ).content
            == ""
        )

    def test_generate_sequential_mode_count_limited_stops_after_count_profiles(
        self, mocker: MockerFixture, generator_mocks: GeneratorMocks, tmp_path: Path
    ) -> None:
        generator_mocks.pipeline_config.profile_selection = make_profile_selection(
            mocker, count=1
        )
        Generator(
            assets=DocsynthAssetsFixture(["foo_001", "foo_002", "foo_003"])
        ).generate()
        assert len(document_files(tmp_path / "output" / "test_batch")) == 1

    def test_generate_profile_files_given_loads_from_specified_files_and_saves_document(
        self, mocker: MockerFixture, generator_mocks: GeneratorMocks, tmp_path: Path
    ) -> None:
        generator_mocks.pipeline_config.profile_selection = make_profile_selection(
            mocker, file=["foo.yml"]
        )
        Generator().generate()
        assert len(document_files(tmp_path / "output" / "test_batch")) == 1

    def test_generate_llm_enabled_sequential_mode_extracts_and_saves_content(
        self, mocker: MockerFixture, generator_mocks: GeneratorMocks, tmp_path: Path
    ) -> None:
        llm_client: Mock = mocker.Mock(spec=LLMClient)
        llm_client.generate.return_value = "<output>foobar waldo</output>"
        mocker.patch.object(Generator, "_init_llm_client", return_value=llm_client)
        Generator().generate()
        output_files: list[Path] = document_files(tmp_path / "output" / "test_batch")
        assert len(output_files) == 1
        assert (
            DocsynthDocument.model_validate(
                json.loads(output_files[0].read_text())
            ).content
            == "foobar waldo"
        )
        assert llm_client.generate.call_args[0][1] is None

    def test_generate_llm_enabled_sequential_mode_no_output_tags_saves_full_response(
        self, mocker: MockerFixture, generator_mocks: GeneratorMocks, tmp_path: Path
    ) -> None:
        llm_client: Mock = mocker.Mock(spec=LLMClient)
        llm_client.generate.return_value = "foo bar baz"
        mocker.patch.object(Generator, "_init_llm_client", return_value=llm_client)
        Generator().generate()
        assert (
            DocsynthDocument.model_validate(
                json.loads(
                    document_files(tmp_path / "output" / "test_batch")[0].read_text()
                )
            ).content
            == "foo bar baz"
        )

    def test_generate_llm_enabled_sequential_mode_generation_exception_skips_document(
        self, mocker: MockerFixture, generator_mocks: GeneratorMocks, tmp_path: Path
    ) -> None:
        llm_client: Mock = mocker.Mock(spec=LLMClient)
        llm_client.generate.side_effect = Exception("thud")
        mocker.patch.object(Generator, "_init_llm_client", return_value=llm_client)
        Generator().generate()
        assert not (tmp_path / "output" / "test_batch").exists()

    def test_generate_llm_enabled_random_mode_extracts_and_saves_content(
        self, mocker: MockerFixture, generator_mocks: GeneratorMocks, tmp_path: Path
    ) -> None:
        generator_mocks.pipeline_config.profile_selection = make_profile_selection(
            mocker, mode="random"
        )
        llm_client: Mock = mocker.Mock(spec=LLMClient)
        llm_client.generate.return_value = "<output>foobar waldo</output>"
        mocker.patch.object(Generator, "_init_llm_client", return_value=llm_client)
        Generator().generate()
        output_files: list[Path] = document_files(tmp_path / "output" / "test_batch")
        assert len(output_files) == 1
        assert (
            DocsynthDocument.model_validate(
                json.loads(output_files[0].read_text())
            ).content
            == "foobar waldo"
        )

    def test_generate_llm_enabled_random_mode_generation_exception_skips_document(
        self, mocker: MockerFixture, generator_mocks: GeneratorMocks, tmp_path: Path
    ) -> None:
        generator_mocks.pipeline_config.profile_selection = make_profile_selection(
            mocker, mode="random"
        )
        llm_client: Mock = mocker.Mock(spec=LLMClient)
        llm_client.generate.side_effect = Exception("thud")
        mocker.patch.object(Generator, "_init_llm_client", return_value=llm_client)
        Generator().generate()
        assert not (tmp_path / "output" / "test_batch").exists()

    def test_generate_llm_enabled_random_mode_no_output_tags_saves_full_response(
        self, mocker: MockerFixture, generator_mocks: GeneratorMocks, tmp_path: Path
    ) -> None:
        generator_mocks.pipeline_config.profile_selection = make_profile_selection(
            mocker, mode="random"
        )
        llm_client: Mock = mocker.Mock(spec=LLMClient)
        llm_client.generate.return_value = "foo bar baz"
        mocker.patch.object(Generator, "_init_llm_client", return_value=llm_client)
        Generator().generate()
        assert (
            DocsynthDocument.model_validate(
                json.loads(
                    document_files(tmp_path / "output" / "test_batch")[0].read_text()
                )
            ).content
            == "foo bar baz"
        )

    def test_generate_via_batch_random_mode_skips_saving_and_runs_batch_inference(
        self, mocker: MockerFixture, generator_mocks: GeneratorMocks, tmp_path: Path
    ) -> None:
        generator_mocks.pipeline_config.profile_selection = make_profile_selection(
            mocker, mode="random"
        )
        llm_client: Mock = mocker.Mock(spec=LLMClient)
        mocker.patch.object(Generator, "_init_llm_client", return_value=llm_client)
        Generator().generate_via_batch("foo-bar", "baz-qux")
        assert not (tmp_path / "output" / "test_batch").exists()
        llm_client.run_batch_inference.assert_called_once_with("foo-bar", "baz-qux")

    def test_generate_via_batch_sequential_mode_skips_saving_and_runs_batch_inference(
        self, mocker: MockerFixture, generator_mocks: GeneratorMocks, tmp_path: Path
    ) -> None:
        llm_client: Mock = mocker.Mock(spec=LLMClient)
        mocker.patch.object(Generator, "_init_llm_client", return_value=llm_client)
        Generator().generate_via_batch("foo-bar", "baz-qux")
        assert not (tmp_path / "output" / "test_batch").exists()
        llm_client.generate.assert_called_once_with(mocker.ANY, "foo_001-nostructure-1")
        llm_client.run_batch_inference.assert_called_once_with("foo-bar", "baz-qux")

    def test_generate_batch_bucket_and_role_given_directly_triggers_batch_mode(
        self, mocker: MockerFixture, generator_mocks: GeneratorMocks, tmp_path: Path
    ) -> None:
        llm_client: Mock = mocker.Mock(spec=LLMClient)
        mocker.patch.object(Generator, "_init_llm_client", return_value=llm_client)
        Generator().generate(batch_bucket="foo-bar", batch_role="baz-qux")
        assert not (tmp_path / "output" / "test_batch").exists()
        llm_client.run_batch_inference.assert_called_once_with("foo-bar", "baz-qux")

    def test_generate_skip_existing_no_prior_output_generates_normally(
        self, mocker: MockerFixture, generator_mocks: GeneratorMocks, tmp_path: Path
    ) -> None:
        generator_mocks.pipeline_config.output = make_output(mocker, skip_existing=True)
        Generator().generate()
        assert len(document_files(tmp_path / "output" / "test_batch")) == 1

    def test_generate_skip_existing_prior_profile_present_filters_it_out(
        self, mocker: MockerFixture, generator_mocks: GeneratorMocks, tmp_path: Path
    ) -> None:
        generator_mocks.pipeline_config.output = make_output(mocker, skip_existing=True)
        output_dir: Path = tmp_path / "output" / "test_batch"
        output_dir.mkdir(parents=True)
        (output_dir / "foobar.json").write_text(json.dumps({"profile": "foo_001"}))
        Generator(assets=DocsynthAssetsFixture(["foo_001", "foo_002"])).generate()
        new_files: list[Path] = [
            json_file
            for json_file in document_files(output_dir)
            if json_file.name != "foobar.json"
        ]
        assert len(new_files) == 1
        assert (
            DocsynthDocument.model_validate(
                json.loads(new_files[0].read_text())
            ).profile
            == "foo_002"
        )

    def test_generate_skip_existing_all_profiles_filtered_out_completes_without_error(
        self, mocker: MockerFixture, generator_mocks: GeneratorMocks, tmp_path: Path
    ) -> None:
        generator_mocks.pipeline_config.output = make_output(mocker, skip_existing=True)
        output_dir: Path = tmp_path / "output" / "test_batch"
        output_dir.mkdir(parents=True)
        (output_dir / "foobar.json").write_text(json.dumps({"profile": "foo_001"}))
        Generator(assets=DocsynthAssetsFixture(["foo_001"])).generate()
        assert list(output_dir.glob("*.json")) == [output_dir / "foobar.json"]

    def test_generate_skip_existing_malformed_existing_file_skips_it_and_generates_normally(
        self, mocker: MockerFixture, generator_mocks: GeneratorMocks, tmp_path: Path
    ) -> None:
        generator_mocks.pipeline_config.output = make_output(mocker, skip_existing=True)
        output_dir: Path = tmp_path / "output" / "test_batch"
        output_dir.mkdir(parents=True)
        (output_dir / "thud.json").write_text("not valid json")
        Generator().generate()
        assert (
            len(
                [
                    json_file
                    for json_file in document_files(output_dir)
                    if json_file.name != "thud.json"
                ]
            )
            == 1
        )


class TestExtractBatchOutput:
    def test_extract_batch_output_llm_disabled_does_nothing(
        self, generator_mocks: GeneratorMocks, tmp_path: Path
    ) -> None:
        Generator().extract_batch_output("foo-bar")
        assert not (tmp_path / "output" / "test_batch").exists()

    def test_extract_batch_output_outputs_available_saves_extracted_documents(
        self, mocker: MockerFixture, generator_mocks: GeneratorMocks, tmp_path: Path
    ) -> None:
        generator_mocks.pipeline_config.llm = mocker.Mock(spec=LLM, enabled=True)
        batch_output = mocker.Mock(recordId="foo_001-nostructure-1")
        batch_output.modelOutput.content = [
            mocker.Mock(text="<output>foobar waldo</output>")
        ]
        batch_output.modelInput.messages = [
            mocker.Mock(content=[mocker.Mock(text="quux")])
        ]
        llm_client: MagicMock = mocker.Mock(spec=LLMClient)
        llm_client.get_batch_inference_outputs.return_value = mocker.Mock(
            outputs=[batch_output]
        )
        mocker.patch.object(Generator, "_init_llm_client", return_value=llm_client)
        Generator().extract_batch_output("foo-bar")
        llm_client.get_batch_inference_outputs.assert_called_once_with("foo-bar")
        output_files: list[Path] = document_files(tmp_path / "output" / "test_batch")
        assert len(output_files) == 1
        document: DocsynthDocument = DocsynthDocument.model_validate(
            json.loads(output_files[0].read_text())
        )
        assert document.document_name == "nostructure"
        assert document.profile == "foo_001"
        assert document.content == "foobar waldo"

    def test_extract_batch_output_no_output_tags_logs_error_and_saves_full_response(
        self, mocker: MockerFixture, generator_mocks: GeneratorMocks, tmp_path: Path
    ) -> None:
        generator_mocks.pipeline_config.llm = mocker.Mock(spec=LLM, enabled=True)
        batch_output = mocker.Mock(recordId="foo_001-nostructure-1")
        batch_output.modelOutput.content = [mocker.Mock(text="foo bar baz")]
        batch_output.modelInput.messages = [
            mocker.Mock(content=[mocker.Mock(text="quux")])
        ]
        llm_client: MagicMock = mocker.Mock(spec=LLMClient)
        llm_client.get_batch_inference_outputs.return_value = mocker.Mock(
            outputs=[batch_output]
        )
        mocker.patch.object(Generator, "_init_llm_client", return_value=llm_client)
        Generator().extract_batch_output("foo-bar")
        assert (
            DocsynthDocument.model_validate(
                json.loads(
                    document_files(tmp_path / "output" / "test_batch")[0].read_text()
                )
            ).content
            == "foo bar baz"
        )

    def test_extract_batch_output_no_outputs_saves_nothing(
        self, mocker: MockerFixture, generator_mocks: GeneratorMocks, tmp_path: Path
    ) -> None:
        generator_mocks.pipeline_config.llm = mocker.Mock(spec=LLM, enabled=True)
        llm_client: MagicMock = mocker.Mock(spec=LLMClient)
        llm_client.get_batch_inference_outputs.return_value = None
        mocker.patch.object(Generator, "_init_llm_client", return_value=llm_client)
        Generator().extract_batch_output("foo-bar")
        assert not (tmp_path / "output" / "test_batch").exists()
