import hashlib
import json
import logging
from datetime import datetime
from pathlib import Path

from docsynth.types.profile import Profile
from docsynth.pipeline import LLM, PipelineConfig
from docsynth.types.wrapper import DocsynthAssets
from docsynth.utils.build_prompt import PromptBuilder
from docsynth.utils.llm_clients import (
    AnthropicClient,
    GeminiClient,
    LLMClient,
    LocalClient,
)
from docsynth.pipeline import LLMProvider
from docsynth.types.documents import DocsynthDocument
from utils.llm import BatchOutputs, LLM as LLMUtils


class Generator:
    """Config driven synthetic document generation"""

    def __init__(self) -> None:
        # basic now for debug
        logging.basicConfig(
            filename="debug.log",
            level=logging.DEBUG,
            format="%(asctime)s - %(levelname)s - %(message)s",
        )
        self.__logger: logging.Logger = logging.getLogger(__name__)
        self.__pipeline_config: PipelineConfig = PipelineConfig()
        self.__llm_client: LLMClient | None = self.__init_llm_client(
            self.__pipeline_config.llm
        )

    def __init_llm_client(self, llm_config: LLM) -> LLMClient | None:
        # initialise chosen LLM client
        llm_client: LLMClient | None = None
        if llm_config.enabled:
            provider: str = llm_config.provider
            try:
                self.__logger.debug(
                    f"Initialising LLM client (provider: {provider})..."
                )
                llm_client = self.__create_llm_client(llm_config)
                if llm_client:
                    self.__logger.debug("LLM client initialised")
                    self.__logger.info(f"LLM client initialised: {provider}")
                    return llm_client
                else:
                    self.__logger.debug(
                        "LLM generation disabled (provider set to 'none')"
                    )
            except Exception as e:
                self.__logger.debug(f"Error initialising LLM client: {e}")
                self.__logger.error(f"Failed to initialize LLM client: {e}")
        else:
            self.__logger.debug("LLM generation disabled (saving prompts only)")
            self.__logger.info("LLM generation disabled")
        return None

    def __save_document(
        self,
        output_dir: str,
        structure_name: str,
        profile_id: str,
        timestamp: str,
        prompt: str,
        content: str | None = None,
    ) -> None:
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        doc_id: str = self.__generate_document_id(prompt, content)
        output: DocsynthDocument = DocsynthDocument(
            doc_id=doc_id,
            document_name=structure_name,
            profile=profile_id,
            timestamp=timestamp,
            prompt=prompt,
        )

        if content is not None:
            output.content = content

        output_path: Path = Path(output_dir) / f"{doc_id}.json"
        with open(output_path, "w") as document:
            document.write(output.model_dump_json(indent=2))

        self.__logger.debug(f"Saved document to {output_path}")

    def __generate_timestamp(self) -> str:
        return datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:19]

    def __generate_document_id(self, prompt: str, content: str | None) -> str:
        hashed_text: str = content if content is not None else prompt
        return hashlib.md5(hashed_text.encode("utf-8")).hexdigest()

    def __get_existing_profile_ids(self, output_dir: str) -> set[str]:
        existing_profile_ids: set[str] = set()
        output_path: Path = Path(output_dir)
        if not output_path.exists():
            return existing_profile_ids

        json_file: Path
        for json_file in output_path.glob("*.json"):
            try:
                data: dict[str, str] = json.loads(json_file.read_text())
                profile_id: str | None = data.get("profile")
                if profile_id:
                    existing_profile_ids.add(profile_id)
            except (json.JSONDecodeError, KeyError) as e:
                self.__logger.warning(
                    f"Could not read profile from {json_file.name}: {e}"
                )
                continue
        return existing_profile_ids

    def __create_llm_client(self, llm_config: LLM) -> LLMClient | None:
        if not llm_config.enabled:
            print("LLM generation disabled")
            return None

        provider: str = llm_config.provider
        config: LLMProvider

        # Return None if no provider configured
        if provider == "none":
            print("LLM provider set to 'none'")
            return None
        elif provider == "gemini":
            config = llm_config.gemini
            if not config.api_key:
                raise ValueError(
                    "llm__gemini__api_key not found in environment variables"
                )
            return GeminiClient(
                model=config.model,
                temperature=config.temperature,
                max_tokens=config.max_tokens,
                api_key=config.api_key,
            )
        elif provider == "anthropic":
            config = llm_config.anthropic
            if not config.api_key:
                raise ValueError(
                    "llm__anthropic__api_key not found in environment variables"
                )
            return AnthropicClient(
                model=config.model,
                temperature=config.temperature,
                max_tokens=config.max_tokens,
                api_key=config.api_key,
            )
        elif provider == "local":
            config = llm_config.local
            if not config.base_url:
                raise ValueError(
                    "llm__local__base_url not found in environment variables"
                )
            base_url: str = config.base_url
            model: str = config.model
            return LocalClient(
                base_url=base_url,
                model=model,
                temperature=config.temperature,
                max_tokens=config.max_tokens,
                api_key=config.api_key or "not-needed",
            )
        else:
            raise ValueError(f"Unknown LLM provider: {provider}")

    def generate(
        self,
        assets: DocsynthAssets,
        bucket: str | None = None,
        bedrock_execution_role: str | None = None,
    ) -> None:
        """Generate one or more synthetic documents

        Args:
            assets (SchemaLlamaAssets): An assets wrapper object extending
                the SchemaLlamaAssets type

        """
        self.__logger.info("Starting document generation pipeline")
        self.__logger.debug("Loading pipeline.yml...")
        self.__logger.debug("Building prompt...")

        enabled_structures: list[str] = (
            self.__pipeline_config.structure_selection.enabled_structures or []
        )

        builder: PromptBuilder = PromptBuilder(assets, enabled_structures)

        profile_files: list[str] = self.__pipeline_config.profile_selection.file
        builder.load_profiles(profile_files)

        if profile_files:
            self.__logger.debug(f"Loaded profiles from: {', '.join(profile_files)}")
        else:
            self.__logger.debug("Loaded all profiles")

        self.__logger.debug(f"Total profiles: {builder.get_profile_count()}")

        output_dir: str = "output/" + self.__pipeline_config.output.subdirectory
        self.__logger.debug(f"Output directory: {output_dir}")

        if self.__pipeline_config.output.skip_existing:
            existing_profile_ids: set[str] = self.__get_existing_profile_ids(output_dir)
            filtered_count: int = builder.filter_existing_profiles(existing_profile_ids)
            self.__logger.info(
                f"Skip existing enabled: filtered out {filtered_count} existing profiles"
            )

        mode: str = str(self.__pipeline_config.profile_selection.mode)
        count: int = self.__pipeline_config.profile_selection.count
        include_style: bool = self.__pipeline_config.prompt_config.include_style
        include_content: bool = self.__pipeline_config.prompt_config.include_content

        total_docs: int = builder.get_profile_count() if count == -1 else count

        action: str = "documents" if self.__llm_client else "prompts"
        self.__logger.debug(f"Generating {total_docs} {action} in '{mode}' mode...")
        self.__logger.debug("#" * 60)

        # TODO: can refactor this as sequential and random share identical code
        batch: bool = bucket is not None and bedrock_execution_role is not None
        i: int
        profile: Profile
        prompt: str
        structure_name: str
        profile_id: str
        batch_entry_id: str
        response: str | None
        extracted: bool
        extraction_status_message: str
        content: str | None = None
        if mode == "sequential":
            for i, profile in enumerate(builder.get_sequential_profiles(), 1):
                if i > total_docs:
                    break

                prompt, structure_name, profile_id = builder.build_prompt(
                    profile, include_style, include_content
                )
                batch_entry_id = f"{profile_id}-{structure_name}-{i}"

                if self.__llm_client:
                    try:
                        self.__logger.info(f"Generating content for {batch_entry_id}")
                        response = self.__llm_client.generate(
                            prompt, batch_entry_id if batch else None
                        )
                        if batch:
                            continue
                        extracted, extraction_status_message, content = (
                            LLMUtils.extract_output_content(str(response))
                        )
                        if extracted:
                            self.__logger.info(extraction_status_message)
                        else:
                            self.__logger.error(extraction_status_message)
                        self.__logger.info(
                            f"Successfully generated content for {batch_entry_id} (length={len(content)} chars)"
                        )
                    except Exception as e:
                        self.__logger.error(
                            f"Error generating content for {batch_entry_id}: {e}"
                        )
                        self.__logger.debug(
                            f"[{i}/{total_docs}] error: {batch_entry_id} - {e}"
                        )
                        continue

                self.__logger.debug(f"[{i}/{total_docs}] Generated: {batch_entry_id}")
                self.__save_document(
                    output_dir,
                    structure_name,
                    profile_id,
                    self.__generate_timestamp(),
                    prompt,
                    content,
                )
        elif mode == "random":
            for i in range(1, total_docs + 1):
                profile = builder.get_random_profile()
                prompt, structure_name, profile_id = builder.build_prompt(
                    profile, include_style, include_content
                )
                batch_entry_id = f"{profile_id}-{structure_name}-{i}"

                content = None
                if self.__llm_client:
                    try:
                        self.__logger.info(f"Generating content for {batch_entry_id}")
                        response = self.__llm_client.generate(prompt, batch_entry_id)
                        if batch:
                            continue
                        extracted, extraction_status_message, content = (
                            LLMUtils.extract_output_content(str(response))
                        )
                        if extracted:
                            self.__logger.info(extraction_status_message)
                        else:
                            self.__logger.error(extraction_status_message)
                        self.__logger.info(
                            f"Successfully generated content for {batch_entry_id} (length={len(content)} chars)"
                        )
                    except Exception as e:
                        self.__logger.error(
                            f"Error generating content for {batch_entry_id}: {e}"
                        )
                        self.__logger.debug(
                            f"[{i}/{total_docs}] error: {batch_entry_id} - {e}"
                        )
                        continue

                self.__logger.debug(f"[{i}/{total_docs}] Generated: {batch_entry_id}")
                self.__save_document(
                    output_dir,
                    structure_name,
                    profile_id,
                    self.__generate_timestamp(),
                    prompt,
                    content,
                )

        self.__logger.debug("#" * 60)
        if (
            self.__llm_client is not None
            and bucket is not None
            and bedrock_execution_role is not None
        ):
            self.__llm_client.run_batch_inference(bucket, bedrock_execution_role)
        else:
            self.__logger.debug(f"Generated {total_docs} {action}")
            self.__logger.debug(f"Saved to: {output_dir}")
            self.__logger.info(
                f"Pipeline completed successfully. Generated {total_docs} {action}"
            )

    def extract_batch_output(self, bucket: str) -> None:
        if self.__llm_client is not None:
            output_dir: str = "output/" + self.__pipeline_config.output.subdirectory
            extracted: bool
            extraction_status_message: str
            content: str | None = None
            bedrock_batch_outputs: BatchOutputs | None = (
                self.__llm_client.get_batch_inference_outputs(bucket)
            )
            if bedrock_batch_outputs is not None:
                for bedrock_batch_output in bedrock_batch_outputs.outputs:
                    extracted, extraction_status_message, content = (
                        LLMUtils.extract_output_content(
                            str(bedrock_batch_output.modelOutput.content[0].text)
                        )
                    )
                    if extracted:
                        self.__logger.info(extraction_status_message)
                    else:
                        self.__logger.error(extraction_status_message)
                    self.__logger.info(
                        f"Successfully extracted content for {bedrock_batch_output.recordId} (length={len(content)} chars)"
                    )
                    profile_id, structure_name, _ = bedrock_batch_output.recordId.split(
                        "-"
                    )
                    self.__save_document(
                        output_dir,
                        structure_name,
                        profile_id,
                        self.__generate_timestamp(),
                        bedrock_batch_output.modelInput.messages[0].content[0].text,
                        content,
                    )

                self.__logger.debug(
                    f"Generated {len(bedrock_batch_outputs.outputs)} {'documents'}"
                )
                self.__logger.debug(f"Saved to: {output_dir}")
                self.__logger.info(
                    f"Pipeline completed successfully. Generated {len(bedrock_batch_outputs.outputs)} {'documents'}"
                )
