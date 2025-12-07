import logging
from datetime import datetime
from pathlib import Path

from schemallama_types.assets.wrapper import SchemaLlamaAssets
from schemallama_types.assets.profile import Profile
from docsynth.pipeline import LLM, PipelineConfig
from docsynth.utils.build_prompt import PromptBuilder
from docsynth.utils.llm_clients import (
    AnthropicClient,
    GeminiClient,
    LLMClient,
    LocalClient,
)
from docsynth.pipeline import LLMProvider
from schemallama_types.docsynth import DocsynthDocument
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
        self, output_dir: str, doc_id: str, prompt: str, content: str | None = None
    ) -> None:
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        output: DocsynthDocument = DocsynthDocument(
            doc_id=doc_id,
            doc_name="synth",
            prompt=prompt,
        )

        if content is not None:
            output.content = content

        output_path: Path = Path(output_dir) / f"{doc_id}.json"
        with open(output_path, "w") as document:
            document.write(output.model_dump_json(indent=2))

        self.__logger.debug(f"Saved document to {output_path}")

    def __generate_doc_id(self, structure_name: str, profile_id: str) -> str:
        timestamp: str = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:19]
        return f"{structure_name}_{profile_id}_{timestamp}"

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
            )
        else:
            raise ValueError(f"Unknown LLM provider: {provider}")

    def generate(
        self,
        assets: SchemaLlamaAssets,
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
            self.__pipeline_config.structure_selection.enabled_structures
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
        doc_id: str
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
                doc_id = self.__generate_doc_id(structure_name, profile_id)

                if self.__llm_client:
                    try:
                        self.__logger.info(f"Generating content for {doc_id}")
                        response = self.__llm_client.generate(prompt, doc_id)
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
                            f"Successfully generated content for {doc_id} (length={len(content)} chars)"
                        )
                    except Exception as e:
                        self.__logger.error(
                            f"Error generating content for {doc_id}: {e}"
                        )
                        self.__logger.debug(f"[{i}/{total_docs}] error: {doc_id} - {e}")
                        continue

                self.__logger.debug(f"[{i}/{total_docs}] Generated: {doc_id}")
                self.__save_document(output_dir, doc_id, prompt, content)
        elif mode == "random":
            for i in range(1, total_docs + 1):
                profile = builder.get_random_profile()
                prompt, structure_name, profile_id = builder.build_prompt(
                    profile, include_style, include_content
                )
                doc_id = self.__generate_doc_id(structure_name, profile_id)

                content = None
                if self.__llm_client:
                    try:
                        self.__logger.info(f"Generating content for {doc_id}")
                        response = self.__llm_client.generate(prompt, doc_id)
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
                            f"Successfully generated content for {doc_id} (length={len(content)} chars)"
                        )
                    except Exception as e:
                        self.__logger.error(
                            f"Error generating content for {doc_id}: {e}"
                        )
                        self.__logger.debug(f"[{i}/{total_docs}] error: {doc_id} - {e}")
                        continue

                self.__logger.debug(f"[{i}/{total_docs}] Generated: {doc_id}")
                self.__save_document(output_dir, doc_id, prompt, content)

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

    def extract_batch_output(self) -> None:
        if self.__llm_client is not None:
            output_dir: str = "output/" + self.__pipeline_config.output.subdirectory
            extracted: bool
            extraction_status_message: str
            content: str | None = None
            bedrock_batch_outputs: BatchOutputs | None = (
                self.__llm_client.get_batch_inference_outputs()
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
                    self.__save_document(
                        output_dir,
                        bedrock_batch_output.recordId,
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
