import json
import logging
import re
from datetime import datetime
from pathlib import Path

from docsynth.config import LLM, PipelineConfig
from schemallama_types.assets import Profile, SchemaLlamaAssets
from docsynth.utils.build_prompt import PromptBuilder
from docsynth.utils.llm_clients import (
    AnthropicClient,
    GeminiClient,
    LLMClient,
    LocalClient,
)
from docsynth.config import LLMProvider


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

    def __extract_output_content(self, response_text: str) -> str:
        pattern: str = r"<OUTPUT>(.*?)</OUTPUT>"
        match: re.Match[str] | None = re.search(pattern, response_text, re.DOTALL)

        if match:
            content: str = match.group(1).strip()
            self.__logger.debug(
                f"Successfully extracted content from <OUTPUT> tags (length={len(content)} chars)"
            )
            return content
        else:
            self.__logger.warning(
                "No <OUTPUT> tags found in response, using full response text"
            )
            return response_text.strip()

    def __save_document(
        self, output_dir: str, doc_id: str, prompt: str, content: str | None = None
    ) -> None:
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        output: dict[str, str] = {
            "doc_id": doc_id,
            "doc_name": "synth",
            "prompt": prompt,
        }

        if content is not None:
            output["content"] = content

        output_path: Path = Path(output_dir) / f"{doc_id}.json"
        with open(output_path, "w") as f:
            json.dump(output, f, indent=2)

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

    def generate(self, assets: SchemaLlamaAssets) -> None:
        """Generate one or more synthetic documents

        Args:
            assets (SchemaLlamaAssets): An assets wrapper object extending
                the SchemaLlamaAssets type

        """
        self.__logger.info("Starting document generation pipeline")
        self.__logger.debug("Loading pipeline.yml...")
        pipeline_config: PipelineConfig = PipelineConfig()

        self.__logger.debug("Building prompt...")

        enabled_structures: list[str] = (
            pipeline_config.structure_selection.enabled_structures
        )

        builder: PromptBuilder = PromptBuilder(assets, enabled_structures)

        profile_files: list[str] = pipeline_config.profile_selection.file
        builder.load_profiles(profile_files)

        if profile_files:
            self.__logger.debug(f"Loaded profiles from: {', '.join(profile_files)}")
        else:
            self.__logger.debug("Loaded all profiles")

        self.__logger.debug(f"Total profiles: {builder.get_profile_count()}")

        # initialise chosen LLM client
        llm_config: LLM = pipeline_config.llm
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
                else:
                    self.__logger.debug(
                        "LLM generation disabled (provider set to 'none')"
                    )
            except Exception as e:
                self.__logger.debug(f"Error initialising LLM client: {e}")
                self.__logger.error(f"Failed to initialize LLM client: {e}")
                return
        else:
            self.__logger.debug("LLM generation disabled (saving prompts only)")
            self.__logger.info("LLM generation disabled")

        output_dir: str = "output/" + pipeline_config.output.subdirectory
        self.__logger.debug(f"Output directory: {output_dir}")

        mode: str = str(pipeline_config.profile_selection.mode)
        count: int = pipeline_config.profile_selection.count
        include_style: bool = pipeline_config.prompt_config.include_style
        include_content: bool = pipeline_config.prompt_config.include_content

        total_docs: int = builder.get_profile_count() if count == -1 else count

        action: str = "documents" if llm_client else "prompts"
        self.__logger.debug(f"Generating {total_docs} {action} in '{mode}' mode...")
        self.__logger.debug("#" * 60)

        # TODO: can refactor this as sequential and random share identical code
        i: int
        profile: Profile
        prompt: str
        structure_name: str
        profile_id: str
        content: str | None = None
        response: str | None
        if mode == "sequential":
            for i, profile in enumerate(builder.get_sequential_profiles(), 1):
                if i > total_docs:
                    break

                prompt, structure_name, profile_id = builder.build_prompt(
                    profile, include_style, include_content
                )
                doc_id: str = self.__generate_doc_id(structure_name, profile_id)

                if llm_client:
                    try:
                        self.__logger.info(f"Generating content for {doc_id}")
                        response = llm_client.generate(prompt)
                        content = self.__extract_output_content(str(response))
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
                if llm_client:
                    try:
                        self.__logger.info(f"Generating content for {doc_id}")
                        response = llm_client.generate(prompt)
                        content = self.__extract_output_content(str(response))
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
        self.__logger.debug(f"Generated {total_docs} {action}")
        self.__logger.debug(f"Saved to: {output_dir}")
        self.__logger.info(
            f"Pipeline completed successfully. Generated {total_docs} {action}"
        )
