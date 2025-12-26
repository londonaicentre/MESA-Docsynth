import json
import logging
import re
from datetime import datetime
from pathlib import Path

import yaml
from dotenv import load_dotenv

from utils.build_prompt import PromptBuilder
from utils.llm_clients import create_llm_client

"""
generate.py - config driven synthetic document generation
"""

load_dotenv()

# basic now for debug
logging.basicConfig(
    filename="debug.log",
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def load_pipeline_config(config_path):
    """
    Loads main configuration as defined in pipeline.yml
    """
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def extract_output_content(response_text):
    """
    Extract content between <output> tags
    """
    pattern = r"<output>(.*?)</output>"
    match = re.search(pattern, response_text, re.DOTALL)

    if match:
        content = match.group(1).strip()
        logger.debug(
            f"Successfully extracted content from <output> tags (length={len(content)} chars)"
        )
        return content
    else:
        logger.warning("No <output> tags found in response, using full response text")
        return response_text.strip()


def save_document(output_dir, doc_id, prompt, content=None):
    """
    Saves output document as JSON file
    If content is None, only saves prompt (debugging prompt-only mode)
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    output = {"doc_id": doc_id, "doc_name": "synth", "prompt": prompt}

    if content is not None:
        output["content"] = content

    output_path = output_dir / f"{doc_id}.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    logger.debug(f"Saved document to {output_path}")


def generate_doc_id(structure_name, profile_id):
    """
    Generate unique document ID as {structure}_{profile}_{timestamp}
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:19]
    return f"{structure_name}_{profile_id}_{timestamp}"


def get_existing_profile_ids(output_dir):
    """
    Scan output directory for existing JSON files and extract profile IDs
    Filename format: {structure}_{profile_id}_{timestamp}.json
    Where timestamp is: YYYYMMDD_HHMMSS_mmm (3 parts)
    """
    existing_profiles = set()

    if not output_dir.exists():
        return existing_profiles

    for json_file in output_dir.glob("*.json"):
        filename = json_file.stem
        parts = filename.split("_")

        if len(parts) >= 5:
            profile_id = "_".join(parts[1:-3])
            if profile_id:
                existing_profiles.add(profile_id)

    return existing_profiles


def main():
    base_dir = Path(__file__).parent

    logger.info("Starting document generation pipeline")
    print("Loading pipeline.yml...")
    pipeline_config = load_pipeline_config(str(base_dir / "pipeline.yml"))

    print("Building prompt...")

    # configuration sections
    prompt_template = pipeline_config["prompt_config"].get("prompt_template", "default")
    enabled_structures = pipeline_config["structure_selection"]["enabled_structures"]

    # domain and file selections
    domain = pipeline_config["profile_selection"]["domain"]
    profile_files = pipeline_config["profile_selection"].get("file")
    style_file = pipeline_config["style_selection"]["file"]
    content_file = pipeline_config["content_selection"]["file"]

    # build prompt
    builder = PromptBuilder(
        template_name=prompt_template,
        enabled_structures=enabled_structures,
        style_file=style_file,
        content_file=content_file,
        domain=domain
    )

    builder.load_profiles(profile_files)

    if profile_files:
        print(f"Loaded profiles from {domain}/{', '.join(profile_files)}")
    else:
        print(f"Loaded all profiles from '{domain}' domain")

    print(f"Total profiles: {builder.get_profile_count()}")
    print(f"Using prompt template: {prompt_template}")

    # initialise chosen LLM client
    llm_config = pipeline_config.get("llm", {})
    llm_client = None

    if llm_config.get("enabled", False):
        provider = llm_config.get("provider", "none")
        try:
            print(f"Initialising LLM client (provider: {provider})...")
            llm_client = create_llm_client(llm_config)
            if llm_client:
                print("LLM client initialised")
                logger.info(f"LLM client initialised: {provider}")
            else:
                print("LLM generation disabled (provider set to 'none')")
        except Exception as e:
            print(f"Error initialising LLM client: {e}")
            logger.error(f"Failed to initialize LLM client: {e}")
            return
    else:
        print("LLM generation disabled (saving prompts only)")
        logger.info("LLM generation disabled")

    output_dir = base_dir / "output" / pipeline_config["output"]["subdirectory"]
    print(f"Output directory: {output_dir}")

    # skip_existing flag to filter profiles
    skip_existing = pipeline_config["output"].get("skip_existing", False)

    if skip_existing:
        existing_profiles = get_existing_profile_ids(output_dir)
        filtered_count = builder.profile_loader.filter_existing_profiles(
            existing_profiles
        )
        print(f"Skip existing enabled: Filtered out {filtered_count} existing profiles")
        logger.info(
            f"Skip existing enabled: {filtered_count} profiles already generated"
        )

    mode = pipeline_config["profile_selection"]["mode"]
    count = pipeline_config["profile_selection"]["count"]
    include_style = pipeline_config["prompt_config"]["include_style"]
    include_content = pipeline_config["prompt_config"]["include_content"]

    total_docs = builder.get_profile_count() if count == -1 else count

    action = "documents" if llm_client else "prompts"
    print(f"Generating {total_docs} {action} in '{mode}' mode...")
    print("#" * 60)

    # todo: can refactor this as sequential and random share identical code
    if mode == "sequential":
        for i, profile in enumerate(builder.get_sequential_profiles(), 1):
            if i > total_docs:
                break

            prompt, structure_name, profile_id = builder.build_prompt(
                profile, include_style, include_content
            )
            doc_id = generate_doc_id(structure_name, profile_id)

            content = None
            if llm_client:
                try:
                    logger.info(f"Generating content for {doc_id}")
                    response = llm_client.generate(prompt)
                    content = extract_output_content(response)
                    logger.info(
                        f"Successfully generated content for {doc_id} (length={len(content)} chars)"
                    )
                except Exception as e:
                    logger.error(f"Error generating content for {doc_id}: {e}")
                    print(f"[{i}/{total_docs}] error: {doc_id} - {e}")
                    continue

            print(f"[{i}/{total_docs}] Generated: {doc_id}")
            save_document(output_dir, doc_id, prompt, content)

    elif mode == "random":
        for i in range(1, total_docs + 1):
            profile = builder.get_random_profile()

            prompt, structure_name, profile_id = builder.build_prompt(
                profile, include_style, include_content
            )
            doc_id = generate_doc_id(structure_name, profile_id)

            content = None
            if llm_client:
                try:
                    logger.info(f"Generating content for {doc_id}")
                    response = llm_client.generate(prompt)
                    content = extract_output_content(response)
                    logger.info(
                        f"Successfully generated content for {doc_id} (length={len(content)} chars)"
                    )
                except Exception as e:
                    logger.error(f"Error generating content for {doc_id}: {e}")
                    print(f"[{i}/{total_docs}] error: {doc_id} - {e}")
                    continue

            print(f"[{i}/{total_docs}] Generated: {doc_id}")
            save_document(output_dir, doc_id, prompt, content)

    print("#" * 60)
    print(f"Generated {total_docs} {action}")
    print(f"Saved to: {output_dir}")
    logger.info(f"Pipeline completed successfully. Generated {total_docs} {action}")


if __name__ == "__main__":
    main()
