# Configuration-Driven Synthetic Document Generation

Configurable pipeline for generating high fidelity synthetic documents that can be turned into training data. Supports multiple domains (e.g., cancer, general medical) with domain-specific profiles and content configurations.

## Overview

The pipeline is configured from `pipeline.yml` and run from `generate.py`.

The following components can be used to add deeper customisation for dynamically creating synthetic data generation prompts:
- Probabilistic sampling from style and content requirements:
  - `./config/style/*.yml`
  - `./config/content/*.yml`
  - `./utils/load_sampling.py`
- Domain-specific profiles (organized by domain, e.g., cancer):
  - `./config/profiles/domain/*.yml`
  - `./utils/load_profiles.py`
- Example structures that are hand-crafted based on real document formats:
  - `./config/structure/*.txt`
  - `./utils/load_structure.py`
- Prompt templates for different domains:
  - `./prompts/*.md`
- Prompt builder that assembles all components into LLM prompts:
  - `./utils/build_prompt.py`

The prompt is passed to an LLM API. Different clients can be defined via `./utils/llm_clients.py`.

## Flowchart

```mermaid
flowchart LR
    A[pipeline.yml] --> G[generate.py]
    B[config/profiles/{domain}/*.yml] --> C[load_profiles.py]
    D[config/structure/*.txt] --> E[load_structure.py]
    F[config/style/*.yml & content/*.yml] --> H[load_sampling.py]
    P[prompts/*.md] --> I[build_prompt.py]
    C --> I
    E --> I
    H --> I
    I --> G
    J[llm_clients.py] --> G
    G --> K[output/*.json]

```

## Quick Start

### 1. Configure the pipeline

Edit `pipeline.yml` to configure LLM provider (gemini, claude, or local), domain selection, profile sampling mode (random/sequential), prompt configuration, and output directory. Configure `.env` with API keys as needed (see `.env.example`).


### 2. Run the generator

```bash
python generate.py
```

Note: while in development, verbose logs are output to `debug.log`.

### 3. View outputs

Generated documents are saved to `./output/{subdirectory}/`:

```json
{
  "doc_id": "{structure}_{profile}_{timestamp}",
  "doc_name": "synth",
  "prompt": "... complete prompt text ...",
  "content": "... generated document ..."
}
```

Note: when `llm.enabled: false` in `pipeline.yml`, only prompts are saved (no `content` field).

## Planned Features

- ?Cleaner primary configuration instead of pipeline.yml 
- Add testing for configuration
- Add Bedrock client
- Add option to directly generate training data
- Better logging/error handling (currently skips)
- Small refactor in generate.py for sequential/random docs
