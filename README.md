# SchemaLlama: Docsynth

Configurable pipeline for generating high fidelity synthetic documents that can be turned into training data.

## Getting started

### AWS

- Obtain a Bedrock API key from an account manager.

### Other providers

If not using AWS Bedrock, obtain suitable credentials for another provider (e.g. Gemini Developer API). If using a local LLM, no/blank credentials will likely be sufficient.

### Assets

The [`DocsynthAssets`](src/docsynth/types/wrapper.py) base class should be extended to wrap assets for a particular domain (e.g. oncology) and pass them to docsynth.
The base class assumes the presence of:

- Primary profiles, one entry per synthetic case, holding whatever domain-specific fields that case needs (formatted according to [`Profiles`](src/docsynth/types/profile.py)):
  - `assets/<use case>/profiles/*.yml`

- Probabilistic sampling from style and content requirements, with domain-defined sections (formatted according to [`Style`](src/docsynth/types/sampling.py) and [`Content`](src/docsynth/types/sampling.py)):
  - `assets/<use case>/style.yml`
  - `assets/<use case>/content.yml`

- Example structures that are hand-crafted based on real clinical document formats:
  - `assets/<use case>/structure/*.txt`

- Prompt templates:
  - `assets/<use case>/prompts/`

In addition, concrete implementations should be provided for abstract methods to handle domain-specific asset creation logic.

### Configuration

Using the format specified in the [`config`](src/docsynth/config.py), create `pipeline.yml` to configure the LLM provider (currently gemini or local), profile sampling mode (random/sequential), prompt configuration, and output directory (see [`pipeline.yml.example`](pipeline.yml.example)). 
If using Gemini or Bedrock, configure `.env` with API key (see [`.env.example`](.env.example)).

## Usage

1. Create a `Generator` object, and call the `generate` method with a child of `DocsynthAssets`:

  ```python
  Generator().generate(MyDocsynthAssets())
  ```

2. Generated documents are saved to `./output/{subdirectory}/{doc_id}.json`:

  ```json
  {
    "doc_id": "9893532233caff98cd083a116b013c0b",
    "document_name": "narrative",
    "document_sourcedb": "DocSynth",
    "profile": "lung_001",
    "timestamp": "20251012_143022_477",
    "prompt": "... complete prompt text ...",
    "content": "... generated clinical document ..."
  }
  ```

Note: when `llm.enabled: false` in `pipeline.yml`, only prompts are saved (no `content` field).

### Flowchart

```mermaid
flowchart LR
    A[pipeline.yml] --> G[generate.py]
    B[profiles/*.yml] --> C[load_profiles.py]
    D[structure/*.txt] --> E[load_structure.py]
    F[style.yml & content.yml] --> H[load_sampling.py]
    C --> I[build_prompt.py]
    E --> I
    H --> I
    I --> G
    J[llm_clients.py] --> G
    G --> K[output/*.json]

```
