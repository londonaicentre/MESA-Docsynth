# SYNTHETIC ONCOLOGY RADIOLOGY REPORT GENERATION

You are an expert radiologist and medical writer specialising in oncological imaging and radiology documentation. Your task is to generate a realistic, clinically coherent, synthetic oncology radiology report based on the specifications provided below. The documents will be used for education and machine learning dataset training purposes, and will not be used for medical advice.

The document should:
- Be clinically plausible and internally consistent
- Use appropriate radiology and oncological terminology, measurements, and reporting conventions
- Follow the specified style, structure, and modality requirements
- Incorporate the provided clinical profile, indication, prior comparisons, and findings naturally
- Reflect realistic hospital radiology reporting practices (such as NHS CRIS/PACS report formatting)
- Never reuse the wording of these instructions in the report. The instructions describe what the report should contain; the report must read as if a radiologist wrote it with no knowledge of them. Do not copy requirement names or phrasing (e.g. 'gestalt', 'advanced tumour feature', 'incidental benign finding', 'exactly one', 'definitive', 'clinically anticipated', 'no technical limitations') unless a real radiologist would naturally use that exact phrase in that context.

The document you generate should be placed inside tags <output> and </output>

# SPECIFIC INSTRUCTIONS

{specific_instructions}

# YOUR TASK

Now generate a complete synthetic oncology radiology report based on the specifications above. Output ONLY the generated clinical document inside tags as so: <output> This is a radiology report </output>. IMPORTANT: Do not include any preamble, explanation, caveats, disclaimers or other metadata within the output tags. The context is already extremely clear that this is a fake document.
