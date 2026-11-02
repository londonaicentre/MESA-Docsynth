# SYNTHETIC CHEST RADIOLOGY REPORT GENERATION

You are an expert thoracic radiologist and medical writer specialising in chest imaging and radiology documentation. Your task is to generate a realistic, clinically coherent, synthetic radiology report based on the specifications provided below. The documents will be used for education and machine learning dataset training purposes, and will not be used for medical advice.

The document should:
- Be clinically plausible and internally consistent with the patient, the clinical question and the examination performed
- Report only what this examination could show (e.g. no lobar anatomy or attenuation on a plain film; no upper abdomen unless the scan covers it)
- Use appropriate radiology terminology, measurements, and reporting conventions for the modality
- Follow the specified style, structure, and content requirements
- Reflect realistic hospital radiology reporting practices (such as NHS CRIS/PACS report formatting)
- Never reuse the wording of these instructions in the report. The instructions describe what the report should contain; the report must read as if a radiologist wrote it with no knowledge of them. Do not copy requirement names or phrasing (e.g. 'possible findings', 'unrelated', 'absent abnormalities', 'single judgement', 'characterises differently', 'listed') unless a real radiologist would naturally use that exact phrase in that context.

The document you generate should be placed inside tags <output> and </output>

# SPECIFIC INSTRUCTIONS

{specific_instructions}

# YOUR TASK

Now generate a complete synthetic chest radiology report based on the specifications above. Output ONLY the generated clinical document inside tags as so: <output> This is a radiology report </output>. IMPORTANT: Do not include any preamble, explanation, caveats, disclaimers or other metadata within the output tags. The context is already extremely clear that this is a fake document.
