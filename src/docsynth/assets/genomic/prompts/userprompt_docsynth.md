# SYNTHETIC GENOMIC TEST REPORT GENERATION

You are an expert clinical geneticist and medical writer specialising in genetic testing and genomic reporting. Your task is to generate a realistic, clinically coherent, synthetic genomic test report based on the specifications provided below. The documents will be used for education and machine learning dataset training purposes, and will not be used for medical advice.

The document should:
- Be clinically plausible and internally consistent with the patient, the clinical question and the examination performed
- Report only what this examination could show (e.g. no variant detection unless the report covers it)
- Use appropriate genomic terminology, and reporting conventions for the modality, especially around gene and protein nomenclature and variant representation
- Follow the specified style, structure, and content requirements
- Reflect realistic hospital genomic reporting practices (such as NHS and Genomics England report formatting)
- Never reuse the wording of these instructions in the report. The instructions describe what the report should contain; the report must read as if a clinical geniticist wrote it with no knowledge of them. Do not copy requirement names or phrasing (e.g. 'possible findings', 'unrelated', 'absent abnormalities', 'single judgement', 'characterises differently', 'listed') unless a real clinical genomics scientist would naturally use that exact phrase in that context.

The document you generate should be placed inside tags <output> and </output> 

# SPECIFIC INSTRUCTIONS

{specific_instructions}

# YOUR TASK

Now generate a complete synthetic genomic test report based on the specifications above. Output ONLY the generated clinical document inside tags as so: <output> This is a genomic test report </output>. IMPORTANT: Do not include any preamble, explanation, caveats, disclaimers or other metadata within the output tags. The context is already extremely clear that this is a fake document.
