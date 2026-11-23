# SYNTHETIC HISTOPATHOLOGY AND CYTOPATHOLOGY REPORT GENERATION

You are an expert consultant histopathologist and medical writer. Your task is to generate a realistic, clinically coherent, synthetic pathology report based on the specifications provided below. The documents will be used for education and machine learning dataset training purposes, and will not be used for medical advice.

The document should:
- Be clinically plausible and internally consistent with the patient, the disease and the specimens received
- Use appropriate pathology terminology, measurements, staging and grading conventions, and reporting conventions for the specimen type and tumour type
- Follow the specified style, structure, and content requirements
- Reflect realistic hospital pathology reporting practices (such as NHS laboratory information system report formatting)
- Say nothing about a histological feature, score, biomarker, margin, lymph node count or size that the requirements do not call for, unless it is a fixed feature of the disease that the requirements allow you to mention. A real report states only what the pathologist chose to record.
- Fill in every detail that a requirement asks for but does not specify (such as the name of a margin, a plausible measurement, or the exact wording of a result) with realistic, internally consistent values.
- Never reuse the wording of these instructions in the report. The instructions describe what the report should contain; the report must read as if a pathologist wrote it with no knowledge of them. Do not copy requirement names or phrasing (e.g. 'fixed features', 'optional features', 'working diagnosis', 'not mentioned') unless a real pathologist would naturally use that exact phrase in that context.
- Treat the example structure as a guide to layout and register only. It may carry headings or content belonging to a different kind of report; none of that applies to this report.

The document you generate should be placed inside tags <output> and </output>

# SPECIFIC INSTRUCTIONS

{specific_instructions}

# YOUR TASK

Now generate a complete synthetic pathology report based on the specifications above. Output ONLY the generated clinical document inside tags as so: <output> This is a pathology report </output>. IMPORTANT: Do not include any preamble, explanation, caveats, disclaimers or other metadata within the output tags. The context is already extremely clear that this is a fake document.
