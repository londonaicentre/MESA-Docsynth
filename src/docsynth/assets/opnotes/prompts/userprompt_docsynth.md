# SYNTHETIC OPERATION NOTE GENERATION

You are an experienced surgeon and medical writer. Your task is to generate a realistic, clinically coherent, synthetic operation note based on the specifications provided below. The documents will be used for education and machine learning dataset training purposes, and will not be used for medical advice.

The document should:
- Be an operation note written by the operating team for a procedure just performed on this patient in an operating theatre (or endoscopy, interventional radiology or cardiac catheter suite)
- Be clinically plausible and internally consistent, with operative steps, findings, instruments, devices and anaesthesia that suit the operation
- Use the terminology, abbreviations and conventions that surgeons in this specialty actually use
- Follow the specified style, structure and content requirements
- Incorporate the provided operation profile naturally
- Reflect realistic NHS operating theatre documentation practices
- Take only the information from the profile, style and content instructions, never their wording. They describe what the note should contain; the note must read as if a surgeon wrote it with no knowledge of them. Do not copy sentences or phrases from the profile (the patient and clinical context, the complication description) or from the instructions (requirement names, descriptions or their examples); re-express the facts in the surgeon's own words, abbreviations and order. Device names, sizes and specifications are the exception: keep them exact. Do not copy requirement names or phrasing (e.g. 'more than planned', 'different operation of similar extent', 'reservation', 'unrelated', 'explicitly', 'not stated') unless a real surgeon would naturally use that exact phrase in that context.
- Record each piece of information where a surgeon naturally would (indication, findings, procedure, complications, plan). Do not add headings or sentences whose only purpose is to show that an instruction was followed, such as an 'Outcome', 'Complexity' or 'Previous operation' section tacked onto the end.

The document you generate should be placed inside tags <output> and </output>

# SPECIFIC INSTRUCTIONS

{specific_instructions}

# YOUR TASK

Now generate a complete synthetic operation note based on the specifications above. Output ONLY the generated clinical document inside tags as so: <output> This is an operation note </output>. IMPORTANT: Do not include any preamble, explanation, caveats, disclaimers or other metadata within the output tags. The context is already extremely clear that this is a fake document.
