This folder contains chest radiology profiles. Each profile is a synthetic patient: who they are, their known or suspected diagnosis, why they are being imaged, and a list of findings that could plausibly appear on their scan, and whether the study is unremarkable.

The description never states what the scan shows. `possible_findings` lists candidates only; content.yml decides how many (if any) are reported as present or explicitly absent, and `unremarkable: true` profiles (~15% of each file) make the study normal, with possible findings mentioned only as absent.

Profile IDs must be unique across all files, as output.skip_existing in pipeline.yml matches on them.
