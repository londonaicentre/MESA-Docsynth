This directory contains chest radiology report layouts, used when listed in structure_selection.enabled_structures in pipeline.yml. Any template may be paired with any profile: they convey structure and register only, and the prompt instructs the generator to substitute the profile's own modality, anatomy and findings.

- `cxr_*`, `itu_mobile`, `ct_chest_systematic`, `hrct_ild`, `ctpa`: synthetic layouts.
- `ct_thorax_postcovid_ild`, `hrct_lch_ph`, `ct_coronary_ctpa_hrct_ssc`, `ctpa_sarcoid_fibrosis`, `hrct_ila`, `hrct_srild_nodules`, `hrct_mda5_ild`, `hrct_ipf_brief`, `hrct_cop`, `hrct_cystic_brief`: near-verbatim variations of real CT/HRCT reports, keeping their layout, typos and quirks, with dates shifted and named locations redacted.
