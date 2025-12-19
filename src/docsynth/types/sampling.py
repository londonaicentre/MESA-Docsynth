from importlib.resources.abc import Traversable

from litellm import BaseModel
from pydantic import ConfigDict
import yaml


class SamplingAttribute(BaseModel):
    probability: float
    description: str


class SamplingElement(BaseModel):
    mutually_exclusive: bool


class DataQualityIssues(SamplingElement):
    none: SamplingAttribute
    missing_dates: SamplingAttribute
    missing_stage_information: SamplingAttribute
    misspelled_terms: SamplingAttribute
    inconsistent_formatting: SamplingAttribute
    truncated_content: SamplingAttribute
    corrupted_values: SamplingAttribute
    missing_sections: SamplingAttribute


class RedactionStyle(SamplingElement):
    fully_redacted: SamplingAttribute
    partially_redacted: SamplingAttribute
    not_redacted: SamplingAttribute


class FooterStyle(SamplingElement):
    complex_footer: SamplingAttribute
    formal_footer: SamplingAttribute
    basic_footer: SamplingAttribute
    no_footer: SamplingAttribute


class HeaderStyle(SamplingElement):
    complex_header: SamplingAttribute
    standard_header: SamplingAttribute
    minimal_header: SamplingAttribute
    no_header: SamplingAttribute


class WritingStyle(SamplingElement):
    concise_factual: SamplingAttribute
    verbose_professional: SamplingAttribute
    narrative_empathetic: SamplingAttribute


class ContentType(SamplingElement):
    initial_consultation: SamplingAttribute
    follow_up_stable: SamplingAttribute
    treatment_review: SamplingAttribute
    treatment_change: SamplingAttribute
    disease_change: SamplingAttribute
    mdt_outcome: SamplingAttribute
    post_acute: SamplingAttribute


class DocumentType(SamplingElement):
    is_a_letter: SamplingAttribute
    is_not_a_letter: SamplingAttribute


class Style(BaseModel):
    document_type: DocumentType
    content_type: ContentType
    writing_style: WritingStyle
    header_style: HeaderStyle
    footer_style: FooterStyle
    redaction_style: RedactionStyle
    data_quality_issues: DataQualityIssues

    def __init__(self, file_path: Traversable) -> None:
        super().__init__(**yaml.safe_load(file_path.read_text()))


class TimelineEvents(SamplingElement):
    clinical_trial_consideration: SamplingAttribute
    clinical_trial_enrollment: SamplingAttribute
    patient_death: SamplingAttribute
    __pydantic_extra__: dict[str, SamplingAttribute]

    model_config = ConfigDict(
        extra="allow",
    )


class PatientFindings(SamplingElement):
    comorbidity: SamplingAttribute
    social_history: SamplingAttribute
    family_history: SamplingAttribute
    symptoms_present: SamplingAttribute
    symptoms_resolved: SamplingAttribute
    physical_exam_positive: SamplingAttribute


class FunctionalStatus(SamplingElement):
    good: SamplingAttribute
    poor: SamplingAttribute
    omit: SamplingAttribute


class MentalState(SamplingElement):
    positive: SamplingAttribute
    distressed: SamplingAttribute
    omit: SamplingAttribute


class DiagnosisDate(SamplingElement):
    include_year_and_month: SamplingAttribute
    include_year_only: SamplingAttribute
    omit: SamplingAttribute


class DiseaseTrajectory(SamplingElement):
    stable: SamplingAttribute
    improving: SamplingAttribute
    slowly_progressing: SamplingAttribute
    rapidly_deteriorating: SamplingAttribute
    remission: SamplingAttribute


class TreatmentComplexity(SamplingElement):
    simple: SamplingAttribute
    moderate: SamplingAttribute
    complex: SamplingAttribute


class Content(BaseModel):
    timeline_events: TimelineEvents
    patient_findings: PatientFindings
    functional_status: FunctionalStatus
    mental_state: MentalState
    diagnosis_date: DiagnosisDate
    disease_trajectory: DiseaseTrajectory
    treatment_complexity: TreatmentComplexity
    __pydantic_extra__: dict[str, dict[str, bool | SamplingAttribute]]

    model_config = ConfigDict(
        extra="allow",
    )

    def __init__(self, file_path: Traversable) -> None:
        super().__init__(**yaml.safe_load(file_path.read_text()))
