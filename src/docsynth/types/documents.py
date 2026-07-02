from pydantic import BaseModel


class DocsynthDocument(BaseModel):
    doc_id: str
    document_name: str
    document_sourcedb: str = "DocSynth"
    profile: str
    timestamp: str
    prompt: str
    content: str | None = None
