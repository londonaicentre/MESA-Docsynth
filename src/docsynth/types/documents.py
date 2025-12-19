from pydantic import BaseModel


class DocsynthDocument(BaseModel):
    doc_id: str
    doc_name: str
    prompt: str
    content: str | None = None
