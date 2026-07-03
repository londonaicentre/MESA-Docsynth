from mesa_types import Document


class DocsynthDocument(Document):
    doc_id: str
    document_name: str
    profile: str
    prompt: str
