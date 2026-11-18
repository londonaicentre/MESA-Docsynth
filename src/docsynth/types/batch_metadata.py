from pydantic import BaseModel


class BatchMetadata(BaseModel):
    batch_id: str
    created_at: str
    num_documents: int
    source_type: str
    domain: str | None = None
    description: str | None = None
    profile_filter: str | None = None
    structure_filter: str | None = None
    style_filter: str | None = None
    content_filter: str | None = None

    def to_text(self) -> str:
        """Render a human-readable summary of the batch

        Returns:
            str: The formatted summary

        """
        lines: list[str] = [f"Document Batch: {self.batch_id}"]
        lines.append("")
        lines.append(f"Created: {self.created_at}")
        lines.append(f"Source: {self.source_type}")
        if self.domain is not None:
            lines.append(f"Domain: {self.domain}")
        lines.extend(
            f"{label} Filter: {pattern}"
            for label, pattern in [
                ("Profile", self.profile_filter),
                ("Structure", self.structure_filter),
                ("Style", self.style_filter),
                ("Content", self.content_filter),
            ]
            if pattern is not None
        )
        lines.append(f"Total Documents: {self.num_documents}")
        if self.description is not None:
            lines.append("")
            lines.append("Description:")
            lines.append(self.description)
        return "\n".join(lines)
