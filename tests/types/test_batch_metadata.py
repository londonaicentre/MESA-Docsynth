import pytest

from docsynth.types.batch_metadata import BatchMetadata


class TestToText:
    @pytest.mark.parametrize(
        ("domain", "description", "expected"),
        [
            (
                "oncology",
                "bar",
                "Document Batch: test_batch-2026-01-17-001\n\n"
                "Created: 2026-01-17T00:00:00\n"
                "Source: DocSynth\n"
                "Domain: oncology\n"
                "Total Documents: 3\n\n"
                "Description:\n"
                "bar",
            ),
            (
                None,
                None,
                "Document Batch: test_batch-2026-01-17-001\n\n"
                "Created: 2026-01-17T00:00:00\n"
                "Source: DocSynth\n"
                "Total Documents: 3",
            ),
        ],
    )
    def test_to_text_optional_fields_given_renders_expected_summary(
        self, domain: str | None, description: str | None, expected: str
    ) -> None:
        assert (
            BatchMetadata(
                batch_id="test_batch-2026-01-17-001",
                created_at="2026-01-17T00:00:00",
                num_documents=3,
                source_type="DocSynth",
                domain=domain,
                description=description,
            ).to_text()
            == expected
        )
