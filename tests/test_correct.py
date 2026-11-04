import hashlib
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from pytest_mock import MockerFixture
from utils.llm import LLM

from docsynth.correct import Corrector
from docsynth.types.correction_rules import (
    DocumentCorrectionAction,
    DocumentCorrectionRule,
    DocumentCorrectionRules,
)
from docsynth.types.documents import DocsynthDocument


class CorrectorFixture(Corrector):
    def read_documents(self) -> list[DocsynthDocument]:
        return self._read_documents()

    def write_document(self, document: DocsynthDocument) -> None:
        self._write_document(document)

    @staticmethod
    def is_unmodified(document: DocsynthDocument) -> bool:
        return Corrector._is_unmodified(document)

    @staticmethod
    def apply_replacements(
        content: str, rules: list[DocumentCorrectionRule]
    ) -> str | None:
        return Corrector._apply_replacements(content, rules)

    @staticmethod
    def review_document(
        system_prompt: str, document: DocsynthDocument
    ) -> list[DocumentCorrectionRule]:
        return Corrector._review_document(system_prompt, document)


def make_document(content: str) -> DocsynthDocument:
    return DocsynthDocument(
        source="DocSynth",
        content=content,
        timestamp="20260117_000000_000",
        doc_id=hashlib.md5(content.encode()).hexdigest(),
        document_name="foo",
        profile="bar_001",
        prompt="baz",
    )


def make_replace(
    find: str, value: str, description: str = "qux", document: str = "foo"
) -> DocumentCorrectionRule:
    return DocumentCorrectionRule(
        name="foo-1",
        description=description,
        document=document,
        action=DocumentCorrectionAction.REPLACE,
        find=find,
        value=value,
    )


def make_regenerate(document: str, description: str = "qux") -> DocumentCorrectionRule:
    return DocumentCorrectionRule(
        name=f"{document[:8]}-1",
        description=description,
        document=document,
        action=DocumentCorrectionAction.REGENERATE,
    )


@dataclass
class ClaudeMocks:
    claude_code: MagicMock


@pytest.fixture
def claude_mocks(mocker: MockerFixture) -> ClaudeMocks:
    return ClaudeMocks(claude_code=mocker.patch.object(LLM, "claude_code"))


@dataclass
class ReviewMocks:
    read_documents: MagicMock
    review_document: MagicMock
    prompt_file: Path
    rules_file: Path


@pytest.fixture
def review_mocks(mocker: MockerFixture, tmp_path: Path) -> ReviewMocks:
    prompt_file: Path = tmp_path / "prompt.txt"
    prompt_file.write_text("foo")
    return ReviewMocks(
        read_documents=mocker.patch.object(
            Corrector,
            "_read_documents",
            return_value=[make_document("bar"), make_document("baz")],
        ),
        review_document=mocker.patch.object(
            Corrector,
            "_review_document",
            side_effect=lambda _, document: [make_regenerate(document.doc_id)],
        ),
        prompt_file=prompt_file,
        rules_file=tmp_path / "rules" / "rules.yaml",
    )


@dataclass
class CorrectWithRulesMocks:
    load: MagicMock
    read_documents: MagicMock
    is_unmodified: MagicMock
    apply_replacements: MagicMock
    write_document: MagicMock


@pytest.fixture
def correct_with_rules_mocks(mocker: MockerFixture) -> CorrectWithRulesMocks:
    return CorrectWithRulesMocks(
        load=mocker.patch.object(
            DocumentCorrectionRules,
            "load",
            return_value=DocumentCorrectionRules(
                rules=[
                    make_replace(
                        "foo", "qux", document=make_document("foo bar").doc_id
                    ),
                    make_regenerate(make_document("baz").doc_id),
                ]
            ),
        ),
        read_documents=mocker.patch.object(
            Corrector,
            "_read_documents",
            return_value=[
                make_document("foo bar"),
                make_document("baz"),
                make_document("quux"),
            ],
        ),
        is_unmodified=mocker.patch.object(
            Corrector, "_is_unmodified", return_value=True
        ),
        apply_replacements=mocker.patch.object(
            Corrector, "_apply_replacements", return_value="qux bar"
        ),
        write_document=mocker.patch.object(Corrector, "_write_document"),
    )


@dataclass
class CorrectWithClaudeMocks:
    claude_code: MagicMock
    load: MagicMock
    read_documents: MagicMock
    is_unmodified: MagicMock
    write_document: MagicMock


@pytest.fixture
def correct_with_claude_mocks(
    mocker: MockerFixture, claude_mocks: ClaudeMocks
) -> CorrectWithClaudeMocks:
    claude_mocks.claude_code.return_value = "<output>corge</output>"
    return CorrectWithClaudeMocks(
        claude_code=claude_mocks.claude_code,
        load=mocker.patch.object(
            DocumentCorrectionRules,
            "load",
            return_value=DocumentCorrectionRules(
                rules=[
                    make_replace(
                        "foo", "qux", document=make_document("foo bar").doc_id
                    ),
                    make_regenerate(make_document("baz").doc_id, "qux; quux"),
                ]
            ),
        ),
        read_documents=mocker.patch.object(
            Corrector,
            "_read_documents",
            return_value=[
                make_document("foo bar"),
                make_document("baz"),
                make_document("quux"),
            ],
        ),
        is_unmodified=mocker.patch.object(
            Corrector, "_is_unmodified", return_value=True
        ),
        write_document=mocker.patch.object(Corrector, "_write_document"),
    )


class TestReadDocuments:
    def test_read_documents_folder_given_returns_sorted_documents_without_metadata(
        self, tmp_path: Path
    ) -> None:
        document: DocsynthDocument
        for document in [make_document("foo"), make_document("bar")]:
            (tmp_path / f"{document.doc_id}.json").write_text(
                document.model_dump_json()
            )
        (tmp_path / "metadata.json").write_text("{}")
        assert [
            document.content
            for document in CorrectorFixture(
                tmp_path, tmp_path / "rules.yaml"
            ).read_documents()
        ] == sorted(
            ["foo", "bar"],
            key=lambda content: hashlib.md5(content.encode()).hexdigest(),
        )


class TestWriteDocument:
    def test_write_document_document_given_writes_it_by_doc_id(
        self, tmp_path: Path
    ) -> None:
        document: DocsynthDocument = make_document("foo")
        CorrectorFixture(tmp_path, tmp_path / "rules.yaml").write_document(document)
        assert (
            DocsynthDocument.model_validate_json(
                (tmp_path / f"{document.doc_id}.json").read_text()
            )
            == document
        )


class TestIsUnmodified:
    @pytest.mark.parametrize(("content", "expected"), [("foo", True), ("bar", False)])
    def test_is_unmodified_content_given_returns_whether_it_hashes_to_doc_id(
        self, content: str, expected: bool
    ) -> None:
        assert (
            CorrectorFixture.is_unmodified(
                make_document("foo").model_copy(update={"content": content})
            )
            is expected
        )


class TestApplyReplacements:
    @pytest.mark.parametrize(
        ("rules", "expected"),
        [
            ([], "foo bar"),
            ([make_replace("foo ", "")], "bar"),
            ([make_replace("foo", "baz"), make_replace("bar", "qux")], "baz qux"),
            ([make_replace("quux", "")], None),
            ([make_replace("o", "")], None),
            ([make_replace("foo", "bar"), make_replace("bar", "")], None),
            ([make_regenerate("foo")], None),
        ],
    )
    def test_apply_replacements_rules_given_returns_replaced_content_or_none(
        self, rules: list[DocumentCorrectionRule], expected: str | None
    ) -> None:
        assert CorrectorFixture.apply_replacements("foo bar", rules) == expected


class TestReviewDocument:
    def test_review_document_prompts_claude_with_generation_prompt_and_document(
        self, claude_mocks: ClaudeMocks
    ) -> None:
        claude_mocks.claude_code.return_value = "<output>[]</output>"
        CorrectorFixture.review_document("foo", make_document("bar"))
        claude_mocks.claude_code.assert_called_once_with(
            "claude-sonnet-5-5",
            "foo",
            "<generation_prompt>\nbaz\n</generation_prompt>\n\n<document>\nbar\n</document>",
        )

    def test_review_document_no_output_tags_raises_value_error(
        self, claude_mocks: ClaudeMocks
    ) -> None:
        claude_mocks.claude_code.return_value = "[]"
        with pytest.raises(ValueError):
            CorrectorFixture.review_document("foo", make_document("bar"))

    def test_review_document_no_corrections_returns_no_rules(
        self, claude_mocks: ClaudeMocks
    ) -> None:
        claude_mocks.claude_code.return_value = "<output>[]</output>"
        assert CorrectorFixture.review_document("foo", make_document("bar")) == []

    def test_review_document_applicable_replacements_returns_named_replace_rules(
        self, claude_mocks: ClaudeMocks
    ) -> None:
        document: DocsynthDocument = make_document("foo bar")
        claude_mocks.claude_code.return_value = (
            '<output>[{"description": "baz", "action": "replace", "find": "foo", "value": "qux"},'
            ' {"description": "quux", "action": "replace", "find": "bar", "value": ""}]</output>'
        )
        assert CorrectorFixture.review_document("foo", document) == [
            DocumentCorrectionRule(
                name=f"{document.doc_id[:8]}-1",
                description="baz",
                document=document.doc_id,
                action=DocumentCorrectionAction.REPLACE,
                find="foo",
                value="qux",
            ),
            DocumentCorrectionRule(
                name=f"{document.doc_id[:8]}-2",
                description="quux",
                document=document.doc_id,
                action=DocumentCorrectionAction.REPLACE,
                find="bar",
                value="",
            ),
        ]

    @pytest.mark.parametrize(
        "second_correction",
        [
            '{"description": "quux", "action": "replace", "find": "corge", "value": ""}',
            '{"description": "quux", "action": "regenerate"}',
        ],
    )
    def test_review_document_inapplicable_or_regenerate_correction_returns_one_regenerate_rule(
        self, claude_mocks: ClaudeMocks, second_correction: str
    ) -> None:
        document: DocsynthDocument = make_document("foo bar")
        claude_mocks.claude_code.return_value = (
            '<output>[{"description": "baz", "action": "replace", "find": "foo", "value": "qux"}, '
            + second_correction
            + "]</output>"
        )
        assert CorrectorFixture.review_document("foo", document) == [
            make_regenerate(document.doc_id, "baz; quux")
        ]


class TestReview:
    def test_review_no_rules_file_reviews_every_document_and_writes_rules(
        self, review_mocks: ReviewMocks, tmp_path: Path
    ) -> None:
        assert (
            Corrector(tmp_path, review_mocks.rules_file).review(
                review_mocks.prompt_file
            )
            == 2
        )
        assert DocumentCorrectionRules.load(
            review_mocks.rules_file
        ) == DocumentCorrectionRules(
            reviewed=[make_document("bar").doc_id, make_document("baz").doc_id],
            rules=[
                make_regenerate(make_document("bar").doc_id),
                make_regenerate(make_document("baz").doc_id),
            ],
        )

    def test_review_rules_file_exists_reviews_only_unreviewed_documents(
        self, review_mocks: ReviewMocks, tmp_path: Path
    ) -> None:
        DocumentCorrectionRules(
            reviewed=[make_document("bar").doc_id],
        ).dump(review_mocks.rules_file)
        assert (
            Corrector(tmp_path, review_mocks.rules_file).review(
                review_mocks.prompt_file
            )
            == 1
        )
        assert review_mocks.review_document.call_count == 1
        assert DocumentCorrectionRules.load(review_mocks.rules_file).reviewed == [
            make_document("bar").doc_id,
            make_document("baz").doc_id,
        ]

    @pytest.mark.parametrize(("count", "dumps"), [(49, 1), (50, 2), (101, 3)])
    def test_review_many_documents_saves_every_fifty_and_at_end(
        self,
        mocker: MockerFixture,
        review_mocks: ReviewMocks,
        tmp_path: Path,
        count: int,
        dumps: int,
    ) -> None:
        review_mocks.read_documents.return_value = [
            make_document(f"foo{index}") for index in range(count)
        ]
        dump: MagicMock = mocker.patch.object(DocumentCorrectionRules, "dump")
        Corrector(tmp_path, review_mocks.rules_file).review(review_mocks.prompt_file)
        assert dump.call_count == dumps


class TestCorrectWithRules:
    def test_correct_with_rules_replace_rules_given_writes_only_their_documents(
        self, correct_with_rules_mocks: CorrectWithRulesMocks, tmp_path: Path
    ) -> None:
        assert Corrector(tmp_path, tmp_path / "rules.yaml").correct_with_rules() == 1
        correct_with_rules_mocks.apply_replacements.assert_called_once_with(
            "foo bar",
            [make_replace("foo", "qux", document=make_document("foo bar").doc_id)],
        )
        correct_with_rules_mocks.write_document.assert_called_once_with(
            make_document("foo bar").model_copy(update={"content": "qux bar"})
        )

    def test_correct_with_rules_document_modified_skips_it(
        self, correct_with_rules_mocks: CorrectWithRulesMocks, tmp_path: Path
    ) -> None:
        correct_with_rules_mocks.is_unmodified.return_value = False
        assert Corrector(tmp_path, tmp_path / "rules.yaml").correct_with_rules() == 0
        correct_with_rules_mocks.write_document.assert_not_called()

    def test_correct_with_rules_replacements_no_longer_apply_raises_value_error(
        self, correct_with_rules_mocks: CorrectWithRulesMocks, tmp_path: Path
    ) -> None:
        correct_with_rules_mocks.apply_replacements.return_value = None
        with pytest.raises(ValueError):
            Corrector(tmp_path, tmp_path / "rules.yaml").correct_with_rules()
        correct_with_rules_mocks.write_document.assert_not_called()


class TestCorrectWithClaude:
    def test_correct_with_claude_regenerate_rule_given_regenerates_only_its_document(
        self, correct_with_claude_mocks: CorrectWithClaudeMocks, tmp_path: Path
    ) -> None:
        assert Corrector(tmp_path, tmp_path / "rules.yaml").correct_with_claude() == 1
        correct_with_claude_mocks.claude_code.assert_called_once_with(
            "claude-haiku-5-5",
            "baz",
            "<document>\nbaz\n</document>\n\n"
            "The document above was generated from the system prompt, but breaks it in these ways:\n\n"
            "- qux; quux\n\n"
            "Rewrite the document so that it meets every requirement of the system prompt, changing only what fixing these problems needs.\n"
            "Reply with the rewritten document inside <output> tags.\n",
        )
        correct_with_claude_mocks.write_document.assert_called_once_with(
            make_document("baz").model_copy(update={"content": "corge"})
        )

    def test_correct_with_claude_document_modified_skips_it(
        self, correct_with_claude_mocks: CorrectWithClaudeMocks, tmp_path: Path
    ) -> None:
        correct_with_claude_mocks.is_unmodified.return_value = False
        assert Corrector(tmp_path, tmp_path / "rules.yaml").correct_with_claude() == 0
        correct_with_claude_mocks.claude_code.assert_not_called()

    def test_correct_with_claude_no_output_tags_raises_value_error(
        self, correct_with_claude_mocks: CorrectWithClaudeMocks, tmp_path: Path
    ) -> None:
        correct_with_claude_mocks.claude_code.return_value = "corge"
        with pytest.raises(ValueError):
            Corrector(tmp_path, tmp_path / "rules.yaml").correct_with_claude()
        correct_with_claude_mocks.write_document.assert_not_called()
