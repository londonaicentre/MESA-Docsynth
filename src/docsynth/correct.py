import hashlib
import json
import logging
from importlib.resources import files
from pathlib import Path

from utils.llm import LLM

from docsynth.types.correction_rules import (
    DocumentCorrectionAction,
    DocumentCorrectionRule,
    DocumentCorrectionRules,
)
from docsynth.types.documents import DocsynthDocument


class Corrector:
    """Corrector of generated documents against the prompts they were generated from.

    Args:
        documents_folder (Path): Folder of generated documents, as saved by
            `Generator`
        rules_file (Path, optional): Rules file that a review writes, and
            corrections are read from. Defaults to
            "rules/document_correction.yaml"

    """

    __REVIEW_MODEL: str = "claude-sonnet-5-5"
    __REGENERATION_MODEL: str = "claude-haiku-5-5"
    __SAVE_INTERVAL: int = 50

    def __init__(
        self,
        documents_folder: Path,
        rules_file: Path = Path("rules/document_correction.yaml"),
    ) -> None:
        self.__logger: logging.Logger = logging.getLogger(__name__)
        if not self.__logger.hasHandlers():
            self.__logger.addHandler(logging.StreamHandler())
            self.__logger.setLevel(logging.INFO)

        self.__documents_folder: Path = documents_folder
        self.__rules_file: Path = rules_file

    def _read_documents(self) -> list[DocsynthDocument]:
        return [
            DocsynthDocument.model_validate_json(document_file.read_text())
            for document_file in sorted(self.__documents_folder.glob("*.json"))
            if document_file.name != "metadata.json"
        ]

    def _write_document(self, document: DocsynthDocument) -> None:
        (self.__documents_folder / f"{document.doc_id}.json").write_text(
            document.model_dump_json(indent=2)
        )

    @staticmethod
    def _is_unmodified(document: DocsynthDocument) -> bool:
        return hashlib.md5(document.content.encode()).hexdigest() == document.doc_id

    @staticmethod
    def _apply_replacements(
        content: str, rules: list[DocumentCorrectionRule]
    ) -> str | None:
        for rule in rules:
            if rule.find is None or rule.value is None or content.count(rule.find) != 1:
                return None
            content = content.replace(rule.find, rule.value)
        return content

    @staticmethod
    def _review_document(
        system_prompt: str, document: DocsynthDocument
    ) -> list[DocumentCorrectionRule]:
        extracted, message, output = LLM.extract_output_content(
            LLM.claude_code(
                Corrector.__REVIEW_MODEL,
                system_prompt,
                f"<generation_prompt>\n{document.prompt}\n</generation_prompt>\n\n"
                f"<document>\n{document.content}\n</document>",
            )
        )
        if not extracted:
            raise ValueError(f"Document {document.doc_id}: {message}")
        rules: list[DocumentCorrectionRule] = [
            DocumentCorrectionRule.model_validate(
                {
                    **correction,
                    "name": f"{document.doc_id[:8]}-{index}",
                    "document": document.doc_id,
                }
            )
            for index, correction in enumerate(json.loads(output), 1)
        ]
        if Corrector._apply_replacements(document.content, rules) is not None:
            return rules
        return [
            DocumentCorrectionRule(
                name=f"{document.doc_id[:8]}-1",
                description="; ".join(rule.description for rule in rules),
                document=document.doc_id,
                action=DocumentCorrectionAction.REGENERATE,
            )
        ]

    def review(self, prompt_file: Path) -> int:
        """Review every document not yet reviewed, adding its rules to the rules file.

        Args:
            prompt_file (Path): Review prompt, to which the output format is
                appended

        Returns:
            int: Number of documents reviewed

        Raises:
            ValueError: If a review has no `<output>` tags

        """
        system_prompt: str = (
            prompt_file.read_text()
            + "\n\n"
            + files("docsynth")
            .joinpath("config/document_correction_format.txt")
            .read_text()
        )
        rules: DocumentCorrectionRules = (
            DocumentCorrectionRules.load(self.__rules_file)
            if self.__rules_file.exists()
            else DocumentCorrectionRules()
        )
        reviewed: set[str] = set(rules.reviewed)
        documents: list[DocsynthDocument] = [
            document
            for document in self._read_documents()
            if document.doc_id not in reviewed
        ]
        index: int
        document: DocsynthDocument
        for index, document in enumerate(documents, 1):
            rules.rules.extend(Corrector._review_document(system_prompt, document))
            rules.reviewed.append(document.doc_id)
            if index % Corrector.__SAVE_INTERVAL == 0:
                rules.dump(self.__rules_file)
                self.__logger.info(f"Reviewed {index}/{len(documents)} documents")
        rules.dump(self.__rules_file)
        return len(documents)

    def correct_with_rules(self) -> int:
        """Apply the `replace` rules to every unmodified document they apply to.

        Returns:
            int: Number of documents corrected

        Raises:
            ValueError: If a document's `replace` rules no longer apply to it

        """
        replacements: dict[str, list[DocumentCorrectionRule]] = {}
        rule: DocumentCorrectionRule
        for rule in DocumentCorrectionRules.load(self.__rules_file).rules:
            if rule.action == DocumentCorrectionAction.REPLACE:
                replacements.setdefault(rule.document, []).append(rule)
        documents: list[DocsynthDocument] = [
            document
            for document in self._read_documents()
            if document.doc_id in replacements and Corrector._is_unmodified(document)
        ]
        document: DocsynthDocument
        for document in documents:
            content: str | None = Corrector._apply_replacements(
                document.content, replacements[document.doc_id]
            )
            if content is None:
                raise ValueError(
                    f"Document {document.doc_id}: replace rules no longer apply"
                )
            self._write_document(document.model_copy(update={"content": content}))
        return len(documents)

    def correct_with_claude(self) -> int:
        """Regenerate, with Claude Code, every unmodified document with a `regenerate` rule.

        Returns:
            int: Number of documents regenerated

        Raises:
            ValueError: If a regeneration has no `<output>` tags

        """
        template: str = (
            files("docsynth").joinpath("config/document_regeneration.txt").read_text()
        )
        problems: dict[str, list[str]] = {}
        rule: DocumentCorrectionRule
        for rule in DocumentCorrectionRules.load(self.__rules_file).rules:
            if rule.action == DocumentCorrectionAction.REGENERATE:
                problems.setdefault(rule.document, []).append(rule.description)
        documents: list[DocsynthDocument] = [
            document
            for document in self._read_documents()
            if document.doc_id in problems and Corrector._is_unmodified(document)
        ]
        index: int
        document: DocsynthDocument
        extracted: bool
        message: str
        content: str
        for index, document in enumerate(documents, 1):
            extracted, message, content = LLM.extract_output_content(
                LLM.claude_code(
                    Corrector.__REGENERATION_MODEL,
                    document.prompt,
                    template.format(
                        document=document.content,
                        problems="\n".join(
                            f"- {problem}" for problem in problems[document.doc_id]
                        ),
                    ),
                )
            )
            if not extracted:
                raise ValueError(f"Document {document.doc_id}: {message}")
            self._write_document(document.model_copy(update={"content": content}))
            self.__logger.info(f"Regenerated {index}/{len(documents)} documents")
        return len(documents)
