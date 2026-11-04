from enum import StrEnum
from pathlib import Path

from pydantic import BaseModel, ConfigDict, model_validator
import yaml


class DocumentCorrectionAction(StrEnum):
    REPLACE = "replace"
    REGENERATE = "regenerate"


class DocumentCorrectionRule(BaseModel):
    """A single correction to a document.

    Attributes:
        name (str): Unique rule name
        description (str): What the rule corrects and why
        document (str): `doc_id` of the document to correct
        action (DocumentCorrectionAction): `replace` (replace `find` with
            `value`) or `regenerate` (regenerate the document against its
            prompt)
        find (str): Text to replace, required by `replace`. Defaults to None
        value (str): Replacement text, required by `replace`. Defaults to None

    """

    model_config = ConfigDict(extra="forbid")
    name: str
    description: str
    document: str
    action: DocumentCorrectionAction
    find: str | None = None
    value: str | None = None

    @model_validator(mode="after")
    def check_arguments(self) -> "DocumentCorrectionRule":
        """Check that `replace` rules give `find` and `value`, and `regenerate` rules do not.

        Returns:
            DocumentCorrectionRule: The validated rule

        Raises:
            ValueError: If `find` or `value` is missing from a `replace` rule,
                or given in a `regenerate` rule

        """
        if any(
            (argument is not None) != (self.action == DocumentCorrectionAction.REPLACE)
            for argument in (self.find, self.value)
        ):
            raise ValueError(
                f"Rule {self.name}: replace requires find and value, and regenerate forbids them"
            )
        return self


class DocumentCorrectionRules(BaseModel):
    """The rules from a review of a folder of documents.

    Attributes:
        reviewed (list[str]): `doc_id` of every reviewed document
        rules (list[DocumentCorrectionRule]): The rules

    """

    reviewed: list[str] = []
    rules: list[DocumentCorrectionRule] = []

    @staticmethod
    def load(rules_file: Path) -> "DocumentCorrectionRules":
        """Load and validate rules from a YAML file.

        Args:
            rules_file (Path): The rules file

        Returns:
            DocumentCorrectionRules: The validated rules

        """
        return DocumentCorrectionRules.model_validate(
            yaml.safe_load(rules_file.read_text())
        )

    def dump(self, rules_file: Path) -> None:
        """Write the rules to a YAML file, leaving out unset arguments.

        Args:
            rules_file (Path): The rules file

        """
        rules_file.parent.mkdir(parents=True, exist_ok=True)
        rules_file.write_text(
            yaml.safe_dump(
                self.model_dump(mode="json", exclude_none=True),
                sort_keys=False,
                allow_unicode=True,
                width=float("inf"),
            )
        )
