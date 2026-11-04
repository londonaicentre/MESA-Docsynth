from pathlib import Path

from pydantic import ValidationError
import pytest

from docsynth.types.correction_rules import (
    DocumentCorrectionAction,
    DocumentCorrectionRule,
    DocumentCorrectionRules,
)


class TestCheckArguments:
    @pytest.mark.parametrize(
        ("action", "find", "value"),
        [
            (DocumentCorrectionAction.REPLACE, "foo", "bar"),
            (DocumentCorrectionAction.REPLACE, "foo", ""),
            (DocumentCorrectionAction.REGENERATE, None, None),
        ],
    )
    def test_check_arguments_valid_arguments_returns_rule(
        self,
        action: DocumentCorrectionAction,
        find: str | None,
        value: str | None,
    ) -> None:
        assert (
            DocumentCorrectionRule(
                name="foo-1",
                description="bar",
                document="baz",
                action=action,
                find=find,
                value=value,
            ).action
            == action
        )

    @pytest.mark.parametrize(
        ("action", "find", "value"),
        [
            (DocumentCorrectionAction.REPLACE, None, "bar"),
            (DocumentCorrectionAction.REPLACE, "foo", None),
            (DocumentCorrectionAction.REGENERATE, "foo", None),
            (DocumentCorrectionAction.REGENERATE, None, "bar"),
        ],
    )
    def test_check_arguments_invalid_arguments_raises_validation_error(
        self,
        action: DocumentCorrectionAction,
        find: str | None,
        value: str | None,
    ) -> None:
        with pytest.raises(ValidationError):
            DocumentCorrectionRule(
                name="foo-1",
                description="bar",
                document="baz",
                action=action,
                find=find,
                value=value,
            )

    def test_check_arguments_unknown_field_raises_validation_error(self) -> None:
        with pytest.raises(ValidationError):
            DocumentCorrectionRule.model_validate(
                {
                    "name": "foo-1",
                    "description": "bar",
                    "document": "baz",
                    "action": "regenerate",
                    "qux": "quux",
                }
            )


class TestLoad:
    def test_load_valid_file_returns_rules(self, tmp_path: Path) -> None:
        rules_file: Path = tmp_path / "rules.yaml"
        rules_file.write_text(
            "reviewed: [qux]\n"
            "rules:\n"
            "  - name: qux-1\n"
            "    description: quux\n"
            "    document: qux\n"
            "    action: regenerate\n"
        )
        rules: DocumentCorrectionRules = DocumentCorrectionRules.load(rules_file)
        assert rules.reviewed == ["qux"]
        assert rules.rules[0].action == DocumentCorrectionAction.REGENERATE

    def test_load_invalid_rule_raises_validation_error(self, tmp_path: Path) -> None:
        rules_file: Path = tmp_path / "rules.yaml"
        rules_file.write_text(
            "rules:\n"
            "  - name: qux-1\n"
            "    description: quux\n"
            "    document: qux\n"
            "    action: replace\n"
        )
        with pytest.raises(ValidationError):
            DocumentCorrectionRules.load(rules_file)


class TestDump:
    def test_dump_rules_given_round_trips_and_omits_unset_arguments(
        self, tmp_path: Path
    ) -> None:
        rules_file: Path = tmp_path / "foo" / "rules.yaml"
        rules: DocumentCorrectionRules = DocumentCorrectionRules(
            reviewed=["qux", "quux"],
            rules=[
                DocumentCorrectionRule(
                    name="qux-1",
                    description="corge",
                    document="qux",
                    action=DocumentCorrectionAction.REPLACE,
                    find="grault: garply",
                    value="",
                ),
                DocumentCorrectionRule(
                    name="quux-1",
                    description="waldo",
                    document="quux",
                    action=DocumentCorrectionAction.REGENERATE,
                ),
            ],
        )
        rules.dump(rules_file)
        assert DocumentCorrectionRules.load(rules_file) == rules
        assert "find" not in rules_file.read_text().split("quux-1")[1]
