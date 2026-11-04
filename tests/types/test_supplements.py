from pathlib import Path

import pytest
from pydantic import ValidationError

from docsynth.types.supplements import Supplement, Supplements


class TestSupplement:
    def test_supplement_extra_field_raises(self) -> None:
        with pytest.raises(ValidationError):
            Supplement.model_validate(
                {
                    "name": "foo",
                    "description": "bar",
                    "count": 1,
                    "requirements": {"baz": "qux"},
                    "quux": "corge",
                }
            )


class TestLoad:
    def test_load_supplement_file_returns_supplements(self, tmp_path: Path) -> None:
        (tmp_path / "supplement.yaml").write_text(
            "supplements:\n"
            "  - name: foo\n"
            "    description: bar\n"
            "    count: 2\n"
            "    requirements:\n"
            "      baz: qux\n"
        )
        assert Supplements.load(tmp_path / "supplement.yaml") == Supplements(
            supplements=[
                Supplement(
                    name="foo", description="bar", count=2, requirements={"baz": "qux"}
                )
            ]
        )
