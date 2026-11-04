from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict


class Supplement(BaseModel):
    """Extra documents to generate, with some requirements pinned.

    Attributes:
        name (str): Unique supplement name
        description (str): The shortfall in the corpus the supplement balances
        count (int): Number of documents to generate
        requirements (dict[str, str]): Option to select, by style or content
            section

    """

    model_config = ConfigDict(extra="forbid")
    name: str
    description: str
    count: int
    requirements: dict[str, str]


class Supplements(BaseModel):
    """The supplements for a folder of documents.

    Attributes:
        supplements (list[Supplement]): The supplements

    """

    model_config = ConfigDict(extra="forbid")
    supplements: list[Supplement]

    @staticmethod
    def load(supplement_file: Path) -> "Supplements":
        """Load and validate supplements from a YAML file.

        Args:
            supplement_file (Path): The supplement file

        Returns:
            Supplements: The validated supplements

        """
        return Supplements.model_validate(yaml.safe_load(supplement_file.read_text()))
