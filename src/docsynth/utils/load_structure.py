import random

from docsynth.types.wrapper import DocsynthAssets


class StructureLoader:
    """Load in relevant structure as prompt

    Args:
        enabled_structures (list): Specified files containing example
            structures to include when building a prompt
        assets (DocsynthAssets): An assets wrapper object extending
            the DocsynthAssets type

    """

    def __init__(self, enabled_structures: list[str], assets: DocsynthAssets):
        self.__assets: DocsynthAssets = assets
        self.enabled_structures: list[str] = enabled_structures
        self.structures: dict[str, str] = {}

    def load_structures(self) -> dict[str, str]:
        """Load structures using asset wrapper

        Returns
            dict: A mapping between structure file names and content

        """
        self.structures = self.__assets.load_structures(self.enabled_structures)
        return self.structures

    def get_random_structure(self) -> tuple[str | None, str | None]:
        """Get a structure at random

        Returns
            tuple: The structure file name and content, or (None, None)
                when no structures are loaded

        """
        if not self.structures:
            return None, None

        filename: str = random.choice(list(self.structures.keys()))
        content: str = self.structures[filename]
        return filename, content

    def format_structure_prompt(self, structure_content: str) -> str:
        """Format an example structure for use in a prompt

        Args:
            structure_content (str): The content of the structure file

        Returns:
            str: The formatted structure

        """
        lines: list[str] = ["## MIMIC THIS DOCUMENT STRUCTURE AND SEMANTIC STYLE ONLY"]
        lines.append("")
        lines.append(
            "Use the example below as a guide for structure and semantic style only: section layout and headings, paragraph/newline patterns, and register/phrasing conventions. Extend it in the same style if you need more space. Do NOT reuse any of its clinical material -- modality, anatomy, findings, measurements, diagnoses, or section labels naming a body part/scan type. All clinical content must come from the profile, style, and content instructions above; substitute the profile's own modality/anatomy into the example's structural pattern."
        )
        lines.append("")
        lines.append("```")
        lines.append(structure_content)
        lines.append("```")
        return "\n".join(lines)

    def get_structure_count(self) -> int:
        """Get the number of loaded structures

        Returns:
            int: Structure number

        """
        return len(self.structures)

    def get_structure_name_without_extension(self, filename: str) -> str:
        """Return a copy of a structure filename without its extension

        Args:
            filename (str): Filename with extension

        Returns:
            str: Structure filename

        """
        return self.__assets.get_structure_name_without_extension(filename)
