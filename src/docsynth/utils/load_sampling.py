import random
from re import fullmatch
from typing import Any

from docsynth.types.wrapper import DocsynthAssets
from docsynth.types.sampling import Content, Style


class ConfigSampler:
    """Probabilistic sampling from config files into prompt

    Args:
        assets (DocsynthAssets): An assets wrapper object extending
            the DocsynthAssets type
        style_filter (str, optional): Regular expression a style
            `<section>.<option>` must fully match to be sampled, applied only
            to the sections it matches. Defaults to None, which samples every
            option
        content_filter (str, optional): Regular expression a content
            `<section>.<option>` must fully match to be sampled, applied only
            to the sections it matches. Defaults to None, which samples every
            option

    Raises:
        ValueError: If a filter matches no option

    """

    def __init__(
        self,
        assets: DocsynthAssets,
        style_filter: str | None = None,
        content_filter: str | None = None,
    ):
        self.style_data: Style = assets.load_style_data()
        self.content_data: Content = assets.load_content_data()
        self.style_filter: str | None = style_filter
        self.content_filter: str | None = content_filter
        ConfigSampler._check_filter(style_filter, self.style_data)
        ConfigSampler._check_filter(content_filter, self.content_data)

    @staticmethod
    def _check_filter(pattern: str | None, data: Style | Content) -> None:
        if pattern is not None and not any(
            fullmatch(pattern, f"{section_name}.{option}")
            for section_name, section in data.root.items()
            for option in section.get_options()
        ):
            raise ValueError(f"No options match filter: {pattern}")

    @staticmethod
    def _filter_section(
        section_name: str, section_data: dict[str, Any], pattern: str | None
    ) -> dict[str, Any]:
        matched: dict[str, Any] = {
            key: value
            for key, value in section_data.items()
            if key != "mutually_exclusive"
            and pattern is not None
            and fullmatch(pattern, f"{section_name}.{key}")
        }
        if not matched:
            return section_data
        return {"mutually_exclusive": section_data["mutually_exclusive"], **matched}

    def _sample_section(
        self, section_data: dict[str, Any], option: str | None = None
    ) -> list[dict[str, str]]:
        mutually_exclusive: bool = section_data.get("mutually_exclusive", False)
        items: dict[str, Any] = {
            k: v
            for k, v in section_data.items()
            if not k.startswith("mutually_exclusive")
        }
        selected: list[dict[str, str]] = []

        if option is not None:
            if option not in items:
                raise ValueError(f"Unknown option: {option}")
            if items[option]["description"]:
                selected.append(
                    {"key": option, "description": items[option]["description"]}
                )
            return selected

        description: str = ""
        if mutually_exclusive:
            choices: list[str] = list(items.keys())
            weights: list[float] = [items[c]["probability"] for c in choices]
            chosen: str = random.choices(choices, weights=weights, k=1)[0]
            description = items[chosen]["description"]
            if description:
                selected.append({"key": chosen, "description": description})
        else:
            key: str
            config: dict[str, Any]
            for key, config in items.items():
                probability: float = config["probability"]
                if random.random() < probability:
                    description = config["description"]
                    if description:
                        selected.append({"key": key, "description": description})

        return selected

    def sample_style_config(
        self, requirements: dict[str, str]
    ) -> dict[str, list[dict[str, str]]]:
        """Process the style config from the asset wrapper object
            to make probabilistic selections of items for the prompt

        Args:
            requirements (dict): Options to select, by section, in place of
                sampling those sections

        Returns:
            dict: Each section of the style asset, matched to probabilistic
                selections (key and description) within that asset

        """
        result: dict[str, list[dict[str, str]]] = {}
        section_name: str
        section_data: dict[str, Any]
        for section_name, section_data in self.style_data.model_dump().items():
            result[section_name] = self._sample_section(
                ConfigSampler._filter_section(
                    section_name, section_data, self.style_filter
                ),
                requirements.get(section_name),
            )
        return result

    def sample_content_config(
        self, requirements: dict[str, str]
    ) -> dict[str, list[dict[str, str]]]:
        """Process the content config from the asset wrapper object
            to make probabilistic selections of items for the prompt

        Args:
            requirements (dict): Options to select, by section, in place of
                sampling those sections

        Returns:
            dict: Each section of the content asset, matched to probabilistic
                selections (key and description) within that asset

        """
        result: dict[str, list[dict[str, str]]] = {}
        section_name: str
        section_data: dict[str, Any]
        for section_name, section_data in self.content_data.model_dump().items():
            result[section_name] = self._sample_section(
                ConfigSampler._filter_section(
                    section_name, section_data, self.content_filter
                ),
                requirements.get(section_name),
            )
        return result

    def format_style_prompt(
        self, sampled_style: dict[str, list[dict[str, str]]]
    ) -> str:
        """Transform probabilistically selected styles into a prompt

        Returns:
            str: The formatted prompt

        """
        lines: list[str] = ["## FOLLOW THESE STYLE REQUIREMENTS"]
        lines.append("")

        section_name: str
        items: list[dict[str, str]]
        for section_name, items in sampled_style.items():
            if items:
                section_title = section_name.replace("_", " ").title()
                lines.append(f"**{section_title}:**")
                for item in items:
                    lines.append(f"- {item['description']}")
                lines.append("")

        return "\n".join(lines).strip()

    def format_content_prompt(
        self, sampled_content: dict[str, list[dict[str, str]]]
    ) -> str:
        """Transform probabilistically selected content into a prompt

        Returns:
            str: The formatted prompt

        """
        lines: list[str] = ["## FOLLOW THESE CONTENT REQUIREMENTS"]
        lines.append("")

        section_name: str
        items: list[dict[str, str]]
        for section_name, items in sampled_content.items():
            if items:
                section_title = section_name.replace("_", " ").title()
                lines.append(f"**{section_title}:**")
                for item in items:
                    lines.append(f"- {item['description']}")
                lines.append("")

        return "\n".join(lines).strip()

    def generate_prompts(
        self, requirements: dict[str, str] | None = None
    ) -> tuple[str, str]:
        """Generated both prompts (style and content) with probabilistic components

        Args:
            requirements (dict, optional): Options to select, by style or
                content section, in place of sampling those sections.
                Defaults to None, which samples every section

        Returns:
            tuple: The style and content prompts

        Raises:
            ValueError: If a requirement names an unknown section or option

        """
        requirements = requirements or {}
        unknown_sections: set[str] = (
            set(requirements) - set(self.style_data.root) - set(self.content_data.root)
        )
        if unknown_sections:
            raise ValueError(f"Unknown sections: {sorted(unknown_sections)}")
        style_config: dict[str, list[dict[str, str]]] = self.sample_style_config(
            requirements
        )
        content_config: dict[str, list[dict[str, str]]] = self.sample_content_config(
            requirements
        )
        style_prompt: str = self.format_style_prompt(style_config)
        content_prompt: str = self.format_content_prompt(content_config)
        return style_prompt, content_prompt
