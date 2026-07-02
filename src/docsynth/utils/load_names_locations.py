import random

from docsynth.types.names_locations import NamesLocations


class NamesLocationsLoader:
    """Random sampling of patient/clinician names and care locations for
    prompt injection

    """

    def __init__(self) -> None:
        self.__names_locations: NamesLocations = NamesLocations()

    def sample(self) -> dict[str, str]:
        """Sample one random name/location per category

        Returns:
            dict: Sampled patient name, clinician name, provider, and
                ward/clinic

        """
        return {
            "patient_name": random.choice(self.__names_locations.patient_names),
            "clinician_name": random.choice(self.__names_locations.clinician_names),
            "provider": random.choice(self.__names_locations.providers),
            "ward_clinic": random.choice(self.__names_locations.wards_clinics),
        }

    def format_prompt(self, sampled: dict[str, str]) -> str:
        """Format sampled names/locations into prompt text

        Args:
            sampled (dict): Sampled names/locations, as returned by
                `sample`

        Returns:
            str: The formatted prompt block

        """
        lines: list[str] = ["## USE THESE NAMES AND LOCATIONS (BUT REDACT AS PROMPTED)"]
        lines.append("")
        lines.append(f"**Patient Name:** {sampled['patient_name']}")
        lines.append(f"**Clinician Name:** {sampled['clinician_name']}")
        lines.append(f"**Hospital/Practice:** {sampled['provider']}")
        lines.append(f"**Ward/Clinic:** {sampled['ward_clinic']}")
        lines.append("")
        return "\n".join(lines)
