import random
from pathlib import Path
import yaml

"""
load_names_locations.py - simple random sampling of names and locations
"""


class NamesLocationsLoader:
    def __init__(self):
        """
        Load names and locations from config file
        """
        base_dir = Path(__file__).parent.parent
        config_path = base_dir / "config" / "names_locations.yml"

        with open(config_path, "r") as f:
            data = yaml.safe_load(f)

        self.patient_names = data["patient_names"]
        self.clinician_names = data["clinician_names"]
        self.providers = data["providers"]
        self.wards_clinics = data["wards_clinics"]

    def sample(self):
        """
        Randomly sample one from each list and return as dict
        """
        return {
            "patient_name": random.choice(self.patient_names),
            "clinician_name": random.choice(self.clinician_names),
            "provider": random.choice(self.providers),
            "ward_clinic": random.choice(self.wards_clinics),
        }

    def format_prompt(self, sampled):
        """
        Format sampled names/locations into prompt text
        """
        lines = ["## USE THESE NAMES AND LOCATIONS (BUT REDACT AS PROMPTED)"]
        lines.append("")
        lines.append(f"**Patient Name:** {sampled['patient_name']}")
        lines.append(f"**Clinician Name:** {sampled['clinician_name']}")
        lines.append(f"**Hospital/Practice:** {sampled['provider']}")
        lines.append(f"**Ward/Clinic:** {sampled['ward_clinic']}")
        lines.append("")
        return "\n".join(lines)
