import random
import re
from pathlib import Path

import yaml

"""
load_profiles.py - loads in cancer & molecular profiles
"""


class ProfileLoader:
    def __init__(self, domain):
        """
        Initialise with specific domain
        """
        if not domain:
            raise ValueError("Domain must be specified for ProfileLoader")

        base_dir = Path(__file__).parent.parent
        self.profiles_base_dir = base_dir / "config" / "profiles"
        self.domain = domain
        self.profiles_dir = self.profiles_base_dir / domain
        self.all_profiles = []
        self.profile_files = []

    def load_all_profiles(self):
        self.profile_files = sorted(self.profiles_dir.glob("*.yml"))
        self.all_profiles = []

        for profile_file in self.profile_files:
            profiles = self._load_profiles_from_file(profile_file)
            self.all_profiles.extend(profiles)

        return self.all_profiles

    def load_profiles_from_files(self, filenames):
        self.all_profiles = []
        for filename in filenames:
            file_path = self.profiles_dir / filename
            if not file_path.exists():
                raise FileNotFoundError(f"Profile file not found: {file_path}")
            profiles = self._load_profiles_from_file(file_path)
            self.all_profiles.extend(profiles)
        return self.all_profiles

    def _load_profiles_from_file(self, file_path):
        """
        Load profiles from YAML file taking all fields
        """
        with open(file_path, "r") as f:
            data = yaml.safe_load(f)

        profiles = []

        for profile_id, profile_data in data.items():
            # extract profile name from profile_id (e.g., "haem_001" -> "haem")
            match = re.match(r'^(.+)_\d+$', profile_id)
            profile_name = match.group(1) if match else profile_id

            profile = {
                "profile_id": profile_id,
                "source_file": file_path.name,
            }

            profile["profile_name"] = profile_name

            if isinstance(profile_data, dict):
                profile.update(profile_data)

            profiles.append(profile)

        return profiles

    def get_random_profile(self):
        if not self.all_profiles:
            raise ValueError("No profiles loaded.")
        return random.choice(self.all_profiles)

    def get_sequential_profiles(self):
        if not self.all_profiles:
            raise ValueError("No profiles loaded.")
        for profile in self.all_profiles:
            yield profile

    def format_profile_prompt(self, profile):
        """
        Format profile into prompt text
        """
        lines = ["## USE THIS PROFILE"]
        lines.append("")

        metadata_keys = {'profile_id', 'source_file'}
        for key, value in profile.items():
            if key not in metadata_keys:
                readable_key = key.replace('_', ' ').title()
                lines.append(f"**{readable_key}:** {value}")

        lines.append("")
        return "\n".join(lines)

    def get_profile_count(self):
        return len(self.all_profiles)

    def filter_existing_profiles(self, existing_profile_ids):
        """
        Remove profiles that already exist from the loaded profiles list
        """
        original_count = len(self.all_profiles)
        self.all_profiles = [
            profile for profile in self.all_profiles
            if profile["profile_id"] not in existing_profile_ids
        ]
        filtered_count = original_count - len(self.all_profiles)
        return filtered_count
