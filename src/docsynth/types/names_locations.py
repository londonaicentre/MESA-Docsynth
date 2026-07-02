from importlib.resources import files

from pydantic_settings import BaseSettings
import yaml


class NamesLocations(BaseSettings):
    patient_names: list[str]
    clinician_names: list[str]
    providers: list[str]
    wards_clinics: list[str]

    def __init__(self) -> None:
        super().__init__(
            **yaml.safe_load(
                files("docsynth").joinpath("config/names_locations.yml").read_text()
            )
        )
