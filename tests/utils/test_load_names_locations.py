import pytest

from docsynth.utils.load_names_locations import NamesLocationsLoader


@pytest.fixture(scope="session")
def names_locations_loader() -> NamesLocationsLoader:
    return NamesLocationsLoader()


def test_sample_returns_one_value_per_category(
    names_locations_loader: NamesLocationsLoader,
) -> None:
    sampled: dict[str, str] = names_locations_loader.sample()
    assert set(sampled.keys()) == {
        "patient_name",
        "clinician_name",
        "provider",
        "ward_clinic",
    }
    assert all(isinstance(value, str) and value for value in sampled.values())


def test_format_prompt_contains_all_sampled_values(
    names_locations_loader: NamesLocationsLoader,
) -> None:
    sampled: dict[str, str] = names_locations_loader.sample()
    formatted: str = names_locations_loader.format_prompt(sampled)
    assert "## USE THESE NAMES AND LOCATIONS (BUT REDACT AS PROMPTED)" in formatted
    assert sampled["patient_name"] in formatted
    assert sampled["clinician_name"] in formatted
    assert sampled["provider"] in formatted
    assert sampled["ward_clinic"] in formatted
