from collections.abc import Callable
from dataclasses import dataclass
from importlib.resources.abc import Traversable
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from pytest_mock import MockerFixture

from docsynth.types.profile import Profile
from docsynth.types.sampling import Content, Style
from docsynth.types.wrapper import DocsynthAssets
from docsynth.assets.genomic.wrapper import GenomicAssets, GenomicProfile


class GenomicAssetsFixture(GenomicAssets):
    def load_profiles_from_file(self, file_path: Traversable) -> list[Profile]:
        return self._load_profiles_from_file(file_path)


@dataclass
class ProfilesMocks:
    Profiles: MagicMock


@pytest.fixture
def profiles_mocks(mocker: MockerFixture) -> ProfilesMocks:
    return ProfilesMocks(mocker.patch("docsynth.assets.genomic.wrapper.Profiles"))


@pytest.fixture
def genomic_profile() -> GenomicProfile:
    return GenomicProfile(
        patient_description="foo",
        test_methodology="bar",
        genetic_finding="baz",
    )


class TestGenomicAssets:
    def test_load_profiles_from_file_genomic_yaml_sets_id_condition_and_source(
        self, profiles_mocks: ProfilesMocks, genomic_profile: GenomicProfile
    ) -> None:
        profiles_mocks.Profiles.__getitem__.return_value.return_value.items = {
            "foo_1": genomic_profile
        }
        file_path: Path = Path("foo_bar.yml")
        profiles: list[Profile] = GenomicAssetsFixture().load_profiles_from_file(
            file_path
        )
        profiles_mocks.Profiles.__getitem__.assert_called_once_with(GenomicProfile)
        profiles_mocks.Profiles.__getitem__.return_value.assert_called_once_with(
            file_path
        )
        assert profiles == [
            genomic_profile.model_copy(
                update={
                    "profile_id": "foo_1",
                    "condition": "foo_bar",
                    "source_file": "foo_bar.yml",
                }
            )
        ]

    def test_format_profile_prompt_genomic_profile_includes_fields(
        self, genomic_profile: GenomicProfile
    ) -> None:
        assert GenomicAssets().format_profile_prompt(genomic_profile) == (
            "## USE THIS GENETIC TEST PROFILE\n\n"
            "**Patient History & Context:** foo\n\n"
            "**Genetic test methodology:** bar\n\n"
            "**Test results and clinical interpretation:** baz"
        )

    def test_format_profile_prompt_other_profile_raises_assertion_error(
        self,
    ) -> None:
        with pytest.raises(AssertionError):
            GenomicAssets().format_profile_prompt(Profile())


class TestGenomicAssetFiles:
    @pytest.fixture(scope="class")
    def genomic_assets(self) -> GenomicAssets:
        return GenomicAssets()

    @pytest.fixture(scope="class")
    def genomic_profiles(self, genomic_assets: GenomicAssets) -> list[Profile]:
        return genomic_assets.load_all_profiles()

    def test_load_all_profiles_packaged_files_loads_complete_profiles(
        self, genomic_profiles: list[Profile]
    ) -> None:
        assert len(genomic_profiles) == 500
        assert all(
            isinstance(profile, GenomicProfile)
            and profile.patient_description
            and profile.test_methodology
            and profile.genetic_finding
            for profile in genomic_profiles
        )

    def test_load_all_profiles_packaged_files_ids_unique_without_hyphens(
        self, genomic_profiles: list[Profile]
    ) -> None:
        ids: list[str] = [str(profile.profile_id) for profile in genomic_profiles]
        assert len(ids) == len(set(ids))
        assert not [profile_id for profile_id in ids if "-" in profile_id]

    def test_load_profiles_from_files_turner_syndrome_loads_its_profiles(
        self, genomic_assets: GenomicAssets
    ) -> None:
        profiles: list[Profile] = genomic_assets.load_profiles_from_files(
            ["turner_syndrome.yml"]
        )
        assert len(profiles) == 10
        assert all(
            isinstance(profile, GenomicProfile)
            and profile.condition == "turner_syndrome"
            for profile in profiles
        )

    @pytest.mark.parametrize(
        "loader", [GenomicAssets.load_style_data, GenomicAssets.load_content_data]
    )
    def test_load_sampling_data_packaged_file_loads_valid_weights(
        self,
        genomic_assets: GenomicAssets,
        loader: Callable[[GenomicAssets], Style | Content],
    ) -> None:
        for name, section in loader(genomic_assets).root.items():
            weights: list[float] = [
                option.probability for option in section.get_options().values()
            ]
            assert weights, name
            assert all(0 <= weight <= 1 for weight in weights), name

    @pytest.mark.parametrize(
        "filename",
        [
            "DNA_1.txt",
            "DNA_2.txt",
            "DNA_3.txt",
            "DNA_4.txt",
            "FISH_1.txt",
            "FISH_2.txt",
            "FISH_3.txt",
            "MLPA.txt",
            "PCR_1.txt",
            "PCR_2.txt",
        ],
    )
    def test_load_structures_packaged_file_one_document_without_hyphen(
        self, genomic_assets: GenomicAssets, filename: str
    ) -> None:
        assert (
            len(
                genomic_assets.load_structures([filename])[filename]
                .strip()
                .splitlines()
            )
            == 1
        )
        assert "-" not in genomic_assets.get_structure_name_without_extension(filename)

    def test_from_domain_genomic_returns_genomic_assets(self) -> None:
        assets: DocsynthAssets = DocsynthAssets.from_domain("genomic")
        assert isinstance(assets, GenomicAssets)
        assert assets.get_domain() == "genomic"
