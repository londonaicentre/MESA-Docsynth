from docsynth.types.sampling import SamplingAttribute, SamplingElement


class TestSamplingElement:
    def test_options_section_returns_options_without_mutually_exclusive(
        self,
    ) -> None:
        assert SamplingElement.model_validate(
            {
                "mutually_exclusive": True,
                "foo": {"probability": 0.5, "description": "bar"},
            }
        ).options == {"foo": SamplingAttribute(probability=0.5, description="bar")}
