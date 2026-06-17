import pytest

pytest.importorskip("crewai")

from tokencircuit import InterventionConfig, instrument_crewai


class TestCrewAIInstrumentation:
    def test_instrument_crewai_with_mock_crew(self):
        crewai = pytest.importorskip("crewai")

        config = InterventionConfig(
            nudge_threshold=1, override_threshold=2, hard_stop_threshold=3,
        )

        crew = crewai.Crew(
            agents=[],
            tasks=[],
        )

        result = instrument_crewai(crew, config=config)
        assert result is crew

    def test_instrument_crewai_with_before_llm_call(self):
        crewai = pytest.importorskip("crewai")

        class HookCrew(crewai.Crew):
            def __init__(self, **kwargs):
                super().__init__(**kwargs)
                self._hooks = []

            def before_llm_call(self, hook):
                self._hooks.append(hook)

        crew = HookCrew(agents=[], tasks=[])
        config = InterventionConfig()

        result = instrument_crewai(crew, config=config)
        assert result is crew
        assert len(crew._hooks) == 1
