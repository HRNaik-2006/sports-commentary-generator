import pytest

from app.core import MatchEvent, build_training_text, format_prompt, seed_examples, validate_event


def test_format_prompt_includes_event_context():
    prompt = format_prompt(MatchEvent(event_type="Buzzer-beater", score="98-97", context="Playoff game"))
    assert "[Soccer]" in prompt
    assert "buzzer-beater" in prompt
    assert "98-97" in prompt
    assert "Playoff game" in prompt


def test_event_validation_rejects_empty_required_field():
    with pytest.raises(ValueError, match="Team cannot be empty"):
        validate_event(MatchEvent(team=" "))


def test_examples_are_nonempty_and_formatted():
    examples = seed_examples()
    assert len(examples) >= 10
    assert all(prompt and commentary.endswith(("!", ".")) for prompt, commentary in examples)
    assert "Event:" in build_training_text(examples)

