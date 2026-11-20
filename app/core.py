"""Domain logic and training data for sports commentary."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable


@dataclass
class MatchEvent:
    sport: str = "Soccer"
    event_type: str = "Goal"
    team: str = "Home United"
    player: str = "Alex Morgan"
    opponent: str = "City Rovers"
    score: str = "2-1"
    time: str = "87th minute"
    context: str = "A tense late-game finish"

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


def validate_event(event: MatchEvent) -> MatchEvent:
    """Validate and normalize a match event, raising ValueError when invalid."""
    for field in ("sport", "event_type", "team", "player", "opponent", "score", "time"):
        value = getattr(event, field)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{field.replace('_', ' ').title()} cannot be empty")
        if len(value) > 100:
            raise ValueError(f"{field.replace('_', ' ').title()} must be 100 characters or fewer")
        setattr(event, field, value.strip())
    if not isinstance(event.context, str):
        raise ValueError("Context must be text")
    event.context = event.context.strip()[:500]
    return event


def format_prompt(event: MatchEvent) -> str:
    event = validate_event(event)
    details = f"{event.time}: {event.player} of {event.team} — {event.event_type.lower()}"
    details += f" against {event.opponent}. Score {event.score}."
    if event.context:
        details += f" {event.context}."
    return f"[{event.sport}] {details}\nCommentary:"


def seed_examples() -> list[tuple[str, str]]:
    """Compact, varied instruction/target pairs for the local adapter drill."""
    rows = [
        ("Soccer|Goal|87th minute|Alex|Home|2-1|late pressure", "GOAL! Alex finds the net in the 87th minute! Home United seize a 2-1 lead as the crowd erupts!"),
        ("Soccer|Goal|12th minute|Mia|Rovers|1-0|quick counter", "A lightning counter and Mia makes no mistake! Rovers strike early to move in front, 1-0!"),
        ("Basketball|Buzzer-beater|0:02|Jordan|Falcons|98-97|playoff game", "Jordan rises at the horn—IT'S GOOD! The Falcons steal it, 98-97, in a finish nobody will forget!"),
        ("Basketball|Score change|4th quarter|Lee|Wolves|101-99|one possession game", "Lee buries the jumper! The Wolves edge ahead 101-99 with the clock becoming the opponent."),
        ("Soccer|Red card|66th minute|Diaz|Athletic|0-0|heated challenge", "The referee reaches for red! Diaz is sent off, and Athletic must see out this tense 0-0 with ten players."),
        ("Soccer|Foul|34th minute|Chen|United|1-1|dangerous area", "A mistimed challenge from Chen gives United a dangerous free kick. At 1-1, this could change everything."),
        ("Hockey|Goal|final minute|Patel|Blizzards|3-2|empty net", "Patel shoots—and scores! The Blizzards make it 3-2 in the final minute, sending the bench into a frenzy!"),
        ("American Football|Touchdown|fourth quarter|Carter|Sharks|24-21|red zone", "Carter powers across the line! Touchdown Sharks! They turn red-zone pressure into a 24-21 lead."),
        ("Soccer|Penalty saved|90th minute|Silva|City|1-1|last chance", "SAVED! Silva guesses right and keeps it out! City are still level at 1-1 in the 90th minute."),
        ("Tennis|Match point|set point|Williams|Garcia|6-4 6-5|championship", "Williams steps in on match point and seals it! A champion's finish after a gripping contest."),
        ("Basketball|Foul|3rd quarter|Brown|Comets|72-72|fast break", "Brown is whistled on the break. The Comets will head to the line with this one locked at 72 apiece."),
        ("Hockey|Power play|second period|Olsen|Wolves|1-1|man advantage", "Olsen lets it fly on the power play—goal! The Wolves break the tie at 1-1 with a crisp finish."),
    ]
    return rows


def build_training_text(examples: Iterable[tuple[str, str]] | None = None) -> str:
    pairs = list(examples if examples is not None else seed_examples())
    if not pairs:
        raise ValueError("At least one training example is required")
    return "\n".join(f"Event: {prompt}\nCommentary: {text}\n" for prompt, text in pairs)

