"""The parts of the threshold tuning that are logic rather than measurement.

Running the engines over an hour of audio is the experiment, and it cannot be
tested in CI. Deciding what the numbers mean can be, and it is the part that
would otherwise be wrong without anyone noticing: a rule that quietly picks the
wrong row turns a measurement into a worse guess than not measuring at all.
"""

from __future__ import annotations

from scripts.tune_wake import Result, pick, table

HOUR = 3600.0


def result(
    threshold: float, woke: int, detected: int, clips: int = 20, answered: int = 0
) -> Result:
    return Result(
        threshold=threshold,
        woke=woke,
        answered=answered,
        detected=detected,
        clips=clips,
        noise_seconds=HOUR,
    )


def test_rates_are_per_hour_whatever_was_recorded() -> None:
    half = Result(threshold=0.5, woke=3, clips=0, noise_seconds=HOUR / 2)
    assert half.woke_per_hour == 6.0


def test_no_noise_recorded_is_zero_rather_than_a_division_error() -> None:
    assert Result(threshold=0.5).woke_per_hour == 0.0
    assert Result(threshold=0.5).detection_rate == 0.0


def test_it_picks_the_lowest_threshold_within_the_budget() -> None:
    # Missing you is the failure you notice every time, so among the rows that
    # are quiet enough, the most sensitive one wins.
    results = [
        result(0.4, woke=9, detected=20),
        result(0.6, woke=1, detected=20),
        result(0.7, woke=0, detected=20),
        result(0.9, woke=0, detected=11),
    ]
    assert pick(results, budget=1.0).threshold == 0.6  # type: ignore[union-attr]


def test_a_quieter_row_loses_if_it_hears_you_less() -> None:
    results = [
        result(0.6, woke=1, detected=20),
        result(0.8, woke=0, detected=14),
    ]
    assert pick(results, budget=1.0).threshold == 0.6  # type: ignore[union-attr]


def test_nothing_within_budget_is_said_rather_than_guessed() -> None:
    results = [result(0.6, woke=20, detected=20), result(0.9, woke=5, detected=9)]
    assert pick(results, budget=1.0) is None


def test_the_table_is_markdown_ready_to_paste_into_the_issue() -> None:
    printed = table([result(0.6, woke=3, answered=1, detected=19)])
    assert printed.splitlines()[0].startswith("| threshold |")
    assert "| 0.60 | 3.0 | 1.0 | 19/20 |" in printed


def test_a_run_with_no_spoken_clips_does_not_invent_a_rate() -> None:
    printed = table([Result(threshold=0.6, woke=2, noise_seconds=HOUR)])
    assert "| - |" in printed


def test_clips_that_never_fire_are_not_a_threshold_problem() -> None:
    # Recommending the lowest row here would read as "the best we can do",
    # when the truth is the clips do not contain the word.
    results = [result(0.3, woke=0, detected=0), result(0.9, woke=0, detected=0)]
    assert pick(results, budget=1.0) is None
