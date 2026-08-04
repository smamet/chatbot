from __future__ import annotations

from evenor.application.progress_log import ProgressLog


def test_progress_log_collects_and_emits() -> None:
    emitted: list[str] = []
    log = ProgressLog(emit=emitted.append)
    log.step("First")
    log.step("Second")
    assert log.messages == ["First", "Second"]
    assert emitted == ["First", "Second"]


def test_progress_log_without_emit() -> None:
    log = ProgressLog()
    log.step("Only stored")
    assert log.messages == ["Only stored"]
