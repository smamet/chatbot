from __future__ import annotations

from evenor.application.email_subject import normalize_subject, subject_similarity


def test_normalize_subject_strips_re_fwd_loop() -> None:
    assert normalize_subject("Re: Fwd: Re: Devis machine") == "devis machine"


def test_normalize_subject_strips_brackets_and_external() -> None:
    assert normalize_subject("[External] [Ticket #12] Question") == "question"


def test_subject_similarity_exact_and_close() -> None:
    assert subject_similarity("Devis pompe", "devis pompe") == 1.0
    assert subject_similarity("Devis pompe A", "Devis pompe B") >= 0.85
