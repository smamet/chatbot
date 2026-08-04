from __future__ import annotations

from evenor.adapters.channels.text_format import (
    format_for_instagram,
    format_for_messenger,
    format_for_whatsapp,
)


def test_format_for_messenger_strips_markdown_and_preserves_bullets() -> None:
    src = "# Title\n- **Room Diffuser**\n- _Mist_"
    assert format_for_messenger(src) == "Title\n• Room Diffuser\n• Mist"


def test_format_for_instagram_matches_messenger_cleanup() -> None:
    src = "## Products\n* `Room`\n* ~Mist~"
    assert format_for_instagram(src) == "Products\n• Room\n• Mist"


def test_format_for_whatsapp_keeps_valid_emphasis() -> None:
    src = "*Room Diffuser* and _Mist_"
    assert format_for_whatsapp(src) == "*Room Diffuser* and _Mist_"


def test_format_for_whatsapp_cleans_malformed_stars() -> None:
    src = "* *Room Diffuser\n\n\n* *Mist"
    assert format_for_whatsapp(src) == "* Room Diffuser\n* Mist"
