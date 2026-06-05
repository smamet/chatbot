from __future__ import annotations

import logging

from chatbot.mail.listener import ImapMailListener

logging.basicConfig(level=logging.INFO)


def main() -> None:
    ImapMailListener().run_forever()


if __name__ == "__main__":
    main()
