"""Standalone WhatsApp hello-world sender.

Usage:
    python scripts/send_whatsapp_hello.py

Reads WHATSAPP_PHONE_NUMBER_ID, WHATSAPP_ACCESS_TOKEN, and WHATSAPP_ADMIN_WA_ID
from the .env file at the project root and sends a test message.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path


def _load_env(path: Path) -> dict[str, str]:
    env: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        env[k.strip()] = v.strip()
    return env


def main() -> None:
    project_root = Path(__file__).resolve().parent.parent
    env_file = project_root / ".env"

    if not env_file.exists():
        print("ERROR: .env file not found at", env_file)
        print("       Copy .env.example to .env and fill in your credentials.")
        sys.exit(1)

    env = _load_env(env_file)

    phone_number_id = env.get("WHATSAPP_PHONE_NUMBER_ID", "").strip()
    access_token = env.get("WHATSAPP_ACCESS_TOKEN", "").strip()
    admin_wa_id = env.get("WHATSAPP_ADMIN_WA_ID", "").strip()

    missing = [name for name, val in [
        ("WHATSAPP_PHONE_NUMBER_ID", phone_number_id),
        ("WHATSAPP_ACCESS_TOKEN",    access_token),
        ("WHATSAPP_ADMIN_WA_ID",     admin_wa_id),
    ] if not val]

    if missing:
        print("ERROR: The following variables are missing or empty in .env:")
        for name in missing:
            print(f"  {name}=")
        print("\nFix: open .env and set these values from the Meta Developer Console:")
        print("  - WHATSAPP_PHONE_NUMBER_ID  → WhatsApp > API Setup > Phone number ID")
        print("  - WHATSAPP_ACCESS_TOKEN     → WhatsApp > API Setup > Temporary/permanent token")
        print("  - WHATSAPP_ADMIN_WA_ID      → Your personal WhatsApp number (e.g. 23052533081)")
        sys.exit(1)

    print(f"Sending to:  {admin_wa_id}")
    print(f"Phone ID:    {phone_number_id}")
    print(f"Token:       {access_token[:20]}...")
    print()

    try:
        import httpx
    except ImportError:
        print("ERROR: httpx is not installed. Run: pip install httpx")
        sys.exit(1)

    url = f"https://graph.facebook.com/v21.0/{phone_number_id}/messages"
    headers = {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json",
    }
    body = {
        "messaging_product": "whatsapp",
        "to": admin_wa_id,
        "type": "text",
        "text": {"preview_url": False, "body": "Hello World 👋 — this is a test from your chatbot admin notifier."},
    }

    try:
        with httpx.Client(timeout=15.0) as client:
            r = client.post(url, headers=headers, content=json.dumps(body))
    except httpx.ConnectError:
        print("ERROR: Could not connect to graph.facebook.com")
        print("       Check your internet connection.")
        sys.exit(1)
    except httpx.TimeoutException:
        print("ERROR: Request timed out after 15 seconds.")
        sys.exit(1)

    if r.status_code == 200:
        msg_id = r.json().get("messages", [{}])[0].get("id", "?")
        print(f"SUCCESS — API accepted the message (id={msg_id})")
        print()
        print("NOTE: If you did NOT receive it on WhatsApp, this is a Meta sandbox limitation.")
        print("  The test phone number (+1 555 046 8875) can only deliver to numbers that are")
        print("  registered as test recipients in the Meta Developer Console.")
        print()
        print("  FIX (pick one):")
        print("  A) Add your number as a recipient:")
        print("     Meta Developer Console → WhatsApp → API Setup → 'To' field → add +23052533081")
        print("     Then re-run this script.")
        print()
        print("  B) Send a message first (open the conversation window):")
        print(f"     From {admin_wa_id}, send any WhatsApp message to the test business number.")
        print("     Then re-run this script.")
        print()
        print("  C) Use a real WhatsApp Business number (not the free test number) —")
        print("     this restriction only applies to the Meta sandbox.")
    else:
        try:
            error = r.json().get("error", {})
            code = error.get("code")
            msg  = error.get("message", r.text)
            sub  = error.get("error_subcode")
        except Exception:
            code, msg, sub = None, r.text, None

        print(f"ERROR {r.status_code}: {msg}")

        hints = {
            190: "Token expired or invalid. Go to Meta Developer Console → WhatsApp → API Setup and generate a new access token, then update WHATSAPP_ACCESS_TOKEN in .env",
            10:  "App not approved or permission missing. Ensure 'whatsapp_business_messaging' permission is granted in Meta App Review.",
            100: "Invalid parameter. Check WHATSAPP_PHONE_NUMBER_ID — it should be the numeric ID from Meta API Setup, not your phone number.",
            131030: "Recipient phone number not in allowed list. In development mode, add the number via Meta Developer Console → WhatsApp → API Setup → Phone numbers.",
            131047: "Re-engagement window expired. The customer must message you first within 24 hours (for templates use a pre-approved template message).",
        }
        hint = hints.get(code) or (hints.get(sub) if sub else None)
        if hint:
            print(f"\nFIX: {hint}")
        else:
            print(f"\nFull response: {r.text}")
        sys.exit(1)


if __name__ == "__main__":
    main()
