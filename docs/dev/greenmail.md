# Email dev — GreenMail + Mailpit

Local end-to-end email bot testing. **Not** deployed in production.

| Service | Role |
|---------|------|
| **GreenMail** | IMAP inbox (worker reads mail) + SMTP inject (simulate inbound client mail) |
| **Mailpit** | Outbound SMTP (approved replies) + web UI to read sent mail |

## Start stack

```bash
./sail up -d --profile dev
```

Starts `greenmail`, `mailpit`, `worker-mail`, `api`, and other services. Both mail services use the Compose `dev` profile.

GreenMail must bind to `0.0.0.0` inside Docker so containers reach it via hostname `greenmail`. Mailpit is reached as `mailpit`.

Ports (localhost only):

| Service | Host URL / port |
|---------|-----------------|
| Mailpit Web UI | http://127.0.0.1:8025 |
| Mailpit SMTP | `127.0.0.1:1025` |
| GreenMail API | http://127.0.0.1:8081 |
| GreenMail SMTP (inject) | `127.0.0.1:3025` |
| GreenMail IMAP (plain) | `127.0.0.1:3143` |
| GreenMail IMAPS | `127.0.0.1:3993` |

Test account (GreenMail IMAP): `bot@test.local` / `secret`

Set in `.env` for dashboard **Test email** tab:

```env
DEV_MODE=true
```

Optional overrides:

```env
DEV_MAIL_INJECT_SMTP_HOST=greenmail
DEV_MAIL_INJECT_SMTP_PORT=3025
DEV_MAILPIT_WEB_URL=http://127.0.0.1:8025
```

## Connector configuration

Configure the bot in **Dashboard → Connectors** (use **Edit** on a row, or autofill buttons in dev):

### Inbound (IMAP) — GreenMail

| Field | Value (Docker network) |
|-------|------------------------|
| IMAP host | `greenmail` |
| IMAP port | `3143` |
| Username | `bot@test.local` |
| Password | `secret` |

### Outbound (SMTP) — Mailpit

| Field | Value |
|-------|-------|
| Send via | SMTP |
| From address | `bot@test.local` |
| SMTP host | `mailpit` |
| SMTP port | `1025` |
| SMTP username | *(empty)* |
| SMTP password | *(empty)* |
| SMTP STARTTLS | **off** |
| Mode | Validation |

GreenMail SMTP port `3025` remains available for **inbound inject only** (Test email tab, CLI). Do not point the outbound connector at GreenMail.

## Test flow

1. Open bot → **Test email** tab (dev only).
2. Fill From (simulated client), Subject, Body → **Send to inbox** (injected via GreenMail SMTP → IMAP).
3. Click **Process now** (or wait for `worker-mail` poll).
4. Open **Validation** → approve reply.
5. Open **Mailpit** (http://127.0.0.1:8025) to read the outbound email.

## CLI inject (alternative)

From the host:

```bash
python -c "
import smtplib
from email.message import EmailMessage
m = EmailMessage()
m['From'] = 'client@example.com'
m['To'] = 'bot@test.local'
m['Subject'] = 'Devis écran 27 pouces'
m.set_content('Bonjour, je voudrais un devis pour 2 écrans 27 pouces.')
with smtplib.SMTP('127.0.0.1', 3025) as s:
    s.send_message(m)
"
```

## Manual worker poll

```bash
./sail exec worker-mail python -m evenor.interfaces.worker_mail --once
```

## Production

- `docker compose up -d` without `--profile dev` → no GreenMail, no Mailpit.
- `worker-mail` uses real IMAP/SMTP from dashboard connectors.
- Keep `smtp_use_tls` enabled for real providers.
