# GreenMail — local email testing (dev only)

GreenMail simulates IMAP + SMTP for end-to-end email bot testing. It is **not** deployed in production.

## Start stack

```bash
./sail up -d --profile dev
```

This starts `greenmail`, `worker-mail`, `api`, and other services. GreenMail is only included with the `dev` Compose profile.

GreenMail must bind to `0.0.0.0` inside Docker so other containers (`api`, `worker-mail`) can reach it via hostname `greenmail`.

Ports (localhost only):

| Service | Host URL / port |
|---------|-----------------|
| Web UI (API) | http://127.0.0.1:8081 |
| SMTP | `127.0.0.1:3025` |
| IMAP (plain) | `127.0.0.1:3143` |
| IMAPS | `127.0.0.1:3993` |

Test account: `bot@test.local` / `secret`

Set in `.env` for dashboard Test email tab:

```env
DEV_MODE=true
```

## Connector configuration

Configure the bot in **Dashboard → Connectors**:

### Inbound (IMAP)

| Field | Value (from Docker network) |
|-------|----------------------------|
| IMAP host | `greenmail` |
| IMAP port | `3143` |
| Username | `bot@test.local` |
| Password | `secret` |

### Outbound (SMTP)

| Field | Value |
|-------|-------|
| Send via | SMTP |
| From address | `bot@test.local` |
| SMTP host | `greenmail` |
| SMTP port | `3025` |
| SMTP STARTTLS | **off** |
| Mode | Validation |

## Test flow

1. Open bot → **Test email** tab (dev only).
2. Fill From (simulated client), Subject, Body → **Send to inbox**.
3. Click **Process now** (or wait ~30s for `worker-mail` poll).
4. Open **Validation** → approve reply.
5. Reply is sent via SMTP back to GreenMail.

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
./sail exec worker-mail python -m chatbot.interfaces.worker_mail --once
```

## Production

- `docker compose up -d` without `--profile dev` → no GreenMail.
- `worker-mail` uses real IMAP/SMTP from dashboard connectors.
- Keep `smtp_use_tls` enabled for real providers.
