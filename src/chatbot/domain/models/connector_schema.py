from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from chatbot.domain.models.connector import ConnectorDirection, ConnectorType


class EmailOutboundProvider(StrEnum):
    SMTP = "smtp"
    MAILJET = "mailjet"
    MAILGUN = "mailgun"


class EmailAuthType(StrEnum):
    PASSWORD = "password"
    MICROSOFT_OAUTH = "microsoft_oauth"
    GOOGLE_OAUTH = "google_oauth"


def _email_auth_type_options() -> tuple[tuple[str, str], ...]:
    return (
        (EmailAuthType.PASSWORD.value, "Password"),
        (EmailAuthType.MICROSOFT_OAUTH.value, "Microsoft OAuth"),
        (EmailAuthType.GOOGLE_OAUTH.value, "Google OAuth"),
    )


@dataclass(frozen=True)
class ConnectorField:
    key: str
    label: str
    help: str
    input_type: str = "text"
    required: bool = False
    placeholder: str = ""
    directions: tuple[str, ...] = (ConnectorDirection.IN.value, ConnectorDirection.OUT.value)
    secret: bool = False
    providers: tuple[str, ...] | None = None
    options: tuple[tuple[str, str], ...] | None = None


def _email_out_providers() -> tuple[tuple[str, str], ...]:
    return (
        (EmailOutboundProvider.SMTP.value, "SMTP"),
        (EmailOutboundProvider.MAILJET.value, "Mailjet"),
        (EmailOutboundProvider.MAILGUN.value, "Mailgun"),
    )


CONNECTOR_SCHEMAS: dict[str, list[ConnectorField]] = {
    ConnectorType.WHATSAPP.value: [
        ConnectorField(
            key="phone_number_id",
            label="Phone Number ID",
            help="WhatsApp Phone Number ID from Meta Business Manager (WhatsApp > API Setup).",
            placeholder="e.g. 123456789012345",
            directions=(ConnectorDirection.OUT.value,),
        ),
        ConnectorField(
            key="access_token",
            label="Access token",
            help="Permanent access token from Meta (System User or WhatsApp app). Leave blank on update to keep current value.",
            input_type="password",
            secret=True,
            directions=(ConnectorDirection.OUT.value,),
        ),
        ConnectorField(
            key="app_secret",
            label="App secret",
            help="Meta App Secret (App Dashboard > Settings > Basic). Used to verify webhook signatures.",
            input_type="password",
            secret=True,
        ),
        ConnectorField(
            key="verify_token",
            label="Verify token",
            help="Arbitrary string you choose; must match the token configured in Meta webhook settings.",
            input_type="password",
            secret=True,
            directions=(ConnectorDirection.IN.value,),
        ),
        ConnectorField(
            key="admin_wa_id",
            label="Admin WhatsApp ID",
            help="Recipient WhatsApp ID for automation alerts (e.g. order notifications from the worker).",
            placeholder="e.g. 2305…",
            directions=(ConnectorDirection.OUT.value,),
        ),
    ],
    ConnectorType.MESSENGER.value: [
        ConnectorField(
            key="page_access_token",
            label="Page access token",
            help="Page Access Token from Meta (Messenger > Settings > Access Tokens). Leave blank on update to keep current value.",
            input_type="password",
            secret=True,
        ),
        ConnectorField(
            key="verify_token",
            label="Verify token",
            help="Must match the verify token in your Meta webhook configuration.",
            input_type="password",
            secret=True,
            directions=(ConnectorDirection.IN.value,),
        ),
        ConnectorField(
            key="app_secret",
            label="App secret",
            help="Meta App Secret for webhook signature verification.",
            input_type="password",
            secret=True,
        ),
    ],
    ConnectorType.INSTAGRAM.value: [
        ConnectorField(
            key="access_token",
            label="Access token",
            help="Instagram Graph API access token. Leave blank on update to keep current value.",
            input_type="password",
            secret=True,
        ),
        ConnectorField(
            key="ig_user_id",
            label="Instagram user ID",
            help="Instagram Business Account ID linked to your Facebook Page.",
            placeholder="e.g. 178414…",
        ),
        ConnectorField(
            key="verify_token",
            label="Verify token",
            help="Must match the verify token in your Meta webhook configuration.",
            input_type="password",
            secret=True,
            directions=(ConnectorDirection.IN.value,),
        ),
        ConnectorField(
            key="app_secret",
            label="App secret",
            help="Meta App Secret for webhook signature verification.",
            input_type="password",
            secret=True,
        ),
    ],
    ConnectorType.EMAIL.value: [
        ConnectorField(
            key="mail_connection_id",
            label="Mail connection",
            help="Reusable OAuth mailbox connection (Microsoft 365 or Gmail). Create one above, then select it here.",
            input_type="select",
            directions=(ConnectorDirection.IN.value, ConnectorDirection.OUT.value),
        ),
        ConnectorField(
            key="auth_type",
            label="Authentication",
            help=(
                "Use Microsoft or Google OAuth for M365 / Gmail mailboxes. "
                "Password auth works for self-hosted or dev mail servers."
            ),
            input_type="select",
            directions=(ConnectorDirection.IN.value, ConnectorDirection.OUT.value),
            options=_email_auth_type_options(),
        ),
        ConnectorField(
            key="microsoft_client_id",
            label="Microsoft client ID",
            help="Entra application (client) ID for this connector.",
            directions=(ConnectorDirection.IN.value, ConnectorDirection.OUT.value),
        ),
        ConnectorField(
            key="microsoft_client_secret",
            label="Microsoft client secret",
            help="Entra client secret. Leave blank on update to keep current value.",
            input_type="password",
            secret=True,
            directions=(ConnectorDirection.IN.value, ConnectorDirection.OUT.value),
        ),
        ConnectorField(
            key="google_client_id",
            label="Google client ID",
            help="Google Cloud OAuth client ID for this connector.",
            directions=(ConnectorDirection.IN.value, ConnectorDirection.OUT.value),
        ),
        ConnectorField(
            key="google_client_secret",
            label="Google client secret",
            help="Google Cloud client secret. Leave blank on update to keep current value.",
            input_type="password",
            secret=True,
            directions=(ConnectorDirection.IN.value, ConnectorDirection.OUT.value),
        ),
        ConnectorField(
            key="imap_host",
            label="IMAP host",
            help="Mail server hostname for incoming mail (e.g. imap.gmail.com, outlook.office365.com).",
            directions=(ConnectorDirection.IN.value,),
        ),
        ConnectorField(
            key="imap_port",
            label="IMAP port",
            help="IMAP port, usually 993 (SSL) or 143.",
            input_type="number",
            placeholder="993",
            directions=(ConnectorDirection.IN.value,),
        ),
        ConnectorField(
            key="username",
            label="Username",
            help="Mailbox login for IMAP (often the full email address).",
            directions=(ConnectorDirection.IN.value,),
        ),
        ConnectorField(
            key="password",
            label="Password",
            help="Mailbox password or app-specific password. Leave blank on update to keep current value.",
            input_type="password",
            secret=True,
            directions=(ConnectorDirection.IN.value,),
        ),
        ConnectorField(
            key="process_since",
            label="Process emails since",
            help=(
                "Only inbound emails received at or after this date/time are sent to the bot. "
                "Older messages are ignored permanently."
            ),
            input_type="datetime-local",
            directions=(ConnectorDirection.IN.value,),
        ),
        ConnectorField(
            key="outbound_provider",
            label="Send via",
            help="Choose one outbound delivery method. Only fields for the selected provider are shown below.",
            input_type="select",
            required=True,
            directions=(ConnectorDirection.OUT.value,),
            options=_email_out_providers(),
        ),
        ConnectorField(
            key="from_addr",
            label="From address",
            help="Sender address (must be verified with your provider).",
            placeholder="noreply@example.com",
            required=True,
            directions=(ConnectorDirection.OUT.value,),
        ),
        ConnectorField(
            key="default_subject",
            label="Default subject",
            help="Default subject for validation replies. Leave empty to auto-use Re: {original subject}.",
            placeholder="Reply from support",
            directions=(ConnectorDirection.OUT.value,),
        ),
        ConnectorField(
            key="smtp_host",
            label="SMTP host",
            help="SMTP server hostname (e.g. smtp.gmail.com, smtp.office365.com).",
            directions=(ConnectorDirection.OUT.value,),
            providers=(EmailOutboundProvider.SMTP.value,),
        ),
        ConnectorField(
            key="smtp_port",
            label="SMTP port",
            help="SMTP port, usually 587 (STARTTLS) or 465 (SSL).",
            input_type="number",
            placeholder="587",
            directions=(ConnectorDirection.OUT.value,),
            providers=(EmailOutboundProvider.SMTP.value,),
        ),
        ConnectorField(
            key="smtp_username",
            label="SMTP username",
            help="Login for SMTP authentication (often the same as your email address).",
            directions=(ConnectorDirection.OUT.value,),
            providers=(EmailOutboundProvider.SMTP.value,),
        ),
        ConnectorField(
            key="smtp_password",
            label="SMTP password",
            help="SMTP password or app-specific password. Leave blank on update to keep current value.",
            input_type="password",
            secret=True,
            directions=(ConnectorDirection.OUT.value,),
            providers=(EmailOutboundProvider.SMTP.value,),
        ),
        ConnectorField(
            key="smtp_use_tls",
            label="SMTP STARTTLS",
            help="Enable STARTTLS (typical for port 587). Disable for local test servers such as GreenMail.",
            input_type="checkbox",
            directions=(ConnectorDirection.OUT.value,),
            providers=(EmailOutboundProvider.SMTP.value,),
        ),
        ConnectorField(
            key="mailjet_api_key",
            label="Mailjet API key",
            help="Public API key from Mailjet account settings. Leave blank on update to keep current value.",
            input_type="password",
            secret=True,
            directions=(ConnectorDirection.OUT.value,),
            providers=(EmailOutboundProvider.MAILJET.value,),
        ),
        ConnectorField(
            key="mailjet_api_secret",
            label="Mailjet API secret",
            help="Secret API key from Mailjet account settings. Leave blank on update to keep current value.",
            input_type="password",
            secret=True,
            directions=(ConnectorDirection.OUT.value,),
            providers=(EmailOutboundProvider.MAILJET.value,),
        ),
        ConnectorField(
            key="mailgun_api_key",
            label="Mailgun API key",
            help="Private API key from Mailgun (Sending > Domain settings). Leave blank on update to keep current value.",
            input_type="password",
            secret=True,
            directions=(ConnectorDirection.OUT.value,),
            providers=(EmailOutboundProvider.MAILGUN.value,),
        ),
        ConnectorField(
            key="mailgun_domain",
            label="Mailgun domain",
            help="Sending domain configured in Mailgun (e.g. mg.example.com).",
            placeholder="mg.example.com",
            directions=(ConnectorDirection.OUT.value,),
            providers=(EmailOutboundProvider.MAILGUN.value,),
        ),
        ConnectorField(
            key="mailgun_region",
            label="Mailgun region",
            help="US or EU API endpoint — must match where your Mailgun account/domain is hosted.",
            input_type="select",
            directions=(ConnectorDirection.OUT.value,),
            providers=(EmailOutboundProvider.MAILGUN.value,),
            options=(("us", "US (api.mailgun.net)"), ("eu", "EU (api.eu.mailgun.net)")),
        ),
    ],
}


def resolve_email_auth_type(config: dict) -> str:
    raw = str(config.get("auth_type", EmailAuthType.PASSWORD.value)).strip().lower()
    try:
        return EmailAuthType(raw).value
    except ValueError:
        return EmailAuthType.PASSWORD.value


def is_oauth_auth_type(auth_type: str) -> bool:
    return auth_type in (EmailAuthType.MICROSOFT_OAUTH.value, EmailAuthType.GOOGLE_OAUTH.value)


def oauth_managed_connector_keys() -> frozenset[str]:
    return frozenset({"oauth_refresh_token", "oauth_access_token", "oauth_token_expires_at"})


def runtime_mail_config_keys() -> frozenset[str]:
    """Ephemeral keys set during OAuth resolution; never persist on connectors."""
    return frozenset({"_resolved_access_token"})

def resolve_email_outbound_provider(config: dict) -> str:
    raw = str(config.get("outbound_provider", EmailOutboundProvider.SMTP.value)).strip().lower()
    try:
        return EmailOutboundProvider(raw).value
    except ValueError:
        return EmailOutboundProvider.SMTP.value


def fields_for(
    connector_type: str,
    direction: str,
    *,
    outbound_provider: str | None = None,
) -> list[ConnectorField]:
    result: list[ConnectorField] = []
    for field in CONNECTOR_SCHEMAS.get(connector_type, []):
        if direction not in field.directions:
            continue
        if field.providers is not None and outbound_provider is not None:
            if field.key != "outbound_provider" and outbound_provider not in field.providers:
                continue
        result.append(field)
    return result


def secret_connector_keys() -> frozenset[str]:
    keys = {
        field.key
        for fields in CONNECTOR_SCHEMAS.values()
        for field in fields
        if field.secret
    }
    return frozenset(keys | {"oauth_refresh_token", "oauth_access_token"})


def connector_schemas_for_template() -> dict[str, list[dict]]:
    return {
        connector_type: [
            {
                "key": field.key,
                "label": field.label,
                "help": field.help,
                "input_type": field.input_type,
                "required": field.required,
                "placeholder": field.placeholder or field.key,
                "directions": list(field.directions),
                "providers": list(field.providers) if field.providers else None,
                "options": [{"value": v, "label": lbl} for v, lbl in field.options]
                if field.options
                else None,
            }
            for field in fields
        ]
        for connector_type, fields in CONNECTOR_SCHEMAS.items()
    }
