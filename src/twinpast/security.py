"""Credential handling shared by the runner and local release checks."""
import base64
import os
import re
from urllib.parse import quote, quote_plus

from .data import require


_CREDENTIAL_FIELDS = {
    "apikey", "key", "token", "accesstoken", "refreshtoken", "accesskey",
    "secret", "secretkey", "clientsecret", "privatekey", "password",
    "credential", "credentials", "authorization", "authorizationheader",
    "authtoken", "bearertoken", "xapikey", "xgoogapikey",
}


def credential_field(name):
    return isinstance(name, str) and re.sub(r"[^a-z0-9]", "", name.lower()) in _CREDENTIAL_FIELDS


def validate_configuration(value):
    """Reject nested credentials, including camel-case and header spellings."""
    if isinstance(value, dict):
        for name, child in value.items():
            require(not credential_field(name), "Credentials must be supplied through TWINPAST_API_KEY")
            validate_configuration(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            validate_configuration(child)


def redact(value):
    """Remove credential fields and common representations of the current key."""
    secret = os.environ.get("TWINPAST_API_KEY")
    variants = set()
    if secret:
        variants = {secret, quote(secret, safe=""), quote_plus(secret),
                    base64.b64encode(secret.encode()).decode()}

    def clean(item):
        if isinstance(item, str):
            for variant in sorted(variants, key=len, reverse=True):
                item = item.replace(variant, "[REDACTED]")
            return item
        if isinstance(item, dict):
            return {clean(name): "[REDACTED]" if credential_field(name) else clean(child)
                    for name, child in item.items()}
        if isinstance(item, (list, tuple)):
            return [clean(child) for child in item]
        return item

    return clean(value)
