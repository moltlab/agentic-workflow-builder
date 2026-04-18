"""Datasource secrets encryption helpers."""

from __future__ import annotations

import json
import os
from hashlib import sha256
from typing import Any, Dict

from cryptography.fernet import Fernet, InvalidToken


def _get_fernet() -> Fernet:
    """
    Build Fernet from DATASOURCE_ENCRYPTION_KEY.

    Accepted formats:
    - a valid Fernet key
    - any passphrase string (will be deterministically derived)
    """
    raw_key = os.getenv("DATASOURCE_ENCRYPTION_KEY", "").strip()
    if not raw_key:
        raise RuntimeError("DATASOURCE_ENCRYPTION_KEY is not set")

    try:
        return Fernet(raw_key.encode("utf-8"))
    except Exception:
        # Derive a stable Fernet-compatible key from passphrase.
        digest = sha256(raw_key.encode("utf-8")).digest()
        import base64

        derived = base64.urlsafe_b64encode(digest)
        return Fernet(derived)


def encrypt_secrets_blob(secrets: Dict[str, Any]) -> str:
    payload = json.dumps(secrets or {}, separators=(",", ":")).encode("utf-8")
    return _get_fernet().encrypt(payload).decode("utf-8")


def decrypt_secrets_blob(payload_ciphertext: str) -> Dict[str, Any]:
    if not payload_ciphertext:
        return {}
    try:
        raw = _get_fernet().decrypt(payload_ciphertext.encode("utf-8"))
    except InvalidToken as exc:
        raise ValueError("Unable to decrypt datasource secrets with current key") from exc
    data = json.loads(raw.decode("utf-8"))
    return data if isinstance(data, dict) else {}

