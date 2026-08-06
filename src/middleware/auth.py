import hashlib
import logging
import secrets
from fastapi import HTTPException, Security, status
from fastapi.security import APIKeyHeader
from src.shared.shared import settings

logger = logging.getLogger(__name__)

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def verify_api_key_with_salt(api_key_to_check: str) -> bool:
    if not api_key_to_check or not settings.api_key_hash or not settings.api_key_salt:
        logger.warning("Missing API key, stored hash, or stored salt.")
        return False

    salted_key_to_check = (api_key_to_check + settings.api_key_salt).encode()
    calculated_hash = hashlib.sha256(salted_key_to_check).hexdigest()
    return secrets.compare_digest(calculated_hash, settings.api_key_hash)


def get_api_key(api_key: str = Security(api_key_header)):
    if not settings.use_api_key:
        return None

    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing API Key!",
            headers={"WWW-Authenticate": "ApiKey"},
        )

    if not verify_api_key_with_salt(api_key_to_check=api_key):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API Key!",
            headers={"WWW-Authenticate": "ApiKey"},
        )

    return api_key