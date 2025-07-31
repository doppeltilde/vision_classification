import hashlib
import logging
from fastapi import HTTPException, Security, status
from fastapi.security import APIKeyHeader
from src.shared.shared import use_api_key, stored_api_key_salt, stored_api_key_hash

logger = logging.getLogger(__name__)

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)
stored_hash = stored_api_key_hash
stored_salt = stored_api_key_salt


def verify_api_key_with_salt(api_key_to_check: str) -> bool:
    if not api_key_to_check or not stored_hash or not stored_salt:
        logger.warning("Missing API key, stored hash, or stored salt.")
        return False

    salted_key_to_check = (api_key_to_check + stored_salt).encode()
    calculated_hash = hashlib.sha256(salted_key_to_check).hexdigest()
    return calculated_hash == stored_hash


def get_api_key(api_key: str = Security(api_key_header)):
    if not use_api_key:
        return None

    if api_key:
        if verify_api_key_with_salt(api_key_to_check=api_key):
            return api_key
        else:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid API Key!",
            )
    else:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing API Key!",
        )
