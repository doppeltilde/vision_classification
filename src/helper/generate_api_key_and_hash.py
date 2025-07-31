import hashlib
import secrets
import logging

logger = logging.getLogger(__name__)


def generate_api_key_and_hash_with_salt():
    api_key = secrets.token_urlsafe(32)

    salt = secrets.token_urlsafe(16)

    salted_key = (api_key + salt).encode()

    key_hash = hashlib.sha256(salted_key).hexdigest()

    logger.debug(f"API Key (store in api gateway): {api_key}")
    logger.debug(f"Hash (store in .env): {key_hash}")
    logger.debug(f"Salt (store in .env): {salt}")
