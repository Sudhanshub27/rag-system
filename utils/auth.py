"""
Authentication and User Session Management.
Provides bcrypt password hashing and persistent user credential storage.
"""

import json
from pathlib import Path

import bcrypt

USERS_FILE = Path("./data/users.json")


def hash_password(password: str) -> str:
    """Hash a plaintext password using bcrypt."""
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def verify_password(password: str, password_hash: str) -> bool:
    """Verify a password against its stored bcrypt hash."""
    try:
        return bcrypt.checkpw(password.encode("utf-8"), password_hash.encode("utf-8"))
    except Exception:
        return False


def _load_users() -> dict:
    """Load persistent users from JSON file, initializing defaults if missing."""
    USERS_FILE.parent.mkdir(exist_ok=True, parents=True)
    if not USERS_FILE.exists():
        # Default seed accounts
        default_users = {
            "demo_user": {
                "username": "demo_user",
                "email": "demo@example.com",
                "password_hash": hash_password("demo123"),
            },
            "alice": {
                "username": "alice",
                "email": "alice@example.com",
                "password_hash": hash_password("alice123"),
            },
            "bob": {
                "username": "bob",
                "email": "bob@example.com",
                "password_hash": hash_password("bob123"),
            },
        }
        USERS_FILE.write_text(json.dumps(default_users, indent=2))
        return default_users
    try:
        return json.loads(USERS_FILE.read_text())
    except Exception:
        return {}


def _save_users(users: dict) -> None:
    """Save user record map to disk."""
    USERS_FILE.parent.mkdir(exist_ok=True, parents=True)
    USERS_FILE.write_text(json.dumps(users, indent=2))


def authenticate_user(username: str, password: str) -> dict | None:
    """
    Authenticate a user by username and password.

    Returns:
        User info dict if valid, else None.
    """
    users = _load_users()
    user = users.get(username.strip().lower())
    if user and verify_password(password, user["password_hash"]):
        return user
    return None


def register_user(username: str, email: str, password: str) -> bool:
    """
    Register a new user with a hashed password.

    Returns:
        True if successfully registered, False if username already exists.
    """
    clean_username = username.strip().lower()
    if not clean_username or len(password) < 4:
        return False

    users = _load_users()
    if clean_username in users:
        return False

    users[clean_username] = {
        "username": clean_username,
        "email": email.strip(),
        "password_hash": hash_password(password),
    }
    _save_users(users)
    return True
