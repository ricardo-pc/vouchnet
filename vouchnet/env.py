"""Load local config from .env, so secrets live per-project instead of globally.

Why not a shell profile: the SDK reads exactly one variable name
(`ANTHROPIC_API_KEY`), so a second `export` of it in ~/.zshrc silently
overwrites the first. Two projects with two keys cannot both work that way --
you get whichever export ran last, which is the one you are least likely to be
thinking about. A .env per project gives each one its own key, scoped to the
directory it belongs to.

Production sets real environment variables (Render, CI), so this is a no-op
there: existing variables always win, and a missing python-dotenv is not an
error.
"""

from __future__ import annotations

import os
from pathlib import Path

_ENV_FILE = Path(__file__).resolve().parent.parent / ".env"


def load_env(path: Path = _ENV_FILE) -> bool:
    """Load `.env` into the environment if present. Returns whether it loaded.

    Never overrides a variable that is already set: a real environment variable
    (production, CI, or an explicit one-off on the command line) outranks the
    local file, so the file cannot silently shadow a deployment's config.
    """
    try:
        from dotenv import load_dotenv
    except ImportError:
        # Optional convenience. Anywhere that supplies real env vars is fine
        # without it, and a clone that hasn't installed it should still run.
        return False
    if not path.is_file():
        return False
    return bool(load_dotenv(path, override=False))


def require(name: str, hint: str = "") -> str:
    """Fetch a required variable, failing with a message that says what to do."""
    value = os.environ.get(name)
    if not value:
        raise SystemExit(
            f"{name} is not set.\n"
            f"  Add it to {_ENV_FILE} (copy .env.example to .env and fill it in)."
            + (f"\n  {hint}" if hint else "")
        )
    return value
