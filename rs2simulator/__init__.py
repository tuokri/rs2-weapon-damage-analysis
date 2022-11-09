from pathlib import Path as _Path

from dotenv import load_dotenv as _load_dotenv

_load_dotenv()

from . import db  # noqa
from . import components  # noqa

ASSETS_DIR = _Path(__file__).parent.resolve() / "assets"

__all__ = [
    "ASSETS_DIR",
    "db",
    "components",
]
