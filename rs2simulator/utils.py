import os
from functools import lru_cache
from pathlib import Path


@lru_cache(maxsize=16)
def read_asset_text(asset_file: str) -> str:
    a = os.environ.get("ASSETS_FOLDER", "assets")
    assets_folder = Path(a).resolve()
    return (assets_folder / asset_file).read_text()
