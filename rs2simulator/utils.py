import os
from functools import lru_cache
from pathlib import Path


@lru_cache(maxsize=16)
def read_asset_text(asset_file: str) -> str:
    assets_folder = Path(os.environ.get("ASSETS_FOLDER")).resolve()
    return (assets_folder / asset_file).read_text()
