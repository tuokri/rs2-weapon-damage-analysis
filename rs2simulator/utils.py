from functools import lru_cache

from rs2simulator import ASSETS_DIR


@lru_cache(maxsize=16)
def read_asset_text(asset_file: str) -> str:
    return (ASSETS_DIR / asset_file).read_text()
