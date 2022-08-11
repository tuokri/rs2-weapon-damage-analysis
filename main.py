import re
from concurrent import futures
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

SRC_DIR = r"C:\Program Files (x86)\Steam\steamapps\common\Rising Storm 2\Development"

DAMAGE_PATTERN = re.compile(r"")
FALLOFF_PATTERN = re.compile(r"")


@dataclass
class BulletData:
    name: str
    damage: int
    damage_falloff: np.ndarray


def process_file(path: Path) -> Optional[BulletData]:
    path = path.resolve()
    print(f"processing: '{path}'")
    file_class = str(path.stem)
    print(f"class is: '{file_class}'")
    with path.open() as file:
        data = file.read()
        for line in data:
            pass

    return BulletData(name=file_class, damage=0, damage_falloff=np.zeros(0))


def main():
    src_files = Path(SRC_DIR).rglob("*.uc")

    executor = ProcessPoolExecutor()
    fs = [executor.submit(process_file, file) for file in src_files]

    result: Optional[BulletData]
    for future in futures.as_completed(fs):
        result = future.result()


if __name__ == "__main__":
    main()
