from pathlib import Path

tests_dir = Path(__file__).parent.resolve()
data_dir = tests_dir / "data"
repo_dir = tests_dir.parent
main_py = repo_dir / "rs2simulator/main.py"
