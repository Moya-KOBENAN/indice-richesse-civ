"""Contrôle conservateur de la liste des fichiers publiables, sans dépendance."""
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
FILES = {"README.md", ".gitignore", ".env.example", "pyproject.toml", "requirements.txt",
         "requirements-lock.txt", "environment.yml", "run_pipeline.py"}
EXTENSIONS = {"src": {".py", ".js"}, "tests": {".py"}, "scripts": {".py"},
              "docs": {".md"}, "config": {".yaml"}, ".github": {".yml"}}


def allowed(name):
    path = Path(name)
    return name in FILES or (len(path.parts) > 1 and
        path.suffix in EXTENSIONS.get(path.parts[0], set()) and
        not any(part.startswith(".") for part in path.parts[1:]))


def main():
    names = {p.relative_to(ROOT).as_posix() for p in ROOT.rglob("*") if p.is_file()
             and not any(x in p.relative_to(ROOT).parts for x in
                         (".git", ".venv", "__pycache__", ".pytest_cache"))}
    if (ROOT / ".git").exists():
        result = subprocess.run(["git", "ls-files", "-z"], cwd=ROOT,
                                check=True, capture_output=True)
        names.update(result.stdout.decode().strip("\0").split("\0"))
    rejected = sorted(name for name in names if name and not allowed(name))
    for name in names:
        path = ROOT / name
        if path.is_file() and path.suffix == ".js":
            content = path.read_text()
            if "ee.Geometry.Point([" in content or "wealth_mean:" in content:
                rejected.append(name + " (données géographiques intégrées)")
    if rejected:
        print("Fichiers hors liste de diffusion :\n" + "\n".join(rejected))
        return 1
    print(f"Contrôle réussi : {len(names)} fichiers, chemins autorisés uniquement.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
