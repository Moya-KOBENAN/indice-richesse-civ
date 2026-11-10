"""Contrôle conservateur de la liste des fichiers publiables, sans dépendance."""
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
FILES = {"README.md", ".gitignore", ".env.example", "pyproject.toml", "requirements.txt",
         "requirements-lock.txt", "environment.yml", "run_pipeline.py"}
EXTENSIONS = {"src": {".py", ".js"}, "tests": {".py"}, "scripts": {".py"},
              "docs": {".md"}, "config": {".yaml"}}


def allowed(name):
    path = Path(name)
    if name.startswith("config/") and name != "config/config.yaml":
        return False
    return name in FILES or (len(path.parts) > 1 and
        path.suffix in EXTENSIONS.get(path.parts[0], set()) and
        not any(part.startswith(".") for part in path.parts[1:]))


def sensitive_content(content):
    """Signaux bloquants : coordonnées littérales et secrets reconnaissables.

    Heuristiques conservatrices, sans prétendre identifier toutes les données.
    Les fichiers suspects ne sont jamais imprimés avec leur contenu.
    """
    patterns = (
        r"ee\.Geometry\.Point\(\s*\[\s*-?\d+(?:\.\d+)?\s*,\s*-?\d",
        r"wealth_mean\s*:\s*-?\d",
        r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----",
        r"gh[pousr]_[A-Za-z0-9]{30,}",
        r"github_pat_[A-Za-z0-9_]{30,}",
    )
    return any(re.search(pattern, content) for pattern in patterns)


def main():
    names = {p.relative_to(ROOT).as_posix() for p in ROOT.rglob("*") if p.is_file()
             and not any(x in p.relative_to(ROOT).parts for x in
                         (".git", ".venv", "__pycache__", ".pytest_cache"))}
    indexed = {}
    if (ROOT / ".git").exists():
        result = subprocess.run(["git", "ls-files", "-z"], cwd=ROOT,
                                check=True, capture_output=True)
        for name in filter(None, result.stdout.decode().split("\0")):
            names.add(name)
            indexed[name] = subprocess.run(["git", "show", ":" + name], cwd=ROOT,
                check=True, capture_output=True).stdout.decode("utf-8", errors="replace")
    rejected = sorted(name for name in names if name and not allowed(name))
    for name in sorted(names):
        path = ROOT / name
        if path.is_symlink():
            rejected.append(name + " (lien symbolique interdit)")
            continue
        contents = [indexed[name]] if name in indexed else []
        if path.is_file() and allowed(name):
            contents.append(path.read_text())
        if any(sensitive_content(content) for content in contents):
            rejected.append(name + " (contenu potentiellement confidentiel)")
    if rejected:
        print("Fichiers hors politique de diffusion :\n" + "\n".join(rejected))
        return 1
    print(f"Contrôle réussi : {len(names)} fichiers, chemins autorisés et contenu contrôlé.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
