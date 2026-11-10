"""Structure du pipeline : scripts présents, chemins cohérents, CLI utilisable."""
from pathlib import Path
import ast
import subprocess
import sys
from zipfile import ZipFile

import pytest
import yaml

from src.utils import paths as project_paths
from src.pipeline import PIPELINES, get_stage, run_stage
from src.data.ehcvm.audit_archive import check_archive, csv_members

ROOT = project_paths.PROJECT_ROOT


def test_registered_stages_resolve_outside_project(monkeypatch, tmp_path):
    """Tout script existe ; le dry-run fonctionne hors du projet sans rien écrire."""
    monkeypatch.chdir(tmp_path)
    for scope, stages in PIPELINES.items():
        for code, stage in stages.items():
            assert stage.path.is_file(), (scope, code)
            inputs_ready = all(path.exists() for path in stage.inputs)
            if inputs_ready and not stage.native_dry_run:
                command = run_stage(scope, code, dry_run=True)
                assert Path(command[1]).is_absolute()
    assert not list(tmp_path.iterdir())


def test_unknown_stage_and_missing_inputs_fail_clearly(monkeypatch):
    with pytest.raises(ValueError, match="inconnue"):
        get_stage("ehcvm", "99")
    stage = PIPELINES["ehcvm"]["02"]
    if not all(path.exists() for path in stage.inputs):
        with pytest.raises(FileNotFoundError, match="Entrées manquantes"):
            run_stage("ehcvm", "02", dry_run=True)


def resolve_path(node, environment):
    """Évaluer uniquement les expressions de chemins, sans importer les CNN."""
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Name):
        return environment[node.id]
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):
        return resolve_path(node.left, environment) / resolve_path(node.right, environment)
    raise KeyError("Expression hors des chemins statiques")


def test_canonical_scripts_use_scoped_paths():
    for scope in ("dhs", "common"):
        for stage in PIPELINES[scope].values():
            # Les scripts partagés (features/evaluation communs, prédiction nationale)
            # reçoivent l'enquête en argument : pas de chemin propre à une branche.
            if stage.unavailable or stage.args or f"/{scope}/" not in f"/{stage.script}" or \
                    stage.script.startswith(("evaluation/common", "models/common/predict")):
                continue
            tree = ast.parse(stage.path.read_text())
            environment = {name: value for name, value in vars(project_paths).items() if isinstance(value, Path)}
            paths = []
            for node in tree.body:
                if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
                    try:
                        value = resolve_path(node.value, environment)
                    except (KeyError, TypeError):
                        continue
                    environment[node.targets[0].id] = value
                    if isinstance(value, Path):
                        paths.append(value)
            assert paths, stage.script
            for path in paths:
                if path.is_relative_to(ROOT / "data/processed") and path != ROOT / "data/processed":
                    assert path.is_relative_to(ROOT / "data/processed" / scope), (stage.script, path)
                if path.is_relative_to(ROOT / "outputs") and path != ROOT / "outputs":
                    # DHS 13 consomme légitimement le modèle national commun.
                    allowed = (scope, "common") if stage.script.endswith("13_extract_national_adapted_features.py") else (scope,)
                    assert any(path.is_relative_to(ROOT / "outputs" / group) for group in allowed), (stage.script, path)


@pytest.mark.parametrize("args,expected", [
    (["--pipeline", "common", "--stage", "12", "--dry-run"], "models/common/12_train_national_viirs_classifier.py"),
    (["--pipeline", "dhs", "--stage", "07", "--dry-run"], "models/dhs/07_ridge_model.py"),
    (["--list"], "compare_feature_sets.py"),
])
def test_cli_from_another_directory(tmp_path, args, expected):
    result = subprocess.run([sys.executable, str(ROOT / "run_pipeline.py"), *args],
                            cwd=tmp_path, text=True, capture_output=True)
    if "--stage" in args:
        # Sans données, le dry-run doit refuser explicitement les entrées absentes.
        # Ce test ne doit pas dépendre de fichiers confidentiels sur la machine.
        from src.pipeline import get_stage
        stage = get_stage(args[1], args[3])
        missing = [p for p in stage.inputs if not p.exists()]
        if missing:
            assert result.returncode != 0
            assert "Entrées manquantes" in result.stderr
        else:
            assert result.returncode == 0, result.stderr
            assert expected in result.stdout
    else:
        assert result.returncode == 0, result.stderr
        assert expected in result.stdout
    assert not list(tmp_path.iterdir())


def test_every_stage_script_defines_entry_point_or_is_historical():
    """Les nouvelles étapes suivent la convention main() + garde __main__."""
    for stages in PIPELINES.values():
        for stage in stages.values():
            text = stage.path.read_text()
            if "def main(" in text:
                assert 'if __name__ == "__main__"' in text or "if __name__=='__main__'" in text, stage.script


def test_config_is_valid():
    config = yaml.safe_load((ROOT / "config/config.yaml").read_text())
    assert config["model"]["regression_model"] == "Ridge"
    assert config["ehcvm"]["archive"].endswith(".zip")


def test_archive_rejects_missing_modules(tmp_path):
    path = tmp_path / "incomplete.zip"
    with ZipFile(path, "w") as archive:
        archive.writestr("enquete/s00_me_civ2021.csv", "grappe\n1\n")
    with pytest.raises(ValueError, match="Modules absents"):
        check_archive(path)


def test_archive_rejects_ambiguous_modules_and_ignores_macos(tmp_path):
    path = tmp_path / "archive.zip"
    with ZipFile(path, "w") as archive:
        archive.writestr("__MACOSX/._file.csv", "metadata")
        archive.writestr("a/file.csv", "value\n1\n")
    with ZipFile(path) as archive:
        assert csv_members(archive) == {"file.csv": "a/file.csv"}
    with ZipFile(path, "a") as archive:
        archive.writestr("b/file.csv", "value\n2\n")
    with ZipFile(path) as archive:
        with pytest.raises(ValueError, match="ambigu"):
            csv_members(archive)
