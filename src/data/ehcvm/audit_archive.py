"""Audit reproductible des CSV EHCVM, sans extraction ni modification du ZIP."""
from pathlib import Path
import argparse
import hashlib
import json
import sys
from zipfile import ZipFile

_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.utils.paths import CONFIG_DIR, EHCVM_REPORTS_DIR

MODULES = ("s00_me", "s11_me", "s12_me", "ehcvm_menage",
           "ehcvm_welfare", "ehcvm_ponderations", "s01_co")
KEYS = ["grappe", "menage", "vague"]


def default_archive():
    from src.utils.config import load_config, resolve_path
    return resolve_path(load_config()["ehcvm"]["archive"])


def csv_members(archive):
    """Ignorer les fichiers de métadonnées macOS et refuser les noms ambigus."""
    members = {}
    for name in archive.namelist():
        if not name.endswith(".csv") or "__MACOSX" in Path(name).parts:
            continue
        basename = Path(name).name
        if basename in members:
            raise ValueError(f"Nom de CSV ambigu dans le ZIP : {basename}")
        members[basename] = name
    return members


def check_archive(path):
    if not path.is_file():
        raise FileNotFoundError(f"Archive EHCVM introuvable : {path}. Utiliser --archive ou config/config.yaml.")
    with ZipFile(path) as archive:
        members = csv_members(archive)
        missing = [name + "_civ2021.csv" for name in MODULES if name + "_civ2021.csv" not in members]
        if missing:
            raise ValueError("Modules absents : " + ", ".join(missing))
    return members


def audit(path):
    import numpy as np
    import pandas as pd
    members = check_archive(path)
    data, stats = {}, {}
    with ZipFile(path) as archive:
        for name in MODULES:
            with archive.open(members[name + "_civ2021.csv"]) as handle:
                frame = pd.read_csv(handle, encoding="cp1252", low_memory=False)
            data[name] = frame
            keys = [key for key in KEYS if key in frame]
            stats[name] = {
                "rows": len(frame), "columns": len(frame.columns),
                "grappes": int(frame.grappe.nunique()),
                "missing_keys": int(frame[keys].isna().any(axis=1).sum()),
                "duplicates_keys": int(frame.duplicated(keys).sum()),
                "vagues": {str(k): int(v) for k, v in frame.vague.value_counts(dropna=False).items()},
            }
            if "menage" in frame:
                stats[name]["duplicates_grappe_menage"] = int(frame.duplicated(["grappe", "menage"]).sum())
    welfare = data["ehcvm_welfare"]
    if welfare[KEYS].isna().any().any() or welfare.duplicated(KEYS).any():
        raise ValueError("Le socle welfare contient des clés absentes ou dupliquées.")
    for name, frame in data.items():
        if set(KEYS).issubset(frame.columns):
            # Ne pas faire correspondre des identifiants manquants entre eux.
            valid = frame.dropna(subset=KEYS)[KEYS].drop_duplicates()
            merged = welfare[KEYS].merge(valid, on=KEYS, how="left", indicator=True, validate="one_to_one")
            stats[name]["welfare_unmatched"] = int((merged._merge == "left_only").sum())
            extra = valid.merge(welfare[KEYS], on=KEYS, how="left", indicator=True, validate="one_to_one")
            stats[name]["extra_valid_keys_vs_welfare"] = int((extra._merge == "left_only").sum())
    gps = data["s00_me"]
    lat = pd.to_numeric(gps.GPS__Latitude, errors="coerce")
    lon = pd.to_numeric(gps.GPS__Longitude, errors="coerce")
    pairs = pd.DataFrame({"lat": lat, "lon": lon})
    cluster_positions = pairs.assign(grappe=gps.grappe).drop_duplicates()
    shared = cluster_positions.duplicated(["lat", "lon"], keep=False)
    stats["gps"] = {
        "pairs_numeric": int(pairs.notna().all(axis=1).sum()),
        "pairs_missing": int(pairs.isna().any(axis=1).sum()),
        "pairs_world_valid": int((lat.between(-90, 90) & lon.between(-180, 180)).sum()),
        "zero_pairs": int(((lat == 0) & (lon == 0)).sum()),
        "unique_numeric_pairs": len(pairs.dropna().drop_duplicates()),
        "grappes_multiple_gps": int((cluster_positions.groupby("grappe").size() > 1).sum()),
        "grappes_shared_coordinates": int(cluster_positions.loc[shared, "grappe"].nunique()),
    }
    assets = data["s12_me"]
    stats["assets"] = {
        "article_codes": int(assets.s12q01.nunique()),
        "duplicate_household_article": int(assets.duplicated(KEYS + ["s12q01"]).sum()),
        "possession_quantity_missing": int(((assets.s12q02 == 1) & assets.s12q03.isna()).sum()),
    }
    stats["positive_welfare"] = {
        column: int((pd.to_numeric(welfare[column], errors="coerce") > 0).sum())
        for column in ("hhweight", "hhsize", "pcexp")
    }
    weights = welfare[KEYS + ["hhweight"]].merge(data["ehcvm_ponderations"], on=KEYS, validate="one_to_one")
    stats["weights_matching_poids"] = int(np.isclose(weights.hhweight, weights.poids, rtol=1e-8, atol=1e-8).sum())
    with path.open("rb") as handle:
        fingerprint = hashlib.file_digest(handle, "sha256").hexdigest()
    stats["source"] = {"archive": str(path), "sha256": fingerprint, "csv_files": len(members), "encoding": "cp1252"}
    return stats


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path)
    parser.add_argument("--output-dir", type=Path, default=EHCVM_REPORTS_DIR)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    path = (args.archive or default_archive()).expanduser().resolve()
    if args.dry_run:
        members = check_archive(path)
        print(f"Archive valide : {path} ({len(members)} CSV). Aucun rapport écrit.")
        return
    results = audit(path)
    destination = args.output_dir.expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    report = destination / "audit_structure.json"
    report.write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n")
    print(f"Audit terminé : {results['ehcvm_welfare']['rows']} ménages, {results['ehcvm_welfare']['grappes']} grappes.")
    print(report)


if __name__ == "__main__":
    main()
